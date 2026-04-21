"""
Optimizador de cortes de vencimiento para buckets de opciones SPX
=================================================================
Maximiza cobertura de strikes penalizando violaciones de butterfly
y calendar spread arbitrage sobre toda la muestra.

Función objetivo (por bucket, promediada):
    score = w_cov1 * norm_strikes + w_cov2 * pct_days_min5
            - lambda1 * pct_butterfly_violations
            - lambda2 * pct_calendar_violations

DISEÑO VECTORIZADO: las métricas se precomputan una vez sobre el
DataFrame completo agregado a nivel (date, T_days), de forma que
la búsqueda exhaustiva solo requiere slicing sobre arrays numpy.
Esto hace viable correr el optimizador sobre la muestra completa
de 34M filas en tiempo razonable.

Uso:
    df = pd.read_parquet("options_data.parquet")
    result = optimize_buckets(df)
    print(result)
"""

import numpy as np
import pandas as pd
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

opt_df = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet")

opt_df_filtered2 = opt_df[opt_df["Date"] >= pd.to_datetime("2019-01-01")]

opt_df_filtered2c = opt_df_filtered2[opt_df_filtered2["CallPut"] == "C"]

opt_df_filtered2p = opt_df_filtered2[opt_df_filtered2["CallPut"] == "P"]



# ─────────────────────────────────────────────
# 1. PREPARACIÓN Y PRECOMPUTO DE MÉTRICAS
# ─────────────────────────────────────────────

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adapta el DataFrame de OptionMetrics al formato interno del optimizador.

    Columnas de entrada esperadas (nomenclatura real del pipeline):
      Date             : fecha de negociación (datetime)
      Strike           : strike ya dividido por 1000
      Days             : días a vencimiento (ajustado por AMSettlement)
      ImpliedVolatility: volatilidad implícita en decimales (e.g. 0.20)
      log_moneyness    : log(Strike/SpotPrice) ya calculado en el pipeline

    Columnas de salida (nombres internos del optimizador):
      date, K, T_days, w, k
    """
    df = df.copy()

    # Varianza total: sigma^2 * T_años (base 252 días hábiles)
    df["T_years"] = df["Days"].astype("float64") / 252.0
    df["w"] = df["ImpliedVolatility"].astype("float64") ** 2 * df["T_years"]

    # Renombrar a nombres internos
    df = df.rename(columns={
        "Date":   "date",
        "Strike": "K",
        "Days":   "T_days",
    })

    # log-moneyness: reutilizar el que ya tienes calculado
    df["k"] = df["log_moneyness"].astype("float64")

    # Filtros de seguridad mínimos
    df = df[(df["T_days"] > 0) & (df["w"] > 0) & (df["ImpliedVolatility"] > 0)]

    return df[["date", "K", "T_days", "w", "k"]].copy()


def _butterfly_viol_group(sub: pd.DataFrame) -> float:
    """
    Segunda diferencia dividida de w(k) para un grupo (date, T_days).
    Retorna fracción de nodos interiores con SDD < 0.
    """
    s = sub.sort_values("k").drop_duplicates("k")
    if len(s) < 3:
        return np.nan
    ks = s["k"].values
    ws = s["w"].values
    ddr = np.diff(ws) / np.diff(ks)           # n-1 diferencias derechas
    ddl = ddr[:-1]                              # izquierdas = derechas desplazadas
    ddr = ddr[1:]
    sdd = (ddr - ddl) / (ks[2:] - ks[:-2])    # segunda diferencia dividida
    return float(np.mean(sdd < 0)) if len(sdd) > 0 else 0.0


def _calendar_viol_group(sub: pd.DataFrame) -> float:
    """
    Fracción de pares de vencimientos consecutivos donde
    mediana(w) decrece dentro de un día.
    """
    med = sub.groupby("T_days")["w"].median().sort_index()
    if len(med) < 2:
        return 0.0
    return float(np.mean(np.diff(med.values) < 0))


def precompute_daily_metrics(df: pd.DataFrame,
                              min_strikes: int = 5) -> pd.DataFrame:
    """
    Precomputa métricas por (date, T_days) una sola vez.
    La búsqueda de cortes operará sobre este DataFrame agregado,
    que es mucho más pequeño que el original.

    Columnas resultado:
      date, T_days, n_strikes, has_min, butterfly_viol, calendar_viol
    """
    print("  Precomputando métricas diarias por vencimiento...")

    # ── Cobertura: número de strikes distintos por (date, T_days) ────────
    coverage = (df.groupby(["date", "T_days"])["K"]
                  .nunique()
                  .reset_index()
                  .rename(columns={"K": "n_strikes"}))
    coverage["has_min"] = (coverage["n_strikes"] >= min_strikes).astype(int)

    # ── Butterfly: por (date, T_days) ────────────────────────────────────
    bfly = (df.groupby(["date", "T_days"])
              .apply(_butterfly_viol_group)
              .reset_index()
              .rename(columns={0: "butterfly_viol"}))

    # ── Calendar: por date (necesita todos los T del día) ────────────────
    cal = (df.groupby("date")
             .apply(_calendar_viol_group)
             .reset_index()
             .rename(columns={0: "calendar_viol"}))

    # ── Merge ─────────────────────────────────────────────────────────────
    metrics = coverage.merge(bfly, on=["date", "T_days"], how="left")
    metrics = metrics.merge(cal, on="date", how="left")

    print(f"  Métricas precomputadas: {len(metrics):,} filas "
          f"({metrics['date'].nunique():,} días, "
          f"{metrics['T_days'].nunique()} vencimientos únicos)")
    return metrics


# ─────────────────────────────────────────────
# 2. FUNCIÓN OBJETIVO (VECTORIZADA)
# ─────────────────────────────────────────────

def score_cuts(cuts: tuple,
               metrics: pd.DataFrame,
               lambda1: float = 1.0,
               lambda2: float = 0.5,
               lambda3: float = 0.5,
               w_cov1: float = 1.0,
               w_cov2: float = 1.0,
               max_strikes: float = 50.0,
               min_last_cut: int = 180) -> float:
    """
    Calcula el score de una configuración de cortes.

    Parámetros
    ----------
    cuts        : tupla de enteros, cortes en días (ordenados)
    metrics     : DataFrame de precompute_daily_metrics
    lambda1     : peso penalización butterfly
    lambda2     : peso penalización calendar spread
    lambda3     : peso penalización desequilibrio de tamaño entre buckets
                  Evita que el algoritmo concentre los cortes en una zona
                  y deje un bucket residual enorme. Típicamente 0.3–1.0.
    w_cov1      : peso cobertura (mean_strikes normalizado)
    w_cov2      : peso cobertura (pct_days >= min_strikes)
    max_strikes : referencia para normalizar mean_strikes

    Retorna
    -------
    score : float (mayor es mejor)
    """
    # Restricción: el último corte debe superar min_last_cut días.
    # Evita que todos los cortes se concentren en vencimientos cortos
    # dejando un bucket residual enorme sin subdividir.
    if max(cuts) < min_last_cut:
        return -99.0

    cuts_full = [0] + list(cuts) + [np.inf]
    bucket_scores = []
    bucket_sizes  = []

    T       = metrics["T_days"].values
    n_total = len(metrics)

    for i in range(len(cuts_full) - 1):
        lo, hi = cuts_full[i], cuts_full[i+1]
        mask = (T > lo) & (T <= hi)
        sub  = metrics[mask]
        bucket_sizes.append(len(sub))

        if len(sub) == 0:
            bucket_scores.append(-10.0)
            continue

        mean_strikes = sub["n_strikes"].mean()
        pct_min      = sub["has_min"].mean()
        bfly_viol    = sub["butterfly_viol"].dropna().mean()
        cal_viol     = sub["calendar_viol"].mean()
        norm_strikes = min(mean_strikes / max_strikes, 1.0)

        s = (w_cov1 * norm_strikes
             + w_cov2 * pct_min
             - lambda1 * bfly_viol
             - lambda2 * cal_viol)
        bucket_scores.append(s)

    # Penalización por desequilibrio: std de proporciones por bucket.
    # Distribución uniforme → balance=0. Bucket residual enorme → balance~0.4.
    proportions = [n / n_total for n in bucket_sizes]
    balance = float(np.std(proportions))

    return float(np.mean(bucket_scores)) - lambda3 * balance


# ─────────────────────────────────────────────
# 3. OPTIMIZADOR PRINCIPAL
# ─────────────────────────────────────────────

def optimize_buckets(df: pd.DataFrame,
                     candidate_cuts: list = None,
                     n_cuts: int = 6,
                     lambda1: float = 1.0,
                     lambda2: float = 0.5,
                     lambda3: float = 0.5,
                     w_cov1: float = 1.0,
                     w_cov2: float = 1.0,
                     min_strikes: int = 5,
                     top_k: int = 10,
                     min_last_cut: int = 180) -> pd.DataFrame:
    """
    Optimiza los cortes de vencimiento por búsqueda exhaustiva
    sobre un conjunto discreto de candidatos.

    Estrategia de eficiencia
    ------------------------
    1. Precomputa métricas por (date, T_days) una sola vez → O(N)
    2. Búsqueda exhaustiva opera solo sobre el DataFrame agregado
       (típicamente ~100K filas vs 34M del original) → O(C * M)
       donde C = combinaciones candidatas, M = filas agregadas

    Parámetros
    ----------
    df              : DataFrame raw, una fila por opción
    candidate_cuts  : cortes candidatos en días
    n_cuts          : número de cortes (buckets = n_cuts + 1). Default=6 → 7 buckets.
    lambda1         : penalización butterfly
    lambda2         : penalización calendar spread
    w_cov1          : peso mean_strikes
    w_cov2          : peso pct_days_min
    min_strikes     : mínimo strikes para "día cubierto"
    top_k           : mejores configuraciones a retornar
    *_col           : nombres de columnas en df

    Retorna
    -------
    DataFrame con las top_k configuraciones y sus scores
    """
    print("=" * 60)
    print("OPTIMIZADOR DE CORTES DE VENCIMIENTO")
    print("=" * 60)

    # 1. Preparar datos
    print("\n[1/3] Preparando datos...")
    df_prep = prepare_data(df)

    # 2. Precomputar métricas
    print("\n[2/3] Precomputando métricas...")
    metrics = precompute_daily_metrics(df_prep, min_strikes=min_strikes)

    # Normalización de strikes basada en percentil 90 observado
    max_strikes = np.percentile(metrics["n_strikes"].values, 90)
    print(f"  Referencia max_strikes (p90): {max_strikes:.1f}")

    # 3. Búsqueda exhaustiva
    if candidate_cuts is None:
        candidate_cuts = [7, 10, 14, 21, 30, 45, 60, 90,
                          120, 150, 180, 210, 252, 300, 365, 450, 520]

    max_T = int(metrics["T_days"].max())
    candidate_cuts = sorted([c for c in candidate_cuts if c < max_T])

    all_combos = list(combinations(candidate_cuts, n_cuts))
    n_total = len(all_combos)

    print(f"\n[3/3] Búsqueda exhaustiva...")
    print(f"  Candidatos: {candidate_cuts}")
    print(f"  Combinaciones: {n_total:,}")
    print(f"  Parámetros: λ1={lambda1}, λ2={lambda2}, λ3={lambda3}, "
          f"w1={w_cov1}, w2={w_cov2}, min_last_cut={min_last_cut}\n")

    scores = []
    for idx, cuts in enumerate(all_combos):
        if idx % max(1, n_total // 20) == 0:
            pct = 100 * idx / n_total
            print(f"  [{pct:5.1f}%] {idx:,}/{n_total:,} combinaciones...")
        s = score_cuts(cuts, metrics,
                       lambda1=lambda1, lambda2=lambda2,
                       lambda3=lambda3,
                       w_cov1=w_cov1, w_cov2=w_cov2,
                       max_strikes=max_strikes,
                       min_last_cut=min_last_cut)
        scores.append(s)

    # Ordenar y formatear resultados
    order = np.argsort(scores)[::-1]
    top_combos = [all_combos[i] for i in order[:top_k]]
    top_scores = [scores[i] for i in order[:top_k]]

    rows = []
    for cuts, sc in zip(top_combos, top_scores):
        row = {f"cut_{i+1}": c for i, c in enumerate(cuts)}
        row["score"] = sc
        # Añadir etiquetas de buckets legibles
        cuts_full = [0] + list(cuts) + ["∞"]
        row["buckets"] = " | ".join(
            f"({cuts_full[i]},{cuts_full[i+1]}]"
            for i in range(len(cuts_full)-1)
        )
        rows.append(row)

    results_df = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print(f"TOP {top_k} CONFIGURACIONES")
    print(f"{'='*60}")
    print(results_df[[c for c in results_df.columns
                       if c != "buckets"]].to_string(index=True))

    return results_df


# ─────────────────────────────────────────────
# 4. EVALUACIÓN DETALLADA DE UNA CONFIGURACIÓN
# ─────────────────────────────────────────────

def evaluate_configuration(df: pd.DataFrame,
                            cuts: tuple,
                            lambda1: float = 1.0,
                            lambda2: float = 0.5,
                            lambda3: float = 0.5,
                            w_cov1: float = 1.0,
                            w_cov2: float = 1.0,
                            min_strikes: int = 5) -> pd.DataFrame:
    """
    Métricas detalladas para una configuración específica.
    Útil para comparar la configuración óptima con la intuitiva.
    """
    df_prep = prepare_data(df)
    metrics = precompute_daily_metrics(df_prep, min_strikes=min_strikes)
    max_strikes = np.percentile(metrics["n_strikes"].values, 90)

    cuts_full = [0] + list(cuts) + [np.inf]
    T = metrics["T_days"].values
    rows = []

    for i in range(len(cuts_full) - 1):
        lo, hi = cuts_full[i], cuts_full[i+1]
        mask = (T > lo) & (T <= hi)
        sub = metrics[mask]
        if len(sub) == 0:
            continue

        mean_str   = sub["n_strikes"].mean()
        pct_min    = sub["has_min"].mean()
        bfly       = sub["butterfly_viol"].dropna().mean()
        cal        = sub["calendar_viol"].mean()
        norm_str   = min(mean_str / max_strikes, 1.0)
        sc = (w_cov1*norm_str + w_cov2*pct_min
              - lambda1*bfly - lambda2*cal)

        label = f"({lo:.0f}, {hi if hi != np.inf else '∞'}]"
        rows.append({
            "bucket":          label,
            "n_obs":           len(sub),
            "mean_strikes":    round(mean_str, 1),
            "pct_days_min5":   f"{pct_min:.1%}",
            "butterfly_viol":  f"{bfly:.3f}",
            "calendar_viol":   f"{cal:.3f}",
            "bucket_score":    round(sc, 4),
        })

    result = pd.DataFrame(rows)
    total_score = score_cuts(cuts, metrics,
                              lambda1=lambda1, lambda2=lambda2,
                              lambda3=lambda3,
                              w_cov1=w_cov1, w_cov2=w_cov2,
                              max_strikes=max_strikes)
    print(f"\nCortes: {cuts}  →  Score global: {total_score:.4f}")
    print(result.to_string(index=False))
    return result


# ─────────────────────────────────────────────
# 5. ANÁLISIS DE SENSIBILIDAD
# ─────────────────────────────────────────────

def sensitivity_analysis(df: pd.DataFrame,
                          candidate_cuts: list = None,
                          n_cuts: int = 6,
                          lambda1_grid: list = None,
                          lambda2_grid: list = None,
                          min_strikes: int = 5) -> pd.DataFrame:
    """
    Analiza si los cortes óptimos cambian al variar lambda1/lambda2.
    Precomputa métricas una sola vez y reutiliza en todo el grid.
    """
    df_prep = prepare_data(df)
    metrics = precompute_daily_metrics(df_prep, min_strikes=min_strikes)
    max_strikes = np.percentile(metrics["n_strikes"].values, 90)

    if candidate_cuts is None:
        candidate_cuts = [7, 10, 14, 21, 30, 45, 60, 90,
                          120, 150, 180, 210, 252, 300, 365]
    candidate_cuts = sorted([c for c in candidate_cuts
                              if c < metrics["T_days"].max()])
    all_combos = list(combinations(candidate_cuts, n_cuts))

    if lambda1_grid is None:
        lambda1_grid = [0.5, 1.0, 1.5, 2.0]
    if lambda2_grid is None:
        lambda2_grid = [0.25, 0.5, 1.0, 1.5]

    rows = []
    total = len(lambda1_grid) * len(lambda2_grid)
    idx = 0

    for l1 in lambda1_grid:
        for l2 in lambda2_grid:
            idx += 1
            print(f"  Sensibilidad {idx}/{total}: λ1={l1}, λ2={l2}")
            best_sc = -np.inf
            best_c  = None
            for cuts in all_combos:
                s = score_cuts(cuts, metrics,
                               lambda1=l1, lambda2=l2,
                               lambda3=0.5,
                               max_strikes=max_strikes)
                if s > best_sc:
                    best_sc = s
                    best_c  = cuts
            row = {"lambda1": l1, "lambda2": l2, "score": best_sc}
            row.update({f"cut_{i+1}": c for i, c in enumerate(best_c)})
            rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 6. EJEMPLO DE USO
# ─────────────────────────────────────────────

if __name__ == "__main__":

    # ── Carga tus datos ──────────────────────────────────────────────────
    # Asegúrate de que opt_df_prueba tiene estas columnas antes de llamar:
    #   Date             : datetime
    #   Strike           : strike / 1000 (ya dividido)
    #   Days             : días a vencimiento (ajustado por AMSettlement)
    #   ImpliedVolatility: vol implícita en decimales
    #   log_moneyness    : log(Strike / SpotPrice)
    #
    # Ejemplo de uso real:
    #
      top = optimize_buckets(opt_df_filtered2p, lambda1=1.0, lambda2=0.5)
      best_cuts = tuple(top.iloc[0][[f"cut_{i+1}" for i in range(6)]].astype(int))
      evaluate_configuration(opt_df_filtered2p, best_cuts)
      evaluate_configuration(opt_df_filtered2p, (15, 45, 105, 183))


    # ────────────────────────────────────────────────────────────────────

    # ── Datos sintéticos que replican tu estructura real ─────────────────
    # np.random.seed(42)
    # n = 200_000
    # dates = pd.date_range("2003-01-01", "2024-12-31", freq="B")
    # T_choices = [5, 8, 12, 14, 21, 30, 45, 60, 90,
    #              120, 150, 180, 252, 365, 450, 520]
    # T_probs = [0.04, 0.04, 0.04, 0.04, 0.08, 0.12, 0.12,
    #            0.08, 0.08, 0.07, 0.06, 0.06, 0.06, 0.05,
    #            0.03, 0.03]

    # S = 4500.0
    # strikes = np.random.choice(np.arange(3000, 6500, 25), n)

    # df_test = pd.DataFrame({
    #     "Date":             np.random.choice(dates, n),
    #     "Strike":           strikes / 1000,           # ya dividido por 1000
    #     "Days":             np.random.choice(T_choices, n, p=T_probs),
    #     "ImpliedVolatility":np.random.uniform(0.10, 0.45, n),
    #     "log_moneyness":    np.log(strikes / (S * 1000)),
    # })

    # # ── Optimización ─────────────────────────────────────────────────────
    # top = optimize_buckets(
    #     df_test,
    #     candidate_cuts=[7, 14, 21, 30, 45, 60, 90, 120, 180, 252, 365],
    #     n_cuts=6,
    #     lambda1=1.0,
    #     lambda2=0.5,
    #     lambda3=0.5,   # penalización desequilibrio de buckets
    #     w_cov1=1.0,
    #     w_cov2=1.0,
    #     min_strikes=5,
    #     top_k=10,
    # )

    # # ── Comparación: óptima vs intuitiva ─────────────────────────────────
    # best_cuts      = tuple(top.iloc[0][[f"cut_{i+1}"
    #                                     for i in range(6)]].astype(int))
    # intuitive_cuts = (15, 45, 105, 183, 365)

    # print("\n\n── CONFIGURACIÓN INTUITIVA ─────────────────────────────")
    # evaluate_configuration(df_test, intuitive_cuts)

    # print("\n── CONFIGURACIÓN ÓPTIMA ────────────────────────────────")
    # evaluate_configuration(df_test, best_cuts)

    # # ── Sensibilidad ─────────────────────────────────────────────────────
    # print("\n── ANÁLISIS DE SENSIBILIDAD ────────────────────────────")
    # sens = sensitivity_analysis(
    #     df_test,
    #     candidate_cuts=[7, 14, 21, 30, 45, 60, 90, 120, 180, 252, 365],
    #     n_cuts=6,
    #     lambda1_grid=[0.5, 1.0, 2.0],
    #     lambda2_grid=[0.25, 0.5, 1.0],
    # )
    # print(sens.to_string(index=False))