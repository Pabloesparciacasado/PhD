# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS COMPARATIVO EXHAUSTIVO DE MÉTODOS DE CÁLCULO DE GREEKS
# Punto de partida: opt_df_filtered
# Métodos: (1) Bucket, (2) Vencimiento exacto, (3) Delta empírica temporal
# Métricas: NaN, gamma negativa, delta fuera de rango, cobertura de moneyness
# VERSIÓN CORREGIDA Y ROBUSTA
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

INPUT_PATH = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"
OUTPUT_FIG = r"C:\Users\pablo.esparcia\Documents\comparacion_exhaustiva_metodos.png"

MIN_STRIKES = 3          # mínimo de strikes por grupo para derivadas transversales
MIN_DS_ABS = 0.001           # umbral absoluto mínimo para método temporal
USE_RELATIVE_DS = False      # si True, usa umbral relativo además del absoluto
MIN_DS_REL = 0.005           # 0.5% del spot previo si USE_RELATIVE_DS = True

V_GRID = [0, 15, 45, 105, 183, 365, np.inf]
V_EDGES = pd.IntervalIndex.from_breaks(V_GRID, closed="right")

MONEYNESS_BINS = [0, 0.70, 0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 1.30, np.inf]
MONEYNESS_LABELS = [
    "<0.70", "0.70-0.80", "0.80-0.90", "0.90-0.95",
    "0.95-1.05", "1.05-1.10", "1.10-1.20", "1.20-1.30", ">1.30"
]

METHOD_NAMES = ["Bucket", "Venc. Exacto", "Delta Empírica"]
METHOD_COLORS = ["#D85A30", "#378ADD", "#1D9E75"]


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

def check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el parquet: {missing}")


def safe_divide(num, den):
    """
    División segura: devuelve NaN cuando el denominador es 0 o no finito.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def assign_bucket(days_series: pd.Series) -> pd.Series:
    return pd.cut(days_series, bins=V_EDGES, labels=False, include_lowest=True)


def assign_moneyness_bin(m_series: pd.Series) -> pd.Series:
    return pd.cut(
        m_series,
        bins=MONEYNESS_BINS,
        labels=MONEYNESS_LABELS,
        include_lowest=True
    )


def filter_groups_min_strikes(df: pd.DataFrame, group_cols: list[str], min_strikes: int) -> pd.DataFrame:
    """
    Conserva solo grupos con al menos min_strikes strikes distintos.
    """
    n_strikes = df.groupby(group_cols)["Strike"].transform("nunique")
    return df[n_strikes >= min_strikes].copy()


def compute_cross_sectional_greeks(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """
    Calcula Delta y Gamma a partir de diferencias finitas centradas
    sobre la dimensión strike dentro de cada grupo.
    
    IMPORTANTE:
    - Este enfoque es razonable solo si dentro del grupo el vencimiento es homogéneo.
    - En método bucket sigue habiendo mezcla de Days dentro del bucket; se mantiene
      porque forma parte del diseño comparativo, pero debe interpretarse con cautela.
    """
    df = df.copy()

    # Orden estable dentro de grupo
    df = df.sort_values(group_cols + ["Strike"]).reset_index(drop=True)

    g = df.groupby(group_cols, sort=False)

    C_next = g["MidPrice"].shift(-1)
    C_prev = g["MidPrice"].shift(1)
    K_next = g["Strike"].shift(-1)
    K_prev = g["Strike"].shift(1)

    h_L = df["Strike"] - K_prev
    h_R = K_next - df["Strike"]
    denom_d1 = K_next - K_prev

    # Primera derivada wrt strike
    df["dc_dk"] = safe_divide(C_next - C_prev, denom_d1)

    # Fórmula usada en tu script original:
    # Delta = (C - K*dC/dK) / S
    # Se mantiene para conservar la lógica comparativa del script.
    df["Delta"] = safe_divide(
        df["MidPrice"] - df["dc_dk"] * df["Strike"],
        df["SpotPrice"]
    )

    # Segunda derivada wrt strike, esquema no uniforme
    term_r = safe_divide(
        C_next - df["MidPrice"],
        h_R * (h_L + h_R)
    )
    term_l = safe_divide(
        C_prev - df["MidPrice"],
        h_L * (h_L + h_R)
    )

    df["d2c_dk2"] = 2.0 * (term_r + term_l)

    # Gamma transformada como en tu script original
    df["Gamma"] = ((df["Strike"] / df["SpotPrice"]) ** 2) * df["d2c_dk2"]

    # Invalidar explícitamente observaciones donde no hay vecindad completa
    invalid_neighbors = (
        K_prev.isna() | K_next.isna() |
        (h_L <= 0) | (h_R <= 0) |
        ~np.isfinite(df["SpotPrice"]) | (df["SpotPrice"] <= 0)
    )

    df.loc[invalid_neighbors, ["dc_dk", "Delta", "d2c_dk2", "Gamma"]] = np.nan

    return df


def compute_temporal_greeks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Delta empírica temporal y una gamma empírica muy ruidosa.
    Se conserva por comparabilidad, pero debe interpretarse con cautela.
    """
    df = df.copy().sort_values(["OptionID", "Date"]).reset_index(drop=True)

    g = df.groupby("OptionID", sort=False)

    df["dC"] = g["MidPrice"].diff()
    df["dS"] = g["SpotPrice"].diff()
    df["SpotPrice_lag"] = g["SpotPrice"].shift(1)

    if USE_RELATIVE_DS:
        min_move_mask = (
            df["dS"].abs() >= MIN_DS_ABS
        ) | (
            safe_divide(df["dS"].abs(), df["SpotPrice_lag"]) >= MIN_DS_REL
        )
    else:
        min_move_mask = df["dS"].abs() >= MIN_DS_ABS

    # Delta temporal
    df["Delta"] = np.where(
        min_move_mask,
        safe_divide(df["dC"], df["dS"]),
        np.nan
    )

    # Gamma temporal empírica
    df["d2C"] = g["MidPrice"].diff().diff()
    df["dS_lag"] = g["SpotPrice"].diff().shift(1)

    if USE_RELATIVE_DS:
        min_move_mask_lag = (
            df["dS_lag"].abs() >= MIN_DS_ABS
        ) | (
            safe_divide(df["dS_lag"].abs(), g["SpotPrice"].shift(2)) >= MIN_DS_REL
        )
    else:
        min_move_mask_lag = df["dS_lag"].abs() >= MIN_DS_ABS

    gamma_mask = (
        min_move_mask &
        min_move_mask_lag &
        np.isfinite(df["dS"]) &
        np.isfinite(df["dS_lag"]) &
        (df["dS"] != 0) &
        (df["dS_lag"] != 0) &
        (df["dS"] * df["dS_lag"] > 0)
    )

    df["Gamma"] = np.where(
        gamma_mask,
        safe_divide(df["d2C"], df["dS"] * df["dS_lag"]),
        np.nan
    )

    return df


def diagnostico_metodo(df: pd.DataFrame, nombre: str,
                       delta_col: str = "Delta",
                       gamma_col: str = "Gamma") -> dict:
    """
    Calcula métricas de diagnóstico para un método dado.
    """

    n_total = len(df)

    # NaN
    nan_delta = df[delta_col].isna().sum()
    nan_gamma = df[gamma_col].isna().sum()
    nan_ambos = (df[delta_col].isna() & df[gamma_col].isna()).sum()

    # Subsets válidos
    valid_delta = df[df[delta_col].notna() & np.isfinite(df[delta_col])].copy()
    valid_gamma = df[df[gamma_col].notna() & np.isfinite(df[gamma_col])].copy()

    n_valid_delta = len(valid_delta)
    n_valid_gamma = len(valid_gamma)

    # Violaciones delta
    viol_call = (
        (valid_delta["CallPut"] == "C") &
        (~valid_delta[delta_col].between(0, 1))
    ).sum()

    viol_put = (
        (valid_delta["CallPut"] == "P") &
        (~valid_delta[delta_col].between(-1, 0))
    ).sum()

    # Gamma negativa
    gamma_neg = (valid_gamma[gamma_col] < 0).sum()

    # Distribución gamma positiva
    gamma_pos_valid = valid_gamma.loc[valid_gamma[gamma_col] >= 0, gamma_col]

    # Cobertura por moneyness: DELTA
    mon_total = (
        df.groupby("Moneyness_bin", observed=True)
        .size()
        .reindex(MONEYNESS_LABELS, fill_value=0)
    )

    mon_cob_delta = (
        valid_delta.groupby("Moneyness_bin", observed=True)
        .size()
        .reindex(MONEYNESS_LABELS, fill_value=0)
    )

    mon_cob_gamma = (
        valid_gamma.groupby("Moneyness_bin", observed=True)
        .size()
        .reindex(MONEYNESS_LABELS, fill_value=0)
    )

    mon_cob_delta_pct = (mon_cob_delta / mon_total.replace(0, np.nan)).fillna(0)
    mon_cob_gamma_pct = (mon_cob_gamma / mon_total.replace(0, np.nan)).fillna(0)

    # Cobertura por bucket
    total_bucket = df.groupby("bucket").size()

    cob_bucket_delta = (
        valid_delta.groupby("bucket").size()
        .div(total_bucket)
        .fillna(0)
    )

    cob_bucket_gamma = (
        valid_gamma.groupby("bucket").size()
        .div(total_bucket)
        .fillna(0)
    )

    return {
        "nombre": nombre,
        "n_total": n_total,

        "nan_delta_n": nan_delta,
        "nan_delta_pct": nan_delta / n_total if n_total > 0 else np.nan,
        "nan_gamma_n": nan_gamma,
        "nan_gamma_pct": nan_gamma / n_total if n_total > 0 else np.nan,
        "nan_ambos_pct": nan_ambos / n_total if n_total > 0 else np.nan,

        "n_valid_delta": n_valid_delta,
        "n_valid_gamma": n_valid_gamma,

        "viol_call_n": viol_call,
        "viol_call_pct": viol_call / n_valid_delta if n_valid_delta > 0 else np.nan,
        "viol_put_n": viol_put,
        "viol_put_pct": viol_put / n_valid_delta if n_valid_delta > 0 else np.nan,
        "viol_delta_total": (viol_call + viol_put) / n_valid_delta if n_valid_delta > 0 else np.nan,

        "gamma_neg_n": gamma_neg,
        "gamma_neg_pct": gamma_neg / n_valid_gamma if n_valid_gamma > 0 else np.nan,

        "gamma_p50": gamma_pos_valid.median() if len(gamma_pos_valid) > 0 else np.nan,
        "gamma_p95": gamma_pos_valid.quantile(0.95) if len(gamma_pos_valid) > 0 else np.nan,
        "gamma_std": gamma_pos_valid.std() if len(gamma_pos_valid) > 0 else np.nan,

        "cob_moneyness_delta": mon_cob_delta_pct,
        "cob_moneyness_gamma": mon_cob_gamma_pct,

        "cob_bucket_delta": cob_bucket_delta,
        "cob_bucket_gamma": cob_bucket_gamma,
    }


def yearly_negative_gamma_series(df: pd.DataFrame) -> pd.Series:
    """
    Serie anual del % de gamma negativa sobre gammas válidas.
    """
    tmp = df.copy()
    tmp = tmp[tmp["Date"].notna()].copy()

    if tmp.empty:
        return pd.Series(dtype=float)

    tmp["year"] = tmp["Date"].dt.year

    def pct_neg(x):
        valid = x["Gamma"][x["Gamma"].notna() & np.isfinite(x["Gamma"])]
        if len(valid) == 0:
            return np.nan
        return (valid < 0).mean() * 100

    return tmp.groupby("year", observed=True).apply(pct_neg)


# ═══════════════════════════════════════════════════════════════════════════════
# CARGA Y PREPARACIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("ANÁLISIS COMPARATIVO DE MÉTODOS DE CÁLCULO DE GREEKS")
print("=" * 80)

opt_df = pd.read_parquet(INPUT_PATH)

required_columns = [
    "Date", "OptionID", "CallPut", "Strike", "Days",
    "MidPrice", "SpotPrice", "Moneyness",
    "OpenInterest", "Volume", "Bid"
]
check_required_columns(opt_df, required_columns)

# Tipos robustos
opt_df = opt_df.copy()
opt_df["Date"] = pd.to_datetime(opt_df["Date"], errors="coerce")

numeric_cols = [
    "Strike", "Days", "MidPrice", "SpotPrice",
    "Moneyness", "OpenInterest", "Volume", "Bid"
]
for col in numeric_cols:
    opt_df[col] = pd.to_numeric(opt_df[col], errors="coerce")

# Filtrado CORREGIDO:
# antes se sobrescribía el primer filtro; ahora se aplican ambos.
opt_df_filtered = opt_df[
    (
        ((opt_df["OpenInterest"] > 0) | (opt_df["Volume"] > 0)) &
        (opt_df["Bid"] > 0)
    ) &
    opt_df["Date"].notna() &
    opt_df["Strike"].notna() &
    opt_df["Days"].notna() &
    opt_df["MidPrice"].notna() &
    opt_df["SpotPrice"].notna() &
    (opt_df["SpotPrice"] > 0) &
    opt_df["Moneyness"].notna() &
    opt_df["CallPut"].isin(["C", "P"])
].reset_index(drop=True)

print(f"Filas totales en origen:     {len(opt_df):,}")
print(f"Filas tras filtrado inicial: {len(opt_df_filtered):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTODO 1: AGRUPACIÓN POR BUCKET DE VENCIMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("MÉTODO 1: AGRUPACIÓN POR BUCKET")
print("─" * 80)

# opt_m1 = opt_df_filtered.copy()
# opt_m1["bucket"] = assign_bucket(opt_m1["Days"])

# # Deduplicar por bucket: conservar strike más líquido
# opt_m1 = (
#     opt_m1.sort_values(
#         ["Date", "bucket", "CallPut", "Strike", "OpenInterest", "Volume"],
#         ascending=[True, True, True, True, False, False]
#     )
#     .drop_duplicates(subset=["Date", "bucket", "CallPut", "Strike"])
#     .reset_index(drop=True)
# )

# # Exigir mínimo de strikes por grupo
# opt_m1 = filter_groups_min_strikes(opt_m1, ["Date", "bucket", "CallPut"], MIN_STRIKES)

# # Greeks
# opt_m1 = compute_cross_sectional_greeks(opt_m1, ["Date", "bucket", "CallPut"])
# opt_m1["Moneyness_bin"] = assign_moneyness_bin(opt_m1["Moneyness"])

# print(f"Filas totales M1: {len(opt_m1):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# MÉTODO 2: VENCIMIENTO EXACTO
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("MÉTODO 2: VENCIMIENTO EXACTO")
print("─" * 80)

opt_m2 = (
    opt_df_filtered.sort_values(
        ["Date", "Days", "CallPut", "Strike", "OpenInterest", "Volume"],
        ascending=[True, True, True, True, False, False]
    )
    .drop_duplicates(subset=["Date", "Days", "CallPut", "Strike"])
    .reset_index(drop=True)
)

opt_m2["bucket"] = assign_bucket(opt_m2["Days"])

# Exigir mínimo de strikes por grupo
opt_m2 = filter_groups_min_strikes(opt_m2, ["Date", "Days", "CallPut"], MIN_STRIKES)

# Greeks
opt_m2 = compute_cross_sectional_greeks(opt_m2, ["Date", "Days", "CallPut"])
opt_m2["Moneyness_bin"] = assign_moneyness_bin(opt_m2["Moneyness"])

print(f"Filas totales M2: {len(opt_m2):,}")


#######################################################################################################
#######################################################################################################
# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS DE LIQUIDEZ Y CALIDAD DE GREEKS (MÉTODO VENCIMIENTO EXACTO)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("ANÁLISIS DE LIQUIDEZ Y SU IMPACTO EN LA CALIDAD DE LOS GREEKS")
print("="*80)

opt_liq = opt_m2.copy()

# ─────────────────────────────────────────────────────────────────────────────
# 1) DEFINICIÓN DE LIQUIDEZ
# ─────────────────────────────────────────────────────────────────────────────

opt_liq["Liquidity"] = opt_liq["OpenInterest"] + opt_liq["Volume"]*0

# Evitar valores extremos absurdos (opcional pero recomendable)
opt_liq = opt_liq[np.isfinite(opt_liq["Liquidity"])]

# ─────────────────────────────────────────────────────────────────────────────
# 2) DISTRIBUCIÓN DE LIQUIDEZ
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "-"*80)
print("DISTRIBUCIÓN DE LIQUIDEZ")
print("-"*80)

liq_bucket = (opt_liq
    .groupby("bucket")
    .agg(
        n_obs=("Liquidity", "size"),
        liq_mean=("Liquidity", "mean"),
        liq_median=("Liquidity", "median"),
        liq_p10=("Liquidity", lambda x: x.quantile(0.1)),
        liq_p90=("Liquidity", lambda x: x.quantile(0.9))
    )
)

print("\nLIQUIDEZ POR BUCKET")
print(liq_bucket)

liq_moneyness = (opt_liq
    .groupby("Moneyness_bin")
    .agg(
        n_obs=("Liquidity", "size"),
        liq_mean=("Liquidity", "mean"),
        liq_median=("Liquidity", "median"),
        liq_p10=("Liquidity", lambda x: x.quantile(0.1)),
        liq_p90=("Liquidity", lambda x: x.quantile(0.9))
    )
    .reindex(MONEYNESS_LABELS)
)

print("\nLIQUIDEZ POR MONEYNESS")
print(liq_moneyness)

# ─────────────────────────────────────────────────────────────────────────────
# 3) BINS DE LIQUIDEZ
# ─────────────────────────────────────────────────────────────────────────────

opt_liq["liq_bin"] = pd.qcut(opt_liq["Liquidity"], q=5, duplicates="drop")

# ─────────────────────────────────────────────────────────────────────────────
# 4) MÉTRICAS DE CALIDAD
# ─────────────────────────────────────────────────────────────────────────────

def metricas_liquidez(df):
    
    valid_delta = df["Delta"].notna() & np.isfinite(df["Delta"])
    valid_gamma = df["Gamma"].notna() & np.isfinite(df["Gamma"])

    delta_viol = (
        ((df["CallPut"]=="C") & (~df["Delta"].between(0,1))) |
        ((df["CallPut"]=="P") & (~df["Delta"].between(-1,0)))
    )

    gamma_neg = df["Gamma"] < 0

    return pd.Series({
        "n_obs": len(df),
        "NaN_Delta_%": df["Delta"].isna().mean(),
        "NaN_Gamma_%": df["Gamma"].isna().mean(),
        "Delta_fuera_rango_%": (
            (delta_viol & valid_delta).sum() / valid_delta.sum()
            if valid_delta.sum() > 0 else np.nan
        ),
        "Gamma_negativa_%": (
            (gamma_neg & valid_gamma).sum() / valid_gamma.sum()
            if valid_gamma.sum() > 0 else np.nan
        ),
        "liq_media": df["Liquidity"].mean()
    })

# ─────────────────────────────────────────────────────────────────────────────
# 5) LIQUIDEZ vs CALIDAD GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

liq_analysis = opt_liq.groupby("liq_bin").apply(metricas_liquidez)

print("\n" + "-"*80)
print("CALIDAD DE GREEKS VS NIVEL DE LIQUIDEZ")
print("-"*80)
print(liq_analysis)

# ─────────────────────────────────────────────────────────────────────────────
# 6) MONEYNESS × LIQUIDEZ
# ─────────────────────────────────────────────────────────────────────────────

cross_moneyness = (opt_liq
    .groupby(["Moneyness_bin", "liq_bin"])
    .apply(metricas_liquidez)
)

print("\n" + "-"*80)
print("MONEYNESS × LIQUIDEZ")
print("-"*80)
print(cross_moneyness)

# ─────────────────────────────────────────────────────────────────────────────
# 7) VENCIMIENTO × LIQUIDEZ
# ─────────────────────────────────────────────────────────────────────────────

cross_bucket = (opt_liq
    .groupby(["bucket", "liq_bin"])
    .apply(metricas_liquidez)
)

print("\n" + "-"*80)
print("BUCKET × LIQUIDEZ")
print("-"*80)
print(cross_bucket)

# ─────────────────────────────────────────────────────────────────────────────
# 8) GRÁFICOS CLAVE
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt

# Gamma negativa vs liquidez
plt.figure(figsize=(8,5))
plt.plot(liq_analysis["liq_media"], liq_analysis["Gamma_negativa_%"], marker="o")
plt.xlabel("Liquidez media")
plt.ylabel("% Gamma negativa")
plt.title("Gamma negativa vs liquidez")
plt.grid()
plt.show()

# Delta fuera de rango vs liquidez
plt.figure(figsize=(8,5))
plt.plot(liq_analysis["liq_media"], liq_analysis["Delta_fuera_rango_%"], marker="o")
plt.xlabel("Liquidez media")
plt.ylabel("% Delta fuera de rango")
plt.title("Delta fuera de rango vs liquidez")
plt.grid()
plt.show()

# NaN vs liquidez
plt.figure(figsize=(8,5))
plt.plot(liq_analysis["liq_media"], liq_analysis["NaN_Gamma_%"], marker="o", label="Gamma")
plt.plot(liq_analysis["liq_media"], liq_analysis["NaN_Delta_%"], marker="o", label="Delta")
plt.xlabel("Liquidez media")
plt.ylabel("% NaN")
plt.title("NaN vs liquidez")
plt.legend()
plt.grid()
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# 9) RESUMEN AUTOMÁTICO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("RESUMEN AUTOMÁTICO: LIQUIDEZ Y CALIDAD")
print("="*80)

for i, row in liq_analysis.iterrows():
    print(f"""
Liquidez (bin): {i}
  Liquidez media:            {row['liq_media']:.2f}
  NaN Delta:                 {row['NaN_Delta_%']:.2%}
  NaN Gamma:                 {row['NaN_Gamma_%']:.2%}
  Delta fuera de rango:      {row['Delta_fuera_rango_%']:.2%}
  Gamma negativa:            {row['Gamma_negativa_%']:.2%}
""")


#######################################################################################################
#######################################################################################################


# # ═══════════════════════════════════════════════════════════════════════════════
# # MÉTODO 3: DELTA EMPÍRICA TEMPORAL
# # ═══════════════════════════════════════════════════════════════════════════════

# print("\n" + "─" * 80)
# print("MÉTODO 3: DELTA EMPÍRICA TEMPORAL")
# print("─" * 80)

# opt_m3 = opt_df_filtered.copy()
# opt_m3["bucket"] = assign_bucket(opt_m3["Days"])

# # Deduplicar por OptionID y Date: conservar observación más líquida
# opt_m3 = (
#     opt_m3.sort_values(
#         ["OptionID", "Date", "OpenInterest", "Volume"],
#         ascending=[True, True, False, False]
#     )
#     .drop_duplicates(subset=["OptionID", "Date"])
#     .reset_index(drop=True)
# )

# opt_m3 = compute_temporal_greeks(opt_m3)
# opt_m3["Moneyness_bin"] = assign_moneyness_bin(opt_m3["Moneyness"])

# print(f"Filas totales M3: {len(opt_m3):,}")


# # ═══════════════════════════════════════════════════════════════════════════════
# # DIAGNÓSTICOS
# # ═══════════════════════════════════════════════════════════════════════════════

# print("\nCalculando diagnósticos...")
# diag_m1 = diagnostico_metodo(opt_m1, "Bucket")
# diag_m2 = diagnostico_metodo(opt_m2, "Venc. Exacto")
# diag_m3 = diagnostico_metodo(opt_m3, "Delta Empírica")
# diags = [diag_m1, diag_m2, diag_m3]

# # Buckets globales para comparar todos los métodos con la misma referencia
# all_buckets = sorted(
#     set(opt_m1["bucket"].dropna().unique())
#     | set(opt_m2["bucket"].dropna().unique())
#     | set(opt_m3["bucket"].dropna().unique())
# )
# buckets_str = [str(b) for b in all_buckets]


# # ═══════════════════════════════════════════════════════════════════════════════
# # TABLAS COMPARATIVAS
# # ═══════════════════════════════════════════════════════════════════════════════

# print("\n" + "=" * 80)
# print("TABLA 1: RESUMEN DE NaN POR MÉTODO")
# print("=" * 80)
# t1 = pd.DataFrame([{
#     "Método": d["nombre"],
#     "N total": f"{d['n_total']:,}",
#     "NaN Delta (N)": f"{d['nan_delta_n']:,}",
#     "NaN Delta (%)": f"{d['nan_delta_pct']:.1%}",
#     "NaN Gamma (N)": f"{d['nan_gamma_n']:,}",
#     "NaN Gamma (%)": f"{d['nan_gamma_pct']:.1%}",
#     "NaN Ambos (%)": f"{d['nan_ambos_pct']:.1%}",
# } for d in diags])
# print(t1.to_string(index=False))

# print("\n" + "=" * 80)
# print("TABLA 2: VIOLACIONES DE DELTA (sobre observaciones válidas, |Δ| finito)")
# print("=" * 80)
# t2 = pd.DataFrame([{
#     "Método": d["nombre"],
#     "N válidas Delta": f"{d['n_valid_delta']:,}",
#     "Viol. Calls (N)": f"{d['viol_call_n']:,}",
#     "Viol. Calls (%)": f"{d['viol_call_pct']:.1%}" if pd.notna(d["viol_call_pct"]) else "N/A",
#     "Viol. Puts (N)": f"{d['viol_put_n']:,}",
#     "Viol. Puts (%)": f"{d['viol_put_pct']:.1%}" if pd.notna(d["viol_put_pct"]) else "N/A",
#     "Viol. Total (%)": f"{d['viol_delta_total']:.1%}" if pd.notna(d["viol_delta_total"]) else "N/A",
# } for d in diags])
# print(t2.to_string(index=False))

# print("\n" + "=" * 80)
# print("TABLA 3: GAMMA NEGATIVA (sobre observaciones válidas, Γ finita)")
# print("=" * 80)
# t3 = pd.DataFrame([{
#     "Método": d["nombre"],
#     "N válidas Gamma": f"{d['n_valid_gamma']:,}",
#     "Gamma neg. (N)": f"{d['gamma_neg_n']:,}",
#     "Gamma neg. (%)": f"{d['gamma_neg_pct']:.1%}" if pd.notna(d["gamma_neg_pct"]) else "N/A",
#     "Gamma p50 (pos)": f"{d['gamma_p50']:.6f}" if pd.notna(d["gamma_p50"]) else "N/A",
#     "Gamma p95 (pos)": f"{d['gamma_p95']:.6f}" if pd.notna(d["gamma_p95"]) else "N/A",
#     "Gamma std (pos)": f"{d['gamma_std']:.6f}" if pd.notna(d["gamma_std"]) else "N/A",
# } for d in diags])
# print(t3.to_string(index=False))

# print("\n" + "=" * 80)
# print("TABLA 4: COBERTURA DE DELTA POR ZONA DE MONEYNESS")
# print("=" * 80)
# t4 = pd.DataFrame({
#     "Moneyness": MONEYNESS_LABELS,
#     "Bucket": [f"{v:.1%}" for v in diag_m1["cob_moneyness_delta"].values],
#     "Venc. Exacto": [f"{v:.1%}" for v in diag_m2["cob_moneyness_delta"].values],
#     "Delta Empír.": [f"{v:.1%}" for v in diag_m3["cob_moneyness_delta"].values],
# })
# print(t4.to_string(index=False))

# print("\n" + "=" * 80)
# print("TABLA 5: COBERTURA DE GAMMA POR ZONA DE MONEYNESS")
# print("=" * 80)
# t5g = pd.DataFrame({
#     "Moneyness": MONEYNESS_LABELS,
#     "Bucket": [f"{v:.1%}" for v in diag_m1["cob_moneyness_gamma"].values],
#     "Venc. Exacto": [f"{v:.1%}" for v in diag_m2["cob_moneyness_gamma"].values],
#     "Delta Empír.": [f"{v:.1%}" for v in diag_m3["cob_moneyness_gamma"].values],
# })
# print(t5g.to_string(index=False))

# print("\n" + "=" * 80)
# print("TABLA 6: COBERTURA DE DELTA POR BUCKET")
# print("=" * 80)
# t6_data = {"Bucket_Rango": buckets_str}
# for d in diags:
#     cob = d["cob_bucket_delta"].reindex(all_buckets).fillna(0)
#     t6_data[d["nombre"]] = [f"{v:.1%}" for v in cob.values]
# t6 = pd.DataFrame(t6_data)
# print(t6.to_string(index=False))


# print("\n" + "=" * 80)
# print("TABLA 7: COBERTURA DE GAMMA POR BUCKET")
# print("=" * 80)
# t7_data = {"Bucket_Rango": buckets_str}
# for d in diags:
#     cob = d["cob_bucket_gamma"].reindex(all_buckets).fillna(0)
#     t7_data[d["nombre"]] = [f"{v:.1%}" for v in cob.values]
# t7 = pd.DataFrame(t7_data)
# print(t7.to_string(index=False))


# # ═══════════════════════════════════════════════════════════════════════════════
# # FIGURA COMPARATIVA
# # ═══════════════════════════════════════════════════════════════════════════════

# fig = plt.figure(figsize=(22, 26))
# fig.suptitle(
#     "Comparación exhaustiva de métodos de cálculo de Greeks\n"
#     "Bucket vs Vencimiento Exacto vs Delta Empírica",
#     fontsize=15,
#     fontweight="bold",
#     y=0.99
# )
# gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.40)

# # A) NaN Delta
# ax_a = fig.add_subplot(gs[0, 0])
# nan_d_pct = [d["nan_delta_pct"] * 100 for d in diags]
# bars = ax_a.bar(METHOD_NAMES, nan_d_pct, color=METHOD_COLORS,
#                 alpha=0.8, edgecolor="black", lw=0.5)
# ax_a.set_title("A) % NaN en Delta", fontsize=11)
# ax_a.set_ylabel("%")
# ax_a.grid(axis="y", alpha=0.3)
# for bar, val in zip(bars, nan_d_pct):
#     ax_a.text(bar.get_x() + bar.get_width() / 2,
#               bar.get_height() + 0.2,
#               f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# # B) NaN Gamma
# ax_b = fig.add_subplot(gs[0, 1])
# nan_g_pct = [d["nan_gamma_pct"] * 100 for d in diags]
# bars = ax_b.bar(METHOD_NAMES, nan_g_pct, color=METHOD_COLORS,
#                 alpha=0.8, edgecolor="black", lw=0.5)
# ax_b.set_title("B) % NaN en Gamma", fontsize=11)
# ax_b.set_ylabel("%")
# ax_b.grid(axis="y", alpha=0.3)
# for bar, val in zip(bars, nan_g_pct):
#     ax_b.text(bar.get_x() + bar.get_width() / 2,
#               bar.get_height() + 0.2,
#               f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# # C) Violaciones delta
# ax_c = fig.add_subplot(gs[0, 2])
# viol_d = [d["viol_delta_total"] * 100 if pd.notna(d["viol_delta_total"]) else 0 for d in diags]
# bars = ax_c.bar(METHOD_NAMES, viol_d, color=METHOD_COLORS,
#                 alpha=0.8, edgecolor="black", lw=0.5)
# ax_c.set_title("C) % Delta fuera de rango\n(sobre válidas)", fontsize=11)
# ax_c.set_ylabel("%")
# ax_c.grid(axis="y", alpha=0.3)
# for bar, val in zip(bars, viol_d):
#     ax_c.text(bar.get_x() + bar.get_width() / 2,
#               bar.get_height() + 0.1,
#               f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# # D) Gamma negativa
# ax_d = fig.add_subplot(gs[1, 0])
# gamma_n = [d["gamma_neg_pct"] * 100 if pd.notna(d["gamma_neg_pct"]) else 0 for d in diags]
# bars = ax_d.bar(METHOD_NAMES, gamma_n, color=METHOD_COLORS,
#                 alpha=0.8, edgecolor="black", lw=0.5)
# ax_d.set_title("D) % Gamma negativa\n(sobre válidas)", fontsize=11)
# ax_d.set_ylabel("%")
# ax_d.grid(axis="y", alpha=0.3)
# for bar, val in zip(bars, gamma_n):
#     ax_d.text(bar.get_x() + bar.get_width() / 2,
#               bar.get_height() + 0.2,
#               f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# # E) Gamma negativa por bucket
# ax_e = fig.add_subplot(gs[1, 1:])
# x = np.arange(len(all_buckets))
# width = 0.25

# for i, (d, df_m, color) in enumerate(zip(diags, [opt_m1, opt_m2, opt_m3], METHOD_COLORS)):
#     vals = []
#     for b in all_buckets:
#         sub = df_m.loc[df_m["bucket"] == b, "Gamma"]
#         valid_sub = sub[sub.notna() & np.isfinite(sub)]
#         vals.append((valid_sub < 0).mean() * 100 if len(valid_sub) > 0 else 0)
#     ax_e.bar(x + i * width, vals, width=width,
#              label=d["nombre"], color=color,
#              alpha=0.8, edgecolor="black", lw=0.3)

# ax_e.set_title("E) % Gamma negativa por bucket y método", fontsize=11)
# ax_e.set_ylabel("%")
# ax_e.set_xticks(x + width)
# ax_e.set_xticklabels(buckets_str, rotation=25, ha="right", fontsize=8)
# ax_e.legend(fontsize=8)
# ax_e.grid(axis="y", alpha=0.3)

# # F) Cobertura de Delta por moneyness
# ax_f = fig.add_subplot(gs[2, :])
# x_m = np.arange(len(MONEYNESS_LABELS))
# width_m = 0.25

# for i, (d, color) in enumerate(zip(diags, METHOD_COLORS)):
#     vals = [d["cob_moneyness_delta"].get(lab, 0) * 100 for lab in MONEYNESS_LABELS]
#     ax_f.bar(x_m + i * width_m, vals, width=width_m,
#              label=d["nombre"], color=color,
#              alpha=0.8, edgecolor="black", lw=0.3)

# ax_f.axhline(80, color="red", ls="--", lw=1.2, label="80% referencia")
# ax_f.set_title("F) % cobertura de Delta por zona de moneyness", fontsize=11)
# ax_f.set_ylabel("% con Delta válida")
# ax_f.set_xticks(x_m + width_m)
# ax_f.set_xticklabels(MONEYNESS_LABELS, rotation=30, ha="right", fontsize=9)
# ax_f.legend(fontsize=8, ncol=4)
# ax_f.grid(axis="y", alpha=0.3)
# ax_f.set_ylim(0, 115)

# # G) Evolución temporal de gamma negativa
# ax_g = fig.add_subplot(gs[3, :])

# for df_m, d, color, ls in zip(
#     [opt_m1, opt_m2, opt_m3],
#     diags,
#     METHOD_COLORS,
#     ["-", "--", "-."]
# ):
#     serie = yearly_negative_gamma_series(df_m)
#     if len(serie) > 0:
#         ax_g.plot(
#             serie.index, serie.values,
#             color=color, lw=2, ls=ls, marker="o", ms=4,
#             label=d["nombre"]
#         )

# ax_g.axvline(2016, color="black", ls=":", lw=1.5)
# ylim_top = ax_g.get_ylim()[1] if len(ax_g.lines) > 0 else 10
# ax_g.text(2016.1, ylim_top * 0.85 if ylim_top > 0 else 5, "SPXW\n2016", fontsize=9)

# ax_g.set_title("G) Evolución temporal de % Gamma negativa por método", fontsize=11)
# ax_g.set_ylabel("% Gamma negativa")
# ax_g.set_xlabel("Año")
# ax_g.legend(fontsize=9, ncol=3)
# ax_g.grid(alpha=0.3)

# plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
# plt.show()


# # ═══════════════════════════════════════════════════════════════════════════════
# # RESUMEN EJECUTIVO FINAL
# # ═══════════════════════════════════════════════════════════════════════════════

# # print("\n" + "=" * 80)
# # print("RESUMEN EJECUTIVO COMPARATIVO")
# # print("=" * 80)

# # for d in diags:
# #     print(f"""
# # ── {d['nombre'].upper()} ──
# #   Observaciones totales:       {d['n_total']:>12,}
# #   NaN Delta:                   {d['nan_delta_pct']:>11.1%}
# #   NaN Gamma:                   {d['nan_gamma_pct']:>11.1%}
# #   Delta fuera de rango:        {d['viol_delta_total']:>11.1%}  (sobre válidas)
# #   Gamma negativa:              {d['gamma_neg_pct']:>11.1%}  (sobre válidas)
# #   Cobertura Delta ATM:         {d['cob_moneyness_delta'].get('0.95-1.05', 0):>11.1%}
# #   Cobertura Delta OTM:         {d['cob_moneyness_delta'].get('0.80-0.90', 0):>11.1%}
# #   Cobertura Gamma ATM:         {d['cob_moneyness_gamma'].get('0.95-1.05', 0):>11.1%}
# #   Cobertura Gamma OTM:         {d['cob_moneyness_gamma'].get('0.80-0.90', 0):>11.1%}
# # """)



# print("\n" + "=" * 80)
# print("RESUMEN EJECUTIVO COMPARATIVO")
# print("=" * 80)

# for d, df_m in zip(diags, [opt_m1, opt_m2, opt_m3]):

#     # ATM: moneyness [0.90, 1.10] — calls y puts juntos
#     atm_mask = df_m["Moneyness"].between(0.90, 1.10)

#     # OTM: puts con K/S < 0.90 y calls con K/S > 1.10
#     otm_mask = (
#         ((df_m["CallPut"] == "P") & (df_m["Moneyness"] < 0.90)) |
#         ((df_m["CallPut"] == "C") & (df_m["Moneyness"] > 1.10))
#     )

#     def cob(mask, col):
#         total = mask.sum()
#         valido = (mask & df_m[col].notna() & np.isfinite(df_m[col])).sum()
#         return valido / total if total > 0 else np.nan

#     print(f"""
# ── {d['nombre'].upper()} ──
#   Observaciones totales:                      {d['n_total']:>12,}
#   NaN Delta:                                  {d['nan_delta_pct']:>11.1%}
#   NaN Gamma:                                  {d['nan_gamma_pct']:>11.1%}
#   Delta fuera de rango:                       {d['viol_delta_total']:>11.1%}  (sobre válidas)
#   Gamma negativa:                             {d['gamma_neg_pct']:>11.1%}  (sobre válidas)

#   Cobertura Delta ATM [0.90-1.10]:            {cob(atm_mask, 'Delta'):>11.1%}
#   Cobertura Delta OTM [put<0.90 / call>1.10]: {cob(otm_mask, 'Delta'):>11.1%}

#   Cobertura Gamma ATM [0.90-1.10]:            {cob(atm_mask, 'Gamma'):>11.1%}
#   Cobertura Gamma OTM [put<0.90 / call>1.10]: {cob(otm_mask, 'Gamma'):>11.1%}
# """)
