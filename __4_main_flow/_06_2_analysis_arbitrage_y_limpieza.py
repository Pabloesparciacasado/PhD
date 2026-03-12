

### 06.02-> Quiero ver también si se incumple algún punto el arbitraje del conjunto creado en 04 y limpiamos

### Analizar si las griegas obtenidas en 07 tienen sentido, comparando con VIX, y bajo supuestos de Gatheral(2004) y Breeden-Liztenberg(1978).
### Comparación de griegas desde el fichero std_option (ATM),(ver si desde el de vol_surface puedo ampliar a más strikes el std_option).

# In[]
"""
Pipeline de diagnóstico y limpieza de la superficie de volatilidad
estandarizada a madurez fija (ej. 30 días).

Identificador de smile: (Date, CallPut)
No requiere columna Expiration ni flag_inside_observed_range.

Checks:
1. Bounds put-call parity   — no-arbitraje estático
2. Monotonicidad en strike  — no calendar spread arbitrage
3. Convexidad en strike     — no butterfly arbitrage, equivalente a
                              densidad RN >= 0 via diferencias finitas
                              sobre Precio_Modelo (sin recalcular spline)

Usa Precio_Modelo directamente — no recalcula BS ni spline.

Nota sobre check RND:
   La convexidad discreta d²P/dK² >= 0 es equivalente a la condición
   de Breeden-Litzenberger q(K) = e^{rT} * d²C/dK² >= 0. Usar un
   segundo spline sobre los precios para estimar d²P/dK² amplifica el
   ruido numérico del spline original — las diferencias finitas sobre
   los precios ya interpolados son más estables y no introducen
   artefactos adicionales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import duckdb

# =============================================================================
# Check 1 — Bounds put-call parity
# =============================================================================
# Call: DF * max(F-K, 0)  <=  C  <=  DF * F
# Put:  DF * max(K-F, 0)  <=  P  <=  DF * K
#
# Fundamento: no-arbitraje estático. Violaciones indican IV fuera del
# rango válido para BS o datos corruptos.
# Tolerancia 1e-6: por debajo del bid-ask spread mínimo en índices.

def check_bounds(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    df = df.copy()
    is_call = df["CallPut"] == "C"

    df["lower_bound"] = np.where(
        is_call,
        df["discount_factor"] * np.maximum(df["forward"] - df["Strike"], 0),
        df["discount_factor"] * np.maximum(df["Strike"]  - df["forward"], 0),
    )
    df["upper_bound"] = np.where(
        is_call,
        df["discount_factor"] * df["forward"],
        df["discount_factor"] * df["Strike"],
    )
    df["flag_bounds_ok"] = (
        (df["Precio_Modelo"] >= df["lower_bound"] - tol) &
        (df["Precio_Modelo"] <= df["upper_bound"] + tol)
    )
    return df


# =============================================================================
# Checks 2 y 3 — Monotonicidad y Convexidad por slice (Date, CallPut)
# =============================================================================
# Monotonicidad:
#   Calls decrecen en K, puts crecen en K.
#   Tolerancia tol_mono en precio absoluto (default 1e-4).
#
# Convexidad discreta con spacing no uniforme:
#   (P[i+1]-P[i])/h2 - (P[i]-P[i-1])/h1 >= -tol_conv
#
#   Fundamento: butterfly arbitrage-free ↔ d²C/dK² >= 0
#   (Breeden & Litzenberger, 1978). La diferencia de pendientes
#   discretas es la aproximación de d²P/dK² sin spline adicional.
#
#   Tolerancia tol_conv (default 1e-3) en diferencia de pendientes:
#   grid uniforme en log-moneyness → spacing variable en K entre
#   ~5 y ~50 puntos de índice. Con h variable, 1e-3 en diferencia
#   de pendientes equivale a ~0.005-0.05 en precio — por debajo
#   del bid-ask en opciones de índice líquidas. Tolerancia 1e-8
#   generaba falsos positivos sistemáticos.
#
#   El check de convexidad es equivalente al check de densidad RN —
#   no se implementa check_rnd separado para evitar amplificar ruido
#   numérico con un segundo spline sobre los precios ya interpolados.

def check_mono_convexity(
    df: pd.DataFrame,
    tol_mono: float = 1e-4,
    tol_conv: float = 1e-3,
) -> pd.DataFrame:
    rows = []

    for (date, callput), grp in df.groupby(["Date", "CallPut"]):
        grp = grp.sort_values("Strike").reset_index(drop=True)
        K = grp["Strike"].to_numpy(dtype=float)
        P = grp["Precio_Modelo"].to_numpy(dtype=float)

        if len(grp) < 3:
            rows.append({
                "Date": date, "CallPut": callput,
                "n_points": len(grp),
                "n_mono_viol": 0, "n_conv_viol": 0,
                "flag_mono_ok": True, "flag_conv_ok": True,
            })
            continue

        # ---- Monotonicidad ----
        dP = np.diff(P)
        mono_viol = int(
            np.sum(dP > tol_mono) if callput == "C"
            else np.sum(dP < -tol_mono)
        )

        # ---- Convexidad discreta (= densidad RN >= 0) ----
        # d²P/dK² aprox via diferencias finitas con spacing no uniforme
        conv_viol = 0
        for i in range(1, len(K) - 1):
            h1 = K[i]     - K[i - 1]
            h2 = K[i + 1] - K[i]
            if h1 <= 0 or h2 <= 0:
                continue
            slope_left  = (P[i]     - P[i - 1]) / h1
            slope_right = (P[i + 1] - P[i])     / h2
            if slope_right - slope_left < -tol_conv:
                conv_viol += 1

        rows.append({
            "Date":         date,
            "CallPut":      callput,
            "n_points":     len(grp),
            "n_mono_viol":  mono_viol,
            "n_conv_viol":  conv_viol,
            "flag_mono_ok": mono_viol == 0,
            "flag_conv_ok": conv_viol == 0,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Pipeline completo
# =============================================================================

def run_surface_analysis(
    df_in: pd.DataFrame,
    tol_bounds: float = 1e-6,
    tol_mono:   float = 1e-4,
    tol_conv:   float = 1e-3,
    drop_bounds_fail: bool = True,
    drop_mono_fail:   bool = True,
    drop_conv_fail:   bool = False,
    verbose:          bool = True,
    plot_sample:      bool = True,
    n_plot:           int  = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Diagnóstico completo y limpieza de la superficie estandarizada a 30d.

    Parameters
    ----------
    df_in             : resultado_europeo — columnas requeridas:
                        Date, CallPut, Strike, forward, rate, T,
                        discount_factor, Precio_Modelo, implied_vol
    tol_bounds        : tolerancia absoluta en precio para bounds (1e-6)
    tol_mono          : tolerancia absoluta en precio para monotonicidad (1e-4)
    tol_conv          : tolerancia en diferencia de pendientes para
                        convexidad (1e-3) — calibrada para spacing no
                        uniforme en K con grid en log-moneyness
    drop_bounds_fail  : excluir slices que violan bounds
    drop_mono_fail    : excluir slices que violan monotonicidad
    drop_conv_fail    : excluir slices que violan convexidad
                        (False por defecto — warning, no exclusión)
    verbose           : imprime resumen y tabla por año
    plot_sample       : plotea slices problemáticos
    n_plot            : número máximo de slices a plotear

    Returns
    -------
    df_clean  : DataFrame limpio listo para griegas y momentos.
                Mismas columnas que df_in más lower_bound, upper_bound,
                flag_bounds_ok.
    diag_df   : DataFrame de diagnóstico con una fila por (Date, CallPut).
    """
    df = df_in.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # ------------------------------------------------------------------
    # Checks
    # ------------------------------------------------------------------
    df           = check_bounds(df, tol=tol_bounds)
    mono_conv_df = check_mono_convexity(df, tol_mono=tol_mono, tol_conv=tol_conv)

    # ------------------------------------------------------------------
    # Bounds flag por slice (Date, CallPut)
    # ------------------------------------------------------------------
    bounds_by_slice = (
        df.groupby(["Date", "CallPut"])["flag_bounds_ok"]
        .all()
        .reset_index()
        .rename(columns={"flag_bounds_ok": "flag_bounds_ok_slice"})
    )

    # ------------------------------------------------------------------
    # Merge diagnóstico
    # ------------------------------------------------------------------
    diag_df = mono_conv_df.merge(
        bounds_by_slice, on=["Date", "CallPut"], how="left"
    )

    # ------------------------------------------------------------------
    # Resumen
    # ------------------------------------------------------------------
    if verbose:
        n_slices = len(diag_df)
        print("=" * 65)
        print("ANÁLISIS DE SUPERFICIE 30D — RESUMEN")
        print("=" * 65)
        print(f"  Slices (Date × CallPut): {n_slices:,}")
        print()

        checks = [
            ("flag_bounds_ok_slice", "1. Bounds put-call parity",  drop_bounds_fail),
            ("flag_mono_ok",         "2. Monotonicidad en strike",  drop_mono_fail),
            ("flag_conv_ok",         "3. Convexidad / densidad RN", drop_conv_fail),
        ]
        for flag, label, drops in checks:
            if flag in diag_df.columns:
                n_fail = int((~diag_df[flag].fillna(True)).sum())
                pct    = n_fail / n_slices * 100
                status = "excluye" if drops else "warning"
                print(f"  {label:38s}: {n_fail:5,} fallos ({pct:5.1f}%)  [{status}]")

        print()
        print("  Tolerancias:")
        print(f"    Bounds        : {tol_bounds:.0e}  (precio absoluto)")
        print(f"    Monotonicidad : {tol_mono:.0e}  (precio absoluto)")
        print(f"    Convexidad    : {tol_conv:.0e}  (diferencia de pendientes)")
        print("=" * 65)

        # Por año
        diag_df["year"] = pd.to_datetime(diag_df["Date"]).dt.year
        by_year = (
            diag_df.groupby("year")
            .agg(
                total    = ("Date",        "size"),
                mono_bad = ("flag_mono_ok", lambda s: (~s).sum()),
                conv_bad = ("flag_conv_ok", lambda s: (~s).sum()),
            )
            .reset_index()
        )
        for col in ["mono_bad", "conv_bad"]:
            by_year[col + "_pct"] = 100 * by_year[col] / by_year["total"]

        print(f"\n  {'Año':>4}  {'Total':>6}  {'Mono%':>6}  {'Conv%':>6}")
        for _, r in by_year.iterrows():
            print(f"  {int(r.year):>4}  {int(r.total):>6}  "
                  f"{r.mono_bad_pct:>6.1f}  {r.conv_bad_pct:>6.1f}")

    # ------------------------------------------------------------------
    # Plots de slices problemáticos
    # ------------------------------------------------------------------
    if plot_sample:
        problem = diag_df[
            (~diag_df["flag_mono_ok"]) |
            (~diag_df["flag_conv_ok"])
        ].head(n_plot)

        if not problem.empty:
            n_cols = min(3, len(problem))
            n_rows = int(np.ceil(len(problem) / n_cols))
            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows)
            )
            axes = np.array(axes).flatten()

            for idx, (_, row) in enumerate(problem.iterrows()):
                sl = df[
                    (df["Date"]    == row["Date"]) &
                    (df["CallPut"] == row["CallPut"])
                ].sort_values("Strike")

                ax = axes[idx]
                ax.plot(sl["Strike"], sl["Precio_Modelo"], "b-", lw=1.5)
                ax.scatter(sl["Strike"], sl["Precio_Modelo"],
                           color="blue", s=20, zorder=5)

                title = (
                    f"{pd.Timestamp(row['Date']).date()} | {row['CallPut']}\n"
                    f"mono={row['n_mono_viol']}  conv={row['n_conv_viol']}"
                )
                ax.set_title(title, fontsize=8)
                ax.set_xlabel("Strike", fontsize=7)
                ax.set_ylabel("Precio_Modelo", fontsize=7)

            for ax in axes[len(problem):]:
                ax.set_visible(False)

            plt.suptitle("Slices problemáticos", fontsize=10, fontweight="bold")
            plt.tight_layout()
            plt.show()

    # ------------------------------------------------------------------
    # DataFrame limpio
    # ------------------------------------------------------------------
    exclude = pd.Series(False, index=diag_df.index)

    if drop_bounds_fail:
        exclude |= ~diag_df["flag_bounds_ok_slice"].fillna(True)
    if drop_mono_fail:
        exclude |= ~diag_df["flag_mono_ok"]
    if drop_conv_fail:
        exclude |= ~diag_df["flag_conv_ok"]

    bad_slices = diag_df[exclude][["Date", "CallPut"]].copy()

    df_clean = df.copy()
    if not bad_slices.empty:
        df_clean = df_clean.merge(
            bad_slices.assign(_drop=True),
            on=["Date", "CallPut"],
            how="left",
        )
        df_clean = df_clean[df_clean["_drop"].isna()].drop(columns="_drop")

    df_clean = df_clean.reset_index(drop=True)

    if verbose:
        n_in   = len(df)
        n_out  = len(df_clean)
        n_drop = n_in - n_out
        print(f"\n  DataFrame limpio:")
        print(f"    Filas entrada  : {n_in:>10,}")
        print(f"    Filas excluidas: {n_drop:>10,}  ({n_drop/n_in*100:.1f}%)")
        print(f"    Filas salida   : {n_out:>10,}  ({n_out/n_in*100:.1f}%)")

    return df_clean, diag_df

con = duckdb.connect()

resultado_europeo = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\superficie_con_precios_shimko_3.parquet')
""").df()

print(resultado_europeo.columns.tolist())

# In[]
df_clean, diag_df = run_surface_analysis(
    df_in            = resultado_europeo,
    drop_bounds_fail = True,
    drop_mono_fail   = True,
    drop_conv_fail   = False,   # warning
    verbose          = True,
    plot_sample      = True,
)

PARQET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\superficie_con_precios_limpio_shimko_3.parquet"
duckdb.from_df(df_clean).write_parquet(PARQET_OUTPUT, compression='snappy')

print("===================================================================")
print("Generada la superficie de volatilidad con pricios limpios con éxito")
print("===================================================================")






print("Fechas únicas en entrada :", resultado_europeo["Date"].nunique())
print("Fechas únicas en df_clean:", df_clean["Date"].nunique())
print()
print("Smiles (Date×CallPut) entrada :", 
      resultado_europeo.groupby(["Date","CallPut"]).ngroups)
print("Smiles (Date×CallPut) df_clean:", 
      df_clean.groupby(["Date","CallPut"]).ngroups)

# ¿Se excluyen más puts o más calls?
bad = diag_df[~diag_df["flag_mono_ok"]]
print(bad["CallPut"].value_counts())

# ¿Hay fechas que siempre fallan?
bad_dates = bad["Date"].value_counts()
print("Fechas que fallan >5 veces:", (bad_dates > 5).sum())

