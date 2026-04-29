"""
Sensibilidades empíricas locales del mercado — evaluación en una fecha
======================================================================
Para cada contrato observado en una fecha de evaluación t0, estima una
regresión temporal local usando su propia historia pasada:

    ΔC = βS ΔS + βS2 (ΔS)^2 + βtau Δtau [+ βIV ΔIV] + ε

Si existe una columna de IV en el dataset, la incluye automáticamente.

Output:
- sensibilidad empírica spot por contrato en t0
- curvatura empírica por contrato en t0
- agregación DEX / GEX por OI
- gráficos tipo LPR:
    1) Curvature empirical vs Moneyness
    2) Spot sensitivity empirical vs Moneyness
    3) GEX por bucket de moneyness
    4) GEX por bucket de madurez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. PARÁMETROS
# =============================================================================

OPTION_TYPE          = "P"         # "C" o "P"
EVAL_DATE            = None        # None = fecha más frecuente del mes elegido
EVAL_MONTH           = None        # None = mes más frecuente del dataset filtrado; o "2022-03"
LOOKBACK_DAYS        = 90          # ventana histórica máxima hacia atrás
MIN_HISTORY_OBS      = 12          # mínimo de observaciones históricas utilizables
MIN_DS_ABS           = 0.005       # filtro mínimo de movimiento absoluto del spot
USE_RELATIVE_DS      = True
MIN_DS_REL           = 0.002       # 0.2%
HALF_LIFE_DAYS       = 20          # ponderación exponencial temporal
RIDGE_LAMBDA         = 1e-8        # regularización
COND_MAX             = 1e10        # umbral de condicionamiento
MIN_OI_TOTAL         = 5           # filtro para puntos agregados
MONEYNESS_ROUND      = 3
TAU_ROUND            = 0

# Nombre opcional de columna de IV. Si no existe, el script sigue sin IV.
IV_CANDIDATE_COLS = [
    "ImpliedVolatility", "IV", "ImpVol", "ImplVol", "SigmaIV", "IV_mid"
]

PATH_DATA   = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"
PATH_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output"

# =============================================================================
# 2. UTILIDADES
# =============================================================================

def safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def pick_iv_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def weighted_mean(x, w):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(x[mask], weights=w[mask])


def build_local_regression_for_option(sub_hist, eval_row, iv_col=None,
                                      min_ds_abs=0.005,
                                      use_relative_ds=True,
                                      min_ds_rel=0.002,
                                      half_life_days=20,
                                      ridge_lambda=1e-8,
                                      cond_max=1e10,
                                      min_history_obs=12):
    """
    sub_hist: histórico pasado de una OptionID (fechas < eval_date)
    eval_row: fila del contrato en fecha evaluación
    """
    if len(sub_hist) < min_history_obs:
        return None

    sub_hist = sub_hist.sort_values("Date").copy()

    # Construimos variaciones first-difference sobre el histórico
    sub_hist["dC"] = sub_hist["MidPrice"].diff()
    sub_hist["dS"] = sub_hist["SpotPrice"].diff()
    sub_hist["dtau"] = sub_hist["Days"].diff()

    if iv_col is not None:
        sub_hist["dIV"] = sub_hist[iv_col].diff()

    sub_hist["Spot_lag"] = sub_hist["SpotPrice"].shift(1)

    abs_mask = sub_hist["dS"].abs() >= min_ds_abs
    if use_relative_ds:
        rel_mask = safe_divide(sub_hist["dS"].abs(), sub_hist["Spot_lag"]).astype(float) >= min_ds_rel
        move_mask = abs_mask | rel_mask
    else:
        move_mask = abs_mask

    cols_needed = ["dC", "dS", "dtau", "Date"]
    if iv_col is not None:
        cols_needed.append("dIV")

    work = sub_hist[move_mask].copy()
    work = work[np.isfinite(work["dC"]) & np.isfinite(work["dS"]) & np.isfinite(work["dtau"])]

    if iv_col is not None:
        work = work[np.isfinite(work["dIV"])]

    if len(work) < min_history_obs:
        return None

    # Variable dependiente
    y = work["dC"].to_numpy(dtype=float)

    # Diseño
    if iv_col is not None:
        X = np.column_stack([
            work["dS"].to_numpy(dtype=float),
            (work["dS"].to_numpy(dtype=float) ** 2),
            work["dtau"].to_numpy(dtype=float),
            work["dIV"].to_numpy(dtype=float)
        ])
        colnames = ["beta_S", "beta_S2", "beta_tau", "beta_IV"]
    else:
        X = np.column_stack([
            work["dS"].to_numpy(dtype=float),
            (work["dS"].to_numpy(dtype=float) ** 2),
            work["dtau"].to_numpy(dtype=float)
        ])
        colnames = ["beta_S", "beta_S2", "beta_tau"]

    # Pesos temporales exponenciales respecto a eval_date
    day_dist = (eval_row["Date"] - work["Date"]).dt.days.to_numpy(dtype=float)
    day_dist = np.maximum(day_dist, 0.0)
    w = np.exp(-np.log(2.0) * day_dist / max(half_life_days, 1e-6))

    # Añadimos intercepto
    X = np.column_stack([np.ones(len(X)), X])

    # WLS ridge
    Xw = X * w[:, None]
    XtWX = Xw.T @ X
    XtWy = Xw.T @ y

    # ridge salvo intercepto
    ridge = ridge_lambda * np.eye(XtWX.shape[0])
    ridge[0, 0] = 0.0
    XtWX_reg = XtWX + ridge

    try:
        cond = np.linalg.cond(XtWX_reg)
        if (not np.isfinite(cond)) or (cond > cond_max):
            return None
        beta = np.linalg.solve(XtWX_reg, XtWy)
    except np.linalg.LinAlgError:
        return None

    out = {
        "alpha": beta[0],
        "n_hist": len(work),
        "cond_local": cond
    }
    for i, name in enumerate(colnames, start=1):
        out[name] = beta[i]

    return out


# =============================================================================
# 3. CARGA Y PREPARACIÓN
# =============================================================================

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

required_cols = [
    "Date", "OptionID", "CallPut", "Strike", "Days",
    "MidPrice", "SpotPrice", "Moneyness",
    "OpenInterest", "Volume", "Bid"
]
missing = [c for c in required_cols if c not in opt_df.columns]
if missing:
    raise ValueError(f"Faltan columnas requeridas: {missing}")

opt_df["Date"] = pd.to_datetime(opt_df["Date"], errors="coerce")

numeric_cols = [
    "Strike", "Days", "MidPrice", "SpotPrice",
    "Moneyness", "OpenInterest", "Volume", "Bid"
]
for col in numeric_cols:
    opt_df[col] = pd.to_numeric(opt_df[col], errors="coerce")

iv_col = pick_iv_column(opt_df, IV_CANDIDATE_COLS)
if iv_col is not None:
    opt_df[iv_col] = pd.to_numeric(opt_df[iv_col], errors="coerce")
    print(f"Columna IV detectada: {iv_col}")
else:
    print("No se detectó columna IV. El modelo se estimará sin ΔIV.")

opt_df_filtered = opt_df[
    opt_df["Date"].notna() &
    opt_df["CallPut"].isin(["C", "P"]) &
    opt_df["Bid"].gt(0) &
    opt_df["SpotPrice"].gt(0) &
    opt_df["MidPrice"].notna() &
    opt_df["Moneyness"].notna() &
    opt_df["Days"].notna() &
    ((opt_df["OpenInterest"] > 0) | (opt_df["Volume"] > 0))
].copy()

opt_df_filtered["YearMonth"] = opt_df_filtered["Date"].dt.to_period("M")

if EVAL_MONTH is None:
    sample_month = opt_df_filtered["YearMonth"].value_counts().idxmax()
else:
    sample_month = pd.Period(EVAL_MONTH, freq="M")

df_month = opt_df_filtered[
    (opt_df_filtered["YearMonth"] == sample_month) &
    (opt_df_filtered["CallPut"] == OPTION_TYPE)
].copy()

if len(df_month) == 0:
    raise ValueError("No hay observaciones para el mes/tipo de opción seleccionados.")

if EVAL_DATE is None:
    eval_date = df_month["Date"].value_counts().idxmax()
else:
    eval_date = pd.Timestamp(EVAL_DATE)

print(f"Mes de evaluación:   {sample_month}")
print(f"Fecha de evaluación: {eval_date.date()}")

df_eval = df_month[df_month["Date"] == eval_date].copy()
if len(df_eval) == 0:
    raise ValueError("No hay observaciones en la fecha de evaluación seleccionada.")

# Deduplicación en fecha de evaluación
df_eval = (
    df_eval.sort_values(
        ["OptionID", "OpenInterest", "Volume"],
        ascending=[True, False, False]
    )
    .drop_duplicates(subset=["OptionID"])
    .reset_index(drop=True)
)

print(f"Contratos en fecha evaluación ({OPTION_TYPE}): {len(df_eval):,}")

# Histórico global para lookback
hist_start = eval_date - pd.Timedelta(days=LOOKBACK_DAYS)
df_hist_pool = opt_df_filtered[
    (opt_df_filtered["CallPut"] == OPTION_TYPE) &
    (opt_df_filtered["Date"] < eval_date) &
    (opt_df_filtered["Date"] >= hist_start)
].copy()

# Deduplicación en histórico por OptionID-Date
df_hist_pool = (
    df_hist_pool.sort_values(
        ["OptionID", "Date", "OpenInterest", "Volume"],
        ascending=[True, True, False, False]
    )
    .drop_duplicates(subset=["OptionID", "Date"])
    .reset_index(drop=True)
)

print(f"Observaciones históricas pool: {len(df_hist_pool):,}")

# =============================================================================
# 4. ESTIMACIÓN EMPÍRICA LOCAL EN LA FECHA DE EVALUACIÓN
# =============================================================================

print("\nEstimando sensibilidades empíricas locales...")

results = []
log_every = max(1, len(df_eval) // 20)

hist_groups = {k: v.copy() for k, v in df_hist_pool.groupby("OptionID", sort=False)}

for i, (_, row) in enumerate(df_eval.iterrows()):
    if i % log_every == 0:
        pct = 100 * i / len(df_eval)
        print(f"  {pct:5.1f}%  ({i:,} / {len(df_eval):,})", end="\r")

    opt_id = row["OptionID"]
    sub_hist = hist_groups.get(opt_id, None)

    if sub_hist is None or len(sub_hist) == 0:
        results.append({
            "OptionID": opt_id,
            "beta_S": np.nan,
            "beta_S2": np.nan,
            "beta_tau": np.nan,
            "beta_IV": np.nan if iv_col is not None else np.nan,
            "n_hist": 0,
            "cond_local": np.nan
        })
        continue

    res = build_local_regression_for_option(
        sub_hist=sub_hist,
        eval_row=row,
        iv_col=iv_col,
        min_ds_abs=MIN_DS_ABS,
        use_relative_ds=USE_RELATIVE_DS,
        min_ds_rel=MIN_DS_REL,
        half_life_days=HALF_LIFE_DAYS,
        ridge_lambda=RIDGE_LAMBDA,
        cond_max=COND_MAX,
        min_history_obs=MIN_HISTORY_OBS
    )

    if res is None:
        results.append({
            "OptionID": opt_id,
            "beta_S": np.nan,
            "beta_S2": np.nan,
            "beta_tau": np.nan,
            "beta_IV": np.nan if iv_col is not None else np.nan,
            "n_hist": 0,
            "cond_local": np.nan
        })
    else:
        results.append({
            "OptionID": opt_id,
            "beta_S": res.get("beta_S", np.nan),
            "beta_S2": res.get("beta_S2", np.nan),
            "beta_tau": res.get("beta_tau", np.nan),
            "beta_IV": res.get("beta_IV", np.nan),
            "n_hist": res.get("n_hist", 0),
            "cond_local": res.get("cond_local", np.nan)
        })

print(f"\n  100.0%  ({len(df_eval):,} / {len(df_eval):,})")

res_df = pd.DataFrame(results)

df = df_eval.merge(res_df, on="OptionID", how="left")

# Sensibilidades empíricas
df["spot_sensitivity_empirical"] = df["beta_S"]
df["curvature_empirical"] = 2.0 * df["beta_S2"]

# nombres para homogeneidad con tu pipeline previo
df["delta_true"] = df["spot_sensitivity_empirical"]
df["gamma_true"] = df["curvature_empirical"]

# =============================================================================
# 5. PUNTOS DE EVALUACIÓN TIPO LPR
# =============================================================================

df["m_eval"] = df["Moneyness"].round(MONEYNESS_ROUND)
df["tau_eval"] = df["Days"].round(TAU_ROUND).astype(int)

df["liq_weight"] = (df["OpenInterest"] + df["Volume"]).clip(lower=1)

eval_points = (
    df.groupby(["m_eval", "tau_eval"], as_index=False)
      .apply(lambda g: pd.Series({
          "OI_total": g["OpenInterest"].sum(),
          "n_contracts": g["OptionID"].nunique(),
          "delta_true": weighted_mean(g["delta_true"].values, g["liq_weight"].values),
          "gamma_true": weighted_mean(g["gamma_true"].values, g["liq_weight"].values),
          "beta_tau": weighted_mean(g["beta_tau"].values, g["liq_weight"].values),
          "beta_IV": weighted_mean(g["beta_IV"].values, g["liq_weight"].values),
          "n_hist_mean": g["n_hist"].mean(),
          "cond_local_median": g["cond_local"].median()
      }))
      .reset_index(drop=True)
)

eval_points = eval_points[eval_points["OI_total"] >= MIN_OI_TOTAL].copy()
eval_points["valid"] = eval_points["delta_true"].notna() & eval_points["gamma_true"].notna()

valid_pct = eval_points["valid"].mean() * 100 if len(eval_points) > 0 else 0.0
print(f"\nPuntos únicos de evaluación: {len(eval_points):,}")
print(f"Puntos válidos: {eval_points['valid'].sum():,} / {len(eval_points):,} ({valid_pct:.1f}%)")

# =============================================================================
# 6. GEX / DEX
# =============================================================================

df["gex_contribution"] = df["gamma_true"] * df["OpenInterest"]
df["dex_contribution"] = df["delta_true"] * df["OpenInterest"]

print("\n=== AGREGACIÓN GEX/DEX ===")
gex_total = df["gex_contribution"].sum(skipna=True)
dex_total = df["dex_contribution"].sum(skipna=True)

label_type = "puts" if OPTION_TYPE == "P" else "calls"
print(f"GEX total ({label_type}): {gex_total:>15,.6f}")
print(f"DEX total ({label_type}): {dex_total:>15,.6f}")

# Buckets de moneyness
bins_m = [0.0, 0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 99]
labels_m = ["<0.80","0.80-0.90","0.90-0.95","ATM[0.95-1.05]",
            "1.05-1.10","1.10-1.20",">1.20"]
df["m_bucket"] = pd.cut(df["Moneyness"], bins=bins_m, labels=labels_m)

gex_by_m = df.groupby("m_bucket", observed=True).agg(
    GEX=("gex_contribution","sum"),
    DEX=("dex_contribution","sum"),
    OI_total=("OpenInterest","sum"),
    n_contracts=("OptionID","nunique")
).round(6)

print("\nGEX/DEX por bucket de moneyness:")
print(gex_by_m.to_string())

# Buckets de tau
bins_tau = [0, 15, 45, 105, 183, 365, 9999]
labels_tau = ["0-15d","15-45d","45-105d","105-183d","183-365d",">365d"]
df["tau_bucket"] = pd.cut(df["Days"], bins=bins_tau, labels=labels_tau)

gex_by_tau = df.groupby("tau_bucket", observed=True).agg(
    GEX=("gex_contribution","sum"),
    DEX=("dex_contribution","sum"),
    OI_total=("OpenInterest","sum"),
    n_contracts=("OptionID","nunique")
).round(6)

print("\nGEX/DEX por bucket de madurez:")
print(gex_by_tau.to_string())

# =============================================================================
# 7. DIAGNÓSTICO
# =============================================================================

print("\n=== DIAGNÓSTICO DE SENSIBILIDADES EMPÍRICAS ===")

v = eval_points[eval_points["valid"]].copy()

if OPTION_TYPE == "C":
    delta_violation_pct = (v["delta_true"] < 0).mean() * 100
    print(f"Spot sensitivity < 0 (violación en calls): {delta_violation_pct:.1f}%")
else:
    delta_violation_pct = (v["delta_true"] > 0).mean() * 100
    print(f"Spot sensitivity > 0 (violación en puts):  {delta_violation_pct:.1f}%")

curv_neg_pct = (v["gamma_true"] < 0).mean() * 100
print(f"Curvature empirical < 0:                   {curv_neg_pct:.1f}%")

v["m_zone"] = pd.cut(
    v["m_eval"],
    bins=[0, 0.85, 0.95, 1.05, 1.15, 99],
    labels=["OTM deep", "OTM", "ATM", "ITM", "ITM deep"]
)

print("\nCurvature empirical < 0 por zona de moneyness:")
print(
    v.groupby("m_zone", observed=True)["gamma_true"]
     .apply(lambda x: f"{(x < 0).mean() * 100:.1f}%")
     .to_string()
)

# =============================================================================
# 8. GRÁFICOS TIPO LPR
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
model_title = "ΔC ~ ΔS + (ΔS)^2 + Δtau"
if iv_col is not None:
    model_title += " + ΔIV"

fig.suptitle(
    f"Sensibilidades empíricas locales — {OPTION_TYPE} — {eval_date.date()}\n"
    f"{model_title}, lookback={LOOKBACK_DAYS}d, min_hist={MIN_HISTORY_OBS}",
    fontsize=12
)

valid_plot = eval_points[eval_points["valid"]].copy()

# Curvature empirical vs moneyness
ax = axes[0, 0]
sc = ax.scatter(
    valid_plot["m_eval"],
    valid_plot["gamma_true"],
    c=valid_plot["tau_eval"],
    cmap="viridis",
    s=6,
    alpha=0.6
)
plt.colorbar(sc, ax=ax, label="tau (días)")
ax.axhline(0, color="red", ls="--", lw=1.2, alpha=0.8)
ax.set_xlabel("Moneyness K/S")
ax.set_ylabel("Curvature empirical")
ax.set_title("Curvature empirical vs Moneyness (color = tau)")
ax.set_xlim(0.70, 1.30)

# Spot sensitivity empirical vs moneyness
ax = axes[0, 1]
ax.scatter(
    valid_plot["m_eval"],
    valid_plot["delta_true"],
    c=valid_plot["tau_eval"],
    cmap="viridis",
    s=6,
    alpha=0.6
)
ax.axhline(0, color="red", ls="--", lw=1.2, alpha=0.8)
ax.set_xlabel("Moneyness K/S")
ax.set_ylabel("Spot sensitivity empirical")
ax.set_title("Spot sensitivity empirical vs Moneyness (color = tau)")
ax.set_xlim(0.70, 1.30)

# GEX por bucket moneyness
ax = axes[1, 0]
gex_plot = gex_by_m.reset_index()
colors = ["red" if g < 0 else "steelblue" for g in gex_plot["GEX"]]
ax.bar(range(len(gex_plot)), gex_plot["GEX"], color=colors, alpha=0.85)
ax.set_xticks(range(len(gex_plot)))
ax.set_xticklabels(gex_plot["m_bucket"], rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("GEX por bucket moneyness")
ax.set_ylabel("GEX (curvature empirical × OI)")

# GEX por bucket madurez
ax = axes[1, 1]
gex_tau_plot = gex_by_tau.reset_index()
colors = ["red" if g < 0 else "steelblue" for g in gex_tau_plot["GEX"]]
ax.bar(range(len(gex_tau_plot)), gex_tau_plot["GEX"], color=colors, alpha=0.85)
ax.set_xticks(range(len(gex_tau_plot)))
ax.set_xticklabels(gex_tau_plot["tau_bucket"], rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("GEX por bucket madurez")
ax.set_ylabel("GEX (curvature empirical × OI)")

plt.tight_layout()
fname_plot = f"{PATH_OUTPUT}\\empirical_local_sensitivities_{eval_date.date()}_{OPTION_TYPE}.png"
plt.savefig(fname_plot, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nGráfico guardado: {fname_plot}")

# =============================================================================
# 9. OUTPUTS
# =============================================================================

fname_eval = f"{PATH_OUTPUT}\\empirical_local_eval_points_{eval_date.date()}_{OPTION_TYPE}.csv"
eval_points.to_csv(fname_eval, index=False)

fname_df = f"{PATH_OUTPUT}\\empirical_local_contracts_{eval_date.date()}_{OPTION_TYPE}.parquet"
df.to_parquet(fname_df, index=False)

print("\nResultados guardados:")
print(f"  {fname_eval}")
print(f"  {fname_df}")
print("\n=== FIN ===")