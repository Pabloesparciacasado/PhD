"""
Delta/Gamma empíricas temporales suavizadas — Modo gráfico tipo LPR
===================================================================
Construye una estimación empírica temporal local de Delta y Gamma
por OptionID usando una regresión local cuadrática sobre variaciones
de precio de la opción frente a variaciones del spot.

Resultado:
- Delta empírica real por contrato
- Gamma empírica real por contrato
- GEX / DEX agregados por OpenInterest
- Gráficos tipo LPR:
    1) Gamma real vs Moneyness
    2) Delta real vs Moneyness
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

OPTION_TYPE      = "P"      # "C" o "P"
MIN_DS_ABS       = 0.001    # movimiento absoluto mínimo en spot
USE_RELATIVE_DS  = False    # si True, usa también umbral relativo
MIN_DS_REL       = 0.005    # 0.5%
HALF_WINDOW      = 3        # semiventana temporal por OptionID
MIN_OBS_WINDOW   = 5        # mínimo obs útiles en ventana local
USE_TIME_WEIGHTS = True     # ponderar por cercanía temporal
MIN_OI_TOTAL     = 5        # filtro para agregación por punto
COND_MAX         = 1e10     # umbral de condición numérica

PATH_DATA   = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"
PATH_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output"

# =============================================================================
# 2. FUNCIONES AUXILIARES
# =============================================================================

def safe_divide(num, den):
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out


def compute_temporal_local_greeks(
    df: pd.DataFrame,
    half_window: int = 3,
    min_obs_window: int = 5,
    min_ds_abs: float = 0.001,
    use_relative_ds: bool = False,
    min_ds_rel: float = 0.005,
    weighted: bool = True,
    cond_max: float = 1e10
) -> pd.DataFrame:
    """
    Estima Delta y Gamma empíricas mediante regresión local temporal
    para cada OptionID:

        dC = b0 + b1*dS + b2*dS^2

    evaluado localmente alrededor de cada fecha.

    Devuelve:
        Delta_emp_local = b1
        Gamma_emp_local = 2*b2
    """
    df = df.copy().sort_values(["OptionID", "Date"]).reset_index(drop=True)

    df["Delta_emp_local"] = np.nan
    df["Gamma_emp_local"] = np.nan
    df["n_obs_local"] = 0
    df["cond_local"] = np.nan

    grouped = df.groupby("OptionID", sort=False).groups

    for opt_id, idx in grouped.items():
        sub = df.loc[idx].copy().sort_values("Date")
        sub_idx = sub.index.to_numpy()

        C = sub["MidPrice"].to_numpy(dtype=float)
        S = sub["SpotPrice"].to_numpy(dtype=float)
        dates = pd.to_datetime(sub["Date"]).to_numpy()

        n = len(sub)
        if n < min_obs_window:
            continue

        for j in range(n):
            left = max(0, j - half_window)
            right = min(n, j + half_window + 1)
            loc = slice(left, right)

            C_win = C[loc]
            S_win = S[loc]
            dates_win = dates[loc]

            C0 = C[j]
            S0 = S[j]
            t0 = dates[j]

            dC = C_win - C0
            dS = S_win - S0

            abs_mask = np.abs(dS) >= min_ds_abs

            if use_relative_ds:
                rel_mask = np.abs(dS) / max(abs(S0), 1e-12) >= min_ds_rel
                move_mask = abs_mask | rel_mask
            else:
                move_mask = abs_mask

            center_mask = np.zeros_like(move_mask, dtype=bool)
            center_mask[j - left] = True

            mask = move_mask | center_mask

            if mask.sum() < min_obs_window:
                continue

            dC_use = dC[mask]
            dS_use = dS[mask]
            dates_use = dates_win[mask]

            X = np.column_stack([
                np.ones(len(dS_use)),
                dS_use,
                dS_use**2
            ])

            if weighted:
                day_dist = np.abs((dates_use - t0).astype("timedelta64[D]").astype(float))
                max_dist = np.nanmax(day_dist)
                if np.isfinite(max_dist) and max_dist > 0:
                    w = 1.0 - day_dist / max_dist
                    w = np.clip(w, 1e-6, None)
                else:
                    w = np.ones(len(dS_use))
            else:
                w = np.ones(len(dS_use))

            Xw = X * w[:, None]
            XtWX = Xw.T @ X
            XtWy = Xw.T @ dC_use

            try:
                cond_number = np.linalg.cond(XtWX)
                if (not np.isfinite(cond_number)) or (cond_number > cond_max):
                    continue

                beta = np.linalg.solve(XtWX, XtWy)
            except np.linalg.LinAlgError:
                continue

            df.loc[sub_idx[j], "Delta_emp_local"] = beta[1]
            df.loc[sub_idx[j], "Gamma_emp_local"] = 2.0 * beta[2]
            df.loc[sub_idx[j], "n_obs_local"] = int(mask.sum())
            df.loc[sub_idx[j], "cond_local"] = cond_number

    return df


# =============================================================================
# 3. CARGA Y FILTRADO
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
sample_month = opt_df_filtered["YearMonth"].value_counts().idxmax()
print(f"Mes de prueba: {sample_month}")

df = opt_df_filtered[
    (opt_df_filtered["YearMonth"] == sample_month) &
    (opt_df_filtered["CallPut"] == OPTION_TYPE)
].copy()

print(f"Observaciones ({OPTION_TYPE}): {len(df):,}")

# =============================================================================
# 4. DEDUPLICACIÓN POR OPTIONID Y DATE
# =============================================================================

# Mantener la observación más líquida si hubiese duplicados
df = (
    df.sort_values(
        ["OptionID", "Date", "OpenInterest", "Volume"],
        ascending=[True, True, False, False]
    )
    .drop_duplicates(subset=["OptionID", "Date"])
    .reset_index(drop=True)
)

print(f"Observaciones tras deduplicar por OptionID-Date: {len(df):,}")

# =============================================================================
# 5. ESTIMACIÓN DE DELTA/GAMMA EMPÍRICAS TEMPORALES
# =============================================================================

print("\nEstimando Delta/Gamma empíricas temporales suavizadas...")

df = compute_temporal_local_greeks(
    df,
    half_window=HALF_WINDOW,
    min_obs_window=MIN_OBS_WINDOW,
    min_ds_abs=MIN_DS_ABS,
    use_relative_ds=USE_RELATIVE_DS,
    min_ds_rel=MIN_DS_REL,
    weighted=USE_TIME_WEIGHTS,
    cond_max=COND_MAX
)

# Renombrar a nombres tipo LPR
df["delta_true"] = df["Delta_emp_local"]
df["gamma_true"] = df["Gamma_emp_local"]
df["price_hat"] = df["MidPrice"]   # aquí no hay precio suavizado estructural; dejamos el observado

print(f"Delta válida: {(df['delta_true'].notna()).mean()*100:.1f}%")
print(f"Gamma válida: {(df['gamma_true'].notna()).mean()*100:.1f}%")

# =============================================================================
# 6. PUNTOS DE EVALUACIÓN PARA GRÁFICOS TIPO LPR
# =============================================================================

# Igual que en LPR: agrupar por moneyness/tau redondeados
df["m_eval"] = df["Moneyness"].round(3)
df["tau_eval"] = df["Days"].round(0).astype(int)

# Agregación local por punto (promedio ponderado por liquidez)
df["liq_weight"] = (df["OpenInterest"] + df["Volume"]).clip(lower=1)

def weighted_mean(x, w):
    mask = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if mask.sum() == 0:
        return np.nan
    return np.average(x[mask], weights=w[mask])

eval_points = (
    df.groupby(["m_eval", "tau_eval"], as_index=False)
      .apply(lambda g: pd.Series({
          "OI_total": g["OpenInterest"].sum(),
          "n_contracts": g["OptionID"].nunique(),
          "delta_true": weighted_mean(g["delta_true"].values, g["liq_weight"].values),
          "gamma_true": weighted_mean(g["gamma_true"].values, g["liq_weight"].values),
          "price_hat": weighted_mean(g["price_hat"].values, g["liq_weight"].values),
          "n_obs_local_mean": g["n_obs_local"].mean(),
          "cond_local_median": g["cond_local"].median()
      }))
      .reset_index(drop=True)
)

eval_points = eval_points[eval_points["OI_total"] >= MIN_OI_TOTAL].copy()
eval_points["valid"] = eval_points["gamma_true"].notna() & eval_points["delta_true"].notna()

valid_pct = eval_points["valid"].mean() * 100 if len(eval_points) > 0 else 0
print(f"Puntos únicos de evaluación: {len(eval_points):,}")
print(f"Puntos válidos: {eval_points['valid'].sum():,} / {len(eval_points):,} ({valid_pct:.1f}%)")

# =============================================================================
# 7. GEX / DEX POR CONTRATO
# =============================================================================

df["gex_contribution"] = df["gamma_true"] * df["OpenInterest"]
df["dex_contribution"] = df["delta_true"] * df["OpenInterest"]

# =============================================================================
# 8. AGREGACIÓN GEX / DEX
# =============================================================================

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

# Buckets de madurez
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
# 9. DIAGNÓSTICO DE VIOLACIONES
# =============================================================================

print("\n=== DIAGNÓSTICO DE VIOLACIONES ===")

v = eval_points[eval_points["valid"]].copy()

if OPTION_TYPE == "C":
    delta_violation_pct = (v["delta_true"] < 0).mean() * 100
    print(f"Delta < 0 (violación en calls): {delta_violation_pct:.1f}%")
else:
    delta_violation_pct = (v["delta_true"] > 0).mean() * 100
    print(f"Delta > 0 (violación en puts):  {delta_violation_pct:.1f}%")

gamma_violation_pct = (v["gamma_true"] < 0).mean() * 100
print(f"Gamma < 0:                    {gamma_violation_pct:.1f}%")

v["m_zone"] = pd.cut(
    v["m_eval"],
    bins=[0, 0.85, 0.95, 1.05, 1.15, 99],
    labels=["OTM deep", "OTM", "ATM", "ITM", "ITM deep"]
)

print("\nGamma < 0 por zona de moneyness:")
print(
    v.groupby("m_zone", observed=True)["gamma_true"]
     .apply(lambda x: f"{(x < 0).mean() * 100:.1f}%")
     .to_string()
)

# =============================================================================
# 10. GRÁFICOS TIPO LPR
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"Delta/Gamma empíricas temporales suavizadas — {OPTION_TYPE} — {sample_month}\n"
    f"half_window={HALF_WINDOW}, min_obs={MIN_OBS_WINDOW}, min_dS_abs={MIN_DS_ABS}",
    fontsize=12
)

valid_plot = eval_points[eval_points["valid"]].copy()

# Gamma empírica vs moneyness
ax = axes[0, 0]
sc = ax.scatter(
    valid_plot["m_eval"],
    valid_plot["gamma_true"],
    c=valid_plot["tau_eval"],
    cmap="viridis",
    s=4,
    alpha=0.6
)
plt.colorbar(sc, ax=ax, label="tau (días)")
ax.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("Moneyness K/S")
ax.set_ylabel("Gamma empírica")
ax.set_title("Gamma empírica vs Moneyness (color = tau)")
ax.set_xlim(0.70, 1.30)

# Delta empírica vs moneyness
ax = axes[0, 1]
ax.scatter(
    valid_plot["m_eval"],
    valid_plot["delta_true"],
    c=valid_plot["tau_eval"],
    cmap="viridis",
    s=4,
    alpha=0.6
)
ax.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("Moneyness K/S")
ax.set_ylabel("Delta empírica")
ax.set_title("Delta empírica vs Moneyness (color = tau)")
ax.set_xlim(0.70, 1.30)

# GEX por bucket moneyness
ax = axes[1, 0]
gex_plot = gex_by_m.reset_index()
colors = ["red" if g < 0 else "steelblue" for g in gex_plot["GEX"]]
ax.bar(range(len(gex_plot)), gex_plot["GEX"], color=colors, alpha=0.8)
ax.set_xticks(range(len(gex_plot)))
ax.set_xticklabels(gex_plot["m_bucket"], rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("GEX por bucket moneyness")
ax.set_ylabel("GEX (gamma empírica × OI)")

# GEX por bucket madurez
ax = axes[1, 1]
gex_tau_plot = gex_by_tau.reset_index()
colors = ["red" if g < 0 else "steelblue" for g in gex_tau_plot["GEX"]]
ax.bar(range(len(gex_tau_plot)), gex_tau_plot["GEX"], color=colors, alpha=0.8)
ax.set_xticks(range(len(gex_tau_plot)))
ax.set_xticklabels(gex_tau_plot["tau_bucket"], rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("GEX por bucket madurez")
ax.set_ylabel("GEX (gamma empírica × OI)")

plt.tight_layout()
fname_plot = f"{PATH_OUTPUT}\\empirical_temporal_greeks_{sample_month}_{OPTION_TYPE}.png"
plt.savefig(fname_plot, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nGráfico guardado: {fname_plot}")

# =============================================================================
# 11. OUTPUTS
# =============================================================================

fname_eval = f"{PATH_OUTPUT}\\empirical_eval_points_{sample_month}_{OPTION_TYPE}.csv"
eval_points.to_csv(fname_eval, index=False)

fname_df = f"{PATH_OUTPUT}\\empirical_contracts_{sample_month}_{OPTION_TYPE}.parquet"
df.to_parquet(fname_df, index=False)

print("\nResultados guardados:")
print(f"  {fname_eval}")
print(f"  {fname_df}")
print("\n=== FIN ===")