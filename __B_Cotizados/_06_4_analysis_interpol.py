"""
compare_surfaces.py
===================
Comparación entre superficie con spline natural y superficie con
Shimko polinomio grado 4.

Comparaciones:
1. IV ATM y smile shape
2. Checks de no-arbitraje (monotonicidad, convexidad)
3. Griegas (delta, gamma_bs, gamma, vega)
4. Correlación y diferencias en cada variable

Requiere:
- volatility_surface_30_flat_natural.parquet  (spline)
- volatility_surface_30_shimko.parquet        (shimko)
- surface_analysis.py
- greeks.py
- model_valuation (tu clase)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import duckdb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from _06_2_analysis_arbitrage_y_limpieza import run_surface_analysis
from _07_greeks import compute_greeks, check_greeks

# ============================================================
# 0. PARÁMETROS
# ============================================================

PATH_SPLINE = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\volatility_surface_30_flat_natural.parquet"
PATH_SHIMKO = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\volatility_surface_30_shimko.parquet"

ATM_BAND   = 0.02   # ±2% alrededor de ATM para comparación ATM
MONO_THRESH = 1e-4
CONV_THRESH = 1e-3

# ============================================================
# 1. CARGAR SUPERFICIES Y CALCULAR PRECIOS BS
# ============================================================

from scipy.stats import norm

def price_bs(df: pd.DataFrame) -> pd.DataFrame:
    """Replica tu price_BS_general sin necesitar model_valuation."""
    df = df.copy()
    F  = df["forward"].to_numpy(dtype=float)
    K  = df["Strike"].to_numpy(dtype=float)
    T  = df["T"].to_numpy(dtype=float)
    iv = df["implied_vol"].to_numpy(dtype=float)
    DF = df["discount_factor"].to_numpy(dtype=float)
    cp = df["CallPut"].to_numpy()

    iv_s = np.maximum(iv, 1e-8)
    d1   = (np.log(F / K) + 0.5 * iv_s**2 * T) / (iv_s * np.sqrt(T))
    d2   = d1 - iv_s * np.sqrt(T)

    price = np.where(
        cp == "C",
        DF * (F * norm.cdf(d1)  - K * norm.cdf(d2)),
        DF * (-F * norm.cdf(-d1) + K * norm.cdf(-d2)),
    )
    df["Precio_Modelo"] = price
    return df


con = duckdb.connect()

print("Cargando superficies...")
spline_raw = con.execute(f"SELECT * FROM read_parquet('{PATH_SPLINE}')").df()
shimko_raw = con.execute(f"SELECT * FROM read_parquet('{PATH_SHIMKO}')").df()

spline_raw["Date"] = pd.to_datetime(spline_raw["Date"])
shimko_raw["Date"] = pd.to_datetime(shimko_raw["Date"])

print(f"  Spline : {len(spline_raw):,} filas, {spline_raw['Date'].nunique()} fechas")
print(f"  Shimko : {len(shimko_raw):,} filas, {shimko_raw['Date'].nunique()} fechas")

# Calcular precios
spline_priced = price_bs(spline_raw)
shimko_priced = price_bs(shimko_raw)

# ============================================================
# 2. NO-ARBITRAJE
# ============================================================

print("\nEjecutando checks de no-arbitraje...")

_, diag_spline = run_surface_analysis(
    spline_priced,
    tol_mono=MONO_THRESH, tol_conv=CONV_THRESH,
    drop_bounds_fail=False, drop_mono_fail=False, drop_conv_fail=False,
    verbose=False, plot_sample=False,
)
_, diag_shimko = run_surface_analysis(
    shimko_priced,
    tol_mono=MONO_THRESH, tol_conv=CONV_THRESH,
    drop_bounds_fail=False, drop_mono_fail=False, drop_conv_fail=False,
    verbose=False, plot_sample=False,
)

n_s = len(diag_spline)
n_h = len(diag_shimko)

print("\n" + "=" * 65)
print("NO-ARBITRAJE — COMPARACIÓN")
print("=" * 65)
print(f"  {'Check':35s}  {'Spline':>10}  {'Shimko':>10}")
print(f"  {'-'*35}  {'-'*10}  {'-'*10}")
for flag, label in [
    ("flag_mono_ok", "Monotonicidad OK (%)"),
    ("flag_conv_ok", "Convexidad OK (%)"),
]:
    pct_s = diag_spline[flag].mean() * 100
    pct_h = diag_shimko[flag].mean() * 100
    print(f"  {label:35s}  {pct_s:>9.1f}%  {pct_h:>9.1f}%")
print("=" * 65)

# ============================================================
# 3. IV ATM Y SMILE SHAPE
# ============================================================

def extract_atm(df: pd.DataFrame, label: str) -> pd.DataFrame:
    atm = (
        df[
            (df["moneyness"] >= 1 - ATM_BAND) &
            (df["moneyness"] <= 1 + ATM_BAND) &
            (df["CallPut"] == "C")
        ]
        .groupby("Date")["implied_vol"]
        .mean()
        .reset_index()
        .rename(columns={"implied_vol": f"iv_atm_{label}"})
    )
    return atm

atm_spline = extract_atm(spline_raw, "spline")
atm_shimko = extract_atm(shimko_raw, "shimko")
atm = atm_spline.merge(atm_shimko, on="Date", how="inner")

corr_atm, _ = stats.pearsonr(atm["iv_atm_spline"], atm["iv_atm_shimko"])
diff_atm     = atm["iv_atm_spline"] - atm["iv_atm_shimko"]
rmse_atm     = np.sqrt(np.mean(diff_atm**2))

print(f"\nIV ATM — Correlación: {corr_atm:.4f} | RMSE: {rmse_atm*100:.3f} vol pts | "
      f"Diff media: {diff_atm.mean()*100:.3f} vol pts")

# Smile shape: OTM skew (IV put 0.9 - IV call 1.1)
def extract_skew(df: pd.DataFrame, label: str) -> pd.DataFrame:
    put_otm = (
        df[(df["moneyness"].between(0.88, 0.92)) & (df["CallPut"] == "P")]
        .groupby("Date")["implied_vol"].mean()
        .reset_index().rename(columns={"implied_vol": "iv_put_otm"})
    )
    call_otm = (
        df[(df["moneyness"].between(1.08, 1.12)) & (df["CallPut"] == "C")]
        .groupby("Date")["implied_vol"].mean()
        .reset_index().rename(columns={"implied_vol": "iv_call_otm"})
    )
    skew = put_otm.merge(call_otm, on="Date", how="inner")
    skew[f"skew_{label}"] = skew["iv_put_otm"] - skew["iv_call_otm"]
    return skew[["Date", f"skew_{label}"]]

skew_spline = extract_skew(spline_raw, "spline")
skew_shimko = extract_skew(shimko_raw, "shimko")
skew = skew_spline.merge(skew_shimko, on="Date", how="inner")

corr_skew, _ = stats.pearsonr(skew["skew_spline"], skew["skew_shimko"])
diff_skew     = skew["skew_spline"] - skew["skew_shimko"]

print(f"Skew OTM  — Correlación: {corr_skew:.4f} | "
      f"Diff media: {diff_skew.mean()*100:.3f} vol pts")

# ============================================================
# 4. GRIEGAS
# ============================================================

print("\nCalculando griegas...")

# Limpiar primero
df_spline_clean, _ = run_surface_analysis(
    spline_priced, verbose=False, plot_sample=False
)
df_shimko_clean, _ = run_surface_analysis(
    shimko_priced, verbose=False, plot_sample=False
)

greeks_spline = compute_greeks(df_spline_clean)
greeks_shimko = compute_greeks(df_shimko_clean)

print("\nSanity checks spline:")
check_greeks(greeks_spline, verbose=True)
print("\nSanity checks Shimko:")
check_greeks(greeks_shimko, verbose=True)

# Comparar griegas ATM
def extract_greeks_atm(df: pd.DataFrame, label: str) -> pd.DataFrame:
    atm = (
        df[
            (df["moneyness"] >= 1 - ATM_BAND) &
            (df["moneyness"] <= 1 + ATM_BAND) &
            (df["CallPut"] == "C")
        ]
        .groupby("Date")[["delta", "vega", "gamma_bs", "gamma"]]
        .mean()
        .reset_index()
    )
    atm.columns = ["Date"] + [f"{c}_{label}" for c in ["delta", "vega", "gamma_bs", "gamma"]]
    return atm

atm_g_spline = extract_greeks_atm(greeks_spline, "spline")
atm_g_shimko = extract_greeks_atm(greeks_shimko, "shimko")
atm_g = atm_g_spline.merge(atm_g_shimko, on="Date", how="inner")

print("\n" + "=" * 75)
print("GRIEGAS ATM — CORRELACIÓN Y DIFERENCIAS")
print("=" * 75)
print(f"  {'Variable':20s}  {'Corr':>8}  {'RMSE':>12}  {'Diff media':>12}")
print(f"  {'-'*20}  {'-'*8}  {'-'*12}  {'-'*12}")
for col in ["delta", "vega", "gamma_bs", "gamma"]:
    c_s = f"{col}_spline"
    c_h = f"{col}_shimko"
    if c_s not in atm_g.columns or c_h not in atm_g.columns:
        continue
    corr, _ = stats.pearsonr(atm_g[c_s].dropna(), atm_g[c_h].dropna())
    diff     = atm_g[c_s] - atm_g[c_h]
    rmse     = np.sqrt(np.mean(diff**2))
    print(f"  {col:20s}  {corr:>8.4f}  {rmse:>12.6f}  {diff.mean():>12.6f}")
print("=" * 75)

# ============================================================
# 5. PLOTS
# ============================================================

fig = plt.figure(figsize=(18, 16))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# ---- IV ATM serie temporal ----
ax = fig.add_subplot(gs[0, :])
ax.plot(atm["Date"], atm["iv_atm_spline"], "b-",  lw=1, alpha=0.8, label="Spline")
ax.plot(atm["Date"], atm["iv_atm_shimko"], "r--", lw=1, alpha=0.8, label="Shimko")
ax.set_title(f"IV ATM — Spline vs Shimko  |  Corr={corr_atm:.4f}  |  RMSE={rmse_atm*100:.2f} vol pts",
             fontsize=10)
ax.set_ylabel("IV ATM")
ax.legend()
ax.grid(alpha=0.3)

# ---- Scatter IV ATM ----
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(atm["iv_atm_spline"], atm["iv_atm_shimko"],
            alpha=0.15, s=5, color="steelblue")
lim = [min(atm["iv_atm_spline"].min(), atm["iv_atm_shimko"].min()),
       max(atm["iv_atm_spline"].max(), atm["iv_atm_shimko"].max())]
ax2.plot(lim, lim, "k--", lw=1, alpha=0.5)
ax2.set_xlabel("IV ATM Spline")
ax2.set_ylabel("IV ATM Shimko")
ax2.set_title(f"Scatter IV ATM  |  Corr={corr_atm:.4f}", fontsize=9)
ax2.grid(alpha=0.3)

# ---- Scatter gamma ----
ax3 = fig.add_subplot(gs[1, 1])
if "gamma_spline" in atm_g.columns and "gamma_shimko" in atm_g.columns:
    ax3.scatter(atm_g["gamma_spline"], atm_g["gamma_shimko"],
                alpha=0.15, s=5, color="tomato")
    lim_g = [min(atm_g["gamma_spline"].min(), atm_g["gamma_shimko"].min()),
             max(atm_g["gamma_spline"].max(), atm_g["gamma_shimko"].max())]
    ax3.plot(lim_g, lim_g, "k--", lw=1, alpha=0.5)
    corr_g, _ = stats.pearsonr(atm_g["gamma_spline"].dropna(),
                                atm_g["gamma_shimko"].dropna())
    ax3.set_xlabel("Gamma Spline")
    ax3.set_ylabel("Gamma Shimko")
    ax3.set_title(f"Scatter Gamma ATM  |  Corr={corr_g:.4f}", fontsize=9)
    ax3.grid(alpha=0.3)

# ---- Scatter delta ----
ax4 = fig.add_subplot(gs[1, 2])
if "delta_spline" in atm_g.columns and "delta_shimko" in atm_g.columns:
    ax4.scatter(atm_g["delta_spline"], atm_g["delta_shimko"],
                alpha=0.15, s=5, color="purple")
    lim_d = [min(atm_g["delta_spline"].min(), atm_g["delta_shimko"].min()),
             max(atm_g["delta_spline"].max(), atm_g["delta_shimko"].max())]
    ax4.plot(lim_d, lim_d, "k--", lw=1, alpha=0.5)
    corr_d, _ = stats.pearsonr(atm_g["delta_spline"].dropna(),
                                atm_g["delta_shimko"].dropna())
    ax4.set_xlabel("Delta Spline")
    ax4.set_ylabel("Delta Shimko")
    ax4.set_title(f"Scatter Delta ATM  |  Corr={corr_d:.4f}", fontsize=9)
    ax4.grid(alpha=0.3)

# ---- No-arbitraje por año ----
diag_spline["year"] = pd.to_datetime(diag_spline["Date"]).dt.year
diag_shimko["year"] = pd.to_datetime(diag_shimko["Date"]).dt.year

by_yr_s = diag_spline.groupby("year").agg(
    mono_pct=("flag_mono_ok", lambda s: (~s).mean() * 100),
    conv_pct=("flag_conv_ok", lambda s: (~s).mean() * 100),
).reset_index()
by_yr_h = diag_shimko.groupby("year").agg(
    mono_pct=("flag_mono_ok", lambda s: (~s).mean() * 100),
    conv_pct=("flag_conv_ok", lambda s: (~s).mean() * 100),
).reset_index()

years = by_yr_s["year"].to_numpy()
x     = np.arange(len(years))
w     = 0.35

ax5 = fig.add_subplot(gs[2, :2])
ax5.bar(x - w/2, by_yr_s["mono_pct"], w, label="Spline", color="steelblue", alpha=0.7)
ax5.bar(x + w/2, by_yr_h["mono_pct"], w, label="Shimko", color="tomato",    alpha=0.7)
ax5.set_xticks(x)
ax5.set_xticklabels(years, rotation=45, fontsize=7)
ax5.set_ylabel("% slices con violación")
ax5.set_title("Monotonicidad — % violaciones por año", fontsize=9)
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3, axis="y")

ax6 = fig.add_subplot(gs[2, 2])
ax6.bar(x - w/2, by_yr_s["conv_pct"], w, label="Spline", color="steelblue", alpha=0.7)
ax6.bar(x + w/2, by_yr_h["conv_pct"], w, label="Shimko", color="tomato",    alpha=0.7)
ax6.set_xticks(x)
ax6.set_xticklabels(years, rotation=45, fontsize=7)
ax6.set_ylabel("% slices con violación")
ax6.set_title("Convexidad — % violaciones por año", fontsize=9)
ax6.legend(fontsize=8)
ax6.grid(alpha=0.3, axis="y")

plt.suptitle("Spline Natural vs Shimko Grado 4 — Comparación completa",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.show()

# ============================================================
# 6. TABLA RESUMEN FINAL
# ============================================================

print("\n" + "=" * 65)
print("RESUMEN FINAL")
print("=" * 65)
print(f"  Fechas comunes          : {len(atm):,}")
print(f"  IV ATM correlación      : {corr_atm:.4f}")
print(f"  IV ATM RMSE             : {rmse_atm*100:.3f} vol pts")
print(f"  Skew OTM correlación    : {corr_skew:.4f}")
print()
print(f"  No-arbitraje spline     : mono={diag_spline['flag_mono_ok'].mean()*100:.1f}%  "
      f"conv={diag_spline['flag_conv_ok'].mean()*100:.1f}%")
print(f"  No-arbitraje shimko     : mono={diag_shimko['flag_mono_ok'].mean()*100:.1f}%  "
      f"conv={diag_shimko['flag_conv_ok'].mean()*100:.1f}%")
print("=" * 65)





# In[]


import duckdb
con = duckdb.connect()

spline = con.execute("""
    SELECT * FROM read_parquet("C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\volatility_surface_30_flat_natural.parquet")
    LIMIT 5
""").df()

shimko = con.execute("""
    SELECT * FROM read_parquet("C:\\\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\volatility_surface_30_shimko.parquet")
    LIMIT 5
""").df()

# Si son iguales, mismo fichero
print((spline == shimko).all().all())

# Comprobar si tienen columnas de Shimko
print("dsigma_dm" in shimko.columns)  # True solo si es Shimko real
# %%
