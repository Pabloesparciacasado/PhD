
### 06.3 Analizar si las griegas obtenidas en 07 tienen sentido, comparando con VIX, y bajo supuestos de Gatheral(2004) y Breeden-Liztenberg(1978).

### Comparación de griegas desde el fichero std_option (ATM),(ver si desde el de vol_surface puedo ampliar a más strikes el std_option).

"""
Comparación de la IV ATM a 30 días de la superficie interpolada
con el VIX oficial de CBOE.

Nota metodológica:
    El VIX no es la IV ATM — es la raíz cuadrada de la varianza
    total integrada sobre toda la smile (CBOE 2003/2014). La
    comparación aquí es IV ATM vs VIX/100, que es una aproximación
    válida como sanity check dado que ambas miden volatilidad
    implícita a 30 días y tienen correlación histórica >0.95.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
from scipy import stats
import duckdb
 
# ============================================================
# 1. CARGAR SUPERFICIE LIMPIA
# ============================================================

con = duckdb.connect()

df_clean = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\volatility_surface_30_flat_natural.parquet')
""").df()

df_clean["Date"] = pd.to_datetime(df_clean["Date"])

# ============================================================
# 2. EXTRAER IV ATM POR FECHA
# ============================================================
# ATM: punto de la grid más cercano a moneyness = 1
# Usamos calls para consistencia (puts dan el mismo resultado
# por put-call parity en ATM)

atm_band = 0.01  # ±1% alrededor de ATM

atm_iv = (
    df_clean[
        (df_clean["moneyness"] >= 1 - atm_band) &
        (df_clean["moneyness"] <= 1 + atm_band) &
        (df_clean["CallPut"] == "C")
    ]
    .groupby("Date")["implied_vol"]
    .mean()
    .reset_index()
    .rename(columns={"implied_vol": "iv_atm"})
)

print(f"Fechas con IV ATM: {len(atm_iv)}")
print(f"Periodo: {atm_iv['Date'].min().date()} — {atm_iv['Date'].max().date()}")

# ============================================================
# 3. DESCARGAR VIX
# ============================================================

date_start = atm_iv["Date"].min().strftime("%Y-%m-%d")
date_end   = atm_iv["Date"].max().strftime("%Y-%m-%d")

print(f"\nDescargando VIX desde {date_start} hasta {date_end}...")

vix_raw = yf.download("^VIX", start=date_start, end=date_end, progress=False)
vix_raw = vix_raw[["Close"]].reset_index()
vix_raw.columns = ["Date", "VIX"]
vix_raw["Date"] = pd.to_datetime(vix_raw["Date"])
vix_raw["vix_iv"] = vix_raw["VIX"] / 100.0  # VIX en % → IV en decimal

print(f"Observaciones VIX descargadas: {len(vix_raw)}")

# ============================================================
# 4. MERGE
# ============================================================

df = atm_iv.merge(vix_raw[["Date", "VIX", "vix_iv"]], on="Date", how="inner")
df = df.dropna(subset=["iv_atm", "vix_iv"])
df["year"] = df["Date"].dt.year

print(f"\nObservaciones en muestra común: {len(df)}")

# ============================================================
# 5. ESTADÍSTICOS GLOBALES
# ============================================================

corr_pearson, p_pearson = stats.pearsonr(df["iv_atm"], df["vix_iv"])
corr_spearman, _        = stats.spearmanr(df["iv_atm"], df["vix_iv"])

# Regresión OLS: iv_atm = alpha + beta * vix_iv + eps
slope, intercept, r_value, p_value, std_err = stats.linregress(
    df["vix_iv"], df["iv_atm"]
)
rmse = np.sqrt(np.mean((df["iv_atm"] - df["vix_iv"])**2))
mae  = np.mean(np.abs(df["iv_atm"] - df["vix_iv"]))

print("\n" + "=" * 55)
print("ESTADÍSTICOS GLOBALES")
print("=" * 55)
print(f"  Observaciones        : {len(df):,}")
print(f"  Correlación Pearson  : {corr_pearson:.4f}  (p={p_pearson:.2e})")
print(f"  Correlación Spearman : {corr_spearman:.4f}")
print(f"  OLS: iv_atm = {intercept:.4f} + {slope:.4f} * vix_iv")
print(f"  R²                   : {r_value**2:.4f}")
print(f"  RMSE                 : {rmse:.4f}  ({rmse*100:.2f} vol points)")
print(f"  MAE                  : {mae:.4f}  ({mae*100:.2f} vol points)")
print("=" * 55)

# ============================================================
# 6. TABLA POR AÑO
# ============================================================

rows_year = []
for year, grp in df.groupby("year"):
    if len(grp) < 5:
        continue
    c_p, _  = stats.pearsonr(grp["iv_atm"], grp["vix_iv"])
    s, i, r, _, _ = stats.linregress(grp["vix_iv"], grp["iv_atm"])
    rmse_y  = np.sqrt(np.mean((grp["iv_atm"] - grp["vix_iv"])**2))
    rows_year.append({
        "Año":   year,
        "N":     len(grp),
        "Corr":  round(c_p, 3),
        "R²":    round(r**2, 3),
        "Alpha": round(i, 4),
        "Beta":  round(s, 4),
        "RMSE":  round(rmse_y, 4),
        "IV_ATM_mean": round(grp["iv_atm"].mean(), 4),
        "VIX_mean":    round(grp["vix_iv"].mean(), 4),
    })

by_year = pd.DataFrame(rows_year)
print("\n" + "=" * 85)
print("TABLA POR AÑO")
print("=" * 85)
print(by_year.to_string(index=False))
print("=" * 85)

# ============================================================
# 7. PLOTS
# ============================================================

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# ---- Panel 1: Serie temporal ----
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df["Date"], df["iv_atm"],  "b-",  lw=1,   alpha=0.8, label="IV ATM (superficie)")
ax1.plot(df["Date"], df["vix_iv"],  "r--", lw=1,   alpha=0.8, label="VIX / 100")
ax1.set_title("IV ATM vs VIX — Serie temporal", fontsize=11, fontweight="bold")
ax1.set_ylabel("Volatilidad implícita")
ax1.legend()
ax1.grid(alpha=0.3)

# ---- Panel 2: Scatter global ----
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(df["vix_iv"], df["iv_atm"],
            alpha=0.15, s=5, color="steelblue")
x_line = np.linspace(df["vix_iv"].min(), df["vix_iv"].max(), 100)
ax2.plot(x_line, intercept + slope * x_line,
         "r-", lw=2, label=f"OLS: α={intercept:.3f}, β={slope:.3f}")
ax2.plot(x_line, x_line, "k--", lw=1, alpha=0.5, label="45° (perfecto)")
ax2.set_xlabel("VIX / 100")
ax2.set_ylabel("IV ATM superficie")
ax2.set_title(f"Scatter global  |  R²={r_value**2:.3f}  |  Corr={corr_pearson:.3f}",
              fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ---- Panel 3: Diferencia (IV_ATM - VIX) ----
ax3 = fig.add_subplot(gs[1, 1])
diff = df["iv_atm"] - df["vix_iv"]
ax3.plot(df["Date"], diff, "purple", lw=0.8, alpha=0.7)
ax3.axhline(0,            color="black", lw=1,   ls="--")
ax3.axhline(diff.mean(),  color="red",   lw=1.5, ls="--",
            label=f"Media = {diff.mean()*100:.2f} vol pts")
ax3.fill_between(df["Date"], diff, 0,
                 where=diff > 0, alpha=0.15, color="red")
ax3.fill_between(df["Date"], diff, 0,
                 where=diff < 0, alpha=0.15, color="blue")
ax3.set_title("Diferencia IV ATM − VIX/100", fontsize=9)
ax3.set_ylabel("Diferencia")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ---- Panel 4: Correlación por año ----
ax4 = fig.add_subplot(gs[2, 0])
ax4.bar(by_year["Año"], by_year["Corr"],
        color="steelblue", alpha=0.7, edgecolor="white")
ax4.axhline(corr_pearson, color="red", ls="--", lw=1.5,
            label=f"Global = {corr_pearson:.3f}")
ax4.set_ylim(0, 1.05)
ax4.set_title("Correlación Pearson por año", fontsize=9)
ax4.set_ylabel("Correlación")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3, axis="y")

# ---- Panel 5: RMSE por año ----
ax5 = fig.add_subplot(gs[2, 1])
ax5.bar(by_year["Año"], by_year["RMSE"] * 100,
        color="tomato", alpha=0.7, edgecolor="white")
ax5.axhline(rmse * 100, color="black", ls="--", lw=1.5,
            label=f"Global = {rmse*100:.2f} vol pts")
ax5.set_title("RMSE por año (vol points)", fontsize=9)
ax5.set_ylabel("RMSE (×100)")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3, axis="y")

plt.suptitle("Validación: IV ATM superficie vs VIX oficial",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()

# ============================================================
# 8. EXPORT
# ============================================================

df.to_csv("vix_comparison_daily.csv", index=False)
by_year.to_csv("vix_comparison_by_year.csv", index=False)
print("\nExportado: vix_comparison_daily.csv, vix_comparison_by_year.csv")