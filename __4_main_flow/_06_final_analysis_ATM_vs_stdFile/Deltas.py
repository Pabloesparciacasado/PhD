"""
Comparación de griegas NTM calcalculado con datos de opciones en bruto vs el generado por optionmetrics (std_option_prices).

"""
# In[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
from scipy.stats import norm
import duckdb


# ============================================================
# 1. CARGAR SUPERFICIES
# ============================================================
atm_band = 0.015 # ±1% alrededor de ATM
con = duckdb.connect()
raw_df_clean = con.execute("""SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\superficie_con_greeks_shimko_2_prueba.parquet')
""").df()
raw_df_clean["Date"] = pd.to_datetime(raw_df_clean["Date"])
raw_df_clean = raw_df_clean[['delta','Date', 'Days', 'T', 'rate', 'moneyness',
       'log_moneyness', 'CallPut', 'implied_vol', 'Precio_Modelo', 'vega', 'gamma_bs', 'vanna_K',
       'volga', 'dsigma_dK', 'd2sigma_dK2', 'gamma']]

# Columnas continuas en ATM → se interpolan con la sonrisa completa C+P
interp_cols = ['delta','vega', 'gamma_bs', 'vanna_K', 'volga',
               'dsigma_dK', 'd2sigma_dK2', 'gamma', 'implied_vol', 'T', 'rate']

# La delta NO se interpola: tiene discontinuidad en ATM
# (put ≈ -0.5, call ≈ +0.5). Se recalcula analíticamente en K=F
# usando iv_atm interpolada con la sonrisa completa.
raw_df_NTM = raw_df_clean[
        (raw_df_clean["moneyness"] >= 1 - atm_band) &
        (raw_df_clean["moneyness"] <= 1 + atm_band)]

def interp_atm(group):
    group = group.sort_values("log_moneyness")
    result = {}
    for col in interp_cols:
        result[col] = np.interp(0.0, group["log_moneyness"], group[col])
    return pd.Series(result)

# raw_df_NTM = raw_df_NTM[raw_df_NTM["CallPut"] == "C"]

atm_greeks = (
    raw_df_NTM
    .groupby("Date")
    .apply(interp_atm)
    .reset_index()
    .rename(columns={
        'implied_vol' : 'iv_atm',
        'vega'        : 'vega_atm',
        'gamma_bs'    : 'gamma_bs_atm',
        'vanna_K'     : 'vanna_atm',
        'volga'       : 'volga_atm',
        'dsigma_dK'   : 'dsigma_dK_atm',
        'd2sigma_dK2' : 'd2sigma_dK2_atm',
        'gamma'       : 'gamma_atm',
        'delta'       : 'delta_atm'
    })
)

# Delta analítica en K=F (ATM exacto): ln(F/K)=0 → d1 = 0.5·σ·√T
_df_r   = np.exp(-atm_greeks["rate"] * atm_greeks["T"])
_d1     = 0.5 * atm_greeks["iv_atm"] * np.sqrt(atm_greeks["T"])
_Nd1    = norm.cdf(_d1)
atm_greeks["delta_bs_atm"] = _df_r * (_Nd1 - 1.0)   # put delta Black-76
#atm_greeks["delta_bs_atm"] = _df_r * (_Nd1)   # call delta Black-76



date_start = atm_greeks["Date"].min().strftime("%Y-%m-%d")
date_end   = atm_greeks["Date"].max().strftime("%Y-%m-%d")



std_df = con.execute("""SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\Acumulado\\std_option_price.parquet')
""").df()

std_df = std_df[std_df["Days"]==30]
std_df["Date"] = pd.to_datetime(std_df["Date"])
std_df = std_df[(std_df["Date"]>=date_start) & (std_df['Date']<=date_end)]
std_df = std_df[std_df['SecurityID'] == 108105]
std_df = std_df[std_df["CallPut"] == "P"]
std_df.drop(columns=['SecurityID', 'Days',"Currency",'Theta',"CallPut"], inplace=True)

std_df.rename(columns={
    'ForwardPrice' : 'ForwardPrice_om',
    'Strike'       : 'Strike_om',
    'Premium'      : 'premium_om',
    'ImpliedVol'   : 'iv_om',
    'Delta'        : 'delta_om',
    'Gamma'        : 'gamma_om',
    'Vega'         : 'vega_om',
}, inplace=True) #calculos de option metrics




# iv_calls = atm_greeks_C.set_index("Date")["iv_atm"]
# iv_puts  = atm_greeks_P.set_index("Date")["iv_atm"]

# print((iv_calls - iv_puts).describe())


# In[]
print("==============================================================")
print(std_df)
print("==============================================================")
print(f"Clongitud de raw: {len(atm_greeks)} // longitud de sdt: {len(std_df)}")
# In[]
# ============================================================
# 2. MERGE
# ============================================================

df_merged = atm_greeks.merge(std_df, on="Date",how="inner")
df_merged["year"] = df_merged["Date"].dt.year

print("==============================================================")
print("visualización de la fusión:")
print(df_merged)
print(f"\nObsColumnas ervaciones en muestra común: {df_merged.columns}")
# In[]
# ============================================================
# 3.1. ESTADÍSTICOS: delta_bs_atm vs delta_om
# ============================================================

Delta = "delta_bs_atm"
df = df_merged.dropna(subset=[Delta, "delta_om"])

corr_pearson, p_pearson = stats.pearsonr(df[Delta], df["delta_om"])
corr_spearman, _        = stats.spearmanr(df[Delta], df["delta_om"])
X_pol = np.column_stack([df["delta_om"]**2, df["delta_om"], np.ones(len(df))])
coeffs, _, _, _ = np.linalg.lstsq(X_pol, df[Delta], rcond=None)
c2, c1, c0 = coeffs

y_pred = X_pol @ coeffs
ss_res = np.sum((df[Delta] - y_pred)**2)
ss_tot = np.sum((df[Delta] - df["delta_om"].mean())**2)
r2_pol = 1 - ss_res / ss_tot

rmse = np.sqrt(np.mean((df[Delta] - df["delta_om"])**2))
mae  = np.mean(np.abs(df[Delta] - df["delta_om"]))
bias = np.mean(df[Delta] - df["delta_om"])

print("\n" + "=" * 55)
print("ESTADÍSTICOS GLOBALES: delta_atm vs delta_om")
print("=" * 55)
print(f"  Observaciones        : {len(df):,}")
print(f"  Correlación Pearson  : {corr_pearson:.4f}  (p={p_pearson:.2e})")
print(f"  Correlación Spearman : {corr_spearman:.4f}")
print(f"  Poly2: delta_atm = {c0:.4f} + {c1:.4f}·x + {c2:.4f}·x²")
print(f"  R² (poly2)           : {r2_pol:.4f}")
print(f"  RMSE                 : {rmse:.6f}")
print(f"  MAE                  : {mae:.6f}")
# print(f"  Bias (atm - om)      : {bias:.6f}")
print("=" * 55)

# %%

# ============================================================
# 3.2. TABLA POR AÑO
# ============================================================

rows_year = []
for year, grp in df.groupby("year"):
    if len(grp) < 5:
        continue
    c_p, _  = stats.pearsonr(grp[Delta], grp["delta_om"])
    s, i, r, _, _ = stats.linregress(grp["delta_om"], grp[Delta])
    rmse_y  = np.sqrt(np.mean((grp[Delta] - grp["delta_om"])**2))
    rows_year.append({
        "Año":          year,
        "N":            len(grp),
        "Corr":         round(c_p, 3),
        "R²":           round(r**2, 3),
        "Alpha":        round(i, 4),
        "Beta":         round(s, 4),
        "RMSE":         round(rmse_y, 6),
        "delta_bs_mean": round(grp[Delta].mean(), 4),
        "delta_om_mean": round(grp["delta_om"].mean(), 4),
    })

by_year = pd.DataFrame(rows_year)
print("\n" + "=" * 85)
print(f"TABLA POR AÑO: {Delta} vs delta_om")
print("=" * 85)
print(by_year.to_string(index=False))
print("=" * 85)

# ============================================================
# 3.3. PLOTS
# ============================================================

fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

# ---- Panel 1: Serie temporal ----
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(df["Date"], df[Delta], "b-",  lw=1, alpha=0.8, label="Delta BS ATM (superficie)")
ax1.plot(df["Date"], (df["delta_om"]),     "r--", lw=1, alpha=0.8, label="Delta OptionMetrics")
ax1.set_title(f"{Delta} vs Delta OM — Serie temporal", fontsize=11, fontweight="bold")
ax1.set_ylabel("Delta")
ax1.legend()
ax1.grid(alpha=0.3)

# ---- Panel 2: Scatter global ----
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(df["delta_om"], df[Delta],
            alpha=0.15, s=5, color="steelblue")
x_line = np.linspace(df["delta_om"].min(), df["delta_om"].max(), 100)
y_line = c0 + c1 * x_line + c2 * x_line**2
ax2.plot(x_line, y_line,
         "r-", lw=2, label=f"Poly2: {c0:.4f} + {c1:.4f}·x + {c2:.4f}·x²")
ax2.plot(x_line, x_line, "k--", lw=1, alpha=0.5, label="45° (perfecto)")
ax2.set_xlabel("Delta OptionMetrics")
ax2.set_ylabel(f"{Delta} (superficie)")
ax2.set_title(f"Scatter global  |  R²={r2_pol:.3f}  |  Corr={corr_pearson:.3f}",
              fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ---- Panel 3: Diferencia (delta_atm - delta_om) ----
ax3 = fig.add_subplot(gs[1, 1])
diff = df[Delta] - df["delta_om"]
ax3.plot(df["Date"], diff, "purple", lw=0.8, alpha=0.7)
ax3.axhline(0,            color="black", lw=1,   ls="--")
ax3.axhline(diff.mean(),  color="red",   lw=1.5, ls="--",
            label=f"Media = {diff.mean():.4f}")
ax3.fill_between(df["Date"], diff, 0,
                 where=diff > 0, alpha=0.15, color="red")
ax3.fill_between(df["Date"], diff, 0,
                 where=diff < 0, alpha=0.15, color="blue")
ax3.set_title(f"Diferencia {Delta} − Delta OM", fontsize=9)
ax3.set_ylabel("Diferencia (delta)")
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
            label=f"Global = {rmse*100:.4f}")
ax5.set_title("RMSE por año (×100)", fontsize=9)
ax5.set_ylabel("RMSE (×100)")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3, axis="y")

plt.suptitle(f"Validación: {Delta} (superficie) vs Delta OptionMetrics",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()