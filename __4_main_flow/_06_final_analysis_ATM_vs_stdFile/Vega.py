"""
Comparación de griegas NTM calcalculado con datos de opciones en bruto vs el generado por optionmetrics (std_option_prices).

"""
# In[]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy import stats
import duckdb


# ============================================================
# 1. CARGAR SUPERFICIES
# ============================================================
atm_band = 0.015  # ±1% alrededor de ATM
con = duckdb.connect()
raw_df_clean = con.execute("""SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\superficie_con_greeks_shimko_2.parquet')
""").df()
raw_df_clean["Date"] = pd.to_datetime(raw_df_clean["Date"])
raw_df_clean = raw_df_clean[['Date', 'Days','moneyness',
       'log_moneyness', 'CallPut','implied_vol', 'Precio_Modelo', 'delta_bs', 'vega', 'gamma_bs', 'vanna_K',
       'volga', 'dsigma_dK', 'd2sigma_dK2', 'delta', 'gamma']]

raw_df_clean = raw_df_clean[raw_df_clean["CallPut"] == "C"]

raw_df_NTM = raw_df_clean[
        (raw_df_clean["moneyness"] >= 1 - atm_band) &
        (raw_df_clean["moneyness"] <= 1 + atm_band)]

greek_cols = ['delta_bs', 'vega', 'gamma_bs', 'vanna_K', 'volga',
              'dsigma_dK', 'd2sigma_dK2', 'delta', 'gamma', 'implied_vol']

atm_greeks = (
    raw_df_NTM
    .groupby("Date")[greek_cols]
    .mean()
    .reset_index()
    .rename(columns={
        'implied_vol' : 'iv_atm',
        'delta_bs'    : 'delta_bs_atm',
        'vega'        : 'vega_atm',
        'gamma_bs'    : 'gamma_bs_atm',
        'vanna_K'     : 'vanna_atm',
        'volga'       : 'volga_atm',
        'dsigma_dK'   : 'dsigma_dK_atm',
        'd2sigma_dK2' : 'd2sigma_dK2_atm',
        'delta'       : 'delta_atm',
        'gamma'       : 'gamma_atm',
    })
)

def interp_atm(group):
    group = group.sort_values("log_moneyness")
    result = {}
    for col in greek_cols:
        result[col] = np.interp(0.0, group["log_moneyness"], group[col])
    return pd.Series(result)

atm_greeks = (
    raw_df_NTM
    .groupby("Date")
    .apply(interp_atm)
    .reset_index()
    .rename(columns={
        'implied_vol' : 'iv_atm',
        'delta_bs'    : 'delta_bs_atm',
        'vega'        : 'vega_atm',
        'gamma_bs'    : 'gamma_bs_atm',
        'vanna_K'     : 'vanna_atm',
        'volga'       : 'volga_atm',
        'dsigma_dK'   : 'dsigma_dK_atm',
        'd2sigma_dK2' : 'd2sigma_dK2_atm',
        'delta'       : 'delta_atm',
        'gamma'       : 'gamma_atm',
    })
)

date_start = atm_greeks["Date"].min().strftime("%Y-%m-%d")
date_end   = atm_greeks["Date"].max().strftime("%Y-%m-%d")



std_df = con.execute("""SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\Acumulado\\std_option_price.parquet')
""").df()

std_df = std_df[std_df["Days"]==30]
std_df["Date"] = pd.to_datetime(std_df["Date"])
std_df = std_df[(std_df["Date"]>=date_start) & (std_df['Date']<=date_end)]
std_df = std_df[std_df['SecurityID'] == 108105]
std_df = std_df[std_df["CallPut"] == "C"]
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

# In[]
    # print("==============================================================")
    # print(std_df)
    # print("==============================================================")
    # print(f"Longitud de raw: {len(atm_greeks)} // longitud de std: {len(std_df)}")
# In[]
# ============================================================
# 2. MERGE
# ============================================================

df_merged = atm_greeks.merge(std_df, on="Date", how="inner")
df_merged["year"] = df_merged["Date"].dt.year

    # print("==============================================================")
    # print("visualización de la fusión:")
    # print(df_merged)
    # print(f"\nColumnas y observaciones en muestra común: {df_merged.columns}")
# In[]
# ============================================================
# 3.1. ESTADÍSTICOS: vega_atm vs vega_om
# ============================================================
df = df_merged.dropna(subset=["vega_atm", "vega_om"])

corr_pearson, p_pearson = stats.pearsonr(df["vega_atm"], df["vega_om"])
corr_spearman, _        = stats.spearmanr(df["vega_atm"], df["vega_om"])

slope, intercept, r_value, p_value, std_err = stats.linregress(
    df["vega_om"], df["vega_atm"]
)
rmse = np.sqrt(np.mean((df["vega_atm"] - df["vega_om"])**2))
mae  = np.mean(np.abs(df["vega_atm"] - df["vega_om"]))
bias = np.mean(df["vega_atm"] - df["vega_om"])

print("\n" + "=" * 55)
print("ESTADÍSTICOS GLOBALES: vega_atm vs vega_om")
print("=" * 55)
print(f"  Observaciones        : {len(df):,}")
print(f"  Correlación Pearson  : {corr_pearson:.4f}  (p={p_pearson:.2e})")
print(f"  Correlación Spearman : {corr_spearman:.4f}")
print(f"  OLS: vega_atm = {intercept:.6f} + {slope:.4f} * vega_om")
print(f"  R²                   : {r_value**2:.4f}")
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
    c_p, _  = stats.pearsonr(grp["vega_atm"], grp["vega_om"])
    s, i, r, _, _ = stats.linregress(grp["vega_om"], grp["vega_atm"])
    rmse_y  = np.sqrt(np.mean((grp["vega_atm"] - grp["vega_om"])**2))
    rows_year.append({
        "Año":           year,
        "N":             len(grp),
        "Corr":          round(c_p, 3),
        "R²":            round(r**2, 3),
        "Alpha":         round(i, 6),
        "Beta":          round(s, 4),
        "RMSE":          round(rmse_y, 6),
        "vega_atm_mean": round(grp["vega_atm"].mean(), 6),
        "vega_om_mean":  round(grp["vega_om"].mean(), 6),
    })

by_year = pd.DataFrame(rows_year)
print("\n" + "=" * 85)
print("TABLA POR AÑO: vega_atm vs vega_om")
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
ax1.plot(df["Date"], df["vega_atm"], "b-",  lw=1, alpha=0.8, label="Vega BS ATM (superficie)")
ax1.plot(df["Date"], df["vega_om"],  "r--", lw=1, alpha=0.8, label="Vega OptionMetrics")
ax1.set_title("Vega BS ATM vs Vega OM — Serie temporal", fontsize=11, fontweight="bold")
ax1.set_ylabel("Vega")
ax1.legend()
ax1.grid(alpha=0.3)

# ---- Panel 2: Scatter global ----
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(df["vega_om"], df["vega_atm"],
            alpha=0.15, s=5, color="steelblue")
x_line = np.linspace(df["vega_om"].min(), df["vega_om"].max(), 100)
ax2.plot(x_line, intercept + slope * x_line,
         "r-", lw=2, label=f"OLS: α={intercept:.6f}, β={slope:.4f}")
ax2.plot(x_line, x_line, "k--", lw=1, alpha=0.5, label="45° (perfecto)")
ax2.set_xlabel("Vega OptionMetrics")
ax2.set_ylabel("Vega BS ATM (superficie)")
ax2.set_title(f"Scatter global  |  R²={r_value**2:.3f}  |  Corr={corr_pearson:.3f}",
              fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ---- Panel 3: Diferencia (vega_atm - vega_om) ----
ax3 = fig.add_subplot(gs[1, 1])
diff = df["vega_atm"] - df["vega_om"]
ax3.plot(df["Date"], diff, "purple", lw=0.8, alpha=0.7)
ax3.axhline(0,            color="black", lw=1,   ls="--")
ax3.axhline(diff.mean(),  color="red",   lw=1.5, ls="--",
            label=f"Media = {diff.mean():.6f}")
ax3.fill_between(df["Date"], diff, 0,
                 where=diff > 0, alpha=0.15, color="red")
ax3.fill_between(df["Date"], diff, 0,
                 where=diff < 0, alpha=0.15, color="blue")
ax3.set_title("Diferencia Vega BS ATM − Vega OM", fontsize=9)
ax3.set_ylabel("Diferencia (vega)")
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
ax5.bar(by_year["Año"], by_year["RMSE"],
        color="tomato", alpha=0.7, edgecolor="white")
ax5.axhline(rmse, color="black", ls="--", lw=1.5,
            label=f"Global = {rmse:.6f}")
ax5.set_title("RMSE por año", fontsize=9)
ax5.set_ylabel("RMSE")
ax5.legend(fontsize=8)
ax5.grid(alpha=0.3, axis="y")

plt.suptitle("Validación: Vega BS ATM (superficie) vs Vega OptionMetrics",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()
