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
# 3.1. ESTADÍSTICOS: delta_atm vs delta_om
# ============================================================
df = df_merged.dropna(subset=["delta_bs_atm", "delta_om"])

corr_pearson, p_pearson = stats.pearsonr(df["delta_bs_atm"], df["delta_om"])
corr_spearman, _        = stats.spearmanr(df["delta_bs_atm"], df["delta_om"])
X_pol = np.column_stack([df["delta_om"]**2, df["delta_om"], np.ones(len(df))])
coeffs, _, _, _ = np.linalg.lstsq(X_pol, df["delta_bs_atm"], rcond=None)
c2, c1, c0 = coeffs

y_pred = X_pol @ coeffs
ss_res = np.sum((df["delta_bs_atm"] - y_pred)**2)
ss_tot = np.sum((df["delta_bs_atm"] - df["delta_bs_atm"].mean())**2)
r2_pol = 1 - ss_res / ss_tot

rmse = np.sqrt(np.mean((df["delta_bs_atm"] - df["delta_om"])**2))
mae  = np.mean(np.abs(df["delta_bs_atm"] - df["delta_om"]))
bias = np.mean(df["delta_bs_atm"] - df["delta_om"])

print("\n" + "=" * 55)
print("ESTADÍSTICOS GLOBALES: delta_bs_atm vs delta_om")
print("=" * 55)
print(f"  Observaciones        : {len(df):,}")
print(f"  Correlación Pearson  : {corr_pearson:.4f}  (p={p_pearson:.2e})")
print(f"  Correlación Spearman : {corr_spearman:.4f}")
print(f"  Poly2: delta_bs_atm = {c0:.4f} + {c1:.4f}·x + {c2:.4f}·x²")
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
    c_p, _  = stats.pearsonr(grp["delta_bs_atm"], grp["delta_om"])
    s, i, r, _, _ = stats.linregress(grp["delta_om"], grp["delta_bs_atm"])
    rmse_y  = np.sqrt(np.mean((grp["delta_bs_atm"] - grp["delta_om"])**2))
    rows_year.append({
        "Año":          year,
        "N":            len(grp),
        "Corr":         round(c_p, 3),
        "R²":           round(r**2, 3),
        "Alpha":        round(i, 4),
        "Beta":         round(s, 4),
        "RMSE":         round(rmse_y, 6),
        "delta_bs_mean": round(grp["delta_bs_atm"].mean(), 4),
        "delta_om_mean": round(grp["delta_om"].mean(), 4),
    })

by_year = pd.DataFrame(rows_year)
print("\n" + "=" * 85)
print("TABLA POR AÑO: delta_bs_atm vs delta_om")
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
ax1.plot(df["Date"], df["delta_bs_atm"], "b-",  lw=1, alpha=0.8, label="Delta BS ATM (superficie)")
ax1.plot(df["Date"], df["delta_om"],     "r--", lw=1, alpha=0.8, label="Delta OptionMetrics")
ax1.set_title("Delta BS ATM vs Delta OM — Serie temporal", fontsize=11, fontweight="bold")
ax1.set_ylabel("Delta")
ax1.legend()
ax1.grid(alpha=0.3)

# ---- Panel 2: Scatter global ----
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(df["delta_om"], df["delta_bs_atm"],
            alpha=0.15, s=5, color="steelblue")
x_line = np.linspace(df["delta_om"].min(), df["delta_om"].max(), 100)
y_line = c0 + c1 * x_line + c2 * x_line**2
ax2.plot(x_line, y_line,
         "r-", lw=2, label=f"Poly2: {c0:.4f} + {c1:.4f}·x + {c2:.4f}·x²")
ax2.plot(x_line, x_line, "k--", lw=1, alpha=0.5, label="45° (perfecto)")
ax2.set_xlabel("Delta OptionMetrics")
ax2.set_ylabel("Delta BS ATM (superficie)")
ax2.set_title(f"Scatter global  |  R²={r2_pol:.3f}  |  Corr={corr_pearson:.3f}",
              fontsize=9)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

# ---- Panel 3: Diferencia (delta_bs_atm - delta_om) ----
ax3 = fig.add_subplot(gs[1, 1])
diff = df["delta_bs_atm"] - df["delta_om"]
ax3.plot(df["Date"], diff, "purple", lw=0.8, alpha=0.7)
ax3.axhline(0,            color="black", lw=1,   ls="--")
ax3.axhline(diff.mean(),  color="red",   lw=1.5, ls="--",
            label=f"Media = {diff.mean():.4f}")
ax3.fill_between(df["Date"], diff, 0,
                 where=diff > 0, alpha=0.15, color="red")
ax3.fill_between(df["Date"], diff, 0,
                 where=diff < 0, alpha=0.15, color="blue")
ax3.set_title("Diferencia Delta BS ATM − Delta OM", fontsize=9)
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

plt.suptitle("Validación: Delta BS ATM (superficie) vs Delta OptionMetrics",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()