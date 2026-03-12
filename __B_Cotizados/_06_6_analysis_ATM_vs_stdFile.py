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
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\superficie_con_greeks_shimko_3.parquet')
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

# def interp_atm(group):
#     group = group.sort_values("log_moneyness")
#     result = {}
#     for col in greek_cols:
#         result[col] = np.interp(0.0, group["log_moneyness"], group[col])
#     return pd.Series(result)

# atm_greeks = (
#     raw_df_NTM
#     .groupby("Date")
#     .apply(interp_atm)
#     .reset_index()
#     .rename(columns={
#         'implied_vol' : 'iv_atm',
#         'delta_bs'    : 'delta_bs_atm',
#         'vega'        : 'vega_atm',
#         'gamma_bs'    : 'gamma_bs_atm',
#         'vanna_K'     : 'vanna_atm',
#         'volga'       : 'volga_atm',
#         'dsigma_dK'   : 'dsigma_dK_atm',
#         'd2sigma_dK2' : 'd2sigma_dK2_atm',
#         'delta'       : 'delta_atm',
#         'gamma'       : 'gamma_atm',
#     })
# )
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
# 3. ESTADÍSTICOS: delta_atm vs delta_om
# ============================================================
df = df_merged.dropna(subset=["delta_bs_atm", "delta_om"])

corr_pearson, p_pearson = stats.pearsonr(df["delta_bs_atm"], df["delta_om"])
corr_spearman, _        = stats.spearmanr(df["delta_bs_atm"], df["delta_om"])

slope, intercept, r_value, p_value, std_err = stats.linregress(
    df["delta_om"], df["delta_bs_atm"]
)
rmse = np.sqrt(np.mean((df["delta_bs_atm"] - df["delta_om"])**2))
mae  = np.mean(np.abs(df["delta_bs_atm"] - df["delta_om"]))
bias = np.mean(df["delta_bs_atm"] - df["delta_om"])

print("\n" + "=" * 55)
print("ESTADÍSTICOS GLOBALES: delta_bs_atm vs delta_om")
print("=" * 55)
print(f"  Observaciones        : {len(df):,}")
print(f"  Correlación Pearson  : {corr_pearson:.4f}  (p={p_pearson:.2e})")
print(f"  Correlación Spearman : {corr_spearman:.4f}")
print(f"  OLS: delta_bs_atm = {intercept:.4f} + {slope:.4f} * delta_om")
print(f"  R²                   : {r_value**2:.4f}")
print(f"  RMSE                 : {rmse:.6f}")
print(f"  MAE                  : {mae:.6f}")
# print(f"  Bias (atm - om)      : {bias:.6f}")
print("=" * 55)

# %%
