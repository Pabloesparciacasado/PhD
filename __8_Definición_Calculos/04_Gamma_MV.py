# In[]: Importamos los datos

import pandas as pd
import numpy as np
import sys
import os
import duckdb

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "Tests Grupo"))

import importlib, funciones_HW
importlib.reload(funciones_HW)
from funciones_HW import bs_price_greeks, rolling_window_estimated, gain_gamma, sse_normalizado


if os.name == 'nt':
    PATH_DATA = r"Y:\OUTPUTS\opt_df_prueba.parquet"
    PATH_CLEAN_DATA = r"Y:\OUTPUTS\REPLICA_HW\opt_pairs.parquet"

    PATH_COEF_OUT = r"Y:\OUTPUTS\REPLICA_HW\HW_coef.csv"
    PATH_OOS_OUT = r"Y:\OUTPUTS\REPLICA_HW\HW_oos.parquet"

    PATH_RESULTS =  r"Y:\OUTPUTS\REPLICA_HW\DG_results.parquet"
    PATH_RESULTS_reg = r"Y:\OUTPUTS\REPLICA_HW\DeltaGamma_results_reg.parquet"

else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_prueba.parquet"

# In[]: Cargamos datos y calculamos griegas:

# pairs = pd.read_parquet(PATH_CLEAN_DATA)
# print("PAIRS cargados")

OOS = pd.read_parquet(PATH_OOS_OUT)
print("Resultados OOS cargados")

COEF = pd.read_csv(PATH_COEF_OUT)
print("Resultados COEF cargados")



col = ['OptionID', 'CallPut', 'Date', 'Days', 'Strike',
       'MidPrice', 'Volume', 'OpenInterest', 'ImpliedVolatility', 'Delta',
       'Gamma', 'Vega', 'Theta', 'Rate', 'SpotPrice', 'Moneyness', 'T', 'ds',
       'df_option', 'vega_norm', 'delta_mv', 'test_month']

pairs = OOS[col].copy()


(volga, vanna, volga_norm)= bs_price_greeks(pairs["SpotPrice"], pairs["Strike"], pairs["T"],
                                pairs["Rate"],pairs["ImpliedVolatility"],pairs["CallPut"])

pairs["Volga"] = volga
pairs["Vanna"] = vanna
pairs["volga_norm"] = volga_norm

desde = "2003-01-02"
hasta = "2024-02-29"

result = rolling_window_estimated(pairs,COEF,desde,hasta)

result["test_month"] = result["test_month"].dt.to_timestamp()

duckdb.from_df(result).write_parquet(
    str(PATH_RESULTS),
    compression="snappy")


#%%
# ============================================================
# Gain Gamma emp vs BS
# ============================================================
# In[] Gain global

gain_raw = result.groupby("CallPut").apply(gain_gamma)  # Series de tuplas
gain = pd.DataFrame(gain_raw.tolist(), index=gain_raw.index, columns=["gain_gamma1_pct", "gain_gamma2_pct", "gain_gamma3_pct","gain_gamma4"]).reset_index()

print(gain.round(1))

# In[]Gain por bucket de delta 
"""
Comparación de métrica de Gains vs HW(2017) paper por buckets
"""

result = result.copy()

result["Delta_bucket"] = result["Delta"].round(1)

result.loc[result["CallPut"] == "C", "Delta_bucket"] = (
    result.loc[result["CallPut"] == "C", "Delta_bucket" ]
    .clip(0.1,0.9))


result.loc[result["CallPut"] == "P","Delta_bucket"] = (
    result.loc[result["CallPut"]=="P", "Delta_bucket"]
    .clip(-0.1,-0.9) )
    # Tabla de Gain por bucket
gain_raw_bucket = result.groupby(["CallPut", "Delta_bucket"]).apply(gain_gamma)  # Series de tuplas

gain_bucket = pd.DataFrame(gain_raw_bucket.tolist(), index=gain_raw_bucket.index,
            columns=["gain_gamma1_pct", "gain_gamma2_pct", "gain_gamma3_pct", "gain_gamma4_pct"]).reset_index()

print(gain_bucket.round(2)) 

# In[] Tabla de SSE normalizados \(SSE_{BS,BS}=1\)

sse_nom = result.groupby("CallPut").apply(sse_normalizado)  # Series de tuplas

sse_nom_map = pd.DataFrame(sse_nom.tolist(), index=sse_nom.index,
                        columns=["SSE_bs_bs_pct", "SSE_bs_emp_pct", "SSE_mv_bs_pct", "SSE_mv_emp_pct"]).reset_index()

print(sse_nom_map.round(2))


# In[] Graficamos los errores de cobertura:

res_C = result[result["CallPut"] == "C"]
res_P = result[result["CallPut"] == "P"]

res_C["fecha"] = pd.to_datetime(res_C["Date"])  # asegúrate de que sea datetime
df = res_C.sort_values("fecha")


columnas = ["eps_delta_bs_gamma_bs", "eps_delta_mv_gamma_emp", "eps_delta_bs_gamma_emp","eps_delta_mv_gamma_bs"]  # columnas a plotear

fig, ax = plt.subplots(figsize=(10, 5))

for col in columnas:
    ax.plot(df["fecha"], df[col], label=col, linewidth=1.5)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()


ax.set_title("Estimated Call hedging errors")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(r"Y:\OUTPUTS\REPLICA_HW\eps_call_gamma.svg")
print("imagen guardada correctamente en formato svg")

plt.show()


res_P["fecha"] = pd.to_datetime(res_P["Date"])  # asegúrate de que sea datetime
df = res_P.sort_values("fecha")


columnas = ["eps_delta_bs_gamma_bs", "eps_delta_mv_gamma_emp", "eps_delta_bs_gamma_emp","eps_delta_mv_gamma_bs"]  # columnas a plotear

fig, ax = plt.subplots(figsize=(10, 5))

for col in columnas:
    ax.plot(df["fecha"], df[col], label=col, linewidth=1.5)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()


ax.set_title("Estimated Put hedging errors")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(r"Y:\OUTPUTS\REPLICA_HW\eps_put_gamma.svg")
print("imagen guardada correctamente en formato svg")

plt.show()


# %%
# ============================================================
# Anális por regímenes de volatilidad y skewness implícita
# ============================================================
# In[] Importamos datos de VIX y SKEW

VIX = pd.read_csv(r"Y:\Maro-Variables\lasdebelén\VIX_cboe.csv")
print("VIX cargada")

vix = VIX[["CLOSE", "DATE"]].copy()
vix["Date"] = pd.to_datetime(vix["DATE"])
vix["vix"] = vix["CLOSE"]
vix = vix[["Date","vix"]]

print(vix["vix"].describe())

SKEW = pd.read_csv(r"Y:\Maro-Variables\lasdebelén\SKEW_cboe.csv")
print("Implied Skewness cargada")

skew = SKEW.copy()
skew = skew.rename(columns = {"DATE" : "Date", "SKEW": "Skew"})
skew["Date"] = pd.to_datetime(skew["Date"])

print(skew["Skew"].describe())

# In[] asignamos a results y marcamos los 3 regimenes de skew y vix:

# Una observación por día
vix_daily = (
    vix[["Date", "vix"]]
    .drop_duplicates("Date")
    .sort_values("Date")
    .copy() )

# Umbrales sobre la distribución temporal
vq33 = vix_daily["vix"].quantile(1/3)
vq67 = vix_daily["vix"].quantile(2/3)

vix_daily["VIX_regime"] = pd.cut(
    vix_daily["vix"],
    bins=[-np.inf, vq33, vq67, np.inf],
    labels=["Low", "Medium", "High"] )

print(vq33, vq67)

# Después se asigna el estado a todas las opciones del día
result = result.merge(
    vix_daily[["Date", "vix", "VIX_regime"]],
    on="Date",
    how="left" )
### Skew
skew_daily = (skew[["Date","Skew"]]
              .drop_duplicates("Date")
              .sort_values("Date")
              .copy())

sq33 = skew_daily["Skew"].quantile(1/3)
sq67 = skew_daily["Skew"].quantile(2/3)


skew_daily["SKW_regime"] = pd.cut(
    skew_daily["Skew"],
    bins=[-np.inf, sq33, sq67, np.inf],
    labels=["Low", "Medium", "High"] )

print(sq33,sq67)

result = result.merge(
    skew_daily[["Date", "Skew", "SKW_regime"]], on="Date" )


result_vix_skw = result.copy()

duckdb.from_df(result_vix_skw).write_parquet(
    str(PATH_RESULTS_reg),
    compression="snappy")


# In[] Mostramos Gains por régimen:

rows = []
regimen= ["Low", "Medium", "High"]
for r1 in regimen:
    for r2 in regimen:

        df = result_vix_skw[(result_vix_skw["VIX_regime"] == r1) &(result_vix_skw["SKW_regime"] == r2) ]


        gain_raw = df.groupby("CallPut").apply(gain_gamma)  # Series de tuplas
        gain = pd.DataFrame(gain_raw.tolist(), index=gain_raw.index, columns=["gain_gamma1_pct", "gain_gamma2_pct", "gain_gamma3_pct","gain_gamma4"]).reset_index()

        gain["VIX_regime"] = r1
        gain["SKW_regime"] = r2

        rows.append(gain)

gain_regimes = pd.concat(rows, ignore_index=True)

# %% printeamos lo de arriba

g = "gain_gamma2_pct"
print("=======================")
print(f"=== {g} ===")
print("=======================")

print("=====Call=====")

table_calls = (gain_regimes[gain_regimes["CallPut"] == "C"].pivot(
        index="VIX_regime",
        columns="SKW_regime",
        values=g )
    .reindex(index=regimen, columns=regimen) )

print(table_calls.round(2))

print("=====Put======")
table_calls = (gain_regimes[gain_regimes["CallPut"] == "P"].pivot(
        index="VIX_regime",
        columns="SKW_regime",
        values=g )
    .reindex(index=regimen, columns=regimen) )

print(table_calls.round(2))



# In[]: Mostramos Gains por régimenes y desagregando Deltas-Buckets

rows = []

regimen = ["Low", "Medium", "High"]

for r1 in regimen:
    for r2 in regimen:

        df_reg = result_vix_skw[
            (result_vix_skw["VIX_regime"] == r1) &
            (result_vix_skw["SKW_regime"] == r2)
        ].copy()

        if df_reg.empty:
            continue

        vs_raw_bucket = (
            df_reg
            .groupby(["CallPut", "Delta_bucket"])
            .apply(gain_gamma)
        )

        gain_bucket = pd.DataFrame(
            vs_raw_bucket.tolist(),
            index=vs_raw_bucket.index,
            columns=[
                "gain_gamma1_pct",
                "gain_gamma2_pct",
                "gain_gamma3_pct",
                "gain_gamma4_pct"
            ]
        ).reset_index()

        gain_bucket["VIX_regime"] = r1
        gain_bucket["SKW_regime"] = r2

        rows.append(gain_bucket)

gain_regimes_bucket = pd.concat(rows, ignore_index=True)

gain_regimes_bucket["Regime"] = (
    "v" + gain_regimes_bucket["VIX_regime"].astype(str)
    + "_"
    + "s" + gain_regimes_bucket["SKW_regime"].astype(str)
)
# %% printeamos lo de arriba

g = "gain_gamma2_pct"
print("=======================")
print(f"=== {g} ===")
print("=======================")

print("=====Call=====")

g = "gain_gamma2_pct"

table_calls_bucket = (gain_regimes_bucket[gain_regimes_bucket["CallPut"] == "C"]
    .pivot(
        index="Delta_bucket",
        columns="Regime",
        values=g
    )
)

print(table_calls_bucket)

g = "gain_gamma2_pct"
print("=======================")
print(f"=== {g} ===")
print("=======================")

print("===== Put =====")

g = "gain_gamma2_pct"

table_puts_bucket = (gain_regimes_bucket[gain_regimes_bucket["CallPut"] == "P"]
    .pivot(
        index="Delta_bucket",
        columns="Regime",
        values=g
    )
)

print(table_puts_bucket)
# %% solo un condicionante

rows = []
regimen= ["Low", "Medium", "High"]

for r2 in regimen:

        vs_raw_bucket = result_vix_skw[result_vix_skw["VIX_regime"]==r2]
        vs_raw_bucket = vs_raw_bucket.groupby(["CallPut", "Delta_bucket"]).apply(gain_gamma)  # Series de tuplas

        gain_bucket = pd.DataFrame(vs_raw_bucket.tolist(), index=vs_raw_bucket.index,
                    columns=["gain_gamma1_pct", "gain_gamma2_pct", "gain_gamma3_pct", "gain_gamma4_pct"]).reset_index()

        gain_bucket["VIX_regime"] = r2
        
        rows.append(gain_bucket)

gain_regimes_bucket = pd.concat(rows, ignore_index=True)




g = "gain_gamma2_pct"
print("=======================")
print(f"=== {g} ===")
print("=======================")

print("=====Call=====")

g = "gain_gamma2_pct"

table_calls_bucket = (gain_regimes_bucket[gain_regimes_bucket["CallPut"] == "C"]
    .pivot(
        index="Delta_bucket",
        columns="VIX_regime",
        values=g
    )
)

print(table_calls_bucket)

# %%
