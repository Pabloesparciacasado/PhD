# In[]: Bibliotecas y Rutas

"""
Análisis de curvaturas y superficie de griegas sobre la moneyness.
"""
import pandas as pd
import sys
import os
import duckdb

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from funciones_HW import dataset_preparation, fit_hw_coefficients, apply_greeks, rolling_window, gain_hw


if os.name == 'nt':
    PATH_DATA = r"Y:\OUTPUTS\opt_df_prueba.parquet"
    PATH_CLEAN_DATA = r"Y:\OUTPUTS\REPLICA_HW\opt_pairs.parquet"
    PATH_COEF_OUT = r"Y:\OUTPUTS\REPLICA_HW\HW_coef.csv"
    PATH_OOS_OUT = r"Y:\OUTPUTS\REPLICA_HW\HW_oos.parquet"

else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_prueba.parquet"


# %% Carga de datos (DuckDB)

print("Cargando datos...")

con = duckdb.connect()
opt_df = con.execute(f"""
    SELECT *
    FROM read_parquet('{PATH_DATA}')
    -- WHERE lo que necesites filtrar, si aplica en este punto
""").df()

# opt_df = pd.read_parquet(PATH_DATA)

opt_df

# In[]: Filtro de columnas y precios spot

#filtro de columnas relevantes:

cols = ['OptionID','CallPut', 'Date','Days', 'Strike',
        'Bid', 'Ask', 'MidPrice','Volume', 'OpenInterest',
        'ImpliedVolatility', 'Delta', 'Gamma', 'Vega', 'Theta' ,
        'Rate', 'SpotPrice','Moneyness' ]

optf_df = opt_df[cols].copy()


desde = "2003-01-02"
hasta = "2024-02-29"

sec_price = pd.read_parquet(os.path.join(r"Y:\OptionMetrics\Acumulado", "security_price.parquet"))
print("Spot Cargado")

sec_price["Date"] = pd.to_datetime(sec_price["Date"], format="%Y-%m-%d")
SP500_price = sec_price[sec_price["SecurityID"] == 108105].reset_index(drop=True)

SP500_price = SP500_price[(SP500_price["Date"] >= pd.Timestamp(desde)) & (SP500_price["Date"] <= pd.Timestamp(hasta))].reset_index(drop=True)


trading_dates = (
    SP500_price["Date"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

next_trading_dates = pd.Series(
    trading_dates.shift(-1).values,
    index=trading_dates.values
)


pairs = dataset_preparation(optf_df, next_trading_dates)

duckdb.from_df(pairs).write_parquet(
    str(PATH_CLEAN_DATA),
    compression="snappy")

print(f"pairs guardado correctamente en: {PATH_CLEAN_DATA}")

# %%

ej_gamma=False
import importlib, funciones_HW
importlib.reload(funciones_HW)
from funciones_HW import dataset_preparation, fit_hw_coefficients, apply_greeks, rolling_window, gain_hw

oos, coef = rolling_window(pairs,ej_gamma,desde,hasta )

oos["test_month"] = oos["test_month"].dt.to_timestamp()

duckdb.from_df(oos).write_parquet(
    str(PATH_OOS_OUT),
    compression="snappy")

print(f"Resultado OOS guardado correctamente en: {PATH_OOS_OUT}")

coef.to_csv(PATH_COEF_OUT, index=False, encoding='utf-8')
print(f"Resultado coef guardado correctamente en: {PATH_COEF_OUT}")


# %% Cargar resultados OOS

OOS = pd.read_parquet(PATH_OOS_OUT)
print("Resultados OOS cargados")

"""

Comparación de métrica de Gains vs HW(2017) paper
Eq. (3):
    Gain = 1 - SSE(eps_MV) / SSE(eps_BS)

"""
# ============================================================
# VALIDACIÓN HULL-WHITE: GAIN
# ============================================================
# %% Gain global (Hull-White)

gain = (
    OOS.groupby("CallPut")
    .apply(gain_hw)
    .rename("Gain")
    .reset_index()
)

gain["Gain_pct"] = gain["Gain"] * 100

print(gain.round(2))

# %% Gain por bucket (setup)
"""
Comparación de métrica de Gains vs HW(2017) paper por buckets
"""

oos = OOS.copy()

# %% Definir Delta_bucket

oos["Delta_bucket"] = oos["Delta"].round(1)

oos.loc[oos["CallPut"] == "C", "Delta_bucket"] = (
    oos.loc[oos["CallPut"] == "C", "Delta_bucket" ]
    .clip(0.1,0.9)
)


oos.loc[oos["CallPut"] == "P","Delta_bucket"] = (

    oos.loc[oos["CallPut"]=="P", "Delta_bucket"]
    .clip(-0.1,-0.9)
)
# %% Tabla de Gain por bucket

gain_bucket = (
    oos.groupby(["CallPut", "Delta_bucket"])
    .apply(gain_hw)
    .rename("Gain")
    .reset_index()
)

gain_bucket["Gain_pct"] = gain_bucket["Gain"] * 100
print(gain_bucket.round(2))


# gain_tab = (
#     gain_bucket
#     .pivot(
#         index="Delta_bucket",
#         columns="CallPut",
#         values="Gain_pct" )
#     .sort_index()
# )
# gain_tab = gain_tab.round(2)
# print(gain_tab)

# %% Cargar coeficientes HW

COEF = pd.read_csv(PATH_COEF_OUT)
print("Resultados COEF cargados")


coef = COEF.copy()

coefc = coef[coef["cp_flag"] == "C"]
coefp = coef[coef["cp_flag"] == "P"]

# %% Plot coeficientes - Calls y Puts

coefc["fecha"] = pd.to_datetime(coefc["train_end"])  # asegúrate de que sea datetime
df = coefc.sort_values("train_end")
df["-b"] = -df["b"]

columnas = ["a", "-b", "c"]  # columnas a plotear

fig, ax = plt.subplots(figsize=(10, 5))

for col in columnas:
    ax.plot(df["fecha"], df[col], label=col, linewidth=1.5)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

ax.set_xlabel("Fecha")
ax.set_ylabel("Valor")
ax.set_title("Estimated Call parameters")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(r"Y:\OUTPUTS\REPLICA_HW\HW_coef_call.svg")
print("imagen guardada correctamente en formato svg")

plt.show()


coefp["fecha"] = pd.to_datetime(coefp["train_end"])  # asegúrate de que sea datetime
df = coefp.sort_values("train_end")

columnas = ["a", "b", "c"]  # columnas a plotear

fig, ax = plt.subplots(figsize=(10, 5))

for col in columnas:
    ax.plot(df["fecha"], df[col], label=col, linewidth=1.5)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

ax.set_title("Estimated Put parameters")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(r"Y:\OUTPUTS\REPLICA_HW\HW_coef_put.svg")
print("imagen guardada correctamente en formato svg")

plt.show()


# %%

