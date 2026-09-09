# In[]: Importamos los datos


import pandas as pd
import numpy as np
import sys
import os
from functools import reduce
import re
import duckdb
from datetime import datetime

from tabulate import tabulate
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy.stats import norm

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent / "Tests Grupo"))


from funciones_HW import bs_price_greeks, rolling_window_estimated, gain_gamma
import importlib, funciones_HW
importlib.reload(funciones_HW)

if os.name == 'nt':
    PATH_DATA = r"Y:\OUTPUTS\opt_df_prueba.parquet"
    PATH_CLEAN_DATA = r"Y:\OUTPUTS\REPLICA_HW\opt_pairs.parquet"

    PATH_COEF_OUT = r"Y:\OUTPUTS\REPLICA_HW\HW_coef.csv"
    PATH_OOS_OUT = r"Y:\OUTPUTS\REPLICA_HW\HW_oos.parquet"

else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_prueba.parquet"



# In[]: Cargamos datos:

# pairs = pd.read_parquet(PATH_CLEAN_DATA)
# print("PAIRS cargados")

OOS = pd.read_parquet(PATH_OOS_OUT)
print("Resultados OOS cargados")

COEF = pd.read_csv(PATH_COEF_OUT)
print("Resultados COEF cargados")


# %%
col = ['OptionID', 'CallPut', 'Date', 'Days', 'Strike',
       'MidPrice', 'Volume', 'OpenInterest', 'ImpliedVolatility', 'Delta',
       'Gamma', 'Vega', 'Theta', 'Rate', 'SpotPrice', 'Moneyness', 'T', 'ds',
       'df_option', 'vega_norm', 'delta_mv', 'test_month']

pairs = OOS[col].copy()


# %%

(volga, vanna, volga_norm)= bs_price_greeks(pairs["SpotPrice"], pairs["Strike"], pairs["T"],
                                pairs["Rate"],pairs["ImpliedVolatility"],pairs["CallPut"])

pairs["Volga"] = volga
pairs["Vanna"] = vanna
pairs["volga_norm"] = volga_norm


result = rolling_window_estimated(pairs,COEF)


# ============================================================
# Gain Gamma emp vs BS
# ============================================================
# %% Gain global (Hull-White)


gain_raw = result.groupby("CallPut").apply(gain_gamma)  # Series de tuplas

gain = pd.DataFrame(gain_raw.tolist(), index=gain_raw.index,
                     columns=["gain_gamma1_pct", "gain_gamma2_pct", "gain_gamma3_pct"]).reset_index()



print(gain.round(1))

# %% Gain por bucket (delta)
"""
Comparación de métrica de Gains vs HW(2017) paper por buckets
"""

result = result.copy()

result["Delta_bucket"] = result["Delta"].round(1)

result.loc[result["CallPut"] == "C", "Delta_bucket"] = (
    result.loc[result["CallPut"] == "C", "Delta_bucket" ]
    .clip(0.1,0.9)
)


result.loc[result["CallPut"] == "P","Delta_bucket"] = (

    result.loc[result["CallPut"]=="P", "Delta_bucket"]
    .clip(-0.1,-0.9)
)
# Tabla de Gain por bucket
gain_raw_bucket = result.groupby(["CallPut", "Delta_bucket"]).apply(gain_gamma)  # Series de tuplas

gain_bucket = pd.DataFrame(gain_raw_bucket.tolist(), index=gain_raw_bucket.index,
                     columns=["gain_gamma1_pct", "gain_gamma2_pct", "gain_gamma3_pct"]).reset_index()

print(gain_bucket.round(2))


# %%

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

# %%
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
