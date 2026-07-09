# In[]: Importamos los datos
import pandas as pd
import numpy as np
import sys
import os
from functools import reduce
import re
import duckdb

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if os.name == 'nt':
    PATH_DATA =  r"Y:\OUTPUTS\Agg_Greeks.csv"
else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_empirical_greeks.parquet"

print("Cargando datos...")
agg_df = pd.read_parquet(PATH_DATA)


agg_df





























# %%
# Columnas redundantes que eliminamos

rem_n_contratos = [
    "n_contratos_very_deep_dspread",
    "n_contratos_deep_dspread",
    "n_contratos_near_dspread",
    "n_contratos_ATM_dspread",
    "n_contratos_delta_OI",
    "n_contratos_delta_VD",
    "n_contratos_gamma_VD",
]

rem_dspread_values = [
    "delta_emp_near_dspread",
    "delta_emp_deep_dspread",
    "delta_emp_very_deep_dspread",
]

rem_gspread_values = [
    "gamma_emp_near_gspread",
    "gamma_emp_deep_gspread",
    "gamma_emp_very_deep_gspread",
]

rem_open_interest_dspread = [
    "OpenInterest_ATM_dspread",
    "OpenInterest_near_dspread",
    "OpenInterest_deep_dspread",
    "OpenInterest_very_deep_dspread",
]

rem_means = [
    "mean_DolarVolume_delta_VD",
    "mean_OpenInterest_delta_OI",
]

rem_col = (
    rem_n_contratos
    + rem_dspread_values
    + rem_gspread_values
    + rem_open_interest_dspread
    + rem_means
)

rename_cols = {
    "n_contratos_gamma_OI": "n_contratos_OI",
    "mean_DolarVolume_gamma_VD": "mean_DolarVolume",
    "mean_OpenInterest_gamma_OI": "mean_OpenInterest",

    "n_contratos_ATM_gspread": "n_contratos_ATM",
    "n_contratos_near_gspread": "n_contratos_near",
    "n_contratos_deep_gspread": "n_contratos_deep",
    "n_contratos_very_deep_gspread": "n_contratos_very_deep",

    "OpenInterest_ATM_gspread": "OpenInterest_ATM",
    "OpenInterest_near_gspread": "OpenInterest_near",
    "OpenInterest_deep_gspread": "OpenInterest_deep",
    "OpenInterest_very_deep_gspread": "OpenInterest_very_deep",
}

agg_df2 = (
    agg_df
    .drop(columns=rem_col, errors="ignore")
    .rename(columns=rename_cols)
)
# %%

# Ordenamos columnas por bloques logicos

id_cols = [
    "Date",
    "CallPut",
    "bucket",
]

weighted_cols = [
    "w_gamma_emp_gamma_OI",
    "w_gamma_emp_gamma_VD",
    "w_delta_emp_delta_OI",
    "w_delta_emp_delta_VD",
]

spread_gamma_cols = [
    "gamma_emp_ATM_gspread",
    "spread_ATM_minus_near_gspread",
    "spread_ATM_minus_deep_gspread",
    "spread_ATM_minus_very_deep_gspread",
]

spread_delta_cols = [
    "delta_emp_ATM_dspread",
    "spread_ATM_minus_near_dspread",
    "spread_ATM_minus_deep_dspread",
    "spread_ATM_minus_very_deep_dspread",
]

n_contratos_cols = [
    "n_contratos_ATM",
    "n_contratos_near",
    "n_contratos_deep",
    "n_contratos_very_deep",
    "n_contratos_OI",
]

peso_cols = [
    "mean_OpenInterest",
    "mean_DolarVolume",
    "OpenInterest_ATM",
    "OpenInterest_near",
    "OpenInterest_deep",
    "OpenInterest_very_deep",
]

orden_columnas = (
    id_cols
    + weighted_cols
    + spread_gamma_cols
    + spread_delta_cols
    + n_contratos_cols
    + peso_cols
)

# Por seguridad: solo usamos columnas que existan
orden_columnas = [col for col in orden_columnas if col in agg_df2.columns]

# Cualquier columna no contemplada se manda al final
otras_cols = [col for col in agg_df2.columns if col not in orden_columnas]

final_df = agg_df2[orden_columnas + otras_cols]
final_df
# %%
