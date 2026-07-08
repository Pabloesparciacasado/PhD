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
agg_df.columns
rem_col = ["n_contratos_very_deep_dspread","n_contratos_deep_dspread","n_contratos_near_dspread","n_contratos_ATM_dspread",
           "n_contratos_delta_OI","n_contratos_delta_VD","n_contratos_gamma_VD","mean_DolarVolume_delta_VD" ]
rename = ["n_contratos_gamma_OI","mean_DolarVolume_gamma_VD"]
agg_df = agg_df.drop(columns =rem_col)
# %%
agg_df.columns

# %%
