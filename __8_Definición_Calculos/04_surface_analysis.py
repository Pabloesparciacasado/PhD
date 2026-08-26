# In[]: Importamos los datos

"""
Análisis de curvaturas y superficie de griegas sobre la moneyness.
"""
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

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


### ANALISIS B

empirical = True
market = "crsp" # "crsp"  "spx"

if os.name == 'nt':
    PATH_DATA = r"Y:\OUTPUTS\opt_df_empirical_greeks.parquet"

else:
    PATH_DATA = r"/Users/pablo/codigos_github/opt_df_empirical_greeks.parquet"
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_empirical_greeks.parquet"


print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

opt_df
# In[]: "Recuperamos el formato de intervalos:"

 
def parse_bound(x):
    x = x.strip().lower()

    if x in ["inf", "+inf", "infinity", "+infinity", "np.inf"]:
        return np.inf
    elif x in ["-inf", "-infinity", "-np.inf"]:
        return -np.inf
    else:
        return float(x)

def parse_interval(s):
    if pd.isna(s):
        return pd.NA

    if isinstance(s, pd.Interval):
        return s

    s = str(s).strip()

    pattern = r"^(\(|\[)\s*([^,]+)\s*,\s*([^\]\)]+)\s*(\)|\])$"
    match = re.match(pattern, s)

    if not match:
        raise ValueError(f"Formato de intervalo no reconocido: {s}")

    left_bracket, left, right, right_bracket = match.groups()

    left = parse_bound(left)
    right = parse_bound(right)

    if left_bracket == "[" and right_bracket == "]":
        closed = "both"
    elif left_bracket == "[" and right_bracket == ")":
        closed = "left"
    elif left_bracket == "(" and right_bracket == "]":
        closed = "right"
    else:
        closed = "neither"

    return pd.Interval(left, right, closed=closed)

opt_df["maturity_bucket"] = opt_df["maturity_bucket"].apply(parse_interval)
opt_df["moneyness_bucket"] = opt_df["moneyness_bucket"].apply(parse_interval)

# %% 




#


