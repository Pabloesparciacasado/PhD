# In[]: "Importamos los datos"
import pandas as pd
import numpy as np
import sys
from   tabulate import tabulate
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import re

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if os.name == 'nt':
    PATH_DATA = r"Y:\OUTPUTS\opt_df_empirical_greeks_sinfiltro.parquet"
    PATH_DATA = r"Y:\OUTPUTS\opt_df_prueba.parquet"
else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_empirical_greeks.parquet"

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

#Añadimos algunas variables de interés:
opt_df["Dummy_Bid"] = opt_df["Bid"] > 0
opt_df["DolarVolume"] = opt_df["Volume"] * opt_df["MidPrice"]

opt_df = opt_df[opt_df["Bid"]>0]

# Asignamos buckets de vencimientos:¡
v_grid = [0, 15, 45, 105, 183, 365, np.inf]
v_edges = pd.IntervalIndex.from_breaks(v_grid, closed="right")
opt_df["maturity_bucket"] = pd.cut(opt_df["Days"], 
    bins=v_edges, 
    labels=False, 
    include_lowest=True)


m_grid = np.round(  np.linspace(0.1,2,int(2/0.1)),2)
m_grid = np.concatenate(([0],m_grid, [np.inf]))
m_edges = pd.IntervalIndex.from_breaks(m_grid, closed="right")
opt_df["moneyness_bucket"] = pd.cut(opt_df["Moneyness"], bins=m_edges, labels=False, include_lowest=True)


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
