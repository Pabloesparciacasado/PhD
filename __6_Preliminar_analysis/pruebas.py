# Volatilidad ATM por bucket: mediana de las opciones más cercanas a moneyness=1
# In[]:
from unittest import result

from scipy.interpolate import CubicSpline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import solve

opt_df = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet")

opt_df_filtered = opt_df[(opt_df["OpenInterest"] > 0) | (opt_df["Volume"] > 0)].reset_index(drop=True)

opt_df_filtered = opt_df[(opt_df["Bid"] > 0)].reset_index(drop=True)

v_grid = [0, 15, 45, 105, 183, 365, np.inf]
v_edges = pd.IntervalIndex.from_breaks(v_grid, closed="right")

v_idx = pd.cut(opt_df_filtered["Days"], bins=v_edges, labels=False, include_lowest=True)
opt_df_filtered["bucket"] = v_idx
opt_df_filtered = opt_df_filtered.sort_values(["Date","Days","CallPut","Strike"]).reset_index(drop=True)


opt_df_filtered_sorted = opt_df_filtered.sort_values(["Date", "bucket", "CallPut", "Strike", "OpenInterest", "Volume"],
                                                     ascending= [True, True, True, True, False, False])
opt_agg = (
    opt_df_filtered_sorted.groupby(["Date","bucket","CallPut","Strike"], sort=False)
    .first().reset_index()
    .sort_values(["Date", "bucket", "CallPut", "Strike"])
    .reset_index(drop=True))

opt_agg


# In[]:

sigma_atm = (opt_agg
             .assign(dist_atm=(opt_agg["Moneyness"] - 1).abs())
             .sort_values("dist_atm")
             .groupby(["Date", "bucket"])
             .agg(sigma_atm=("ImpliedVolatility", "first"),
                  SpotPrice=("SpotPrice", "first"))
             .reset_index())


# bucket_centers usando los mismos Interval que tienes
bucket_centers = {
    pd.Interval(0.0,   15.0,  closed="right"): 7.5   / 252,
    pd.Interval(15.0,  45.0,  closed="right"): 30.0  / 252,
    pd.Interval(45.0,  105.0, closed="right"): 75.0  / 252,
    pd.Interval(105.0, 183.0, closed="right"): 144.0 / 252,
    pd.Interval(183.0, 365.0, closed="right"): 274.0 / 252,
    pd.Interval(365.0, np.inf,closed="right"): 500.0 / 252,
}

# Forzar conversión a float directamente sin map
sigma_atm["T_years"] = sigma_atm["bucket"].apply(
    lambda b: float(b.mid) if not np.isinf(b.right) else (float(b.left) + 500) / 2
) / 252.0

print(sigma_atm["T_years"].isna().sum())
print(sigma_atm["T_years"].dtype)
print(sigma_atm[["bucket","T_years"]].drop_duplicates())


sigma_atm["T_years"] = sigma_atm["bucket"].map(bucket_centers)
print(sigma_atm["T_years"].isna().sum())  # debe ser 0

# Theta ATM
sigma_atm["Theta_ATM"] = (
    - sigma_atm["SpotPrice"]
    * sigma_atm["sigma_atm"]
    / (2 * np.sqrt(sigma_atm["T_years"] * 252))
)

# Mediana por bucket
theta_by_bucket = (sigma_atm
                   .groupby("bucket")["Theta_ATM"]
                   .median()
                   .reset_index())

print(theta_by_bucket)

# In[]:

sigma_atm = (opt_agg
             .assign(dist_atm=(opt_agg["Moneyness"] - 1).abs())
             .sort_values("dist_atm")
             .groupby(["Date", "bucket"])
             .agg(sigma_atm=("ImpliedVolatility", "first"),
                  SpotPrice=("SpotPrice", "first"))
             .reset_index())

# Theta analítica de BS en ATM (call, sin dividendos, r≈0 simplificado)
# Theta = -S * sigma * phi(d1) / (2 * sqrt(T))
# En ATM con r=0: d1 = sigma*sqrt(T)/2 ≈ 0 para T pequeño
# Simplificación: Theta_ATM ≈ -S * sigma / (2 * sqrt(T * 252))

sigma_atm[""] = sigma_atm["bucket"].apply(
    lambda b: {
        "(0.0, 15.0]":    7.5 / 252,    # punto medio del bucket
        "(15.0, 45.0]":   30.0 / 252,
        "(45.0, 105.0]":  75.0 / 252,
        "(105.0, 183.0]": 144.0 / 252,
        "(183.0, 365.0]": 274.0 / 252,
        "(365.0, inf]":   500.0 / 252,
    }.get(str(b), np.nan)
)

sigma_atm["Theta_ATM"] = (
    - sigma_atm["SpotPrice"]
    * sigma_atm["sigma_atm"]
    / (2 * np.sqrt(sigma_atm["T_years"] * 252))
)

# Mediana de theta ATM por bucket sobre toda la muestra
theta_by_bucket = (sigma_atm
                   .groupby("bucket")["Theta_ATM"]
                   .median()
                   .reset_index())

print(theta_by_bucket)