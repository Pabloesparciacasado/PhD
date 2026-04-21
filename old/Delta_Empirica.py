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
#v_grid = [1,9,29,59,89,179,364, np.inf]

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



# Delta empírica contrato a contrato
opt_df_sorted = opt_df_filtered.sort_values(["OptionID","Date"])

opt_df_sorted["dC"] = opt_df_sorted.groupby("OptionID")["MidPrice"].diff()
opt_df_sorted["dS"] = opt_df_sorted.groupby("OptionID")["SpotPrice"].diff()

# Filtrar días con movimiento suficiente del subyacente
MIN_DS = 5.0  # mínimo 5 puntos de movimiento en SPX

opt_df_sorted["Delta_emp"] = np.where(
    opt_df_sorted["dS"].abs() >= MIN_DS,
    opt_df_sorted["dC"] / opt_df_sorted["dS"],
    np.nan
)

# Comparar con delta de OptionMetrics
valid = opt_df_sorted[
    opt_df_sorted["Delta_emp"].notna() &
    opt_df_sorted["Delta"].notna()
].copy()

print(f"Observaciones válidas: {len(valid):,}")
print(f"Correlación con Delta OptionMetrics: "
      f"{valid['Delta'].corr(valid['Delta_emp']):.3f}")

print("\nDistribución Delta empírica:")
print(valid["Delta_emp"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))

print("\nMAE respecto a OptionMetrics:")
print((valid["Delta_emp"] - valid["Delta"]).abs().median())

# Por bucket
valid["bucket"] = pd.cut(valid["Days"], bins=v_edges,
                          labels=False, include_lowest=True)
print("\nCorrelación por bucket:")
print(valid.groupby("bucket").apply(
    lambda x: x["Delta"].corr(x["Delta_emp"])
).round(3))



# Filtros adicionales para reducir ruido en las colas
MIN_DS    = 10.0   # subir umbral de movimiento mínimo
MAX_DELTA = 1.5    # cota superior de delta válida

opt_df_sorted["Delta_emp_clean"] = np.where(
    (opt_df_sorted["dS"].abs() >= MIN_DS) &
    (opt_df_sorted["Delta_emp"].abs() <= MAX_DELTA),
    opt_df_sorted["Delta_emp"],
    np.nan
)

valid_clean = opt_df_sorted[
    opt_df_sorted["Delta_emp_clean"].notna() &
    opt_df_sorted["Delta"].notna()
].copy()

print(f"Observaciones tras filtro: {len(valid_clean):,}")
print(f"Cobertura: {len(valid_clean)/len(opt_df_sorted):.1%}")
print(f"Correlación: {valid_clean['Delta'].corr(valid_clean['Delta_emp_clean']):.3f}")
print(f"MAE: {(valid_clean['Delta_emp_clean'] - valid_clean['Delta']).abs().median():.4f}")

# Fracción de delta fuera de rango
mask_call = valid_clean["CallPut"] == "C"
mask_put  = valid_clean["CallPut"] == "P"
print(f"\nCalls con Delta_emp fuera de [0,1]: "
      f"{(~valid_clean[mask_call]['Delta_emp_clean'].between(0,1)).mean():.1%}")
print(f"Puts con Delta_emp fuera de [-1,0]: "
      f"{(~valid_clean[mask_put]['Delta_emp_clean'].between(-1,0)).mean():.1%}")


# ¿Dónde se concentran las violaciones?
violaciones_call = valid_clean[
    mask_call & ~valid_clean["Delta_emp_clean"].between(0, 1)
].copy()

violaciones_put = valid_clean[
    mask_put & ~valid_clean["Delta_emp_clean"].between(-1, 0)
].copy()

print("=== CALLS CON DELTA FUERA DE [0,1] ===")
print(f"N: {len(violaciones_call):,}")
print(f"\nPor moneyness:")
print(violaciones_call["Moneyness"].describe(
    percentiles=[.05,.25,.5,.75,.95]).round(3))
print(f"\nPor dS (movimiento subyacente):")
print(violaciones_call["dS"].describe(
    percentiles=[.05,.25,.5,.75,.95]).round(2))
print(f"\nPor bucket:")
print(violaciones_call.groupby("bucket").size() /
      valid_clean[mask_call].groupby("bucket").size())
print(f"\nPor año:")
print(violaciones_call.groupby(
    violaciones_call["Date"].dt.year).size() /
    valid_clean[mask_call].groupby(
    valid_clean["Date"].dt.year).size())



print("=============== Diagnostico 2 =======================")

# Filtro 1: Moneyness — eliminar opciones muy OTM
MIN_MONEYNESS = 0.85
MAX_MONEYNESS = 1.15

# Filtro 2: dS mínimo más agresivo para reducir ruido
MIN_DS = 15.0

# Filtro 3: Eliminar bucket (0,15] donde gamma alta
# hace que la aproximación lineal sea mala
MIN_DAYS = 15

opt_df_sorted["Delta_emp_v2"] = np.where(
    (opt_df_sorted["dS"].abs() >= MIN_DS) &
    (opt_df_sorted["Delta_emp"].abs() <= 1.5) &
    (opt_df_sorted["Moneyness"].between(MIN_MONEYNESS, MAX_MONEYNESS)) &
    (opt_df_sorted["Days"] >= MIN_DAYS),
    opt_df_sorted["Delta_emp"],
    np.nan
)

valid_v2 = opt_df_sorted[
    opt_df_sorted["Delta_emp_v2"].notna() &
    opt_df_sorted["Delta"].notna()
].copy()

mask_call_v2 = valid_v2["CallPut"] == "C"
mask_put_v2  = valid_v2["CallPut"] == "P"

print(f"Observaciones: {len(valid_v2):,}")
print(f"Cobertura: {len(valid_v2)/len(opt_df_sorted):.1%}")
print(f"Correlación: {valid_v2['Delta'].corr(valid_v2['Delta_emp_v2']):.3f}")
print(f"MAE: {(valid_v2['Delta_emp_v2']-valid_v2['Delta']).abs().median():.4f}")

print(f"\nCalls fuera de [0,1]:  "
      f"{(~valid_v2[mask_call_v2]['Delta_emp_v2'].between(0,1)).mean():.1%}")
print(f"Puts fuera de [-1,0]:  "
      f"{(~valid_v2[mask_put_v2]['Delta_emp_v2'].between(-1,0)).mean():.1%}")

print("\nCorrelación por bucket:")
print(valid_v2.groupby("bucket").apply(
    lambda x: x["Delta"].corr(x["Delta_emp_v2"])
).round(3))

print("\nCobertura por bucket:")
print(valid_v2.groupby("bucket").size() /
      opt_df_sorted.groupby("bucket").size())


# In[]:
print("Fracción cubierta:")


# ¿Qué fracción del OI total está cubierta?
total_OI = opt_df_sorted["OpenInterest"].sum()
cubierto_OI = opt_df_sorted[
    opt_df_sorted["Delta_emp_v2"].notna()
]["OpenInterest"].sum()

print(f"% OI cubierto por delta empírica válida: {cubierto_OI/total_OI:.1%}")

# Por bucket
print("\nPor bucket:")
print(opt_df_sorted.groupby("bucket").apply(
    lambda x: x[x["Delta_emp_v2"].notna()]["OpenInterest"].sum() /
              x["OpenInterest"].sum()
).round(3))
# %%
