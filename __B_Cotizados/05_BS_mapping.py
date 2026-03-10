# In[]
import pandas as pd
import numpy as np
from scipy.stats import norm
import sys
import duckdb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __3_Functions.valuation import model_valuation


con = duckdb.connect()

vol_surface = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\pablo.esparcia\\Documents\OptionMetrics\\output\\volatility_surface.parquet')
""").df()
print(vol_surface.columns)
#### Aplicamos BSM para mapear las volatilidades a precios de opciones. #####


# Algoritmo de estandarización de maturities para la volatilidad
# In[]




# In[]
model = model_valuation(curve_df = vol_surface)
resultado_europeo = model.price_BS_general(vol_surface)
resultado_europeo = resultado_europeo.rename(columns={"BS_Price": "Precio_Modelo"})

# In[]

print(resultado_europeo[resultado_europeo["CallPut"] == "P"].sort_values("forward").head(30))

# In[]

len(resultado_europeo)

# In[]





















# vol_surface_2 = vol_surface[(vol_surface["flag_wing_clipped"]) == False]
# #print(vol_surface_2.head(50).sort_values("Days"))

# clipped_extrap = vol_surface[
#     (vol_surface["flag_wing_clipped"]) & 
#     (vol_surface["flag_inside_observed_range"])
# ]
# print(clipped_extrap)


# # In[]

# # Coge una smile cualquiera
# grupo = list(vol_surface.groupby(["Date", "Expiration"]))
# smile = grupo[0][1]

# print("n_grid points:", len(smile))
# print("m_obs_min:", smile["m_obs_min"].iloc[0])
# print("m_obs_max:", smile["m_obs_max"].iloc[0])
# print("moneyness min:", smile["moneyness"].min())
# print("moneyness max:", smile["moneyness"].max())
# print(smile["flag_inside_observed_range"].value_counts())

# # In[]


# stats_by_smile = (
#     vol_surface.groupby(["Date", "Expiration"])["flag_wing_clipped"]
#     .value_counts()
#     .unstack(fill_value=0)
# )

# print(stats_by_smile)
# # %%

# summary = (
#     vol_surface.groupby(["Date","Expiration"])
#     .agg(
#         n_points=("flag_wing_clipped","size"),
#         n_clipped=("flag_wing_clipped","sum"),
#         has_clipping=("flag_wing_clipped","any")
#     )
# )

# print(summary.head())


# # %%
# has_false = (
#     vol_surface.groupby(["Date", "Expiration"])["flag_wing_clipped"]
#     .apply(lambda s: (~s).any())
# )

# all_false = vol_surface.groupby(["Date", "Expiration"])["flag_wing_clipped"].any()
# all_false.mean()
# # %%

# %%
