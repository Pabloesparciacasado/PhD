# In[]
import pandas as pd
import numpy as np
import sys
import duckdb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __2_Files.option_price import OptionPrice
from __2_Files.forward_price import ForwardPrice
from __3_Functions.interpolation import interpolate_rates_surface
from __2_Files.zero_curve import ZeroCurve



#------------------configuración y cargade datos ------------------

PARQUET_RUTA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\option_price.parquet"


COLUMNAS = [
    'OptionID','SecurityID', 'Date', 'Expiration', 'Strike', 'CallPut', 'Bid', 'Ask', 'Volume', 'OpenInterest','AMSettlement','ExpiryIndicator',
    'ImpliedVolatility', 'Delta','Gamma','Vega','Theta']

desde = "2003-01-02"
hasta = "2024-02-29"

op = OptionPrice()
op.cargar_parquet(PARQUET_RUTA, desde, hasta,security_id=108105, columnas=COLUMNAS)

opt_df = op.df

# Reducir memoria: float64→float32 (~40% menos RAM)
_float_cols = ['Bid', 'Ask', 'Strike', 'ImpliedVolatility', 'Delta', 'Gamma', 'Vega', 'Theta']
for _c in _float_cols:
    if _c in opt_df.columns:
        opt_df[_c] = opt_df[_c].astype('float32')

opt_df["Date"] = pd.to_datetime(opt_df["Date"], format="%Y-%m-%d")
opt_df["Expiration"] = pd.to_datetime(opt_df["Expiration"], format="%Y-%m-%d")
opt_df["Days"] = ((opt_df["Expiration"] - opt_df["Date"]).dt.days - opt_df["AMSettlement"])
opt_df["Strike"] = opt_df["Strike"]/1000
opt_df = opt_df[opt_df["ImpliedVolatility"] != -99.99]

opt_df["MidPrice"] = (opt_df["Bid"] + opt_df["Ask"]) / 2
opt_df["horquilla"] = (opt_df["Ask"] - opt_df["Bid"])/ opt_df["MidPrice"]

opt_df = opt_df[(opt_df["Bid"]>=0) & (opt_df["Ask"] > opt_df["Bid"])]

# In[]

# PARQUET_FP = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\forward_price_filtered.parquet"
# fp = ForwardPrice()
# fp.cargar_parquet(PARQUET_FP,desde, hasta)
# fp_df = fp.df

# PARQUET_ZC = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\zero_curve.parquet"
# zc  = ZeroCurve(sep='\t')
# zc.cargar_parquet(PARQUET_ZC, desde, hasta)
# zc_df = zc.df


# opt_bloque = []

# for fecha, opt_fecha in opt_df.groupby("Date"):
   
#     fp_fecha = fp_df[fp_df["Date"] == fecha]
#     fp_fecha = fp_fecha.sort_values("Days")  
#     if fp_fecha.empty:
        
#         continue
#     curva_fecha = zc_df[(zc_df["Date"] == fecha) & (zc_df["Currency"] == 333)]
#     if curva_fecha.empty:
       
#         continue
#     opt_fecha["forward_index"] = np.interp(opt_fecha["Days"], fp_fecha["Days"], fp_fecha["ForwardPrice"])
#     opt_fecha["Rate"] = interpolate_rates_surface(zc_df, opt_fecha, fecha, 333, 365)

#     opt_bloque.append(opt_fecha)

# opt_df = pd.concat(opt_bloque, ignore_index=True)

# opt_df["Moneyness"] = opt_df["Strike"] / opt_df["forward_index"]
# opt_df["log_moneyness"] = np.log(opt_df["Moneyness"])
# opt_df["flag_otm"] = (
#     ((opt_df["CallPut"] == "P") & (opt_df["Moneyness"] <= 1)) |
#     ((opt_df["CallPut"] == "C") & (opt_df["Moneyness"] >= 1))
# )
# %%
opt_df
# %%

sec_price = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\security_price.parquet")

# %%
sec_price["Date"] = pd.to_datetime(sec_price["Date"], format="%Y-%m-%d")

SP500_price = sec_price[sec_price["SecurityID"] == 108105].reset_index(drop=True)
SP500_price = SP500_price[(SP500_price["Date"] >= desde) & (SP500_price["Date"] <= hasta)].reset_index(drop=True)

opt_df_prueba = pd.merge(opt_df, SP500_price[["Date", "ClosePrice"]], on="Date", how="outer", indicator=True)
# %%

print(opt_df_prueba["_merge"].value_counts())

opt_df_prueba.rename(columns={"ClosePrice": "SpotPrice"}, inplace=True)



opt_df_prueba["Moneyness"] = opt_df_prueba["Strike"] / opt_df_prueba["SpotPrice"]
opt_df_prueba["log_moneyness"] = np.log(opt_df_prueba["Moneyness"])
opt_df_prueba["flag_otm"] = (
    ((opt_df_prueba["CallPut"] == "P") & (opt_df_prueba["Moneyness"] <= 1)) |
    ((opt_df_prueba["CallPut"] == "C") & (opt_df_prueba["Moneyness"] >= 1))
)


PARQUET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"
duckdb.from_df(opt_df_prueba).write_parquet(PARQUET_OUTPUT, compression='snappy')
print("=" * 100)
print(f"Fichero guardado correctamente en: {PARQUET_OUTPUT}")

# %%
