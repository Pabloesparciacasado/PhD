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

PARQUET_FP = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\forward_price_filtered.parquet"
fp = ForwardPrice()
fp.cargar_parquet(PARQUET_FP,desde, hasta)
fp_df = fp.df

PARQUET_ZC = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\zero_curve.parquet"
zc  = ZeroCurve(sep='\t')
zc.cargar_parquet(PARQUET_ZC, desde, hasta)
zc_df = zc.df


opt_bloque = []

for fecha, opt_fecha in opt_df.groupby("Date"):
   
    fp_fecha = fp_df[fp_df["Date"] == fecha]
    fp_fecha = fp_fecha.sort_values("Days")  
    if fp_fecha.empty:
        
        continue
    curva_fecha = zc_df[(zc_df["Date"] == fecha) & (zc_df["Currency"] == 333)]
    if curva_fecha.empty:
       
        continue
    opt_fecha["forward_index"] = np.interp(opt_fecha["Days"], fp_fecha["Days"], fp_fecha["ForwardPrice"])
    opt_fecha["Rate"] = interpolate_rates_surface(zc_df, opt_fecha, fecha, 333, 365)

    opt_bloque.append(opt_fecha)

opt_df = pd.concat(opt_bloque, ignore_index=True)

opt_df["Moneyness"] = opt_df["Strike"] / opt_df["forward_index"]
opt_df["log_moneyness"] = np.log(opt_df["Moneyness"])
opt_df["flag_otm"] = (
    ((opt_df["CallPut"] == "P") & (opt_df["Moneyness"] <= 1)) |
    ((opt_df["CallPut"] == "C") & (opt_df["Moneyness"] >= 1))
)
# %%
opt_df
# %%
