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
    'SecurityID','OptionID' ,'Date', 'Expiration', 'Strike', 'CallPut', 'Bid', 'Ask', 'Volume', 'OpenInterest','AMSettlement','ExpiryIndicator',
    'ImpliedVolatility', 'Delta','Gamma','Vega','Theta']

desde = "2003-01-02"
hasta = "2024-02-29"

op = OptionPrice()
op.cargar_parquet(PARQUET_RUTA, desde, hasta,security_id=108105, columnas=COLUMNAS)
opt_df = op.df
opt_df = opt_df[opt_df["ImpliedVolatility"] != -99.99]


# PARQET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Descriptivos\opt_df.parquet"
# duckdb.from_df(opt_df).write_parquet(PARQET_OUTPUT, compression='snappy')
# print("DataFrame generado y almacenado correctamente")

# In[]



opt_df["Date"] = pd.to_datetime(opt_df["Date"], format="%Y-%m-%d")
opt_df["Expiration"] = pd.to_datetime(opt_df["Expiration"], format="%Y-%m-%d")
opt_df["Days"] = ((opt_df["Expiration"] - opt_df["Date"]).dt.days - opt_df["AMSettlement"])
opt_df["Strike"] = opt_df["Strike"]/1000



# In[]

####### Descriptivo 1: Strikes más cotizados.

unique_strikes = op.df["Strike"].astype(int).unique().tolist()
print(unique_strikes)





# In[]
x = opt_df["Strike"]*1000
# In[]
print(x.describe())
opt_df["Strike"].describe()
# %%
