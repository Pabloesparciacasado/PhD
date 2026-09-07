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


if os.name == 'nt':
    PATH_DATA = r"Y:\OUTPUTS\REPLICA_HW\opt_df_prueba.parquet"
    PATH_COEF_OUT = r"Y:\OUTPUTS\REPLICA_HW\HW_coef.csv"
    PATH_OOS_OUT = r"Y:\OUTPUTS\REPLICA_HW\HW_oos.parquet"

else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_prueba.parquet"
# %% Carga de datos (DuckDB)

print("Cargando datos...")

con = duckdb.connect()
opt_df = con.execute(f"""
    SELECT *
    FROM read_parquet('{PATH_DATA}')
    -- WHERE lo que necesites filtrar, si aplica en este punto
""").df()

# opt_df = pd.read_parquet(PATH_DATA)

opt_df

# In[]: Filtro de columnas y precios spot

#filtro de columnas relevantes:

cols = ['OptionID','CallPut', 'Date','Days', 'Strike',
        'Bid', 'Ask', 'MidPrice','Volume', 'OpenInterest',
        'ImpliedVolatility', 'Delta', 'Gamma', 'Vega', 'Theta' ,
        'Rate', 'SpotPrice','Moneyness' ]

optf_df = opt_df[cols].copy()


desde = "2003-01-02"
hasta = "2024-02-29"

sec_price = pd.read_parquet(os.path.join(r"Y:\OptionMetrics\Acumulado", "security_price.parquet"))
print("Spot Cargado")

sec_price["Date"] = pd.to_datetime(sec_price["Date"], format="%Y-%m-%d")
SP500_price = sec_price[sec_price["SecurityID"] == 108105].reset_index(drop=True)

SP500_price = SP500_price[(SP500_price["Date"] >= pd.Timestamp(desde)) & (SP500_price["Date"] <= pd.Timestamp(hasta))].reset_index(drop=True)

# next_trading_dates  =  SP500_price["Date"].dropna().unique()


# next_trading_dates = pd.Series(next_trading_dates.shift(-1), index=next_trading_dates)
# print(next_trading_dates)

trading_dates = (
    SP500_price["Date"]
    .dropna()
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

next_trading_dates = pd.Series(
    trading_dates.shift(-1).values,
    index=trading_dates.values
)


#%% Función: dataset_preparation
"""
Los datos importados tan solo tiene el filtro de quela implied volatility,
    se ha puesto formato fecha a tau (Days) ,
    el strike divido /1000,
    enemos el precio del subyacente,
    Y tenemos los datos filtrados para el SP500.

    VER "01_limpieza_preliminar"
"""

def dataset_preparation(df,next_trad_date):

    # Hacemos filtro de pairs dates. Necesitamos al menos dos fechas consecutivas de trading.
    df = df.sort_values( ["OptionID", "Date"]).copy()
        
    g = df.groupby("OptionID", sort=False)

    df["price_next"] = g["MidPrice"].shift(-1)
    df["spot_next"]  = g["SpotPrice"].shift(-1)
    df["next_date"]  = g["Date"].shift(-1)

    df["next_market_date"] = df["Date"].map(next_trad_date)
#####-> poner el dropnan
    df = df[df["next_date"] == df["next_market_date"]].copy()

    df = df[df["Days"]>=14].copy()

    df["T"] = df["Days"] / 365
    
    # BID / ASK /VI and greeks
    df = df[ (df["Bid"]>0) & (df["Ask"]>0) & (df["ImpliedVolatility"] != -99.99)].copy()

    # Quitamos deltas extremas:
    df = df[ (np.abs(df["Delta"].round(2))>=0.05) & ( np.abs(df["Delta"].round(2))<=0.95) ].copy()

    #Normalizamos el spot para el primer día del par sea de 1:

    df["ds"] = (df["spot_next"]- df["SpotPrice"])/df["SpotPrice"]
    df["df_option"] = (df["price_next"] - df["MidPrice"]) / df["SpotPrice"]
    df["vega_norm"] = df["Vega"] / df["SpotPrice"]

    # Metemos la ecuación de regresión:

        # Y = Δf - delta_BS ΔS
          # Y = vega*ds/T *(a + bDelta +cDelta^2)
          # Y = a* vegads/T  + b*vega*ds/T*Delta +c*vega*ds/TDelta^2)

    df["Y"] = df["df_option"] -  df["Delta"] * df["ds"]
    df["Z"] = df["vega_norm"]* df["ds"]/np.sqrt(df["T"])
    df["X_a"] = df["Z"]
    df["X_b"] = df["Z"]*df["Delta"]
    df["X_c"] = df["Z"]*df["Delta"]**2

    return df.reset_index(drop=True)


# ============================================================
#  ESTIMACIÓN DE a, b, c
# ============================================================

def fit_hw_coefficients(df_train):
    """
    Estima Eq. (6) SIN intercepto:

        Y = a*X_a + b*X_b + c*X_c + epsilon

    equivalente a:

        Δf - delta_BS ΔS = [vega / sqrt(T)] [ΔS/S](a + b delta_BS + c delta_BS²)  + epsilon
       
    """

    cols = [ "Y","X_a","X_b","X_c" ]
    d = (  df_train[cols].replace([np.inf, -np.inf], np.nan).dropna() )
      
# LSTSQ (más abajo) no permite meter DataFrames, es preferible trabajar con las matrices de Numpy
  # y usamos np.linalg.lstsq se basa en la norma 2 y devuelve información clave del álgebra su diagnóstico como rango... y es más rápdido.
        
    y = d["Y"].to_numpy()

    X = d[ ["X_a", "X_b", "X_c"] ].to_numpy() 
        
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    return { "a": beta[0], "b": beta[1],"c": beta[2],"N": len(d) }


# ============================================================
# una vez estimados los parametros. calculamos los errores:
# ============================================================

def apply_hw_delta(df_test, params):

    out = df_test.copy()

    a = params["a"]
    b = params["b"]
    c = params["c"]

    delta = out["Delta"]

    # --------------------------------------------------------
    # Función Cuadrática de HW
    # --------------------------------------------------------

    out["hw_poly"] = ( a + b * delta + c * delta**2 )
        
    # ========================================================
    # Derivada empírica de IV respecto al spot:
    # ∂E[sigma_imp] / ∂S
    # ========================================================

    out["sigma_S_hat"] = ( out["hw_poly"] /(out["SpotPrice"]* np.sqrt(out["T"])) )

    # ========================================================
    # Minimum Variance Delta:
    # delta_MV= delta_BS + Vega * sigma_S_hat
    # ========================================================

    out["delta_mv"] = (out["Delta"] + out["Vega"] * out["sigma_S_hat"])
        
    # --------------------------------------------------------
    # Hedging errors usando variables normalizadas
    # --------------------------------------------------------

    out["eps_bs"] = (  out["df_option"] - out["Delta"] * out["ds"] )

    out["eps_mv"] = (  out["df_option"] -  out["delta_mv"] * out["ds"] )
       

    return out



# ============================================================
# ROLLING WINDOW. Dates from HW(2017)
# ============================================================

def hw_rolling_window(
    pairs,
    f1="2004-01-01", # primera fecha de estimación, tras el tiempo de espera en la primera estimación-
    f2="2015-08-31",
    window_months_est= 36,
    month_pred=1,
    cp_col="CallPut"
):
    """
    HW:
        36 meses de estimación -> parámetros aplicados al siguiente mes.
        
    Calls y puts se estiman separadamente.
    """

    
    df = pairs.copy()


    #filtro harcodeado de fechas usadas en HW:

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"]>=pd.Timestamp(f1)) & (df["Date"]<=pd.Timestamp(f2))] #pandas intenta interpretar el string como fecha, pero con Timestamp mejor


    first_month = pd.Timestamp(df["Date"].min()).to_period("M")
    first_test_month = (first_month + window_months_est)
    last_month = pd.Timestamp(df["Date"].max()).to_period("M")

    months = pd.period_range(start=first_test_month, end=last_month, freq="M")
    
    results = []
    coefficients = []

    for m in months:
        train_start = (m - window_months_est).start_time
        train_end= (m).start_time 

        testing_start = (m).start_time
        testing_end = (m + month_pred).start_time

        train = df[ (df["Date"]>=train_start) &( df["Date"]<train_end) ]
        test = df[ (df["Date"]>=testing_start) &( df["Date"]<testing_end) ]

        # HW estima puts y calls separadamente
        for cp in ["C", "P"]:

            train_cp = train[ train[cp_col] == cp ]
            test_cp = test[ test[cp_col] == cp ]
                
            if len(train_cp) == 0 or len(test_cp) == 0:
                continue

            params = fit_hw_coefficients( train_cp ) 

            coefficients.append({
                "test_month": m,
                "cp_flag": cp,
                "train_start": train_start,
                "train_end": train_end,
                **params  })
           
            pred = apply_hw_delta( test_cp, params)

            pred["test_month"] = m

            results.append(pred)

    oos = pd.concat( results, ignore_index=True )

    coef = pd.DataFrame(coefficients)

    return oos, coef



pairs = dataset_preparation(optf_df, next_trading_dates)

oos, coef = hw_rolling_window( pairs )


oos["test_month"] = oos["test_month"].dt.to_timestamp()

duckdb.from_df(oos).write_parquet(
    str(PATH_OOS_OUT),
    compression="snappy")

print(f"Resultado OOS guardado correctamente en: {PATH_OOS_OUT}")

coef.to_csv(PATH_COEF_OUT, index=False, encoding='utf-8')
print(f"Resultado coef guardado correctamente en: {PATH_COEF_OUT}")


# %% Cargar resultados OOS

OOS = pd.read_parquet(PATH_OOS_OUT)
print("Resultados OOS cargados")

"""

Comparación de métrica de Gains vs HW(2017) paper
Eq. (3):
    Gain = 1 - SSE(eps_MV) / SSE(eps_BS)

"""
# ============================================================
# VALIDACIÓN HULL-WHITE: GAIN
# ============================================================
# %% Gain global (Hull-White)

def gain_hw(df):
    """
    Hull & White (2017), Eq. (3):

        Gain = 1 - SSE(eps_MV) / SSE(eps_BS)
    """

    sse_mv = np.sum(df["eps_mv"] ** 2)
    sse_bs = np.sum(df["eps_bs"] ** 2)

    return 1 - sse_mv / sse_bs


gain = (
    OOS.groupby("CallPut")
    .apply(gain_hw)
    .rename("Gain")
    .reset_index()

)

gain["Gain_pct"] = gain["Gain"] * 100

print(gain.round(0))

# %% Gain por bucket (setup)
"""

Comparación de métrica de Gains vs HW(2017) paper por buckets

"""

oos = OOS.copy()

# %% Definir Delta_bucket

oos["Delta_bucket"] = oos["Delta"].round(1)

oos.loc[oos["CallPut"] == "C", "Delta_bucket"] = (
    oos.loc[oos["CallPut"] == "C", "Delta_bucket" ]
    .clip(0.1,0.9)
)


oos.loc[oos["CallPut"] == "P","Delta_bucket"] = (

    oos.loc[oos["CallPut"]=="P", "Delta_bucket"]
    .clip(-0.1,-0.9)
)
# %% Tabla de Gain por bucket
gain_bucket = (
    oos.groupby(["CallPut", "Delta_bucket"])
    .apply(gain_hw)
    .rename("Gain")
    .reset_index()
)

gain_bucket["Gain_pct"] = gain_bucket["Gain"] * 100
print(gain_bucket.round(2))


gain_tab = (
    gain_bucket
    .pivot(
        index="Delta_bucket",
        columns="CallPut",
        values="Gain_pct" )
    .sort_index()
)
gain_tab = gain_tab.round(2)
print(gain_tab)

# %% Cargar coeficientes HW

COEF = pd.read_csv(PATH_COEF_OUT)
print("Resultados COEF cargados")


coef = COEF.copy()

coefc = coef[coef["cp_flag"] == "C"]
coefp = coef[coef["cp_flag"] == "P"]

# %% Plot coeficientes - Calls y Puts

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

coefc["fecha"] = pd.to_datetime(coefc["train_end"])  # asegúrate de que sea datetime
df = coefc.sort_values("train_end")
df["-b"] = -df["b"]

columnas = ["a", "-b", "c"]  # columnas a plotear

fig, ax = plt.subplots(figsize=(10, 5))

for col in columnas:
    ax.plot(df["fecha"], df[col], label=col, linewidth=1.5)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

ax.set_xlabel("Fecha")
ax.set_ylabel("Valor")
ax.set_title("Estimated Call parameters")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(r"Y:\OUTPUTS\REPLICA_HW\HW_coef_call.svg")
print("imagen guardada correctamente en formato svg")

plt.show()


coefp["fecha"] = pd.to_datetime(coefp["train_end"])  # asegúrate de que sea datetime
df = coefp.sort_values("train_end")

columnas = ["a", "b", "c"]  # columnas a plotear

fig, ax = plt.subplots(figsize=(10, 5))

for col in columnas:
    ax.plot(df["fecha"], df[col], label=col, linewidth=1.5)

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate()

ax.set_xlabel("Fecha")
ax.set_ylabel("Valor")
ax.set_title("Estimated Put parameters")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(r"Y:\OUTPUTS\REPLICA_HW\HW_coef_put.svg")
print("imagen guardada correctamente en formato svg")

plt.show()


# %%
