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



###################################################################
######################### FASE 1 ##################################
###################################################################

#------------------configuración y cargade datos ------------------

PARQUET_RUTA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\option_price.parquet"

COLUMNAS = [
    'SecurityID', 'Date', 'Expiration', 'Strike', 'CallPut', 'Bid', 'Ask', 'Volume', 'OpenInterest','AMSettlement','ExpiryIndicator',
    'ImpliedVolatility', 'Delta','Gamma','Vega','Theta']

desde = "2003-01-02"
hasta = "2024-02-29"
# In[]
op = OptionPrice()
op.cargar_parquet(PARQUET_RUTA, desde, hasta,security_id=108105, columnas=COLUMNAS)
# In[]
opt_df = op.df[op.df['SecurityID'] == 108105]

opt_df["Date"] = pd.to_datetime(opt_df["Date"], format="%Y-%m-%d")
opt_df["Expiration"] = pd.to_datetime(opt_df["Expiration"], format="%Y-%m-%d")
opt_df["Days"] = ((opt_df["Expiration"] - opt_df["Date"]).dt.days - opt_df["AMSettlement"])
opt_df["Strike"] = opt_df["Strike"]/1000

#Columnas que nos interesan
opt_df = opt_df[["Date", "Expiration", "Strike", "CallPut", "Bid", "Ask", "Volume", "OpenInterest","Days","ImpliedVolatility", "ExpiryIndicator", "AMSettlement"]]

# Calulamos Mid Price y filtramos NID > 0 , por mid price > 0, volumen > 0 y open interest > 0, implied volatility > 0.
opt_df = opt_df[opt_df["Bid"] > 0]
opt_df = opt_df[opt_df["Ask"] >= opt_df["Bid"]]
opt_df["MidPrice"] = (opt_df["Bid"] + opt_df["Ask"]) / 2
opt_df["horquilla"] = (opt_df["Ask"] - opt_df["Bid"])/ opt_df["MidPrice"]
opt_df = opt_df[(opt_df["MidPrice"] > 0) & (opt_df["Volume"] > 0) & (opt_df["OpenInterest"] > 0)]
opt_df = opt_df[opt_df["ImpliedVolatility"] != -99.99]
opt_df = opt_df[opt_df["ImpliedVolatility"] > 0]



##############filtros de valores extremos (outliers) en MidPrice e Implied Volatility, por debajo del percentil 1 y por encima del percentil 99
#pendiente

opt_df.describe()

# In[]:

#---------Importamos el módulo de forward price para el subyacente -------------------

PARQUET_FP = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\forward_price_filtered.parquet"
fp = ForwardPrice()
fp.cargar_parquet(PARQUET_FP,desde, hasta)
fp_df = fp.df
PARQUET_ZC = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\zero_curve.parquet"
zc  = ZeroCurve(sep='\t')
zc.cargar_parquet(PARQUET_ZC, desde, hasta)


print(zc.df.head())
# In[]:
#---------Asignamos al dataframes de opciones los forward price -------------------

opt_bloque = []

for fecha, opt_fecha in opt_df.groupby("Date"):
    #print(f"Procesando fecha {fecha.date()} con {len(opt_fecha)} opciones...")
    fp_fecha = fp_df[fp_df["Date"] == fecha]
    fp_fecha = fp_fecha.sort_values("Days")  # Aseguramos que esté ordenado por tiempo a vencimiento
    if fp_fecha.empty:
        print(f"No hay datos de forward price para la fecha {fecha}")
        continue
    curva_fecha = zc.df[(zc.df["Date"] == fecha) & (zc.df["Currency"] == 333)]
    if curva_fecha.empty:
        print(f"No hay zero curve (Currency=333) para la fecha {fecha}")
        continue
    opt_fecha["forward_index"] = np.interp(opt_fecha["Days"], fp_fecha["Days"], fp_fecha["ForwardPrice"])
    opt_fecha["Rate"] = interpolate_rates_surface(zc.df, opt_fecha, fecha, 333, 365)

    opt_bloque.append(opt_fecha)

opt_df_2 = pd.concat(opt_bloque, ignore_index=True)

opt_df_2["Moneyness"] = opt_df_2["Strike"] / opt_df_2["forward_index"]
opt_df_2["log_moneyness"] = np.log(opt_df_2["Moneyness"])
opt_df_2["flag_otm"] = (
    ((opt_df_2["CallPut"] == "P") & (opt_df_2["Moneyness"] <= 1)) |
    ((opt_df_2["CallPut"] == "C") & (opt_df_2["Moneyness"] >= 1))
)
opt_df_2 = opt_df_2[(opt_df_2["Moneyness"] >= 0.3) & (opt_df_2["Moneyness"] <= 1.7)]



# In[]:
############ Filtros de no-arbitraje estático para opciones cotizadas (bounds estáticos) ############
def static_arbitrage_bounds(
    df: pd.DataFrame,
    filtrar: bool = False,
    abs_tol: float = 1e-10
) -> pd.DataFrame:
    """
    Aplica bounds estáticos de no arbitraje a calls y puts en un único DataFrame.

    Requiere las columnas:
    - CallPut  : 'C' o 'P'
    - Strike
    - MidPrice
    - Rate
    - Days
    - forward_index
    - flag_otm : bool

    Bounds usados
    -------------
    PUTS OTM:
        lower = exp(-rT) * max(K - F, 0)
        upper = K * exp(-rT)

    CALLS OTM:
        lower = exp(-rT) * max(F - K, 0)
        upper = F * exp(-rT)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de opciones.
    filtrar : bool, default False
        Si True, devuelve solo observaciones OTM que cumplen bounds.
        Si False, devuelve todo el DataFrame con flags añadidos.
    abs_tol : float, default 1e-10
        Tolerancia numérica para comparación de bounds.

    Returns
    -------
    pd.DataFrame
    """
    out = df.copy()

    # Tiempo y factor de descuento
    
    out["DiscountFactor"] = np.exp(-out["Rate"] * out["Days"] / 365.0)

    is_put = out["CallPut"] == "P"
    is_call = out["CallPut"] == "C"

    # Inicializamos columnas
    out["lower_bound"] = np.nan
    out["upper_bound"] = np.nan

    # PUTS
    out.loc[is_put, "lower_bound"] = (
        np.maximum(
            out.loc[is_put, "Strike"] - out.loc[is_put, "forward_index"],
            0.0
        ) * out.loc[is_put, "DiscountFactor"]
    )
    out.loc[is_put, "upper_bound"] = (
        out.loc[is_put, "Strike"] * out.loc[is_put, "DiscountFactor"]
    )

    # CALLS
    out.loc[is_call, "lower_bound"] = (
        np.maximum(
            out.loc[is_call, "forward_index"] - out.loc[is_call, "Strike"],
            0.0
        ) * out.loc[is_call, "DiscountFactor"]
    )
    out.loc[is_call, "upper_bound"] = (
        out.loc[is_call, "forward_index"] * out.loc[is_call, "DiscountFactor"]
    )

    # Flags individuales
    out["flag_lower_bound_ok"] = (
        out["MidPrice"] >= out["lower_bound"] - abs_tol
    )
    out["flag_upper_bound_ok"] = (
        out["MidPrice"] <= out["upper_bound"] + abs_tol
    )

    # Flag global bounds
    out["flag_static_bounds_ok"] = (
        out["flag_lower_bound_ok"] & out["flag_upper_bound_ok"]
    )

    # Flag final: OTM + bounds
    out["flag_otm_bounds_ok"] = (
        out["flag_otm"] & out["flag_static_bounds_ok"]
    )

    if filtrar:
        out = out[out["flag_otm_bounds_ok"]].copy()

    return out

opt_df_2 = static_arbitrage_bounds(opt_df_2, filtrar=False)
#opt_df_2 = static_arbitrage_bounds(opt_df_2, filtrar=True)
opt_df_2 = opt_df_2.drop(columns=["DiscountFactor", "lower_bound", "upper_bound"])


#############################################################################################################
#############################################################################################################
#############################################################################################################


# In[]:
############ HACEMOS UN CHECK DE DUPLICADOS EN LOS STRIKES DENTRO DE CADA GRUPO FECHA/EXPIRY/CALLPUT ############

print(opt_df_2.groupby(["Date", "Expiration", "CallPut", "Strike", "AMSettlement"]).size().value_counts())

opt_df_3 = (
    opt_df_2
    .sort_values(["Date", "Expiration", "CallPut", "Strike", "AMSettlement"])  # Ordenamos por AMSettlement para priorizar AM sobre PM
    .drop_duplicates(
        subset=["Date", "Expiration", "CallPut", "Strike"],
        keep="first"
    )
    .copy()
)


opt_df_clean = opt_df_3[opt_df_3["flag_otm_bounds_ok"]].copy()


PARQET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\opt_df_clean.parquet"
duckdb.from_df(opt_df_clean).write_parquet(PARQET_OUTPUT, compression='snappy')
print("DataFrame generado y almacenado correctamente")



# In[]:


#### Summary:

slice_summary = (
    opt_df_clean.groupby(["Date", "Days"])
    .agg(
        n_obs=("Strike", "size"),
        n_puts=("CallPut", lambda x: (x == "P").sum()),
        n_calls=("CallPut", lambda x: (x == "C").sum()),
        min_k=("log_moneyness", "min"),
        max_k=("log_moneyness", "max"),
        min_strike=("Strike", "min"),
        max_strike=("Strike", "max"),
        median_rel_spread=("horquilla", "median"),
        min_iv=("ImpliedVolatility", "min"),
        max_iv=("ImpliedVolatility", "max"),
    )
    .reset_index()
)
slice_summary["flag_enough_obs"] = slice_summary["n_obs"] >= 3
slice_summary["flag_both_sides"] = (slice_summary["min_k"] < 0) & (slice_summary["max_k"] > 0)
slice_summary["flag_reasonable_k_range"] = (slice_summary["max_k"] - slice_summary["min_k"]) >= 0.15
slice_summary["flag_slice_usable"] = (
    slice_summary["flag_enough_obs"] &
    slice_summary["flag_both_sides"] &
    slice_summary["flag_reasonable_k_range"]
)



slice_summary.to_csv(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\slice_summary_1.csv", index=False)

print("Proceso de carga y filtrado de opciones cotizadas finalizado.")

























































########################## OLD ##########################



# In[]:
# -------- Filtros de no-arbitraje estático para PUTS --------

def aplicar_boundary_puts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros de no-arbitraje estático para puts OTM.
    
    Bounds utilizados:
    1. OTM: solo puts con Strike <= forward_index
    2. Lower bound: P >= e^{-rT} * max(K - F, 0)  → como usamos OTM, K<=F → lower bound = 0
    3. Upper bound: P <= K * e^{-rT}
    4. Monotonicity: precio put creciente en K (dentro de cada grupo fecha/expiry)
    5. Convexity: precio put convexo en K (dentro de cada grupo fecha/expiry)
    
    Parameters
    ----------
    df : DataFrame con columnas [Date, Expiration, Strike, CallPut, MidPrice, 
                                  forward_index, TimeToExpiration]
                                  
    Returns
    -------
    DataFrame filtrado
    """
    
    # Trabajamos solo con puts
    puts = df[df["CallPut"] == "P"].copy()
    
    # 1. OTM: Strike <= ForwardPrice para puts
    puts = puts[puts["flag_otm"] == True]
    
    # 2. Upper bound: P <= K * e^{-rT}
    
    puts["DiscountFactor"] = np.exp(-puts["Rate"] * puts["Days"] / 365)
    puts = puts[puts["MidPrice"] <= puts["Strike"] * puts["DiscountFactor"]]
    
    # 3. Lower bound: P >= max(K - F, 0) * e^{-rT}
    # Para puts OTM (K <= F) el lower bound es 0, pero lo dejamos explícito
    lower = np.maximum(puts["Strike"] - puts["forward_index"], 0) * puts["DiscountFactor"]
    puts = puts[puts["MidPrice"] >= lower]
    
    # # 4. Monotonicity y Convexity (por grupo fecha + expiry)
    # puts = puts.sort_values(["Date", "Expiration", "Strike"])
    
    # valid_idx = []
    # for (fecha, expiry), grupo in puts.groupby(["Date", "Expiration"]):
    #     if len(grupo) < 2:
    #         continue
        
    #     strikes = grupo["Strike"].values
    #     prices  = grupo["MidPrice"].values
        
    #     # Monotonicity: precio put debe ser creciente en K
    #     mono_mask = np.ones(len(grupo), dtype=bool)
    #     for i in range(1, len(grupo)):
    #         if prices[i] < prices[i-1]:
    #             mono_mask[i] = False
        
    #     # Convexity: segunda diferencia >= 0
    #     conv_mask = np.ones(len(grupo), dtype=bool)
    #     for i in range(1, len(grupo) - 1):
    #         h1 = strikes[i]   - strikes[i-1]
    #         h2 = strikes[i+1] - strikes[i]
    #         second_diff = (prices[i+1] - prices[i]) / h2 - (prices[i] - prices[i-1]) / h1
    #         if second_diff < -1e-6:  # tolerancia numérica
    #             conv_mask[i] = False
        
    #     mask = mono_mask & conv_mask
    #     valid_idx.extend(grupo.index[mask].tolist())
    
    # return puts.loc[valid_idx].drop(columns=["DiscountFactor"])
    # return puts.drop(columns=["DiscountFactor"])

# puts_clean = aplicar_boundary_puts(opt_df_2)

# print(f"Puts antes de boundaries: {len(opt_df_2[opt_df_2['CallPut']=='P'])}")
# print(f"Puts después de boundaries: {len(puts_clean)}")



# In[]:
# -------- Filtros de no-arbitraje estático para CALLS --------



def aplicar_boundary_calls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica filtros de no-arbitraje estático para calls OTM.
    
    Bounds utilizados:
    1. OTM: solo calls con Strike >= forward_index
    2. Upper bound: C <= F * e^{-rT}  (= S * e^{-qT})
    3. Lower bound: C >= max(F - K, 0) * e^{-rT} → para OTM es 0, trivial
    4. Monotonicity: precio call decreciente en K
    5. Convexity: precio call convexo en K
    
    Parameters
    ----------
    df : DataFrame con columnas [Date, Expiration, Strike, CallPut, MidPrice,
                                  forward_index, TimeToExpiration, Rate]
                                  
    Returns
    -------
    DataFrame filtrado
    """
    
    # Trabajamos solo con calls
    calls = df[df["CallPut"] == "C"].copy()
    
    # 1. OTM: Strike >= ForwardPrice para calls
    calls = calls[calls["flag_otm"] == True]
    
    # 2. Factor de descuento
    calls["DiscountFactor"] = np.exp(-calls["Rate"] * calls["Days"] / 365)
    
    # 3. Upper bound: C <= F * e^{-rT}
    calls = calls[calls["MidPrice"] <= calls["forward_index"] * calls["DiscountFactor"]]
    
    # 4. Lower bound: C >= max(F - K, 0) * e^{-rT}
    # Para calls OTM (K >= F) el lower bound es 0, pero lo dejamos explícito
    lower = np.maximum(calls["forward_index"] - calls["Strike"], 0) * calls["DiscountFactor"]
    calls = calls[calls["MidPrice"] >= lower]
    
    # # 5. Monotonicity y Convexity (por grupo fecha + expiry)
    # calls = calls.sort_values(["Date", "Expiration", "Strike"])
    
    # valid_idx = []
    # for (fecha, expiry), grupo in calls.groupby(["Date", "Expiration"]):
    #     if len(grupo) < 2:
    #         continue
        
    #     strikes = grupo["Strike"].values
    #     prices  = grupo["MidPrice"].values
        
    #     # Monotonicity: precio call debe ser DECRECIENTE en K
    #     mono_mask = np.ones(len(grupo), dtype=bool)
    #     for i in range(1, len(grupo)):
    #         if prices[i] > prices[i-1]:
    #             mono_mask[i] = False
        
    #     # Convexity: segunda diferencia >= 0 (igual que puts)
    #     conv_mask = np.ones(len(grupo), dtype=bool)
    #     for i in range(1, len(grupo) - 1):
    #         h1 = strikes[i]   - strikes[i-1]
    #         h2 = strikes[i+1] - strikes[i]
    #         second_diff = (prices[i+1] - prices[i]) / h2 - (prices[i] - prices[i-1]) / h1
    #         if second_diff < -1e-6:
    #             conv_mask[i] = False
        
    #     mask = mono_mask & conv_mask
    #     valid_idx.extend(grupo.index[mask].tolist())
    
    # return calls.loc[valid_idx].drop(columns=["DiscountFactor"])
    return calls.drop(columns=["DiscountFactor"])


# calls_clean = aplicar_boundary_calls(opt_df_2)

# print(f"Calls antes de boundaries: {len(opt_df_2[opt_df_2['CallPut']=='C'])}")
# print(f"Calls después de boundaries: {len(calls_clean)}")






# %%
