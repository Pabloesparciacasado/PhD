# In[]
import pandas as pd
import numpy as np
import sys
import os
import duckdb


from pathlib import Path



if os.name == 'nt':
    PATH_DATA = r"Y:\OUTPUTS\opt_df_prueba.parquet"
    PATH_OUTPUT = r"Y:\OUTPUTS\opt_df_empirical_greeks_sinfiltro.parquet"
else:
    PATH_DATA = r"/Volumes/data/OptionMetrics/OUTPUTS/opt_df_prueba.parquet"
    PATH_OUTPUT = r"/Volumes/data/OptionMetrics/OUTPUTS/opt_df_empirical_greeks.parquet"

PATH_DATA   = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)


#Añadimos algunas variables de interés:
opt_df["Dummy_Bid"] = opt_df["Bid"] > 0
opt_df["DolarVolume"] = opt_df["Volume"] *opt_df["MidPrice"]



# Asignamos buckets de vencimientos:¡
v_grid = [0, 15, 45, 105, 183, 365, np.inf]
v_edges = pd.IntervalIndex.from_breaks(v_grid, closed="right")
opt_df["maturity_bucket"] = pd.cut(opt_df["Days"], 
    bins=v_edges, 
    labels=False, 
    include_lowest=True)


# Asignamos buckets de moneyness (de momento en relación al spot):
# m_grid = np.round(np.arange(0.1, 2.1, 0.1), 2)
# m_grid = np.concatenate(([0], m_grid, [np.inf]))

# opt_df["moneyness_bucket"] = pd.cut(
#     opt_df["Moneyness"],
#     bins=m_grid,
#     right=True,
#     labels=False,
#     include_lowest=True
# )

m_grid = np.round(  np.linspace(0.1,2,int(2/0.1)),2)
m_grid = np.concatenate(([0],m_grid, [np.inf]))
m_edges = pd.IntervalIndex.from_breaks(m_grid, closed="right")
opt_df["moneyness_bucket"] = pd.cut(opt_df["Moneyness"], bins=m_edges, labels=False, include_lowest=True)


print(opt_df[["Days", "maturity_bucket", "Moneyness", "moneyness_bucket"]].head(10))
print("Datos cargados y buckets asignados.")

# In[]:

# ============================================================
# OPCIÓN 1: Delta empírica a nivel contrato individual
# ============================================================

def delta_empirica_op1(data):
    df = data.sort_values(["OptionID", "Date"]).copy()

    df["MidPrice_lag"]         = df.groupby("OptionID")["MidPrice"].shift(1)
    df["SpotPrice_lag"]        = df.groupby("OptionID")["SpotPrice"].shift(1)
    df["moneyness_bucket_lag"] = df.groupby("OptionID")["moneyness_bucket"].shift(1)

    df = df.dropna(subset=["MidPrice_lag", "SpotPrice_lag", "moneyness_bucket_lag"])
    df = df[df["moneyness_bucket"] == df["moneyness_bucket_lag"]]

    dS = df["SpotPrice"] - df["SpotPrice_lag"]
    df = df[dS.abs() > 0].copy()
    dS = df["SpotPrice"] - df["SpotPrice_lag"]
    dC = df["MidPrice"]  - df["MidPrice_lag"]

    df["delta_emp"] = dC / dS

    # Mantenemos SpotPrice para gamma
    return df[["Date", "OptionID", "CallPut", "moneyness_bucket", "OpenInterest", "SpotPrice", "delta_emp"]]


# ============================================================
# OPCIÓN 2: Delta empírica agregada por bucket
# ============================================================

def delta_empirica_op2(delta_op1):
    def wavg(group):
        w = group["OpenInterest"]
        d = group["delta_emp"]
        if w.sum() == 0:
            return np.nan
        return (w * d).sum() / w.sum()

    resultado = (delta_op1
        .groupby(["Date", "CallPut", "moneyness_bucket"])
        .apply(wavg)
        .reset_index()
        .rename(columns={0: "delta_emp_bucket"})
    )
    return resultado


# ============================================================
# OPCIÓN 3: Delta empírica sobre precio agregado por bucket
# ============================================================

def delta_empirica_op3(data):
    df = data.copy()

    agg = (df.groupby(["Date", "CallPut", "moneyness_bucket"])
        .apply(lambda g: pd.Series({
            "MidPrice_agg": (g["OpenInterest"] * g["MidPrice"]).sum() / g["OpenInterest"].sum()
                             if g["OpenInterest"].sum() > 0 else np.nan,
            "SpotPrice":    g["SpotPrice"].iloc[0]
        }))
        .reset_index()
    )

    agg = agg.sort_values(["CallPut", "moneyness_bucket", "Date"])

    agg["MidPrice_agg_lag"] = agg.groupby(["CallPut", "moneyness_bucket"])["MidPrice_agg"].shift(1)
    agg["SpotPrice_lag"]    = agg.groupby(["CallPut", "moneyness_bucket"])["SpotPrice"].shift(1)

    agg = agg.dropna(subset=["MidPrice_agg_lag", "SpotPrice_lag"])

    dS = agg["SpotPrice"]    - agg["SpotPrice_lag"]
    dC = agg["MidPrice_agg"] - agg["MidPrice_agg_lag"]

    agg = agg[dS.abs() > 0].copy()
    dS  = agg["SpotPrice"]    - agg["SpotPrice_lag"]
    dC  = agg["MidPrice_agg"] - agg["MidPrice_agg_lag"]

    agg["delta_emp_bucket"] = dC / dS

    # Mantenemos SpotPrice para gamma
    return agg[["Date", "CallPut", "moneyness_bucket", "MidPrice_agg", "SpotPrice", "delta_emp_bucket"]]


# ============================================================
# OPCIÓN 1: Gamma empírica a nivel contrato individual
# ============================================================

def gamma_empirica_op1(delta_op1):
    df = delta_op1.sort_values(["OptionID", "Date"]).copy()

    df["delta_emp_lag"] = df.groupby("OptionID")["delta_emp"].shift(1)
    df["SpotPrice_lag"] = df.groupby("OptionID")["SpotPrice"].shift(1)

    df = df.dropna(subset=["delta_emp_lag", "SpotPrice_lag"])

    dS     = df["SpotPrice"] - df["SpotPrice_lag"]
    ddelta = df["delta_emp"] - df["delta_emp_lag"]

    df = df[dS.abs() > 0].copy()
    dS     = df["SpotPrice"] - df["SpotPrice_lag"]
    ddelta = df["delta_emp"] - df["delta_emp_lag"]

    df["gamma_emp"] = ddelta / dS

    return df[["Date", "OptionID", "CallPut", "moneyness_bucket", "OpenInterest", "gamma_emp"]]


# ============================================================
# OPCIÓN 2: Gamma empírica agregada por bucket
# ============================================================

def gamma_empirica_op2(gamma_op1):
    def wavg(group):
        w = group["OpenInterest"]
        g = group["gamma_emp"]
        if w.sum() == 0:
            return np.nan
        return (w * g).sum() / w.sum()

    resultado = (gamma_op1
        .groupby(["Date", "CallPut", "moneyness_bucket"])
        .apply(wavg)
        .reset_index()
        .rename(columns={0: "gamma_emp_bucket"})
    )
    return resultado


# ============================================================
# OPCIÓN 3: Gamma empírica sobre delta agregada por bucket
# ============================================================

def gamma_empirica_op3(delta_op2, data):
    df = delta_op2.copy()

    spot = data[["Date", "SpotPrice"]].drop_duplicates("Date")
    df   = df.merge(spot, on="Date", how="left")

    df = df.sort_values(["CallPut", "moneyness_bucket", "Date"])

    df["delta_lag"]     = df.groupby(["CallPut", "moneyness_bucket"])["delta_emp_bucket"].shift(1)
    df["SpotPrice_lag"] = df.groupby(["CallPut", "moneyness_bucket"])["SpotPrice"].shift(1)

    df = df.dropna(subset=["delta_lag", "SpotPrice_lag"])

    dS     = df["SpotPrice"]        - df["SpotPrice_lag"]
    ddelta = df["delta_emp_bucket"] - df["delta_lag"]

    df = df[dS.abs() > 0].copy()
    dS     = df["SpotPrice"]        - df["SpotPrice_lag"]
    ddelta = df["delta_emp_bucket"] - df["delta_lag"]

    df["gamma_emp_bucket"] = ddelta / dS

    return df[["Date", "CallPut", "moneyness_bucket", "SpotPrice", "gamma_emp_bucket"]]


# ============================================================
# FUNCIÓN DE CÁLCULO DE DELTA Y GAMMA EMPÍRICAS
# ============================================================

def calcular_greeks_empiricas(data):
    data = data.copy()
    delta_1 = delta_empirica_op1(data)
    delta_2 = delta_empirica_op2(delta_1)
    delta_3 = delta_empirica_op3(data)

    gamma_1 = gamma_empirica_op1(delta_1)
    gamma_2 = gamma_empirica_op2(gamma_1)
    gamma_3 = gamma_empirica_op3(delta_2, data)

    data = data.merge(
        delta_1[["Date", "OptionID", "delta_emp"]],
        on=["Date", "OptionID"],
        how="left"
    )

    data = data.merge(
        delta_2[["Date", "CallPut", "moneyness_bucket", "delta_emp_bucket"]].rename(columns={"delta_emp_bucket": "delta_emp_op2"}),
        on=["Date", "CallPut", "moneyness_bucket"],
        how="left"
    )

    data = data.merge(
        delta_3[["Date", "CallPut", "moneyness_bucket", "delta_emp_bucket"]].rename(columns={"delta_emp_bucket": "delta_emp_op3"}),
        on=["Date", "CallPut", "moneyness_bucket"],
        how="left"
    )

    data = data.merge(
        gamma_1[["Date", "OptionID", "gamma_emp"]],
        on=["Date", "OptionID"],
        how="left"
    )

    data = data.merge(
        gamma_2[["Date", "CallPut", "moneyness_bucket", "gamma_emp_bucket"]].rename(columns={"gamma_emp_bucket": "gamma_emp_op2"}),
        on=["Date", "CallPut", "moneyness_bucket"],
        how="left"
    )

    data = data.merge(
        gamma_3[["Date", "CallPut", "moneyness_bucket", "gamma_emp_bucket"]].rename(columns={"gamma_emp_bucket": "gamma_emp_op3"}),
        on=["Date", "CallPut", "moneyness_bucket"],
        how="left"
    )
    print("===============Cálculo de greeks empíricas completado.===============")

    resultados = {
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
        "gamma_1": gamma_1,
        "gamma_2": gamma_2,
        "gamma_3": gamma_3,
    }

    return data, resultados


# ============================================================================
# EJECUCIÓN DE DELTA Y GAMMA EMPÍRICAS: todos los buckets de vencimiento
# ============================================================================

maturity_buckets = opt_df["maturity_bucket"].dropna().unique().sort_values()

resultados_por_bucket = []

for bucket in maturity_buckets[0:]:
    print(f"Calculando greeks empíricas para bucket de vencimiento: {bucket}:")
    
    data_bucket = opt_df[(opt_df["maturity_bucket"] == bucket) ] # & (opt_df["Bid"] > 0)
    opt_df_con_greeks, resultados_greeks = calcular_greeks_empiricas(data_bucket)
    resultados_por_bucket.append(opt_df_con_greeks)

opt_df_con_greeks = pd.concat(resultados_por_bucket, ignore_index=True)

# In[]:

# Guardamos el panel con las greeks empíricas:
opt_df_con_greeks = opt_df_con_greeks.copy()

opt_df_con_greeks["maturity_bucket"] = opt_df_con_greeks["maturity_bucket"].astype(str)
opt_df_con_greeks["moneyness_bucket"] = opt_df_con_greeks["moneyness_bucket"].astype(str)

duckdb.from_df(opt_df_con_greeks).write_parquet(
    str(PATH_OUTPUT),
    compression="snappy"
)

print(f"Fichero guardado correctamente en: {PATH_OUTPUT}")

# %%
