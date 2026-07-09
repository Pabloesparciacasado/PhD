# In[]: Importamos los datos
import pandas as pd
import numpy as np
import sys
import os
from functools import reduce
import re
import duckdb


from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if os.name == 'nt':
    PATH_DATA = r"Y:\OUTPUTS\opt_df_empirical_greeks.parquet"
    OUT_PATH =  r"Y:\OUTPUTS\Agg_Greeks.csv"
else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_empirical_greeks.parquet"

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

# #Añadimos algunas variables de interés:
# opt_df["Dummy_Bid"] = opt_df["Bid"] > 0
opt_df["DolarVolume"] = opt_df["Volume"] * opt_df["MidPrice"]



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


# Recuperamos el formato de intervalos
 
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

# In[]: Limpieza de datos

opt_df = opt_df.drop(columns=["delta_emp_op2","delta_emp_op3","gamma_emp_op2", "gamma_emp_op3","Moneyness_Forward","log_moneyness_Forward"])

mask = opt_df["maturity_bucket"] == pd.Interval(15, 45.0, closed="right")
opt_df_filtrado = opt_df[mask]

print(opt_df_filtrado.isnull().sum())
opt_df_filtrado.shape


opt_df_filtrado = opt_df_filtrado.dropna()
opt_df_filtrado.shape
# In[]: Funciones:

def WA_diaria(df,variable, greek_emp, greek_teo):
    resultados = []

    for (dt,cp), group in df.groupby(["Date","CallPut"]):
        grupo_valid = group[group[greek_emp].notna() & group[greek_teo].notna()].copy()
        if grupo_valid.empty:
            continue

        oi = grupo_valid[variable]
        if oi.sum() == 0:
            continue
        
        # grupo_valid[greek_teo] = grupo_valid[greek_teo]*grupo_valid["SpotPrice"]**2 if "Gamma" == greek_teo else grupo_valid[greek_teo]
        # grupo_valid[greek_emp] = grupo_valid[greek_emp]*grupo_valid["SpotPrice"]**2 if "Gamma" == greek_teo else grupo_valid[greek_emp]

        resultados.append({
            "Date":       dt,
            "CallPut":    cp,
            f"w_{greek_emp}":    (oi * grupo_valid[greek_emp]).sum() / oi.sum(),
            # greek_teo:    (oi * grupo_valid[greek_teo]).sum() / oi.sum(),
            f"mean_{variable}":     oi.mean(),
            "n_contratos": len(grupo_valid)
        })

    df_out = pd.DataFrame(resultados)
    return df_out


def gamma_spread_left(df, variable, greek_emp, bucket_col="Moneyness"):
    """
    Calculamos la diferencia diaria agrupando por tipo de opcion.
    Primero agrupamos con una media ponderada dentro de cada grupo de rango de moneyness:
        ATM --> (0.9-1.1]
        OTM_PUT --> [0.7-0.9)
        deep_OTM_PUT --> [0.5-0.7)
        very_deep_OTM_PUT -->[0.0-0.5)
        IMT_CALL --> [0.7-0.9)
        deep_IMT_CAL --> [0.5-0.7)
        very_deep_IMT_CAL -->[0.0-0.5)

    Calcularemos la diferencia para el valor obtenido en el rango ATM, con respecto cada una de las regiones (separando por call y put)
    Para definir ambas zonas de dinero, puedo hacer media ponderada por bucket.

    """
        
    cols_needed = ["Date", "CallPut", bucket_col, variable, greek_emp]
    df_valid = df.dropna(subset=cols_needed).copy()
    df_valid = df_valid[df_valid[variable] > 0]

    bins = [0.0, 0.5, 0.7, 0.9, 1.1]
    labels = ["very_deep", "deep", "near", "ATM"]

    df_valid["range"] = pd.cut(
        df_valid[bucket_col],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )

    # df_valid = df_valid.dropna(subset=["range"])

    df_valid["weighted_greek"] = df_valid[variable] * df_valid[greek_emp]

    df_bucket = (
        df_valid
        .groupby(["Date", "CallPut", "range"], observed=True, as_index=False)
        .agg(
            weighted_greek=("weighted_greek", "sum"),
            weight_sum=(variable, "sum"),
            weight_mean=(variable, "mean"),
            n_contratos=(greek_emp, "size")
        )
    )

    df_bucket[greek_emp] = df_bucket["weighted_greek"] / df_bucket["weight_sum"]

    resultados_spread = []

    for (dt, cp), group in df_bucket.groupby(["Date", "CallPut"]):
        atm = group[group["range"] == "ATM"]

        if atm.empty:
            continue

        greek_atm = atm[greek_emp].iloc[0]

        for bucket in ["very_deep", "deep", "near"]:
            row = group[group["range"] == bucket]

            if row.empty:
                continue

            greek_bucket = row[greek_emp].iloc[0]

            resultados_spread.append({
                "Date": dt,
                "CallPut": cp,
                "bucket": bucket,
                f"{greek_emp}_ATM": greek_atm,
                f"{greek_emp}_{bucket}": greek_bucket,
                f"spread_ATM_minus_{bucket}": greek_atm - greek_bucket,
                "n_contratos_ATM": atm["n_contratos"].iloc[0],
                f"n_contratos_{bucket}": row["n_contratos"].iloc[0],
                f"{variable}_ATM": atm["weight_mean"].iloc[0],
                f"{variable}_{bucket}": row["weight_mean"].iloc[0],
            })

    return pd.DataFrame(resultados_spread)



# %% Ejecución:
opt_df_filtrado[opt_df_filtrado["OpenInterest"] > 0]

g_spread_left = gamma_spread_left(opt_df_filtrado, "OpenInterest","gamma_emp" )
d_spread_left = gamma_spread_left(opt_df_filtrado, "OpenInterest","delta_emp" )

serie_gamma_OI = WA_diaria(opt_df_filtrado,"OpenInterest", "gamma_emp", "Gamma")
serie_gamma_VD = WA_diaria(opt_df_filtrado,"DolarVolume", "gamma_emp", "Gamma")

serie_delta_OI = WA_diaria(opt_df_filtrado,"OpenInterest", "delta_emp", "Delta")
serie_delta_VD = WA_diaria(opt_df_filtrado,"DolarVolume", "delta_emp", "Delta")

# %% Unimos resultados:

keys = ["Date", "CallPut", "bucket"]

dfs = [
    g_spread_left.add_suffix("_gspread").rename(columns={
        "Date_gspread": "Date",
        "CallPut_gspread": "CallPut",
        "bucket_gspread": "bucket"
    }),

    d_spread_left.add_suffix("_dspread").rename(columns={
        "Date_dspread": "Date",
        "CallPut_dspread": "CallPut",
        "bucket_dspread": "bucket"

    })
]

final_df = reduce(
    lambda x, y: pd.merge(x, y, on=keys, how="outer"),
    dfs
)

dfs = [
    final_df,

    serie_gamma_OI.add_suffix("_gamma_OI").rename(columns={
        "Date_gamma_OI": "Date",
        "CallPut_gamma_OI": "CallPut",
    }),

    serie_gamma_VD.add_suffix("_gamma_VD").rename(columns={
        "Date_gamma_VD": "Date",
        "CallPut_gamma_VD": "CallPut",
    }),

    serie_delta_OI.add_suffix("_delta_OI").rename(columns={
        "Date_delta_OI": "Date",
        "CallPut_delta_OI": "CallPut",
    }),

    serie_delta_VD.add_suffix("_delta_VD").rename(columns={
        "Date_delta_VD": "Date",
        "CallPut_delta_VD": "CallPut",
    })
]

keys = ["Date", "CallPut"]

final_df = reduce(
    lambda x, y: pd.merge(x, y, on=keys, how="outer"),
    dfs
)

# %% Ordenamos y guardamos
# Columnas redundantes que eliminamos

rem_n_contratos = [
    "n_contratos_very_deep_dspread",
    "n_contratos_deep_dspread",
    "n_contratos_near_dspread",
    "n_contratos_ATM_dspread",
    "n_contratos_delta_OI",
    "n_contratos_delta_VD",
    "n_contratos_gamma_VD",
]

rem_dspread_values = [
    "delta_emp_near_dspread",
    "delta_emp_deep_dspread",
    "delta_emp_very_deep_dspread",
]

rem_gspread_values = [
    "gamma_emp_near_gspread",
    "gamma_emp_deep_gspread",
    "gamma_emp_very_deep_gspread",
]

rem_open_interest_dspread = [
    "OpenInterest_ATM_dspread",
    "OpenInterest_near_dspread",
    "OpenInterest_deep_dspread",
    "OpenInterest_very_deep_dspread",
]

rem_means = [
    "mean_DolarVolume_delta_VD",
    "mean_OpenInterest_delta_OI",
]

rem_col = (
    rem_n_contratos
    + rem_dspread_values
    + rem_gspread_values
    + rem_open_interest_dspread
    + rem_means
)

rename_cols = {
    "n_contratos_gamma_OI": "n_contratos_OI",
    "mean_DolarVolume_gamma_VD": "mean_DolarVolume",
    "mean_OpenInterest_gamma_OI": "mean_OpenInterest",

    "n_contratos_ATM_gspread": "n_contratos_ATM",
    "n_contratos_near_gspread": "n_contratos_near",
    "n_contratos_deep_gspread": "n_contratos_deep",
    "n_contratos_very_deep_gspread": "n_contratos_very_deep",

    "OpenInterest_ATM_gspread": "OpenInterest_ATM",
    "OpenInterest_near_gspread": "OpenInterest_near",
    "OpenInterest_deep_gspread": "OpenInterest_deep",
    "OpenInterest_very_deep_gspread": "OpenInterest_very_deep",
}

final_df = (
    final_df
    .drop(columns=rem_col, errors="ignore")
    .rename(columns=rename_cols)
)

# Ordenamos columnas por bloques logicos

id_cols = [
    "Date",
    "CallPut",
    "bucket",
]

weighted_cols = [
    "w_gamma_emp_gamma_OI",
    "w_gamma_emp_gamma_VD",
    "w_delta_emp_delta_OI",
    "w_delta_emp_delta_VD",
]

spread_gamma_cols = [
    "gamma_emp_ATM_gspread",
    "spread_ATM_minus_near_gspread",
    "spread_ATM_minus_deep_gspread",
    "spread_ATM_minus_very_deep_gspread",
]

spread_delta_cols = [
    "delta_emp_ATM_dspread",
    "spread_ATM_minus_near_dspread",
    "spread_ATM_minus_deep_dspread",
    "spread_ATM_minus_very_deep_dspread",
]

n_contratos_cols = [
    "n_contratos_ATM",
    "n_contratos_near",
    "n_contratos_deep",
    "n_contratos_very_deep",
    "n_contratos_OI",
]

peso_cols = [
    "mean_OpenInterest",
    "mean_DolarVolume",
    "OpenInterest_ATM",
    "OpenInterest_near",
    "OpenInterest_deep",
    "OpenInterest_very_deep",
]

orden_columnas = (
    id_cols
    + weighted_cols
    + spread_gamma_cols
    + spread_delta_cols
    + n_contratos_cols
    + peso_cols
)

# Por seguridad: solo usamos columnas que existan
orden_columnas = [col for col in orden_columnas if col in final_df.columns]

# Cualquier columna no contemplada se manda al final
otras_cols = [col for col in final_df.columns if col not in orden_columnas]

final_df = final_df[orden_columnas + otras_cols]
final_df






duckdb.from_df(final_df).write_parquet(
    str(OUT_PATH),
    compression="snappy"
)

print(f"Fichero guardado correctamente en: {OUT_PATH}")

# %%
