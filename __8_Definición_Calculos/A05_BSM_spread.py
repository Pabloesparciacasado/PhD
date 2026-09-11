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
    PATH_RESULTS =  r"Y:\OUTPUTS\REPLICA_HW\DeltaGamma_results_reg.parquet"

    OUT_PATH =  r"Y:\OUTPUTS\WA_mv_greeks.csv"
else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_empirical_greeks.parquet"

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_RESULTS)

# #Añadimos algunas variables de interés:
# Ya traemos los dtaos filtrados según HW(2017)
opt_df["DolarVolume"] = opt_df["Volume"] * opt_df["MidPrice"]


opt_df

# In[]: Funciones:

# Dos posibilidades de cálculo:

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
            f"w_{greek_teo}":    (oi * grupo_valid[greek_teo]).sum() / oi.sum(),
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

## DEJAMOS ABIERTO:

# 1) PCA para obtener la pendiente
# 2) LPKR o solo Kernel ponderando por OI.
# 3) WLS 
# redefinición de las categorías de moneyness. 

# %% Ejecución:

serie_gamma_OI = WA_diaria(opt_df,"OpenInterest", "gamma_emp", "Gamma")
serie_gamma_VD = WA_diaria(opt_df,"DolarVolume", "gamma_emp", "Gamma")

serie_delta_OI = WA_diaria(opt_df,"OpenInterest", "delta_mv", "Delta")
serie_delta_VD = WA_diaria(opt_df,"DolarVolume", "delta_mv", "Delta")

# %% Unimos resultados:



dfs = [
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
    "w_delta_mv_delta_OI",
    "w_delta_mv_delta_VD",
    "w_Gamma_gamma_OI",
    "w_Gamma_gamma_VD",
    "w_Delta_delta_OI",
    "w_Delta_delta_VD",
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

final_df.to_csv(OUT_PATH, index=False, encoding='utf-8')

print(f"Fichero guardado correctamente en: {OUT_PATH}")

# %%
