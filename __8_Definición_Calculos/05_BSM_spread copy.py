# %% Importaciones y rutas
import os
from pathlib import Path
from functools import reduce
import pandas as pd
import numpy as np

BASE = Path(r"Y:\OUTPUTS") if os.name == "nt" else Path("/Volumes/data/OUTPUTS")
PATH_RESULTS = BASE / "REPLICA_HW" / "DeltaGamma_results_reg.parquet"
OUT_PATH = BASE / "WA_mv_greeks_prueba.csv"

VOLUME_DATA = r"Y:\Maro-Variables\SPY_diario.csv"
# %% Medias ponderadas diarias por Date y CallPut

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
            f"w_{greek_emp}":    (oi * grupo_valid[greek_emp]).sum() /oi.sum(),
            f"w_{greek_teo}":    (oi * grupo_valid[greek_teo]).sum() / oi.sum(),
            f"sum_{variable}":     oi.sum(),
            "n_contratos": len(grupo_valid)
        })

    df_out = pd.DataFrame(resultados)
    return df_out

# %% Volumen externo de SPY: una fila por sesión.
def cargar_adv_spy(path, rolling_days=21):
    """Close de SPY x Volume de SPY; promedio de sesiones previas.

    Usar Close sin ajuste por dividendos. El CSV debe incluir todas las
    sesiones; una fila ausente no se detecta a partir del CSV por sí solo.
    """
    spy = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    spy.columns = spy.columns.str.strip()
    requeridas = {"Date", "Close", "Volume"}
    if not requeridas.issubset(spy.columns):
        raise ValueError(f"El CSV debe incluir {sorted(requeridas)}")
    fechas = spy["Date"].astype(str).str.strip()
    if fechas.str.fullmatch(r"\d{2}/\d{2}/\d{4}").all():
        spy["Date"] = pd.to_datetime(fechas, format="%d/%m/%Y")
    else:
        spy["Date"] = pd.to_datetime(fechas, errors="raise")
    spy["Date"] = spy["Date"].dt.normalize()
    if spy["Date"].isna().any() or spy["Date"].duplicated().any():
        raise ValueError("Hay fechas ausentes o duplicadas en SPY")
    for col in ["Close", "Volume"]:
        spy[col] = pd.to_numeric(spy[col], errors="raise")
    if not np.isfinite(spy[["Close", "Volume"]].to_numpy()).all():
        raise ValueError("SPY contiene precios o volúmenes ausentes/no finitos")
    if spy["Close"].le(0).any() or spy["Volume"].lt(0).any():
        raise ValueError("SPY contiene precios no positivos o volumen negativo")
    spy = spy.set_index("Date").sort_index()
    spy["DollarVolume_SPY"] = spy["Close"] * spy["Volume"]
    # Calcular ANTES de cruzar con las opciones: conserva la historia previa.
    return (
        spy["DollarVolume_SPY"]
        .rolling(rolling_days, min_periods=rolling_days)
        .mean().shift(1).rename("ADV_SPY_previo")
    )


# %% Exposición SPX normalizada por liquidez SPY (proxy).
def greek_net_exposure(
    data, greek, rolling_days=21, empirical=True, adv_previo=None,
):
    data = data.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise").dt.normalize()
    if data["Date"].isna().any():
        raise ValueError("Hay fechas ausentes en las opciones")
    columnas = {"gamma": ("gamma_emp", "Gamma"), "delta": ("delta_mv", "Delta")}
    if greek not in columnas:
        raise ValueError("greek debe ser gamma o delta")
    emp_col, bs_col = columnas[greek]
    greek_col = emp_col if empirical else bs_col
    nombre = ("" if empirical else "BS_") + greek.capitalize() + "_Exposure"
    if not data["CallPut"].isin(["C", "P"]).all():
        raise ValueError("CallPut debe contener C/P")
    if "SecurityID" in data and data["SecurityID"].nunique(dropna=False) != 1:
        raise ValueError("Filtra un único subyacente SPX")
    for col in [emp_col, bs_col, "OpenInterest", "SpotPrice"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if np.isinf(data[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinitos en {col}")
    if data["OpenInterest"].dropna().lt(0).any():
        raise ValueError("OpenInterest no puede ser negativo")
    if data["SpotPrice"].isna().any() or data["SpotPrice"].le(0).any():
        raise ValueError("SpotPrice debe ser válido y positivo")
    if data.groupby("Date")["SpotPrice"].nunique().gt(1).any():
        raise ValueError("Hay varios SpotPrice por fecha")
    spot = data.groupby("Date")["SpotPrice"].first().sort_index()
    if adv_previo is None:
        adv_previo = cargar_adv_spy(VOLUME_DATA, rolling_days)
    if not adv_previo.index.is_unique:
        raise ValueError("Fechas duplicadas en ADV")
    faltantes = spot.index.difference(adv_previo.index)
    if len(faltantes):
        raise ValueError(f"Faltan sesiones en SPY: {faltantes[:5].tolist()}")
    # Comparación empírica/BSM sobre los mismos contratos por tipo de griega.
    valid = data.dropna(subset=[emp_col, bs_col, "OpenInterest"]).copy()
    signo = valid["CallPut"].map({"C": 1, "P": -1})
    valid["_exposure"] = signo * valid[greek_col] * valid["OpenInterest"]
    exposure = valid.groupby("Date")["_exposure"].sum(min_count=1).reindex(spot.index)
    adv = adv_previo.reindex(spot.index)
    # Multiplicador 100 y movimiento del 1% se cancelan para gamma.
    # Delta: sustitución literal de Gamma por Delta en la misma ecuación;
    # no es la exposición delta monetaria convencional.
    resultado = exposure * spot**2 / adv.where(adv > 0)
    return resultado.rename(nombre).rename_axis("Date").reset_index()

# %% Carga y cálculo de las medias ponderadas
opt_df = pd.read_parquet(PATH_RESULTS)
opt_df["Date"] = pd.to_datetime(opt_df["Date"])
opt_df["DolarVolume"] = opt_df["Volume"] * opt_df["MidPrice"]
keys = ["Date", "CallPut"]
dfs = []
for greek, emp, bs in [("gamma", "gamma_emp", "Gamma"), ("delta", "delta_mv", "Delta")]:
    for peso, sufijo in [("OpenInterest", "OI"), ("DolarVolume", "VD")]:
        serie = WA_diaria(opt_df, peso, emp, bs)
        nombres = {c: f"{c}_{greek}_{sufijo}" for c in serie.columns if c not in keys}
        dfs.append(serie.rename(columns=nombres))
final_df = reduce(lambda x, y: x.merge(y, on=keys, how="outer", validate="one_to_one"), dfs)

# %% Cuatro exposiciones diarias, alineadas por fecha e integradas en final_df
adv_spy = cargar_adv_spy(VOLUME_DATA, rolling_days=21)
series_exposure = [
    greek_net_exposure(opt_df, greek, rolling_days=21, empirical=empirical, adv_previo=adv_spy)
    for greek in ["gamma", "delta"] for empirical in [True, False]
]
net_exposure = reduce(lambda x, y: x.merge(y, on="Date", how="outer", validate="one_to_one"), series_exposure)
final_df = final_df.merge(net_exposure, on="Date", how="outer", validate="many_to_one")
final_df = final_df.drop(columns=[
    "n_contratos_delta_OI", "n_contratos_delta_VD", "n_contratos_gamma_VD", "sum_OpenInterest_gamma_OI","sum_DolarVolume_gamma_VD"
]).rename(columns={
    "n_contratos_gamma_OI": "n_contratos_OI",
    "sum_OpenInterest_delta_OI": "sum_OpenInterest",
    "sum_DolarVolume_delta_VD": "sum_DolarVolume",
})
final_df = final_df.sort_values(keys).reset_index(drop=True)
final_df.to_csv(OUT_PATH, index=False, encoding="utf-8")
print(f"Fichero guardado correctamente en: {OUT_PATH}")