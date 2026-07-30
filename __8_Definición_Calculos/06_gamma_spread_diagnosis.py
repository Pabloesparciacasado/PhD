# In[]: Importamos los datos
import pandas as pd
import numpy as np
import sys
import os
from functools import reduce
import re
import duckdb

from tabulate import tabulate
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.stats.sandwich_covariance import cov_hac

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if os.name == 'nt':
    PATH_DATA =  r"Y:\OUTPUTS\Agg_Greeks.csv"
    DIAG_PATH =  r"Y:\Financial_Data"
else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_empirical_greeks.parquet"

print("Cargando datos...")
agg_df = pd.read_parquet(PATH_DATA)

agg_df


ff_factors = os.path.join(DIAG_PATH, "F-F_Research_Data_Factors.csv")

ff_df = pd.read_csv(ff_factors, skiprows=3)
ff_df = ff_df.rename(columns={ff_df.columns[0]: "Date"})
ff_df = ff_df.iloc[0:1199,:]

ff_df["periodo_mes"] = pd.to_datetime(ff_df["Date"].astype(str), format="%Y%m").dt.to_period("M")



# In[]: "Análisis de continuidad de las configuraciones del spread (ATM - near/deep/very_deep)"

def _aux_con_spread_valor(df, tipo, name = "spread_valor"):
    """
    Añade la columna 'spread_valor', coalescendo por fila la columna
    spread_ATM_minus_{bucket}_{tipo} que corresponde según el "bucket" de esa fila.
    """
    col_por_bucket = {
        "very_deep": f"spread_ATM_minus_very_deep_{tipo}",
        "deep":      f"spread_ATM_minus_deep_{tipo}",
        "near":      f"spread_ATM_minus_near_{tipo}",
    }
    df = df.copy()
    df["spread_valor"] = np.nan
    for bucket, col in col_por_bucket.items():
        if col not in df.columns:
            continue
        mask = df["bucket"] == bucket
        df.loc[mask, name] = df.loc[mask, col]
    return df


def analisis_continuidad_spread(df, tipo="gspread", group_cols=("CallPut", "bucket")):
    """
    Evalúa la continuidad temporal de cada configuración del spread (near, deep, very_deep)
    replicando el análisis de rachas de 03_panel_diagnosis.py (analisis_vencimiento), pero
    aplicado a agg_df: en vez de "moneyness_bucket" usamos "bucket" (la categoría de
    comparación frente a ATM), y evaluamos por separado Calls y Puts.

    tipo: "gspread" (gamma) o "dspread" (delta)
    """
    group_cols = list(group_cols)
    df = _aux_con_spread_valor(df, tipo)

    fechas_ordenadas = pd.Series(sorted(df["Date"].unique()))
    total_dias = len(fechas_ordenadas)

    def rachas_stats(grupo):
        dias_con_datos = set(grupo.loc[grupo["spread_valor"].notna(), "Date"])
        serie = fechas_ordenadas.isin(dias_con_datos).astype(int).values

        rachas, racha_actual = [], 0
        for v in serie:
            if v == 1:
                racha_actual += 1
            else:
                if racha_actual > 0:
                    rachas.append(racha_actual)
                racha_actual = 0
        if racha_actual > 0:
            rachas.append(racha_actual)

        return pd.Series({
            "dias_con_datos": len(dias_con_datos),
            "total_dias":     total_dias,
            "pct_cobertura":  len(dias_con_datos) / total_dias * 100 if total_dias else 0.0,
            "n_rachas":       len(rachas),
            "racha_max":      max(rachas) if rachas else 0,
            "racha_min":      min(rachas) if rachas else 0,
            "racha_media":    np.mean(rachas) if rachas else 0.0,
        })

    # groupby hace exactamente lo que antes hacíamos a mano con drop_duplicates + máscaras:
    # separa df en un sub-dataframe por cada combinación de group_cols y le aplica rachas_stats.
    continuidad = (
        df.groupby(group_cols)[["Date", "spread_valor"]]
          .apply(rachas_stats)
          .reset_index()
    )

    print(f"\n=== Continuidad por configuración del spread — {tipo} ===")
    print(tabulate(continuidad, headers="keys", tablefmt="rounded_outline", floatfmt=".1f", showindex=False))

    return continuidad


def grafico_continuidad_spread(df, tipo="gspread", group_cols=("CallPut", "bucket")):
    """
    Visualiza, para cada combinación de group_cols, qué días tienen dato válido de spread.
    Permite detectar visualmente huecos de continuidad (equivalente al gráfico de
    cobertura de panel_diagnosis, pero por configuración de spread en vez de moneyness_bucket).
    """
    group_cols = list(group_cols)
    df = _aux_con_spread_valor(df, tipo)

    validos = df.loc[df["spread_valor"].notna(), group_cols + ["Date"]]
    grupos = validos.groupby(group_cols, sort=True)

    fig, ax = plt.subplots(figsize=(14, 0.5 * grupos.ngroups + 2))

    etiquetas = []
    for i, (clave, sub) in enumerate(grupos):
        clave = clave if isinstance(clave, tuple) else (clave,)
        etiquetas.append(" / ".join(str(v) for v in clave))
        ax.scatter(sub["Date"], [i] * len(sub), s=2, marker="|")

    ax.set_yticks(range(len(etiquetas)))
    ax.set_yticklabels(etiquetas, fontsize=8)
    ax.set_title(f"Cobertura diaria por configuración del spread — {tipo}")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()


# --- Ejecución: continuidad del gamma spread y del delta spread ---
continuidad_gspread = analisis_continuidad_spread(agg_df, tipo="gspread")
continuidad_dspread = analisis_continuidad_spread(agg_df, tipo="dspread")

grafico_continuidad_spread(agg_df, tipo="gspread")
grafico_continuidad_spread(agg_df, tipo="dspread")

# In[]: Frecuencia mensual de todas las variables (weighted average, spreads gamma/delta)

def serie_mensual(df, value_col, callput="P", out_name=None):
    """
    Construye una serie mensual para 'value_col' quedándose, para cada mes, con
    el último día disponible en los datos (el último día de cotización dentro
    del mes, no necesariamente el último día del calendario).

    Cada variable se calcula de forma independiente: no todas tienen datos el
    mismo último día del mes (distinta cobertura/huecos), así que el "último
    día" se determina por separado para cada una antes de unirlas por periodo_mes.
    """
    out_name = out_name or value_col

    sub = df.loc[(df["CallPut"] == callput) & df[value_col].notna(), ["Date", value_col]].copy()
    sub["Date"] = pd.to_datetime(sub["Date"])
    sub = sub.sort_values("Date")
    sub["periodo_mes"] = sub["Date"].dt.to_period("M")

    mensual = (
        sub
        .groupby("periodo_mes", as_index=False)
        .tail(1)
        [["periodo_mes", value_col]]
        .rename(columns={value_col: out_name})
        .reset_index(drop=True)
    )

    return mensual


# Nombre final -> columna real en agg_df
variables_mensuales = {
    # niveles ATM
    "ATM_Put_gamma":           "gamma_emp_ATM_gspread",
    "ATM_Put_delta":           "delta_emp_ATM_dspread",
    # spreads gamma (ATM - bucket)
    "near_otm_Put_gamma":      "spread_ATM_minus_near_gspread",
    "deep_otm_Put_gamma":      "spread_ATM_minus_deep_gspread",
    "very_deep_otm_Put_gamma": "spread_ATM_minus_very_deep_gspread",
    # spreads delta (ATM - bucket)
    "near_otm_Put_delta":      "spread_ATM_minus_near_dspread",
    "deep_otm_Put_delta":      "spread_ATM_minus_deep_dspread",
    "very_deep_otm_Put_delta": "spread_ATM_minus_very_deep_dspread",
    # medias ponderadas (nivel, no spread)
    "w_gamma_OI":              "w_gamma_emp_gamma_OI",
    "w_gamma_VD":              "w_gamma_emp_gamma_VD",
    "w_delta_OI":              "w_delta_emp_delta_OI",
    "w_delta_VD":              "w_delta_emp_delta_VD",
}


series_mensuales = [
    serie_mensual(agg_df, col, callput="P", out_name=nombre)
    for nombre, col in variables_mensuales.items()
    if col in agg_df.columns
]

# Cada serie puede tener meses distintos disponibles (huecos distintos); las unimos
# con outer join por periodo_mes para no perder observaciones de ninguna variable.
panel_mensual = reduce(
    lambda izq, der: izq.merge(der, on="periodo_mes", how="outer"),
    series_mensuales
)

panel_mensual = (
    panel_mensual
    .merge(ff_df.drop(columns=["Date"]), on="periodo_mes", how="inner")
    .sort_values("periodo_mes")
    .reset_index(drop=True)
)

# Reordenamos: periodo_mes primero, luego el resto
primero = ["periodo_mes"]
panel_mensual = panel_mensual[primero + [c for c in panel_mensual.columns if c not in primero]]
panel_mensual


# In[]: Regresión predictiva IS — Mkt-RF_t = a + b * near_otm_Put_gamma_{t-1} + e_t

def regresion_predictiva(df, dep_var="Mkt-RF", pred_var="near_otm_Put_delta", 
                         lag=0, horizon=1, nw_lags=None):
    df = df.copy().sort_values("periodo_mes").reset_index(drop=True)

    df[dep_var]  = pd.to_numeric(df[dep_var], errors="coerce")
    df[pred_var] = pd.to_numeric(df[pred_var], errors="coerce")

    # Retorno acumulado sobre los siguientes q=horizon meses.
    # rolling(window=horizon).sum() suma los q meses previos.
    # shift(-horizon) desplaza esa suma para que aparezca en la fecha t
    # y represente la suma de r_{t+1}, ..., r_{t+horizon}.
    dep_col = f"{dep_var}_h{horizon}"
    df[dep_col] = df[dep_var].shift(-1).rolling(window=horizon).sum().shift(-(horizon-1))
    
    pred_lag_col = f"{pred_var}_lag{lag}"
    df[pred_lag_col] = df[pred_var].shift(lag)

    df = df.dropna(subset=[dep_col, pred_lag_col])

    X = add_constant(df[[pred_lag_col]])
    y = df[dep_col]

    # Newey-West: por convención maxlags = horizon (o horizon-1)
    if nw_lags is None:
        nw_lags = horizon
    
    model  = OLS(y, X).fit()
    cov_nw = cov_hac(model, nlags=nw_lags)
    se_nw  = np.sqrt(np.diag(cov_nw))
    t_nw   = model.params / se_nw

    resultados = pd.DataFrame({
        "Variable": X.columns,
        "Coef":     model.params.values,
        "SE":  se_nw,
        "t-stat":   t_nw,
        "Sig":      ["***" if abs(t) > 2.576 else
                     "**"  if abs(t) > 1.960 else
                     "*"   if abs(t) > 1.645 else ""
                     for t in t_nw]
    })

    print(f"\n=== Regresión IS — {dep_var}_t ~ {pred_var}_(t-{lag}) ===")
    print(tabulate(resultados, headers="keys", tablefmt="rounded_outline", floatfmt=".4f", showindex=False))
    print(f"R^2: {model.rsquared:.4f}   Obs: {int(model.nobs)}")

    return resultados, model


# %% Ejecutamos la regresión:
variables= list(variables_mensuales.keys())
""" En primer lugar, queremos evaluar si la variable agregada en niveles (no en retornos ni diferencias), predice el mercado
    # Evaluamos para distintos horizontes, empezamos con un mes vista y definciones del mercado; según CRSP y SP500.
    # Hacemos la evaluación IS con OLS:
"""

for n in variables:
    for h in [1, 3, 12, 24, 36]:      # horizontes
        for l in [0]:            # rezagos del predictor
            resultados_is, modelo_is = regresion_predictiva(
                panel_mensual, dep_var="Mkt-RF", pred_var=n, 
                lag=l, horizon=h, nw_lags=h)
