# In[]: Importamos los datos
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

empirical = True
market = "spx" # "crsp"  "spx"
agregacion_mensual = "media"  # "ultimo": pipeline actual; "media": media diaria del mes
callput = "C" 

if agregacion_mensual not in {"ultimo", "media"}:
    raise ValueError("agregacion_mensual debe ser 'ultimo' o 'media'")

if os.name == 'nt':
    if empirical:
        PATH_DATA =  r"Y:\OUTPUTS\WA_mv_greeks_prueba.csv"
    else:
        PATH_DATA =  r"Y:\OUTPUTS\WA_mv_greeks_prueba.csv"

    DIAG_PATH =  r"Y:\Famma-French"
else:
    if empirical:
        PATH_DATA =  r"/Volumes/data/OUTPUTS/Agg_Greeks.csv"
    else:
        PATH_DATA =  r"/Volumes/data/OUTPUTS/Agg_Greeks_BS.csv"

    DIAG_PATH =  r"/Volumes/data/Financial_Data"


print("Cargando datos...")
agg_df = pd.read_csv(PATH_DATA)

agg_df

ff_factors = os.path.join(
    DIAG_PATH, "F-F_Research_Data_Factors.csv"
)

ff_df = pd.read_csv(ff_factors, skiprows=3)
ff_df = ff_df.rename(columns={ff_df.columns[0]: "Date"})

# Conservamos únicamente fechas mensuales con formato YYYYMM.
fecha_ff = ff_df["Date"].astype(str).str.strip()
ff_df = ff_df.loc[fecha_ff.str.fullmatch(r"\d{6}", na=False)].copy()

ff_df["periodo_mes"] = pd.to_datetime(ff_df["Date"].astype(str).str.strip(), format="%Y%m").dt.to_period("M")

assert not ff_df["periodo_mes"].duplicated().any(), \
    "Hay meses duplicados en Fama-French"

spx = pd.read_parquet(os.path.join(r"Y:\OptionMetrics\Acumulado", "security_price.parquet"))

spx = spx.loc[spx["SecurityID"] == 108105].reset_index(drop=True)
spx = spx[[ 'Date', 'Bid', 'Ask', 'OpenPrice',
       'ClosePrice', 'TotalReturn', 'AdjustmentFactor','AdjustmentFactor2']]

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

def serie_mensual_general(df, value_col, out_name=None):
    """
    Construye una serie mensual para `value_col`, tomando para cada mes el
    último dato no nulo disponible.

    No filtra por CallPut, por lo que sirve para cualquier serie contenida
    en un DataFrame con columnas `Date` y `value_col`.
    """
    out_name = out_name or value_col

    sub = df.loc[
        df[value_col].notna(),
        ["Date", value_col]
    ].copy()

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


def media_mensual_aritmetic(df, value_col, callput=None, out_name=None):
    """Media simple de los valores diarios disponibles de una serie en cada mes.

    Filtra por CallPut cuando se indica. Para series globales como net exposure,
    elimina las copias idénticas por fecha para dar el mismo peso a cada día.
    """
    out_name = out_name or value_col
    sub = df if callput is None else df.loc[df["CallPut"] == callput]
    sub = sub.loc[sub[value_col].notna(), ["Date", value_col]].copy()
    sub["Date"] = pd.to_datetime(sub["Date"])
    sub = sub.drop_duplicates()
    if sub["Date"].duplicated().any():
        raise ValueError(f"Hay varios valores de {value_col} para una misma fecha")
    sub["periodo_mes"] = sub["Date"].dt.to_period("M")

    return (
        sub.groupby("periodo_mes", as_index=False)[value_col]
        .mean()
        .rename(columns={value_col: out_name})
    )


def retorno_mensual_compuesto(df, return_col, out_name=None):
    """
    Calcula el retorno mensual compuesto a partir de retornos diarios simples.

    Para cada mes:
        R_mensual = producto(1 + r_diario) - 1

    Los valores ausentes se ignoran. Si un mes no contiene ningún retorno
    válido, su resultado será NaN.
    """
    out_name = out_name or return_col

    sub = df.loc[
        df[return_col].notna(),
        ["Date", return_col]
    ].copy()

    sub["Date"] = pd.to_datetime(sub["Date"])
    sub["periodo_mes"] = sub["Date"].dt.to_period("M")

    mensual = (
        sub
        .groupby("periodo_mes")[return_col]
        .apply(lambda retornos: (1 + retornos).prod() - 1)
        .rename(out_name)
        .reset_index()
    )

    return mensual

# %%

dd = agg_df[["Date", "CallPut", "w_Gamma_gamma_OI","w_gamma_emp_gamma_OI","sum_OpenInterest","w_delta_mv_delta_OI", "w_Delta_delta_OI"]]

dd["position"] = np.where(dd["CallPut"]=="C",1,-1)

total_oi_por_fecha = dd.groupby("Date")["sum_OpenInterest"].transform("sum")


dd["w_Gamma_gamma_OI_pos"] = (dd["w_Gamma_gamma_OI"] * dd["position"]) / total_oi_por_fecha
dd["w_gamma_emp_gamma_OI_pos"] = (dd["w_gamma_emp_gamma_OI"] * dd["position"]) / total_oi_por_fecha

# 5. Volvemos a usar .transform() para sumar los resultados por fecha directamente en la tabla
dd["imbalance_gamma"] = dd.groupby("Date")["w_Gamma_gamma_OI_pos"].transform("sum")
dd["imbalance_gamma_emp"] = dd.groupby("Date")["w_gamma_emp_gamma_OI_pos"].transform("sum")

agg_df["imbalance_gamma"] = dd["imbalance_gamma"]*10000000
agg_df["imbalance_gamma_emp "] = dd["imbalance_gamma_emp"]*10000000 



#%%
# Nombre final -> columna real en agg_df
if empirical:
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
    "w_delta_OI":              "w_delta_mv_delta_OI",
    "w_delta_VD":              "w_delta_mv_delta_VD",
    "imbalance_gamma":         "imbalance_gamma_emp"
}
else:
    variables_mensuales = {
    # niveles ATM
    "ATM_Put_gamma":           "Gamma_ATM_gspread",
    "ATM_Put_delta":           "Delta_ATM_dspread",
    # spreads gamma (ATM - bucket)
    "near_otm_Put_gamma":      "spread_ATM_minus_near_gspread",
    "deep_otm_Put_gamma":      "spread_ATM_minus_deep_gspread",
    "very_deep_otm_Put_gamma": "spread_ATM_minus_very_deep_gspread",
    # spreads delta (ATM - bucket)
    "near_otm_Put_delta":      "spread_ATM_minus_near_dspread",
    "deep_otm_Put_delta":      "spread_ATM_minus_deep_dspread",
    "very_deep_otm_Put_delta": "spread_ATM_minus_very_deep_dspread",
    # medias ponderadas (nivel, no spread)
    "w_gamma_OI":              "w_Gamma_gamma_OI",
    "w_gamma_VD":              "w_Gamma_gamma_VD",
    "w_delta_OI":              "w_Delta_delta_OI",
    "w_delta_VD":              "w_Delta_delta_VD",
    "imbalance_gamma":         "imbalance_gamma"
}

agregador_por_tipo = media_mensual_aritmetic if agregacion_mensual == "media" else serie_mensual
agregador_general = media_mensual_aritmetic if agregacion_mensual == "media" else serie_mensual_general

series_mensuales = [
    agregador_por_tipo(agg_df, col, callput=callput, out_name=nombre)
    for nombre, col in variables_mensuales.items()
    if col in agg_df.columns ]

variables_exposure = ["imbalance_gamma",
    "Gamma_Exposure", "Delta_Exposure", "BS_Gamma_Exposure", "BS_Delta_Exposure",
]
faltan_exposure = [col for col in variables_exposure if col not in agg_df.columns]
if faltan_exposure:
    raise ValueError(
        f"Faltan las columnas de net exposure: {faltan_exposure}. "
        "Ejecuta 05_BSM_spread copy.py para actualizar WA_mv_greeks.csv."
    )

series_mensuales.extend(
    agregador_por_tipo(agg_df, col)
    for col in variables_exposure
)

# Cada serie puede tener meses distintos disponibles (huecos distintos); las unimos
# con outer join por periodo_mes para no perder observaciones de ninguna variable.
panel_mensual = reduce(
    lambda izq, der: izq.merge(der, on="periodo_mes", how="outer", validate="one_to_one"),
    series_mensuales
)

panel_mensual = (
    panel_mensual
    .merge(ff_df.drop(columns=["Date"]), on="periodo_mes", how="outer", validate="one_to_one")
    .sort_values("periodo_mes")
    .reset_index(drop=True)
)

spx_m = retorno_mensual_compuesto(
    spx,
    return_col="TotalReturn",
    out_name="R_SPX_m"
)
panel_mensual = (
    panel_mensual
    .merge(spx_m, on="periodo_mes", how="outer", validate="one_to_one")
    .sort_values("periodo_mes")
    .reset_index(drop=True)
)

# Aseguramos una fila por cada mes del calendario.
calendario = pd.period_range(
    start=panel_mensual["periodo_mes"].min(),
    end=panel_mensual["periodo_mes"].max(),
    freq="M"
)

panel_mensual = (
    panel_mensual
    .set_index("periodo_mes")
    .reindex(calendario)
    .rename_axis("periodo_mes")
    .reset_index()
)
# Reordenamos: periodo_mes primero, luego el resto
# Fama-French viene en porcentaje: lo convertimos a decimal.
panel_mensual["Mkt-RF"] = pd.to_numeric(panel_mensual["Mkt-RF"], errors="raise") / 100

panel_mensual["RF"] = pd.to_numeric(panel_mensual["RF"], errors="raise") / 100

panel_mensual

# In[]: Regresión predictiva IS — Mkt-RF_t = a + b * near_otm_Put_gamma_{t-1} + e_t

def regresion_predictiva(df, dep_var="Mkt-RF", pred_var="near_otm_Put_delta", 
                         lag=0, horizon=1, nw_lags=None):
    df = df.copy().sort_values("periodo_mes").reset_index(drop=True)

    df[dep_var]  = pd.to_numeric(df[dep_var], errors="coerce")
    df[pred_var] = pd.to_numeric(df[pred_var], errors="coerce")

    h = horizon
    dep_col = f"{dep_var}_h{horizon}"
    gross = (1 + df[dep_var])
    # producto móvil de q factores brutos futuros, colocado en t
    idx = pd.api.indexers.FixedForwardWindowIndexer(window_size=h)
    df[dep_col] = (
    gross.shift(-1)
         .rolling(window=idx, min_periods=h)
         .apply(np.prod, raw=True)
            ) - 1 
    
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


def exportar_regresiones_excel(
    registros,
    ruta_excel,
    titulo=None,
):
    """
    Genera un Excel consolidado con:

    - Resultados: coeficientes y errores estándar por horizonte.
    - Detalle: base numérica completa.
    - Notas: ficha metodológica.

    Requiere xlsxwriter:
        pip install xlsxwriter
    """

    if not registros:
        raise ValueError(
            "No hay resultados de regresiones para exportar."
        )

    ruta_excel = Path(ruta_excel)
    ruta_excel.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detalle = pd.DataFrame(registros)

    detalle = detalle.sort_values(
        ["Predictor", "Horizonte", "Lag"]
    ).reset_index(drop=True)

    dep_vars = detalle["Variable dependiente"].unique()

    if len(dep_vars) == 1:
        dep_var = dep_vars[0]
    else:
        dep_var = " / ".join(dep_vars)

    if titulo is None:
        titulo = f"Regresiones predictivas IS · {dep_var}"

    horizontes = sorted(
        detalle["Horizonte"].unique()
    )

    predictores = list(
        detalle["Predictor"].drop_duplicates()
    )

    # Tabla principal: β + estrellas y SE entre paréntesis
    tabla_resultados = pd.DataFrame(
        index=predictores,
        columns=horizontes,
        dtype=object,
    )

    tabla_r2 = pd.DataFrame(
        index=predictores,
        columns=horizontes,
        dtype=float,
    )

    for _, fila in detalle.iterrows():
        predictor = fila["Predictor"]
        horizonte = fila["Horizonte"]

        coef = fila["Coeficiente"]
        se = fila["Error estándar"]
        sig = fila["Significancia"]

        tabla_resultados.loc[predictor, horizonte] = (
            f"{coef:.4f}{sig}\n"
            f"({se:.4f})"
        )

        tabla_r2.loc[predictor, horizonte] = fila["R²"]

    tabla_resultados.index.name = "Predictor"
    tabla_r2.index.name = "Predictor"

    # Modelo con mayor |t|
    idx_max = detalle["t-stat"].abs().idxmax()
    mejor_modelo = detalle.loc[idx_max]

    n_significativas = (
        detalle["Significancia"]
        .fillna("")
        .ne("")
        .sum()
    )

    with pd.ExcelWriter(
        ruta_excel,
        engine="xlsxwriter",
    ) as writer:

        workbook = writer.book

        # Paleta
        navy = "#17324D"
        teal = "#0F766E"
        medium_blue = "#49677F"
        pale_blue = "#EDF3F8"
        pale_teal = "#E8F4F2"
        light_gray = "#F7F9FB"
        border = "#D5DCE3"
        gold = "#D9A441"
        pale_gold = "#FFF8E8"
        dark_gray = "#5B6573"

        # Formatos
        fmt_title = workbook.add_format({
            "bold": True,
            "font_size": 18,
            "font_color": "white",
            "bg_color": navy,
            "valign": "vcenter",
        })

        fmt_subtitle = workbook.add_format({
            "italic": True,
            "font_size": 10,
            "font_color": navy,
            "bg_color": "#DCE7F0",
            "valign": "vcenter",
        })

        fmt_header = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": teal,
            "align": "center",
            "valign": "vcenter",
            "bottom": 1,
            "bottom_color": teal,
        })

        fmt_header_blue = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": medium_blue,
            "align": "center",
            "valign": "vcenter",
        })

        fmt_section = workbook.add_format({
            "bold": True,
            "font_size": 12,
            "font_color": "white",
            "bg_color": navy,
        })

        fmt_predictor = workbook.add_format({
            "bold": True,
            "font_color": navy,
            "bg_color": pale_blue,
            "valign": "vcenter",
        })

        fmt_result = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "bottom": 1,
            "bottom_color": border,
        })

        fmt_result_alt = workbook.add_format({
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
            "bg_color": light_gray,
            "bottom": 1,
            "bottom_color": border,
        })

        fmt_card_label = workbook.add_format({
            "bold": True,
            "font_color": dark_gray,
            "bg_color": light_gray,
            "border": 1,
            "border_color": border,
        })

        fmt_card_value = workbook.add_format({
            "bold": True,
            "font_size": 12,
            "font_color": teal,
            "bg_color": light_gray,
            "border": 1,
            "border_color": border,
            "align": "right",
        })

        fmt_gold_header = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": gold,
        })

        fmt_gold_value = workbook.add_format({
            "font_color": navy,
            "bg_color": pale_gold,
        })

        fmt_note = workbook.add_format({
            "italic": True,
            "font_size": 9,
            "font_color": dark_gray,
            "bg_color": pale_gold,
            "border": 1,
            "border_color": "#E9D7A5",
            "text_wrap": True,
            "valign": "vcenter",
        })

        fmt_detail_header = workbook.add_format({
            "bold": True,
            "font_color": "white",
            "bg_color": navy,
            "align": "center",
            "valign": "vcenter",
            "text_wrap": True,
        })

        fmt_decimal = workbook.add_format({
            "num_format": "0.0000",
        })

        fmt_integer = workbook.add_format({
            "num_format": "#,##0",
        })

        fmt_percent = workbook.add_format({
            "num_format": "0.00%",
            "align": "center",
        })

        # =====================================================
        # Hoja Resultados
        # =====================================================
        sheet_name = "Resultados"
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        worksheet.hide_gridlines(2)

        ncols = len(horizontes) + 1
        last_col = ncols - 1

        worksheet.merge_range(
            0, 0, 0, last_col,
            titulo,
            fmt_title,
        )

        worksheet.set_row(0, 34)

        worksheet.merge_range(
            1, 0, 1, last_col,
            "Predictor en t frente al rendimiento "
            "acumulado futuro · estimación OLS",
            fmt_subtitle,
        )

        worksheet.set_row(1, 23)

        # Tarjetas de resumen
        worksheet.write(3, 0, "Especificaciones", fmt_card_label)
        worksheet.write(3, 1, len(detalle), fmt_card_value)

        worksheet.write(3, 2, "Horizontes", fmt_card_label)
        worksheet.write(3, 3, len(horizontes), fmt_card_value)

        worksheet.write(
            4, 0,
            "Significativas (10%)",
            fmt_card_label,
        )

        worksheet.write(
            4, 1,
            int(n_significativas),
            fmt_card_value,
        )

        worksheet.write(4, 2, "Mayor |t|", fmt_card_label)
        worksheet.write(
            4, 3,
            abs(mejor_modelo["t-stat"]),
            fmt_card_value,
        )

        if last_col >= 4:
            worksheet.merge_range(
                3, 4, 3, last_col,
                "Modelo con mayor |t|",
                fmt_gold_header,
            )

            texto_mejor = (
                f"{mejor_modelo['Predictor']} · "
                f"h={int(mejor_modelo['Horizonte'])} "
                f"(t={mejor_modelo['t-stat']:.3f})"
            )

            worksheet.merge_range(
                4, 4, 4, last_col,
                texto_mejor,
                fmt_gold_value,
            )

        # Tabla de coeficientes
        start_row = 7

        worksheet.write(
            start_row,
            0,
            "Predictor",
            fmt_header,
        )

        for j, horizonte in enumerate(horizontes, start=1):
            worksheet.write(
                start_row,
                j,
                f"Horizonte {horizonte}",
                fmt_header,
            )

        for i, predictor in enumerate(predictores, start=1):
            row = start_row + i

            worksheet.write(
                row,
                0,
                predictor.replace("_", " "),
                fmt_predictor,
            )

            formato_fila = (
                fmt_result_alt
                if i % 2 == 0
                else fmt_result
            )

            for j, horizonte in enumerate(horizontes, start=1):
                valor = tabla_resultados.loc[
                    predictor,
                    horizonte,
                ]

                worksheet.write(
                    row,
                    j,
                    valor if pd.notna(valor) else "—",
                    formato_fila,
                )

            worksheet.set_row(row, 32)

        # Tabla de R²
        r2_title_row = start_row + len(predictores) + 3

        worksheet.merge_range(
            r2_title_row,
            0,
            r2_title_row,
            last_col,
            "Bondad de ajuste (R²)",
            fmt_section,
        )

        r2_header_row = r2_title_row + 1

        worksheet.write(
            r2_header_row,
            0,
            "Predictor",
            fmt_header_blue,
        )

        for j, horizonte in enumerate(horizontes, start=1):
            worksheet.write(
                r2_header_row,
                j,
                f"Horizonte {horizonte}",
                fmt_header_blue,
            )

        for i, predictor in enumerate(predictores, start=1):
            row = r2_header_row + i

            worksheet.write(
                row,
                0,
                predictor.replace("_", " "),
                fmt_predictor,
            )

            for j, horizonte in enumerate(horizontes, start=1):
                valor = tabla_r2.loc[predictor, horizonte]

                if pd.isna(valor):
                    worksheet.write_blank(
                        row,
                        j,
                        None,
                        fmt_percent,
                    )
                else:
                    worksheet.write(
                        row,
                        j,
                        valor,
                        fmt_percent,
                    )

        # Escala de color de R²
        first_r2_row = r2_header_row + 1
        last_r2_row = r2_header_row + len(predictores)

        worksheet.conditional_format(
            first_r2_row,
            1,
            last_r2_row,
            last_col,
            {
                "type": "3_color_scale",
                "min_color": "#FFFFFF",
                "mid_color": pale_teal,
                "max_color": "#59A89C",
            },
        )

        # Nota final
        note_row = last_r2_row + 2

        worksheet.merge_range(
            note_row,
            0,
            note_row + 1,
            last_col,
            "Notas: errores estándar entre paréntesis. "
            "* p<0,10; ** p<0,05; *** p<0,01. "
            "La hoja «Detalle» conserva coeficientes, "
            "errores estándar, estadísticos t, R² y "
            "observaciones como valores numéricos.",
            fmt_note,
        )

        worksheet.set_column(0, 0, 28)
        worksheet.set_column(1, last_col, 16)
        worksheet.freeze_panes(start_row + 1, 1)

        # =====================================================
        # Hoja Detalle
        # =====================================================
        detalle.to_excel(
            writer,
            sheet_name="Detalle",
            index=False,
            startrow=0,
        )

        ws_detalle = writer.sheets["Detalle"]
        ws_detalle.hide_gridlines(2)
        ws_detalle.freeze_panes(1, 0)
        ws_detalle.autofilter(
            0, 0,
            len(detalle),
            len(detalle.columns) - 1,
        )

        for col, nombre in enumerate(detalle.columns):
            ws_detalle.write(
                0,
                col,
                nombre,
                fmt_detail_header,
            )

        ancho_columnas = {
            "Variable dependiente": 20,
            "Predictor": 27,
            "Horizonte": 11,
            "Lag": 8,
            "NW lags": 10,
            "Coeficiente": 14,
            "Error estándar": 15,
            "t-stat": 12,
            "Significancia": 13,
            "R²": 10,
            "Observaciones": 14,
            "Constante": 14,
            "SE constante": 14,
            "t constante": 14,
            "Sig. constante": 14,
        }

        for col, nombre in enumerate(detalle.columns):
            ancho = ancho_columnas.get(nombre, 14)

            if nombre in {
                "Coeficiente",
                "Error estándar",
                "t-stat",
                "R²",
                "Constante",
                "SE constante",
                "t constante",
            }:
                formato = fmt_decimal

            elif nombre in {
                "Horizonte",
                "Lag",
                "NW lags",
                "Observaciones",
            }:
                formato = fmt_integer

            else:
                formato = None

            ws_detalle.set_column(
                col,
                col,
                ancho,
                formato,
            )

        # Resaltar significancia
        sig_col = detalle.columns.get_loc("Significancia")

        ws_detalle.conditional_format(
            1,
            sig_col,
            len(detalle),
            sig_col,
            {
                "type": "text",
                "criteria": "containing",
                "value": "*",
                "format": workbook.add_format({
                    "bold": True,
                    "font_color": teal,
                    "bg_color": pale_teal,
                }),
            },
        )

        # =====================================================
        # Hoja Notas
        # =====================================================
        ws_notas = workbook.add_worksheet("Notas")
        writer.sheets["Notas"] = ws_notas
        ws_notas.hide_gridlines(2)

        ws_notas.merge_range(
            "A1:F1",
            "Ficha metodológica",
            fmt_title,
        )

        ws_notas.write("A3", "Elemento", fmt_header)
        ws_notas.write("B3", "Descripción", fmt_header)

        ficha = [
            (
                "Variable dependiente",
                f"{dep_var} acumulado al horizonte indicado",
            ),
            (
                "Regresor",
                "Variable de opciones contemporánea "
                "o rezagada según la especificación",
            ),
            (
                "Horizontes",
                ", ".join(map(str, horizontes)),
            ),
            (
                "Especificaciones",
                len(detalle),
            ),
            (
                "Errores estándar",
                "Newey-West / HAC",
            ),
            (
                "Convención",
                "Coeficiente con error estándar "
                "entre paréntesis",
            ),
            (
                "Significancia",
                "* p<0,10; ** p<0,05; *** p<0,01",
            ),
        ]

        for row, (elemento, descripcion) in enumerate(
            ficha,
            start=3,
        ):
            ws_notas.write(
                row,
                0,
                elemento,
                fmt_predictor,
            )

            ws_notas.write(
                row,
                1,
                descripcion,
            )

        ws_notas.set_column("A:A", 24)
        ws_notas.set_column("B:B", 65)

    return ruta_excel

def regresion_predictiva(
    df,
    dep_var="Mkt-RF",
    pred_var="near_otm_Put_delta",
    lag=0,
    horizon=1,
    nw_lags=None,
    imprimir=True,
    excel_collector=None,
):
    """
    Ejecuta una regresión predictiva con errores estándar Newey-West.

    Si se proporciona `excel_collector`, añade los resultados a la lista
    para generar posteriormente un Excel consolidado.
    """

    df = df.copy().sort_values("periodo_mes").reset_index(drop=True)

    df[dep_var] = pd.to_numeric(df[dep_var], errors="coerce")
    df[pred_var] = pd.to_numeric(df[pred_var], errors="coerce")

    # Rendimiento acumulado futuro en exceso de RF.
    dep_col = f"{dep_var}_exceso_h{horizon}"

    rf = pd.to_numeric(df["RF"], errors="raise")

    # Reconstruimos el retorno del mercado si recibimos Mkt-RF.
    if dep_var == "Mkt-RF":
        retorno_mercado = df[dep_var] + rf
    else:
        # En la ejecución SPX, dep_var es R_SPX_m.
        retorno_mercado = df[dep_var]

    idx = pd.api.indexers.FixedForwardWindowIndexer(
        window_size=horizon
    )

    acumulado_mercado = (
        (1 + retorno_mercado)
        .shift(-1)
        .rolling(window=idx, min_periods=horizon)
        .apply(np.prod, raw=True)
        - 1
    )

    acumulado_rf = (
        (1 + rf)
        .shift(-1)
        .rolling(window=idx, min_periods=horizon)
        .apply(np.prod, raw=True)
        - 1
    )

    df[dep_col] = acumulado_mercado - acumulado_rf

    # Predictor rezagado
    pred_lag_col = f"{pred_var}_lag{lag}"
    df[pred_lag_col] = df[pred_var].shift(lag)

    df_reg = df.dropna(
        subset=[dep_col, pred_lag_col]
    ).copy()

    if df_reg.empty:
        raise ValueError(
            f"No hay observaciones para {pred_var}, "
            f"h={horizon}, lag={lag}"
        )

    # Comprobamos que la muestra conserve meses consecutivos.
    meses = pd.PeriodIndex(df_reg["periodo_mes"], freq="M")

    if len(meses) > 1 and not np.all(np.diff(meses.asi8) == 1):
        raise ValueError(
            "La muestra tiene meses duplicados o huecos internos. "
            "Revisa las fechas antes de aplicar HAC."
        )

    # Matriz de regresores y variable dependiente.
    X = add_constant(df_reg[[pred_lag_col]])
    y = df_reg[dep_col]

    # Evitamos estimar una pendiente sin variación del predictor.
    if df_reg[pred_lag_col].nunique() < 2:
        raise ValueError("El predictor no tiene variación en esta muestra")

    # Número de rezagos para Newey-West.
    if nw_lags is None:
        nw_lags = horizon

    if len(df_reg) <= X.shape[1]:
        raise ValueError("No hay suficientes observaciones para estimar")

    if nw_lags >= len(df_reg):
        raise ValueError("El número de rezagos HAC debe ser menor que N")

    # OLS con inferencia HAC almacenada en el propio modelo.
    model = OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": nw_lags,
            "kernel": "bartlett",
            "use_correction": True,
        },
        use_t=False,
    )

    se_nw = model.bse
    t_nw = model.tvalues

    # Valores p de la inferencia HAC.
    p_nw = model.pvalues

    def estrellas(p):
        if p < 0.01:
            return "***"
        if p < 0.05:
            return "**"
        if p < 0.10:
            return "*"
        return ""

    significancia = p_nw.apply(estrellas)

    resultados = pd.DataFrame({
        "Variable": model.params.index,
        "Coef": model.params.values,
        "SE": se_nw.values,
        "t-stat": t_nw.values,
        "p-valor": p_nw.values,
        "Sig": significancia.values,
    })

    # Fechas de las filas t utilizadas en la regresión.
    inicio_t = str(df_reg["periodo_mes"].min())
    fin_t = str(df_reg["periodo_mes"].max())

    nombre_dependiente = (
        f"Exceso acumulado "
        f"{'CRSP' if dep_var == 'Mkt-RF' else dep_var} vs RF"
    )

    if imprimir:
        print(
            f"\n=== Regresión IS — "
            f"{nombre_dependiente}, t+1 a t+{horizon} "
            f"~ {pred_var}_(t-{lag}) ==="
        )

        print(
            tabulate(
                resultados,
                headers="keys",
                tablefmt="rounded_outline",
                floatfmt=".4f",
                showindex=False,
            )
        )

        print(
            f"R^2: {model.rsquared:.4f}   "
            f"Obs: {int(model.nobs)}   "
            f"NW lags: {nw_lags}"
        )

        print(
            f"Periodo de origen t: {inicio_t} a {fin_t}"
        )

    # Guardar información para el Excel consolidado.
    if excel_collector is not None:
        excel_collector.append({
            "Variable dependiente": nombre_dependiente,
            "Predictor": pred_var,
            "Horizonte": horizon,
            "Lag": lag,
            "NW lags": nw_lags,

            "Coeficiente": model.params[pred_lag_col],
            "Error estándar": se_nw[pred_lag_col],
            "t-stat": t_nw[pred_lag_col],
            "p-valor": p_nw[pred_lag_col],
            "Significancia": significancia[pred_lag_col],

            "R²": model.rsquared,
            "Observaciones": int(model.nobs),
            "Inicio t": inicio_t,
            "Fin t": fin_t,

            "Constante": model.params["const"],
            "SE constante": se_nw["const"],
            "t constante": t_nw["const"],
            "p constante": p_nw["const"],
            "Sig. constante": significancia["const"],
        })

    return resultados, model

# %% Ejecutamos la regresión:
""" En primer lugar, queremos evaluar si la variable agregada en niveles (no en retornos ni diferencias), predice el mercado
    # Evaluamos para distintos horizontes, empezamos con un mes vista y definciones del mercado; según CRSP.
    # Hacemos la evaluación IS con OLS:
"""

variables = ['w_gamma_OI', 'w_gamma_VD',"w_delta_OI","w_delta_VD"]
variables += (
    ["Gamma_Exposure", "Delta_Exposure", "imbalance_gamma"] if empirical
    else ["BS_Gamma_Exposure", "BS_Delta_Exposure"]
)
horizontes = [1, 2, 3, 12, 24, 36, 48]
lags = [0]
# Lista que acumulará todas las regresiones
resultados_excel = []

for variable in variables:
    for horizonte in horizontes:
        for lag in lags:

            print(f"Horizonte {horizonte}")

            resultados_is, modelo_is = regresion_predictiva(
                panel_mensual,
                dep_var="Mkt-RF" if market == "crsp" else "R_SPX_m" ,
                pred_var=variable,
                lag=lag,
                horizon=horizonte,
                nw_lags=horizonte,
                imprimir=True,
                excel_collector=resultados_excel,
            )

# Generar un único Excel al terminar


if empirical:
    ruta = exportar_regresiones_excel(
    registros=resultados_excel,
    ruta_excel=(
        f"Y:/OUTPUTS/Resultados/"
        f"AAregresiones_predictivas_IS_{market}_{agregacion_mensual}_{callput}.xlsx"
    ),
)
else:
    ruta = exportar_regresiones_excel(
    registros=resultados_excel,
    ruta_excel=(
        f"Y:/OUTPUTS/Resultados/"
        f"AAregresiones_predictivas_IS_{market}_BS_{agregacion_mensual}_{callput}.xlsx"
    ),
)

print(f"\nExcel generado correctamente en:\n{ruta}")



# %%
