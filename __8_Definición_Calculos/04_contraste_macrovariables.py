# In[]
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
import matplotlib.dates as mdates

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


if os.name == 'nt':
    PATH_DATA_OPT = r"Y:\OUTPUTS\opt_df_empirical_greeks.parquet"
else:
    PATH_DATA_OPT = r"/Volumes/data/OptionMetrics/OUTPUTS/opt_df_empirical_greeks.parquet"

print("Cargando datos...")
opt_df_greek = pd.read_parquet(PATH_DATA_OPT)


print(opt_df_greek.info())
print(opt_df_greek.describe())
print(opt_df_greek.isnull().sum())

#Nos quedamos con los datos del bucket de vencimiento 15-45 días y delta no NaN:

opt_df_greek_filt = opt_df_greek[(opt_df_greek["maturity_bucket"] == '(0.0, 15.0]') & (opt_df_greek["delta_emp"].notna()) & (opt_df_greek["gamma_emp"].notna())]

opt_df_greek_filt



#####################################################################################
# Análisis 1: Descripción de las variables empíricas vs teóricas por tipo de opción
#####################################################################################

opt_df_greek_filt.groupby("CallPut")[["delta_emp", "gamma_emp", "Delta", "Gamma"]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])


desc = opt_df_greek_filt.groupby("CallPut")[[
    "delta_emp", "gamma_emp", "Delta", "Gamma"
]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

desc = desc.stack(level=0).reset_index()
desc = desc.rename(columns={"level_1": "Greek"})
desc

# Filtrado al percentil 95% y 5%:

opt_df_greek_filt_95 = opt_df_greek_filt[
    (opt_df_greek_filt["delta_emp"] <= opt_df_greek_filt["delta_emp"].quantile(0.95)) &
    (opt_df_greek_filt["delta_emp"] >= opt_df_greek_filt["delta_emp"].quantile(0.05)) &
    (opt_df_greek_filt["gamma_emp"] <= opt_df_greek_filt["gamma_emp"].quantile(0.95)) &
    (opt_df_greek_filt["gamma_emp"] >= opt_df_greek_filt["gamma_emp"].quantile(0.05))
]


opt_df_greek_filt_95
# %% 
##################################################################################### 
# Análisis 1 Parte 2: Descriptivos con dataframe filtrado al percentil 95% y 5%
#####################################################################################

desc = opt_df_greek_filt_95.groupby("CallPut")[[
    "delta_emp", "gamma_emp", "Delta", "Gamma"
]].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])

desc = desc.stack(level=0).reset_index()
desc = desc.rename(columns={"level_1": "Greek"})
desc



# %%
#####################################################################################
# Análisis 2: GRÁFICOS DE SERIES TEMPORAL MENSUALES.
#####################################################################################

#####################################################################################
# Análisis 2a: Serie temporal del contrato más cercano a la mediana de moneyness
# Comparativa delta/gamma empírica vs teórica (BS) --- DIARIOS ---
#####################################################################################

def contrato_mediana_moneyness(df, greek_emp, greek_teo):
    """
    Para cada (Date, CallPut):
    1. Calcula la mediana del moneyness ese día
    2. Selecciona el contrato cuyo moneyness esté más próximo a esa mediana
    3. Devuelve el valor de la greek empírica y teórica de ese contrato
    """
    resultados = []

    for (date, cp), grupo in df.groupby(["Date", "CallPut"]):
        grupo_valid = grupo[grupo[greek_emp].notna()].copy()
        if grupo_valid.empty:
            continue

        mediana_m = grupo_valid["Moneyness"].median()
        idx_closest = (grupo_valid["Moneyness"] - mediana_m).abs().idxmin()
        fila = grupo_valid.loc[idx_closest]

        resultados.append({
            "Date":          date,
            "CallPut":       cp,
            "Moneyness":     fila["Moneyness"],
            "mediana_m":     mediana_m,
            greek_emp:       fila[greek_emp],
            greek_teo:       fila[greek_teo],
        })

    return pd.DataFrame(resultados)


serie_delta = contrato_mediana_moneyness(opt_df_greek_filt_95, "delta_emp", "Delta")
serie_gamma = contrato_mediana_moneyness(opt_df_greek_filt_95, "gamma_emp", "Gamma")


# ---- Gráfico Delta ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for cp, ax in zip(["C", "P"], axes):
    data_cp = serie_delta[serie_delta["CallPut"] == cp].sort_values("Date")
    ax.plot(data_cp["Date"], data_cp["delta_emp"], color="steelblue", linewidth=0.8, label="Delta empírica")
    ax.plot(data_cp["Date"], data_cp["Delta"],     color="firebrick",  linewidth=0.8, label="Delta BS", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Delta — {'Call' if cp == 'C' else 'Put'} (contrato más próximo a mediana de moneyness)")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal Delta empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()

# ---- Gráfico Gamma ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for cp, ax in zip(["C", "P"], axes):
    data_cp = serie_gamma[serie_gamma["CallPut"] == cp].sort_values("Date")
    ax.plot(data_cp["Date"], data_cp["gamma_emp"], color="darkorange", linewidth=0.8, label="Gamma empírica")
    ax.plot(data_cp["Date"], data_cp["Gamma"],     color="firebrick",  linewidth=0.8, label="Gamma BS", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Gamma — {'Call' if cp == 'C' else 'Put'} (contrato más próximo a mediana de moneyness)")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal Gamma empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()
# %%
#####################################################################################
# Análisis 2b: Serie temporal MENSUAL — media de delta y gamma sobre todo el moneyness
# ponderada por OI, comparativa empírica vs teórica (BS)
#####################################################################################
import matplotlib.dates as mdates
# Añadimos columna año-mes
opt_df_greek_filt_95["YearMonth"] = opt_df_greek_filt_95["Date"].dt.to_period("M")

# Media mensual ponderada por OI para cada (YearMonth, CallPut)
def media_mensual_OI(df, greek_emp, greek_teo):
    resultados = []

    for (ym, cp), grupo in df.groupby(["YearMonth", "CallPut"]):
        grupo_valid = grupo[grupo[greek_emp].notna() & grupo[greek_teo].notna()].copy()
        if grupo_valid.empty:
            continue

        oi = grupo_valid["OpenInterest"]
        if oi.sum() == 0:
            continue
        
        # grupo_valid[greek_teo] = grupo_valid[greek_teo]*grupo_valid["SpotPrice"]**2 if "Gamma" == greek_teo else grupo_valid[greek_teo]
        # grupo_valid[greek_emp] = grupo_valid[greek_emp]*grupo_valid["SpotPrice"]**2 if "Gamma" == greek_teo else grupo_valid[greek_emp]

        resultados.append({
            "YearMonth":  ym,
            "CallPut":    cp,
            greek_emp:    (oi * grupo_valid[greek_emp]).sum() / oi.sum(),
            greek_teo:    (oi * grupo_valid[greek_teo]).sum() / oi.sum(),
            "n_contratos": len(grupo_valid)
        })

    df_out = pd.DataFrame(resultados)
    df_out["Date"] = df_out["YearMonth"].dt.to_timestamp()
    return df_out

# %%
serie_delta_m = media_mensual_OI(opt_df_greek_filt_95, "delta_emp", "Delta")
serie_gamma_m = media_mensual_OI(opt_df_greek_filt_95, "gamma_emp", "Gamma")


# ---- Gráfico Delta ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for cp, ax in zip(["C", "P"], axes):
    data_cp = serie_delta_m[serie_delta_m["CallPut"] == cp].sort_values("Date")
    ax.plot(data_cp["Date"], data_cp["delta_emp"], color="steelblue", linewidth=1.0, label="Delta empírica")
    ax.plot(data_cp["Date"], data_cp["Delta"],     color="firebrick",  linewidth=1.0, label="Delta BS", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.xaxis.set(major_locator=mdates.YearLocator(1),
             major_formatter=mdates.DateFormatter("%Y"))
    ax.set_title(f"Delta — {'Call' if cp == 'C' else 'Put'} (media mensual ponderada por OI, todo el moneyness)")
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Delta empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()

# ---- Gráfico Gamma ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for cp, ax in zip(["C", "P"], axes):
    data_cp = serie_gamma_m[serie_gamma_m["CallPut"] == cp].sort_values("Date")
    ax.plot(data_cp["Date"], data_cp["gamma_emp"], color="darkorange", linewidth=1.0, label="Gamma empírica")
    ax.plot(data_cp["Date"], data_cp["Gamma"],     color="firebrick",  linewidth=1.0, label="Gamma BS", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Gamma — {'Call' if cp == 'C' else 'Put'} (media mensual ponderada por OI, todo el moneyness)")
    ax.xaxis.set(major_locator=mdates.YearLocator(1),
             major_formatter=mdates.DateFormatter("%Y"))
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Gamma empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()

# %%

#####################################################################################
# Análisis 2c: Serie temporal MENSUAL — media aritmética de delta y gamma sobre todo el moneyness
# Comparativa empírica vs teórica (BS)
#####################################################################################

# Añadimos columna año-mes
opt_df_greek_filt_95["YearMonth"] = opt_df_greek_filt_95["Date"].dt.to_period("M")

# Media mensual ponderada por OI para cada (YearMonth, CallPut)
def media_mensual_aritmetic(df, greek_emp, greek_teo):
    resultados = []

    for (ym, cp), grupo in df.groupby(["YearMonth", "CallPut"]):
        grupo_valid = grupo[grupo[greek_emp].notna() & grupo[greek_teo].notna()].copy()
        if grupo_valid.empty:
            continue

        oi = grupo_valid["OpenInterest"]
        if oi.sum() == 0:
            continue

        resultados.append({
            "YearMonth":  ym,
            "CallPut":    cp,
            greek_emp:    (grupo_valid[greek_emp]).mean(),
            greek_teo:    (grupo_valid[greek_teo]).mean(),
            "n_contratos": len(grupo_valid)
        })

    df_out = pd.DataFrame(resultados)
    df_out["Date"] = df_out["YearMonth"].dt.to_timestamp()
    return df_out

# %%
serie_delta_m = media_mensual_aritmetic(opt_df_greek_filt_95, "delta_emp", "Delta")
serie_gamma_m = media_mensual_aritmetic(opt_df_greek_filt_95, "gamma_emp", "Gamma")


# ---- Gráfico Delta ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for cp, ax in zip(["C", "P"], axes):
    data_cp = serie_delta_m[serie_delta_m["CallPut"] == cp].sort_values("Date")
    ax.plot(data_cp["Date"], data_cp["delta_emp"], color="steelblue", linewidth=1.0, label="Delta empírica")
    ax.plot(data_cp["Date"], data_cp["Delta"],     color="firebrick",  linewidth=1.0, label="Delta BS", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Delta — {'Call' if cp == 'C' else 'Put'} (media aritmética para todo el moneyness)")
    ax.xaxis.set(major_locator=mdates.YearLocator(1),
             major_formatter=mdates.DateFormatter("%Y"))
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Delta empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()

# ---- Gráfico Gamma ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for cp, ax in zip(["C", "P"], axes):
    data_cp = serie_gamma_m[serie_gamma_m["CallPut"] == cp].sort_values("Date")
    ax.plot(data_cp["Date"], data_cp["gamma_emp"], color="darkorange", linewidth=1.0, label="Gamma empírica")
    ax.plot(data_cp["Date"], data_cp["Gamma"],     color="firebrick",  linewidth=1.0, label="Gamma BS", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Gamma — {'Call' if cp == 'C' else 'Put'} (media aritmética para todo el moneyness)")
    ax.xaxis.set(major_locator=mdates.YearLocator(1),
             major_formatter=mdates.DateFormatter("%Y"))
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Gamma empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()
# %%

#######################################################################################
# Análisis 3: Correlaciones sensibilidades empíricas vs teóricas
#######################################################################################

#"solo printeamos"

import seaborn as sns
corr_greeks = opt_df_greek_filt_95[["delta_emp","delta_emp_op2","delta_emp_op3", "Delta", "gamma_emp", "gamma_emp_op2","gamma_emp_op3","Gamma"]].corr(method="pearson")


corr_greeks.style.text_gradient(vmin=-1, vmax=1,cmap="coolwarm").set_caption("Correlaciones sensibilidades empíricas vs teóricas")
plt.figure(figsize=(8, 6))
sns.heatmap(corr_greeks, annot=True, vmin=-1, vmax=1)
plt.title("Correlaciones sensibilidades empíricas vs teóricas")
plt.show()

# %%

corr_greeks.style.highlight_quantile(
    axis=None,
    q_left=0.75,
    q_right=1,
    props="font-weight:bold;color:#4fc8d1")
# %%
corr_greeks.style.background_gradient(vmin=-1, vmax=1,cmap="coolwarm")

# %%
serie_delta1_m = media_mensual_OI(opt_df_greek_filt_95, "delta_emp",     "Delta")
serie_delta2_m = media_mensual_OI(opt_df_greek_filt_95, "delta_emp_op2", "Delta")
serie_delta3_m = media_mensual_OI(opt_df_greek_filt_95, "delta_emp_op3", "Delta")
serie_gamma1_m = media_mensual_OI(opt_df_greek_filt_95, "gamma_emp",     "Gamma")
serie_gamma2_m = media_mensual_OI(opt_df_greek_filt_95, "gamma_emp_op2", "Gamma")
serie_gamma3_m = media_mensual_OI(opt_df_greek_filt_95, "gamma_emp_op3", "Gamma")



agregado = pd.DataFrame({
    "YearMonth": serie_delta1_m["YearMonth"],
    "CallPut": serie_delta1_m["CallPut"],
    "DeltaT": serie_delta2_m["Delta"],
    "Delta1": serie_delta1_m["delta_emp"],
    "Delta2": serie_delta2_m["delta_emp_op2"],
    "Delta3": serie_delta3_m["delta_emp_op3"],
    "GammaT": serie_gamma2_m["Gamma"],
    "Gamma1": serie_gamma1_m["gamma_emp"],
    "Gamma2": serie_gamma2_m["gamma_emp_op2"],
    "Gamma3": serie_gamma3_m["gamma_emp_op3"]
})

agregado[["Delta1", "Delta2", "Delta3", "Gamma1", "Gamma2", "Gamma3"]].corr(method="pearson").style.text_gradient(vmin=-1, vmax=1,cmap="coolwarm").set_caption("Correlaciones con teóricas medias aritméticas")

# In[]:
#######################################################################################
# Análisis 4: Formas de cálculo de frecuencias mensuales de las sensibilidades empíricas
#######################################################################################
"""
Para este análsis parto del df ya filtrado por ambas colas al 5% de los datos.
"""
# Opción 1: Media mensual ponderada por OI (ya calculada en serie_delta_m y serie_gamma_m)
    #valores en el dataframe: agregado

# Opción 2: Como diferencia entre el primer día del mes y el último día del mes.
    # Nos permite capturar el cambio en información del mercado, tanto por OI como por sensibilidades (pendiente ver variantes relacionadas para aislar efectos)
opt_df_greek_filt_95["YearMonth"] = opt_df_greek_filt_95["Date"].dt.to_period("M")

"""
Necesitamos agrupar todas las sensibilidades para ese día, haciendo por ejemplo la media por OI:
1: Partimos de que las opciones de este bucket de tiempo a vencimiento son homogéneas en cuanto a su efecto temporal.
2: Calculamos una griega promedio del día por OI.
3: Empleamos la diferencia entre final y principio de mes para obtener la frecuencia mensual. (Se están metiendo los cambios tanto en OI como en precio)

"""


# In[]:

data_example = opt_df_greek_filt_95.sort_values(by="Date")
data_example = data_example.groupby("YearMonth")


def last_minus_first(arr):
    return arr.iloc[-1] - arr.iloc[0]

data_example["delta_emp"].agg(last_minus_first)



# In[]:

def diferencia_mensual_OI(df, greek_emp, greek_teo):
    """
    1. OI-weighted mean por (Date, CallPut)  → serie diaria
    2. last-minus-first por (YearMonth, CallPut) → frecuencia mensual
    """
    resultados_diarios = []

    for (date, cp), grupo in df.groupby(["Date", "CallPut"]):
        grupo_valid = grupo[grupo[greek_emp].notna() & grupo[greek_teo].notna()].copy()
        if grupo_valid.empty:
            continue

        oi = grupo_valid["OpenInterest"]
        if oi.sum() == 0:
            continue

        resultados_diarios.append({
            "Date":      date,
            "YearMonth": grupo_valid["YearMonth"].iloc[0],
            "CallPut":   cp,
            greek_emp:   (oi * grupo_valid[greek_emp]).sum() / oi.sum(),
            greek_teo:   (oi * grupo_valid[greek_teo]).sum() / oi.sum(),
        })

    daily = pd.DataFrame(resultados_diarios).sort_values("Date")

    # Paso 2: last - first por mes
    resultados_mensuales = []
    for (ym, cp), grupo in daily.groupby(["YearMonth", "CallPut"]):
        if len(grupo) < 2:
            continue
        resultados_mensuales.append({
            "YearMonth": ym,
            "CallPut":   cp,
            greek_emp:   ((grupo[greek_emp].iloc[-1] - grupo[greek_emp].iloc[0])),
            greek_teo:   ((grupo[greek_teo].iloc[-1] - grupo[greek_teo].iloc[0])),
        })

    df_out = pd.DataFrame(resultados_mensuales)
    df_out["Date"] = df_out["YearMonth"].dt.to_timestamp()
    return df_out


serie_delta_diff = diferencia_mensual_OI(opt_df_greek_filt_95, "delta_emp", "Delta")
serie_gamma_diff = diferencia_mensual_OI(opt_df_greek_filt_95, "gamma_emp", "Gamma")


fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for cp, ax in zip(["C", "P"], axes):
    data_cp = serie_delta_diff[serie_delta_diff["CallPut"] == cp].sort_values("Date")
    ax.plot(data_cp["Date"], data_cp["delta_emp"], color="steelblue", linewidth=1.0, label="Delta empírica")
    ax.plot(data_cp["Date"], data_cp["Delta"],     color="firebrick",  linewidth=1.0, label="Delta BS", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Delta — {'Call' if cp == 'C' else 'Put'} (Variación Mensual relativa (media diaria) para todo el moneyness)")
    ax.xaxis.set(major_locator=mdates.YearLocator(1),
             major_formatter=mdates.DateFormatter("%Y"))
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Delta empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()

# ---- Gráfico Gamma ----
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for cp, ax in zip(["C", "P"], axes):
    data_cp = serie_gamma_diff[serie_gamma_diff["CallPut"] == cp].sort_values("Date")
    ax.plot(data_cp["Date"], data_cp["gamma_emp"], color="darkorange", linewidth=1.0, label="Gamma empírica")
    ax.plot(data_cp["Date"], data_cp["Gamma"],     color="firebrick",  linewidth=1.0, label="Gamma BS", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"Gamma — {'Call' if cp == 'C' else 'Put'} (media aritmética para todo el moneyness)")
    ax.xaxis.set(major_locator=mdates.YearLocator(1),
             major_formatter=mdates.DateFormatter("%Y"))
    ax.legend()
    ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Gamma empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()

# In[]:
#serie_gamma_diff[["gamma_emp","Gamma"]].corr(method="pearson").style.text_gradient(vmin=-1, vmax=1,cmap="coolwarm").set_caption("Correlaciones con teóricas con diferencia en media diaria a inicio y final del mes")
serie_gamma_diff["gamma_emp"].autocorr(lag=1)

from statsmodels.tsa.stattools import adfuller, acf

# ADF test
serie_calls = serie_gamma_diff[serie_gamma_diff["CallPut"] == "C"]["gamma_emp"].dropna()
adf_result = adfuller(serie_calls)
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

# ACF hasta lag 12 para ver si hay estacionalidad
acf_vals = acf(serie_calls, nlags=12)
print(acf_vals)
# In[]:

































# %%
#####################################################################################
# Análisis 4: Descomposición vanna/charm — brecha entre gamma realizada y BS
#####################################################################################

serie_greek_95 = opt_df_greek_filt_95.merge(agregado[["YearMonth", "CallPut"]], on=["YearMonth", "CallPut"], how="left")
serie_greek_95
# %%
serie_delta1_m
# %%
opt_df_greek_filt_95
# %%
