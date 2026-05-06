# In[]
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, acf, arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tabulate import tabulate
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats

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

opt_df_greek_filt = opt_df_greek[(opt_df_greek["maturity_bucket"] == '(15.0, 45.0]') & (opt_df_greek["delta_emp"].notna()) & (opt_df_greek["gamma_emp"].notna())]

# %%
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

# In[]


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
# Análisis 3: Correlaciones sensibilidades empíricas vs teóricas y análsis de autocorrelaciones
#######################################################################################

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

agregado[["Delta1", "Delta2", "Delta3", "Gamma1", "Gamma2", "Gamma3"]].corr(method="pearson").style.text_gradient(vmin=-1, vmax=1,cmap="coolwarm").set_caption("Correlaciones con teóricas medias ponderadas por OI")
from statsmodels.tsa.stattools import adfuller, acf
# In[]:
# ADF test

gamma_serie_callsl = serie_gamma1_m[serie_gamma1_m["CallPut"] == "C"]["gamma_emp"].dropna()
gamma_serie_putsl = serie_gamma1_m[serie_gamma1_m["CallPut"] == "P"]["gamma_emp"].dropna()

delta_serie_callsl = serie_delta1_m[serie_delta1_m["CallPut"] == "C"]["delta_emp"].dropna()
delta_serie_putsl = serie_delta1_m[serie_delta1_m["CallPut"] == "P"]["delta_emp"].dropna()



# Mostramos descritpivos de la serie temporal para GAMMA Call
print("Gamma Call")
gamma_serie_callsl_diff = gamma_serie_callsl.diff().dropna()

adf_result = adfuller(gamma_serie_callsl_diff,regression="ctt")
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

print("Critical Values:")
print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(gamma_serie_callsl, ax=axes[0], lags=48, alpha=0.05, 
         use_vlines=True, fft=True, title='Autocorrelation', 
         zero=False, bartlett_confint=True)

plot_pacf(gamma_serie_callsl, ax=axes[1], lags=48, alpha=0.05, 
          method='ols', use_vlines=True, 
          title='Partial Autocorrelation', zero=False)

plt.tight_layout()
plt.show()

min_order = arma_order_select_ic(gamma_serie_callsl, max_ar=4, max_ma=2, ic='bic', trend='n')
print(min_order.bic)

# Mostramos descritpivos de la serie temporal para GAMMA Put

print("Gamma Put")
gamma_serie_putsl_diff = gamma_serie_putsl.diff().dropna()

adf_result = adfuller(gamma_serie_putsl_diff,regression="ctt")
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

print("Critical Values:")
print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(gamma_serie_putsl, ax=axes[0], lags=48, alpha=0.05, 
         use_vlines=True, fft=True, title='Autocorrelation', 
         zero=False, bartlett_confint=True)

plot_pacf(gamma_serie_putsl, ax=axes[1], lags=48, alpha=0.05, 
          method='ols', use_vlines=True, 
          title='Partial Autocorrelation', zero=False)

plt.tight_layout()
plt.show()

min_order = arma_order_select_ic(gamma_serie_putsl, max_ar=4, max_ma=2, ic='bic', trend='n')
print(min_order.bic)

# Mostramos descritpivos de la serie temporal para DELTA Call

print("Delta Call")
delta_serie_callsl_diff = delta_serie_callsl.diff().dropna()

adf_result = adfuller(delta_serie_callsl_diff,regression="ctt")
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

print("Critical Values:")
print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(delta_serie_callsl, ax=axes[0], lags=48, alpha=0.05, 
         use_vlines=True, fft=True, title='Autocorrelation', 
         zero=False, bartlett_confint=True)

plot_pacf(delta_serie_callsl, ax=axes[1], lags=48, alpha=0.05, 
          method='ols', use_vlines=True, 
          title='Partial Autocorrelation', zero=False)

plt.tight_layout()
plt.show()

min_order = arma_order_select_ic(delta_serie_callsl, max_ar=4, max_ma=2, ic='bic', trend='n')
print(min_order.bic)

# Mostramos descritpivos de la serie temporal para DELTA Put

print("Delta Put")
delta_serie_putsl_diff = delta_serie_putsl.diff().dropna()
adf_result = adfuller(delta_serie_putsl_diff,regression="ctt")
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

print("Critical Values:")
print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(delta_serie_putsl, ax=axes[0], lags=48, alpha=0.05, 
         use_vlines=True, fft=True, title='Autocorrelation', 
         zero=False, bartlett_confint=True)

plot_pacf(delta_serie_putsl, ax=axes[1], lags=48, alpha=0.05, 
          method='ols', use_vlines=True, 
          title='Partial Autocorrelation', zero=False)

plt.tight_layout()
plt.show()

min_order = arma_order_select_ic(delta_serie_putsl, max_ar=4, max_ma=2, ic='bic', trend='n')
print(min_order.bic)



    ### Ajustar ARIMA(2,1,0) (por ejemplo)
modelo = ARIMA(gamma_serie_callsl, order=(10,15,0)).fit()
print(modelo.summary())


# In[]:
#######################################################################################
# Análisis 4: Otras formas de cálculo de frecuencias mensuales de las sensibilidades empíricas
#######################################################################################

# Opción 1: Media mensual ponderada por OI (ya calculada en serie_delta_m y serie_gamma_m)
    #valores en el dataframe: agregado

# Opción 2: Como diferencia entre el primer día del mes y el último día del mes.
    # Nos permite capturar el cambio en información del mercado, tanto por OI como por sensibilidades (pendiente ver variantes relacionadas para aislar efectos)

"""

Necesitamos agrupar todas las sensibilidades para ese día, haciendo por ejemplo la media por OI:
1: Partimos de que las opciones de este bucket de tiempo a vencimiento son homogéneas en cuanto a su efecto temporal.
2: Calculamos una griega promedio del día por OI.
3: Empleamos la diferencia entre final y principio de mes para obtener la frecuencia mensual. (Se están metiendo los cambios tanto en OI como en precio)

"""

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

opt_df_greek_filt_95["YearMonth"] = opt_df_greek_filt_95["Date"].dt.to_period("M")

serie_delta_diff = diferencia_mensual_OI(opt_df_greek_filt_95, "delta_emp", "Delta")
serie_gamma_diff = diferencia_mensual_OI(opt_df_greek_filt_95, "gamma_emp", "Gamma")

agregado_diff = pd.DataFrame({
    "YearMonth": serie_delta_diff["YearMonth"],
    "CallPut": serie_delta_diff["CallPut"],
    "DeltaT": serie_delta_diff["Delta"],
    "Delta1": serie_delta_diff["delta_emp"],
    "GammaT": serie_gamma_diff["Gamma"],
    "Gamma1": serie_gamma_diff["gamma_emp"]
})




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

serie_gamma_diff[["gamma_emp","Gamma"]].corr(method="pearson").style.text_gradient(vmin=-1, vmax=1,cmap="coolwarm").set_caption("Correlaciones con teóricas con diferencia en media diaria a inicio y final del mes")
# In[]:
from statsmodels.tsa.stattools import adfuller, acf, arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tabulate import tabulate
from statsmodels.tsa.arima.model import ARIMA
# ADF test
gamma_serie_calls = serie_gamma_diff[serie_gamma_diff["CallPut"] == "C"]["gamma_emp"].dropna()
gamma_serie_puts = serie_gamma_diff[serie_gamma_diff["CallPut"] == "P"]["gamma_emp"].dropna()

delta_serie_calls = serie_delta_diff[serie_delta_diff["CallPut"] == "C"]["delta_emp"].dropna()
delta_serie_puts = serie_delta_diff[serie_delta_diff["CallPut"] == "P"]["delta_emp"].dropna()



# Mostramos descritpivos de la serie temporal para GAMMA Call
print("Gamma Call")
adf_result = adfuller(gamma_serie_calls,regression="ctt")
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

print("Critical Values:")
print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(gamma_serie_calls, ax=axes[0], lags=48, alpha=0.05, 
         use_vlines=True, fft=True, title='Autocorrelation', 
         zero=False, bartlett_confint=True)

plot_pacf(gamma_serie_calls, ax=axes[1], lags=48, alpha=0.05, 
          method='ols', use_vlines=True, 
          title='Partial Autocorrelation', zero=False)

plt.tight_layout()
plt.show()

min_order = arma_order_select_ic(gamma_serie_calls, max_ar=4, max_ma=2, ic='bic', trend='n')
print(min_order.bic)

# Mostramos descritpivos de la serie temporal para GAMMA Put

print("Gamma Put")
adf_result = adfuller(gamma_serie_puts,regression="ctt")
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

print("Critical Values:")
print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(gamma_serie_puts, ax=axes[0], lags=48, alpha=0.05, 
         use_vlines=True, fft=True, title='Autocorrelation', 
         zero=False, bartlett_confint=True)

plot_pacf(gamma_serie_puts, ax=axes[1], lags=48, alpha=0.05, 
          method='ols', use_vlines=True, 
          title='Partial Autocorrelation', zero=False)

plt.tight_layout()
plt.show()

min_order = arma_order_select_ic(gamma_serie_puts, max_ar=4, max_ma=2, ic='bic', trend='n')
print(min_order.bic)

# Mostramos descritpivos de la serie temporal para DELTA Call

print("Delta Call")
adf_result = adfuller(delta_serie_calls,regression="ctt")
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

print("Critical Values:")
print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(delta_serie_calls, ax=axes[0], lags=48, alpha=0.05, 
         use_vlines=True, fft=True, title='Autocorrelation', 
         zero=False, bartlett_confint=True)

plot_pacf(delta_serie_calls, ax=axes[1], lags=48, alpha=0.05, 
          method='ols', use_vlines=True, 
          title='Partial Autocorrelation', zero=False)

plt.tight_layout()
plt.show()

min_order = arma_order_select_ic(delta_serie_calls, max_ar=4, max_ma=2, ic='bic', trend='n')
print(min_order.bic)

# Mostramos descritpivos de la serie temporal para DELTA Put

print("Delta Put")
adf_result = adfuller(delta_serie_puts,regression="ctt")
print(f"ADF statistic: {adf_result[0]:.4f}")
print(f"p-value: {adf_result[1]:.4f}")

print("Critical Values:")
print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))


fig, axes = plt.subplots(2, 1, figsize=(14, 8))

plot_acf(delta_serie_puts, ax=axes[0], lags=48, alpha=0.05, 
         use_vlines=True, fft=True, title='Autocorrelation', 
         zero=False, bartlett_confint=True)

plot_pacf(delta_serie_puts, ax=axes[1], lags=48, alpha=0.05, 
          method='ols', use_vlines=True, 
          title='Partial Autocorrelation', zero=False)

plt.tight_layout()
plt.show()

min_order = arma_order_select_ic(delta_serie_puts, max_ar=4, max_ma=2, ic='bic', trend='n')
print(min_order.bic)



# %%

############################################################################
# Comparación del método 0 y del A:
############################################################################


# Varianza de cada serie
print("Varianza F2L:", delta_serie_puts.var())
print("Varianza Delta Monthly:", delta_serie_putsl_diff.var())

# Correlación entre ambas
print("Correlación:", delta_serie_puts.corr(delta_serie_putsl_diff))

# %%

###################### PRUEBAS #####################################################
from arch import arch_model

# ARMA(p,q) con distribución skew t-Student y varianza constante
modelo = arch_model(delta_serie_putsl_diff, mean='ARX', lags=0,
                   vol='Constant', dist='skewt')
res = modelo.fit(disp='off')
print(res.summary())

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

residuos = res.resid / res.conditional_volatility  # residuos estandarizados

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. Histograma con densidades superpuestas
ax = axes[0]
ax.hist(residuos, bins=40, density=True, alpha=0.5, color='steelblue', label='Residuos')

x = np.linspace(residuos.min(), residuos.max(), 300)

# Normal
ax.plot(x, stats.norm.pdf(x), color='firebrick', linewidth=1.5, label='Normal(0,1)')

# t-Student con nu estimado
nu = res.params['eta']
ax.plot(x, stats.t.pdf(x, df=nu), color='darkorange', linewidth=1.5, label=f't-Student(ν={nu:.2f})')

ax.set_title('Distribución de residuos estandarizados')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. QQ-plot vs Normal
ax = axes[1]
stats.probplot(residuos, dist="norm", plot=ax)
ax.set_title('QQ-plot vs Normal')
ax.grid(True, alpha=0.3)

# 3. QQ-plot vs t-Student
ax = axes[2]
stats.probplot(residuos, dist=stats.t, sparams=(nu,), plot=ax)
ax.set_title(f'QQ-plot vs t-Student(ν={nu:.2f})')
ax.grid(True, alpha=0.3)

fig.suptitle('Diagnóstico de residuos — Innovación gamma calls', fontsize=13)
plt.tight_layout()
plt.show()


# In[]:
###################################################################################################
# Opción 3: Para cada calculo una medida de "greek" imbalance al estilo Barbon(2020)

"""
1: Para cada día, calculo la Net "Gamma" Exposure como:
2: Primero cada uno de los puntos(m,t) ponderada por OI multiplicado por el underlying y asumiendo que las puts son vendidas.
3: La suma de todos los puntos ponderados por call y puts, se dividen por el dollar volume medio del mes anterior (21 business days) y se multiplica por el underlying
R: Se obtiene una medida de ka cantidad que se necesita cubrir ante un 1% de cambio en el underlying como fracción del volumen medio del mes anterior.
"""

def greek_net_exposure(data, greek: str, rolling_days: int, empirical: bool = True) -> pd.DataFrame:
    """
    Calcula el net exposure diario de una greek empírica (delta o gamma) normalizado
    por el volumen medio del período anterior, siguiendo la metodología de Soebhag (2023)
    y Barbon & Buraschi (2021).

    La medida resultante representa la cantidad neta que los dealers necesitan cubrir
    ante un 1% de movimiento en el subyacente, expresada como fracción del volumen
    medio diario del período anterior (semielasticidad).

    Las posiciones en puts se asumen vendidas (signo negativo), reflejando que los
    dealers son contrapartida neta de los compradores de protección.

    Parámetros
    ----------
    data : pd.DataFrame
        DataFrame con observaciones a nivel contrato. Debe contener las columnas:
        'Date', 'CallPut', 'OpenInterest', 'SpotPrice', 'DolarVolume',
        'delta_emp' (si greek='delta') o 'gamma_emp' (si greek='gamma').
    greek : str
        Greek a calcular. Opciones: 'delta' o 'gamma'.
    rolling_days : int
        Número de días de trading para la ventana móvil del volumen medio (típicamente 21,
        equivalente a 1 mes de trading).
    empirical : bool, optional
        Si True (default), usa la greek empírica ('delta_emp', 'gamma_emp').
        Si False, usa la greek teórica de Black-Scholes ('Delta', 'Gamma').

    Returns
    -------
    pd.DataFrame
        DataFrame diario con las columnas:
        - Date: fecha.
        - DolarVolume: volumen medio diario en dólares del día t.
        - Avg_DolarVolume_t-1: media móvil del volumen de los rolling_days anteriores.
        - Gamma_Exposure / Delta_Exposure: net exposure normalizado.
          Para gamma: sum(gamma_i * OI_i * S²) / (100² * AvgDolVol).
          Para delta: sum(delta_i * OI_i * S)  / (100  * AvgDolVol).
        - SpotPrice: precio del subyacente en t.

    Nota
    -----
    - El volumen medio se calcula sobre todos los contratos (calls y puts conjuntamente)
      como media aritmética diaria, dado que representa la liquidez global del mercado.
    """

    data = data.copy()

    def rolling_mean(group):
        return group.rolling(rolling_days).mean()

    # Volumen medio rolling
    datadiario = (data
        .groupby("Date")["DolarVolume"]
        .mean()
        .reset_index()
    )
    datadiario["Avg_DolarVolume_t-1"] = datadiario["DolarVolume"].transform(rolling_mean)

    # Configuración por greek
    if greek == "gamma":
        greek_col    = "gamma_emp" if empirical else "Gamma"
        exposure_col = "Gamma_Exposure" if empirical else "BS_Gamma_Exposure"
        scale        = 100**2
        
        data[exposure_col] = np.where(
            data["CallPut"] == "C",
            data[greek_col] * data["OpenInterest"] * data["SpotPrice"],
            data[greek_col] * data["OpenInterest"] * data["SpotPrice"] * (-1)
            )

    elif greek == "delta":
        greek_col    = "delta_emp" if empirical else "Delta"
        exposure_col = "Delta_Exposure" if empirical else "BS_Delta_Exposure"
        scale        = 100

        data[exposure_col] = np.where(
            data["CallPut"] == "C",
            data[greek_col] * data["OpenInterest"],
            data[greek_col] * data["OpenInterest"]* (-1)
            )
    else:
        raise ValueError(f"greek debe ser 'gamma' o 'delta', recibido: {greek}")

    # Spot diario
    spot_diario = data[["Date", "SpotPrice"]].drop_duplicates("Date")

    # Agregación diaria
    df = (data.groupby("Date")[exposure_col]
        .sum()
        .reset_index()
        .sort_values("Date")
        .merge(spot_diario,  on="Date", how="left")
        .merge(datadiario,   on="Date", how="left")
        .dropna()
        .reset_index(drop=True)
    )

    # Normalización: exposure * S / (scale * AvgDolVol)
    df[exposure_col] = df[exposure_col] * df["SpotPrice"] / (scale*df["Avg_DolarVolume_t-1"])

    return df[["Date", "DolarVolume", "Avg_DolarVolume_t-1", exposure_col, "SpotPrice"]]

def diferencia_mensual(df:pd.DataFrame, greek_emp:str, greek_teo = None) -> pd.DataFrame:
    """
    1. OI-weighted mean por (Date, CallPut)  → serie diaria
    2. last-minus-first por (YearMonth, CallPut) → frecuencia mensual
    """

    df["YearMonth"] = df["Date"].dt.to_period("M")

    df = pd.DataFrame(df).sort_values("Date")

    # Paso 2: last - first por mes
    resultados_mensuales = []
    for ym, grupo in df.groupby("YearMonth"):
        if len(grupo) < 2:
            continue
        resultados_mensuales.append({
            "YearMonth": ym,
            greek_emp:   ((grupo[greek_emp].iloc[-1] - grupo[greek_emp].iloc[0])),
            greek_teo:   ((grupo[greek_teo].iloc[-1] - grupo[greek_teo].iloc[0])),
        })

    df_out = pd.DataFrame(resultados_mensuales)
    df_out["Date"] = df_out["YearMonth"].dt.to_timestamp()
    return df_out

# %%
net_exposure = pd.DataFrame({
    "Date":              (greek_net_exposure(opt_df_greek_filt_95, "delta", 21))["Date"],
    "Delta_Exposure":    (greek_net_exposure(opt_df_greek_filt_95, "delta", 21))["Delta_Exposure"],
    "BS_Delta_Exposure": (greek_net_exposure(opt_df_greek_filt_95, "delta", 21, empirical=False))["BS_Delta_Exposure"],
    "Gamma_Exposure":    (greek_net_exposure(opt_df_greek_filt_95, "gamma", 21))["Gamma_Exposure"],
    "BS_Gamma_Exposure": (greek_net_exposure(opt_df_greek_filt_95, "gamma", 21, empirical=False))["BS_Gamma_Exposure"],
})


net_exposure_gamma_m = diferencia_mensual(net_exposure,"Gamma_Exposure", "BS_Gamma_Exposure")
net_exposure_delta_m = diferencia_mensual(net_exposure,"Delta_Exposure", "BS_Delta_Exposure")

net_exposure_m = pd.DataFrame({
    "Date":           net_exposure_delta_m["Date"],
    "Delta_Exposure_m":    net_exposure_delta_m["Delta_Exposure"],
    "BS_Delta_Exposure_m": net_exposure_delta_m["BS_Delta_Exposure"],
    "Gamma_Exposure_m":    net_exposure_gamma_m["Gamma_Exposure"],
    "BS_Gamma_Exposure_m": net_exposure_gamma_m["BS_Gamma_Exposure"],
})
net_exposure_m

# In[]:

########################
# Graficamos
########################

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

ax.plot(net_exposure_m["Date"], net_exposure_m["Delta_Exposure_m"], color="steelblue", linewidth=1.0, label="Delta empírica")
ax.plot(net_exposure_m["Date"], net_exposure_m["BS_Delta_Exposure_m"], color="firebrick", linewidth=1.0, label="Delta BS", alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_title(
    "Delta Net Exposure mensual\n"
    "(fracción del volumen medio mensual a cubrir ante un movimiento del 1% en el subyacente)"
)
ax.xaxis.set(major_locator=mdates.MonthLocator(bymonth=[1, 6]),
             major_formatter=mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Delta empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()

# ---- Gráfico Gamma ----


fig, ax = plt.subplots(1, 1, figsize=(14, 8))

ax.plot(net_exposure_m["Date"], net_exposure_m["Gamma_Exposure_m"], color="steelblue", linewidth=1.0, label="Gamma empírica")
ax.plot(net_exposure_m["Date"], net_exposure_m["BS_Gamma_Exposure_m"], color="firebrick", linewidth=1.0, label="Gamma BS", alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_title(
    "Gamma Net Exposure mensual\n"
    "(fracción del volumen medio mensual a cubrir ante un movimiento del 1% en el subyacente)"
)
ax.xaxis.set(major_locator=mdates.MonthLocator(bymonth=[1, 6]),
             major_formatter=mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Gamma empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()
# %%

########################
# Graficamos
########################

fig, ax = plt.subplots(1, 1, figsize=(14, 8))

ax.plot(net_exposure_m["Date"], net_exposure_m["Delta_Exposure_m"], color="steelblue", linewidth=1.0, label="Delta empírica")
ax.plot(net_exposure_m["Date"], net_exposure_m["BS_Delta_Exposure_m"], color="firebrick", linewidth=1.0, label="Delta BS", alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_title(
    "Delta Net Exposure mensual\n"
    "(fracción del volumen medio mensual a cubrir ante un movimiento del 1% en el subyacente)"
)
ax.xaxis.set(major_locator=mdates.MonthLocator(bymonth=[1, 6]),
             major_formatter=mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Delta empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()

# ---- Gráfico Gamma ----


fig, ax = plt.subplots(1, 1, figsize=(14, 8))

ax.plot(net_exposure_m["Date"], net_exposure_m["Gamma_Exposure_m"], color="steelblue", linewidth=1.0, label="Gamma empírica")
ax.plot(net_exposure_m["Date"], net_exposure_m["BS_Gamma_Exposure_m"], color="firebrick", linewidth=1.0, label="Gamma BS", alpha=0.8)
ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
ax.set_title(
    "Gamma Net Exposure mensual\n"
    "(fracción del volumen medio mensual a cubrir ante un movimiento del 1% en el subyacente)"
)
ax.xaxis.set(major_locator=mdates.MonthLocator(bymonth=[1, 6]),
             major_formatter=mdates.DateFormatter("%Y-%m"))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle("Serie temporal mensual — Gamma empírica vs BS", fontsize=13)
plt.tight_layout()
plt.show()
# %%


########################
# Graficamos2
########################

tramos = [("2003-01", "2015-12"), ("2016-01", "2025-12")]

def plot_exposure(df, col_emp, col_bs, label_emp, label_bs, titulo, supertitulo, tramos):
    for t_ini, t_fin in tramos:
        data_tramo = df[(df["Date"] >= t_ini) & (df["Date"] <= t_fin)]
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        ax.plot(data_tramo["Date"], data_tramo[col_emp], color="steelblue", linewidth=1.0, label=label_emp)
        ax.plot(data_tramo["Date"], data_tramo[col_bs],  color="firebrick",  linewidth=1.0, label=label_bs, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_title(f"{titulo}\n({t_ini} — {t_fin})")
        ax.xaxis.set(major_locator=mdates.MonthLocator(bymonth=[1, 6]),
                     major_formatter=mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.suptitle(supertitulo, fontsize=13)
        plt.tight_layout()
        plt.show()


# ---- Delta ----
plot_exposure(
    df         = net_exposure_m,
    col_emp    = "Delta_Exposure_m",
    col_bs     = "BS_Delta_Exposure_m",
    label_emp  = "Delta empírica",
    label_bs   = "Delta BS",
    titulo     = "Delta Net Exposure mensual\n(fracción del volumen medio mensual a cubrir ante un movimiento del 1% en el subyacente)",
    supertitulo= "Serie temporal mensual — Delta empírica vs BS",
    tramos     = tramos
)

# ---- Gamma ----
plot_exposure(
    df         = net_exposure_m,
    col_emp    = "Gamma_Exposure_m",
    col_bs     = "BS_Gamma_Exposure_m",
    label_emp  = "Gamma empírica",
    label_bs   = "Gamma BS",
    titulo     = "Gamma Net Exposure mensual\n(fracción del volumen medio mensual a cubrir ante un movimiento del 1% en el subyacente)",
    supertitulo= "Serie temporal mensual — Gamma empírica vs BS",
    tramos     = tramos
)


# %%
from statsmodels.tsa.stattools import adfuller, acf, arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from tabulate import tabulate

def diagnostico_serie_temporal(serie, nombre):
    """
    ADF test, ACF, PACF y selección de orden ARMA para una serie temporal.
    """
    serie = serie.dropna()

    print(f"\n{'='*60}")
    print(f"DIAGNÓSTICO — {nombre}")
    print(f"{'='*60}")

    # ADF test
    adf_result = adfuller(serie, regression="c")
    print(f"ADF statistic : {adf_result[0]:.4f}")
    print(f"p-value       : {adf_result[1]:.4f}")
    print("Critical Values:")
    print(tabulate([adf_result[4]], headers="keys", tablefmt="grid"))

    # ACF y PACF
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    plot_acf(serie, ax=axes[0], lags=48, alpha=0.05,
             use_vlines=True, fft=True, title=f"ACF — {nombre}",
             zero=False, bartlett_confint=True)
    plot_pacf(serie, ax=axes[1], lags=48, alpha=0.05,
              method="ols", use_vlines=True,
              title=f"PACF — {nombre}", zero=False)
    plt.tight_layout()
    plt.show()

    # Selección de orden ARMA
    min_order = arma_order_select_ic(serie, max_ar=4, max_ma=2, ic="bic", trend="n")
    print(f"Orden ARMA óptimo (BIC):")
    print(min_order.bic)


# ============================================================
# EJECUCIÓN
# ============================================================

diagnostico_serie_temporal(net_exposure_m["Delta_Exposure_m"],    "Delta Net Exposure — Empírica")
diagnostico_serie_temporal(net_exposure_m["BS_Delta_Exposure_m"],  "Delta Net Exposure — BS")
diagnostico_serie_temporal(net_exposure_m["Gamma_Exposure_m"],     "Gamma Net Exposure — Empírica")
diagnostico_serie_temporal(net_exposure_m["BS_Gamma_Exposure_m"],  "Gamma Net Exposure — BS")

# %%


#####################################################################################
# Análisis 4: Comparación con variables macro:
#####################################################################################
"""
Contrasto la opción última (D en el documento) y A. Al ser ya estacionarias y no necesita controlar por autocorrlación (como el vix)
Selección de variables:


"""

































# %%

#####################################################################################
# Análisis 5: Descomposición vanna/charm — brecha entre gamma realizada y BS
#####################################################################################

serie_greek_95 = opt_df_greek_filt_95.merge(agregado[["YearMonth", "CallPut"]], on=["YearMonth", "CallPut"], how="left")
serie_greek_95
# %%
serie_delta1_m
# %%
opt_df_greek_filt_95
# %%
