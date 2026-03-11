

### 06.01-> Aquí quiero ver el tema de las maturities que dejo fuera en el filtrado de moneyness en 01 porque me quiero centrar entorno a 30 días

### Quiero ver también si se incumple algún punto el arbitraje del conjunto creado en 04 
### Comparación de griegas desde el fichero std_option (ATM),(ver si desde el de vol_surface puedo ampliar a más strikes el std_option).
### Analizar si las griegas obtenidas en 04 tienen sentido.

# In[]
import pandas as pd
import numpy as np
import sys
import duckdb
from pathlib import Path
import matplotlib.pyplot as plt


sys.path.insert(0, str(Path(__file__).parent.parent))

from __2_Files.option_price import OptionPrice
from __2_Files.forward_price import ForwardPrice
from __3_Functions.interpolation import interpolate_rates_surface
from __2_Files.zero_curve import ZeroCurve


####################################################################
 # veamos las maturities que tenemos directamente raw_data_tratado
####################################################################

#------------------configuración y cargade datos ------------------



############################################
# CARGA DE DATOS
############################################

con = duckdb.connect()

quoted_option = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\opt_df_clean.parquet')
""").df()

# filtros equivalentes a los que estás usando
quoted_option = quoted_option[quoted_option["flag_otm"]]
quoted_option = quoted_option[(quoted_option["Moneyness"] >= 0.3) & (quoted_option["Moneyness"] <= 1.7)]

df = quoted_option.copy()
df["Date"] = pd.to_datetime(df["Date"])

############################################
# 1. ANÁLISIS DE COBERTURA TEMPORAL
############################################

target_days = 30

rows = []

for date, df_date in df.groupby("Date"):

    days = np.sort(df_date["Days"].unique())

    menores = days[days < target_days]
    mayores = days[days > target_days]

    T1 = menores.max() if len(menores) else None
    T2 = mayores.min() if len(mayores) else None

    rows.append({
        "Date": date,
        "T1": T1,
        "T2": T2,
        "dist_below": target_days - T1 if T1 is not None else None,
        "dist_above": T2 - target_days if T2 is not None else None,
        "has_below": T1 is not None,
        "has_above": T2 is not None
    })

coverage = pd.DataFrame(rows)

############################################
# 2. RESUMEN GLOBAL
############################################

print("\nCobertura temporal:")
print(
    coverage[["has_below","has_above"]]
    .value_counts()
)

############################################
# 3. DISTANCIAS A 30 DÍAS
############################################

print("\nDistancia a 30 días:")
print(
    coverage[["dist_below","dist_above"]]
    .describe()
)

############################################
# 4. ANÁLISIS POR AÑO
############################################

coverage["year"] = coverage["Date"].dt.year

print("\nCobertura por año:")
print(
    coverage.groupby("year")[["has_below","has_above"]]
    .mean()
)

############################################
# 5. INTERVALO ENTRE EXPIRIES
############################################

usable = coverage[
    coverage["has_below"] &
    coverage["has_above"]
].copy()

usable["interval"] = usable["T2"] - usable["T1"]

print("\nTamaño del intervalo entre expiries:")
print(
    usable["interval"].describe()
)

############################################
# 6. STRIKES POR EXPIRY
############################################

strike_stats = (
    df.groupby(["Date","Days"])
    .agg(n_strikes=("Strike","nunique"))
    .reset_index()
)

print("\nNúmero de strikes por expiry:")
print(
    strike_stats["n_strikes"].describe()
)

############################################
# 7. HEATMAP DATE × DAYS
############################################

coverage_heat = (
    df.groupby(["Date", "Days"])
    .size()
    .reset_index(name="n_options")
)

coverage_heat["present"] = 1

pivot = coverage_heat.pivot_table(
    index="Date",
    columns="Days",
    values="present",
    fill_value=0
)

pivot_plot = pivot.loc[:, 0:120]

plt.figure(figsize=(14,8))

plt.imshow(
    pivot_plot.T,
    aspect="auto",
    origin="lower",
    cmap="viridis"
)

plt.colorbar(label="Presence of options")

plt.axhline(30, color="red", linestyle="--", linewidth=2)

plt.xlabel("Time (Date index)")
plt.ylabel("Days to expiration")
plt.title("Coverage of option maturities (Date × Days)")

plt.show()

############################################
# 8. HEATMAP DE DENSIDAD DE OPCIONES
############################################

pivot_density = coverage_heat.pivot_table(
    index="Date",
    columns="Days",
    values="n_options",
    fill_value=0
)

plt.figure(figsize=(14,8))

plt.imshow(
    pivot_density.loc[:,0:120].T,
    aspect="auto",
    origin="lower",
    cmap="magma"
)

plt.colorbar(label="Number of options")

plt.axhline(30, color="white", linestyle="--")

plt.title("Density of option maturities")

plt.xlabel("Time")
plt.ylabel("Days")

plt.show()

############################################
# 9. DISTANCIA MÍNIMA A 30 DÍAS
############################################

dist = (
    df.groupby("Date")["Days"]
    .apply(lambda x: np.min(np.abs(x - target_days)))
)

plt.figure(figsize=(8,4))

plt.hist(dist, bins=40)

plt.title("Minimum distance to 30 days")

plt.xlabel("Days")

plt.show()

print("\nResumen distancia mínima a 30 días:")
print(dist.describe())

############################################
# GUARDAR RESULTADO
############################################

coverage_summary = coverage.copy()

coverage_summary["valid_for_interp"] = (
    coverage_summary["has_below"] &
    coverage_summary["has_above"]
)

coverage_summary.to_csv(
    "coverage_30d_analysis.csv",
    index=False
)

print("\nArchivo coverage_30d_analysis.csv generado.")