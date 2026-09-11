# In[]
import pandas as pd
import numpy as np
import sys
from tabulate import tabulate
import matplotlib.pyplot as plt


from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


PATH_DATA   = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"
print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

# In[]

#Añadimos algunas variables de interés:
opt_df["Dummy_Bid"] = opt_df["Bid"] > 0
opt_df["DolarVolume"] = opt_df["Volume"] *opt_df["MidPrice"]


# In[]

# Asignamos buckets de vencimientos:
v_grid = [0, 15, 45, 105, 183, 365, np.inf]
v_edges = pd.IntervalIndex.from_breaks(v_grid, closed="right")
opt_df["maturity_bucket"] = pd.cut(opt_df["Days"], bins=v_edges, labels=False, include_lowest=True)

# Asignamos buckets de moneyness (de momento en relación al spot):

m_grid = np.round(np.linspace(0.1,2,int(2/0.1)),2)
m_edges = pd.IntervalIndex.from_breaks(m_grid, closed="right")
opt_df["moneyness_bucket"] = pd.cut(opt_df["Moneyness"], bins=m_edges, labels=False, include_lowest=True)
print(opt_df[["Days", "maturity_bucket", "Moneyness", "moneyness_bucket"]].head(10))
print("Datos cargados y buckets asignados.")

# # In[]

# # Nos quedamos con los vencimientos de 15 a 45 días naturales.
# data_15_45 = opt_df[opt_df["maturity_bucket"] == pd.Interval(15, 45, closed="right")]

#     #También podría hacer: data_15_45 = opt_df[opt_df["Days"].between(15, 45)]


# # In[]


# # Generamos las métricas por grupo de moneyness:
# todos_dias = data_15_45["Date"].unique()
# todos_buckets = data_15_45["moneyness_bucket"].unique()
# idx_completo = pd.MultiIndex.from_product(
#     [todos_buckets, todos_dias], 
#     names=["moneyness_bucket", "Date"]
# )

# # Generamos las métricas por grupo de moneyness con reindex:
# grouped_15_45_v4 = (data_15_45
#     .groupby(["moneyness_bucket", "Date"]).agg(
#         n_contracts  = ("OptionID",      "count"),
#         oi_dsum      = ("OpenInterest",  "sum"),
#         dolvol_dsum  = ("DolarVolume",   "sum"),
#         dbid_dmean   = ("Dummy_Bid",     "mean")
#     )
#     .reindex(idx_completo, fill_value=0) #####Importante esto para asegurar que tenemos filas para todos los días y buckets, incluso si no hay contratos.
#     .reset_index()
# )

# grouped_tabla1_3 = grouped_15_45_v4.groupby("moneyness_bucket").agg(
#     max_n_contracts = ("n_contracts",  "max"),
#     min_n_contracts = ("n_contracts",  "min"),
#     max_oi_dsum     = ("oi_dsum",      "max"),
#     min_oi_dsum     = ("oi_dsum",      "min"),
#     max_dolvol_dsum = ("dolvol_dsum",  "max"),
#     min_dolvol_dsum = ("dolvol_dsum",  "min"),
#     max_dbid_dmean  = ("dbid_dmean",   "max"),
#     min_dbid_dmean  = ("dbid_dmean",   "min")
# ).reset_index()

# grouped_tabla1_3[["max_dbid_dmean", "min_dbid_dmean"]] *= 100
# print(tabulate(grouped_tabla1_3, headers="keys", tablefmt="rounded_outline", floatfmt=".3f", showindex=False))
# # In[]

# # Y si quito OI<=1?


# # Y si quito Bid=0?




# ### No lo hago porque claramente se reduce más.
# # Tan solo notar que en el caso del filtro de OI, aumenta el el Bid_Dummy maximo.

# # In[]
# #Veamos si hay suficiente continuidaad en el panel para cada uno de los buckets de moneyness:



# import matplotlib.pyplot as plt
# # Serie temporal del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros)

# data_15_45_v2 = data_15_45[data_15_45["Bid"] >=0]

# grouped_15_45 = (data_15_45_v2.groupby(["moneyness_bucket", "Date"]).agg(
#         n_contracts  = ("OptionID",      "count"),
#         oi_dsum      = ("OpenInterest",  "sum"),
#         dolvol_dsum  = ("DolarVolume",   "sum"),
#         dbid_dmean   = ("Dummy_Bid",     "mean")
#     ).reset_index(inplace=False)
# )

# # Serie temporal del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros)
# rango_moneyness = grouped_15_45.groupby("Date").agg(
#     m_min = ("moneyness_bucket", "min"),
#     m_max = ("moneyness_bucket", "max"),
#     n_buckets = ("moneyness_bucket", "count")
# ).reset_index()

# # Convertir Interval a extremo derecho
# rango_moneyness["m_min"] = rango_moneyness["m_min"].apply(lambda x: x.right)
# rango_moneyness["m_max"] = rango_moneyness["m_max"].apply(lambda x: x.right)


# fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# axes[0].plot(rango_moneyness["Date"], rango_moneyness["m_min"], label="m_min")
# axes[1].plot(rango_moneyness["Date"], rango_moneyness["m_max"], label="m_max")
# axes[2].plot(rango_moneyness["Date"], rango_moneyness["n_buckets"], label="n_buckets", color="green")

# for ax, title in zip(axes, ["Bucket mínimo cubierto", "Bucket máximo cubierto", "Nº buckets con contratos"]):
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
# # In[]:
# ##################################################################################################################
# # Prueba para diferentes vencimientos 45 a 105 días naturales.:
# ####################################################################################################################

# data_45_105 = opt_df[opt_df["maturity_bucket"] == pd.Interval(45, 105, closed="right")]



# # Generamos las métricas por grupo de moneyness:
# todos_dias = data_45_105["Date"].unique()
# todos_buckets = data_45_105["moneyness_bucket"].unique()
# idx_completo = pd.MultiIndex.from_product(
#     [todos_buckets, todos_dias], 
#     names=["moneyness_bucket", "Date"]
# )

# # Generamos las métricas por grupo de moneyness con reindex:
# grouped_45_105 = (data_45_105
#     .groupby(["moneyness_bucket", "Date"]).agg(
#         n_contracts  = ("OptionID",      "count"),
#         oi_dsum      = ("OpenInterest",  "sum"),
#         dolvol_dsum  = ("DolarVolume",   "sum"),
#         dbid_dmean   = ("Dummy_Bid",     "mean")
#     )
#     .reindex(idx_completo, fill_value=0) #####Importante esto para asegurar que tenemos filas para todos los días y buckets, incluso si no hay contratos.
#     .reset_index()
# )

# grouped_tabla2 = grouped_45_105.groupby("moneyness_bucket").agg(
#     max_n_contracts = ("n_contracts",  "max"),
#     min_n_contracts = ("n_contracts",  "min"),
#     max_oi_dsum     = ("oi_dsum",      "max"),
#     min_oi_dsum     = ("oi_dsum",      "min"),
#     max_dolvol_dsum = ("dolvol_dsum",  "max"),
#     min_dolvol_dsum = ("dolvol_dsum",  "min"),
#     max_dbid_dmean  = ("dbid_dmean",   "max"),
#     min_dbid_dmean  = ("dbid_dmean",   "min")
# ).reset_index()

# grouped_tabla2[["max_dbid_dmean", "min_dbid_dmean"]] *= 100
# print(tabulate(grouped_tabla2, headers="keys", tablefmt="rounded_outline", floatfmt=".3f", showindex=False))

# # Y si quito OI<=1?


# # Y si quito Bid=0?



# #Veamos si hay suficiente continuidaad en el panel para cada uno de los buckets de moneyness:

# import matplotlib.pyplot as plt
# # Serie temporal del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros)

# data_45_105_v2 = data_45_105[data_45_105["Bid"] >=0]

# grouped_45_105 = (data_45_105_v2.groupby(["moneyness_bucket", "Date"]).agg(
#         n_contracts  = ("OptionID",      "count"),
#         oi_dsum      = ("OpenInterest",  "sum"),
#         dolvol_dsum  = ("DolarVolume",   "sum"),
#         dbid_dmean   = ("Dummy_Bid",     "mean")
#     ).reset_index(inplace=False)
# )

# # Serie temporal del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros)
# rango_moneyness = grouped_45_105.groupby("Date").agg(
#     m_min = ("moneyness_bucket", "min"),
#     m_max = ("moneyness_bucket", "max"),
#     n_buckets = ("moneyness_bucket", "count")
# ).reset_index()

# # Convertir Interval a extremo derecho
# rango_moneyness["m_min"] = rango_moneyness["m_min"].apply(lambda x: x.right)
# rango_moneyness["m_max"] = rango_moneyness["m_max"].apply(lambda x: x.right)


# fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# axes[0].plot(rango_moneyness["Date"], rango_moneyness["m_min"], label="m_min")
# axes[1].plot(rango_moneyness["Date"], rango_moneyness["m_max"], label="m_max")
# axes[2].plot(rango_moneyness["Date"], rango_moneyness["n_buckets"], label="n_buckets", color="green")

# for ax, title in zip(axes, ["Bucket mínimo cubierto", "Bucket máximo cubierto", "Nº buckets con contratos"]):
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
# # In[]:
# ##################################################################################################################
# # Prueba para diferentes vencimientos 0 a 15 días naturales.:
# ####################################################################################################################

# data_0_15 = opt_df[opt_df["maturity_bucket"] == pd.Interval(0, 15, closed="right")]



# # Generamos las métricas por grupo de moneyness:
# todos_dias = data_0_15["Date"].unique()
# todos_buckets = data_0_15["moneyness_bucket"].unique()
# idx_completo = pd.MultiIndex.from_product(
#     [todos_buckets, todos_dias], 
#     names=["moneyness_bucket", "Date"]
# )

# # Generamos las métricas por grupo de moneyness con reindex:
# grouped_0_15 = (data_0_15
#     .groupby(["moneyness_bucket", "Date"]).agg(
#         n_contracts  = ("OptionID",      "count"),
#         oi_dsum      = ("OpenInterest",  "sum"),
#         dolvol_dsum  = ("DolarVolume",   "sum"),
#         dbid_dmean   = ("Dummy_Bid",     "mean")
#     )
#     .reindex(idx_completo, fill_value=0) #####Importante esto para asegurar que tenemos filas para todos los días y buckets, incluso si no hay contratos.
#     .reset_index()
# )

# grouped_tabla3 = grouped_0_15.groupby("moneyness_bucket").agg(
#     max_n_contracts = ("n_contracts",  "max"),
#     min_n_contracts = ("n_contracts",  "min"),
#     max_oi_dsum     = ("oi_dsum",      "max"),
#     min_oi_dsum     = ("oi_dsum",      "min"),
#     max_dolvol_dsum = ("dolvol_dsum",  "max"),
#     min_dolvol_dsum = ("dolvol_dsum",  "min"),
#     max_dbid_dmean  = ("dbid_dmean",   "max"),
#     min_dbid_dmean  = ("dbid_dmean",   "min")
# ).reset_index()

# grouped_tabla3[["max_dbid_dmean", "min_dbid_dmean"]] *= 100
# print(tabulate(grouped_tabla3, headers="keys", tablefmt="rounded_outline", floatfmt=".3f", showindex=False))

# # Y si quito OI<=1?


# # Y si quito Bid=0?



# #Veamos si hay suficiente continuidaad en el panel para cada uno de los buckets de moneyness:

# import matplotlib.pyplot as plt
# # Serie temporal del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros)

# data_0_15_v2 = data_0_15[data_0_15["Bid"] >=0]

# grouped_0_15 = (data_0_15_v2.groupby(["moneyness_bucket", "Date"]).agg(
#         n_contracts  = ("OptionID",      "count"),
#         oi_dsum      = ("OpenInterest",  "sum"),
#         dolvol_dsum  = ("DolarVolume",   "sum"),
#         dbid_dmean   = ("Dummy_Bid",     "mean")
#     ).reset_index(inplace=False)
# )

# # Serie temporal del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros)
# rango_moneyness = grouped_0_15.groupby("Date").agg(
#     m_min = ("moneyness_bucket", "min"),
#     m_max = ("moneyness_bucket", "max"),
#     n_buckets = ("moneyness_bucket", "count")
# ).reset_index()

# # Convertir Interval a extremo derecho
# rango_moneyness["m_min"] = rango_moneyness["m_min"].apply(lambda x: x.right)
# rango_moneyness["m_max"] = rango_moneyness["m_max"].apply(lambda x: x.right)


# fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# axes[0].plot(rango_moneyness["Date"], rango_moneyness["m_min"], label="m_min")
# axes[1].plot(rango_moneyness["Date"], rango_moneyness["m_max"], label="m_max")
# axes[2].plot(rango_moneyness["Date"], rango_moneyness["n_buckets"], label="n_buckets", color="green")

# for ax, title in zip(axes, ["Bucket mínimo cubierto", "Bucket máximo cubierto", "Nº buckets con contratos"]):
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()
# # In[]:
# ##################################################################################################################
# # Prueba para diferentes vencimientos 105 a 183 días naturales.:
# ####################################################################################################################

# data_105_183 = opt_df[opt_df["maturity_bucket"] == pd.Interval(105, 183, closed="right")]



# # Generamos las métricas por grupo de moneyness:
# todos_dias = data_105_183["Date"].unique()
# todos_buckets = data_105_183["moneyness_bucket"].unique()
# idx_completo = pd.MultiIndex.from_product(
#     [todos_buckets, todos_dias], 
#     names=["moneyness_bucket", "Date"]
# )

# # Generamos las métricas por grupo de moneyness con reindex:
# grouped_105_183 = (data_105_183
#     .groupby(["moneyness_bucket", "Date"]).agg(
#         n_contracts  = ("OptionID",      "count"),
#         oi_dsum      = ("OpenInterest",  "sum"),
#         dolvol_dsum  = ("DolarVolume",   "sum"),
#         dbid_dmean   = ("Dummy_Bid",     "mean")
#     )
#     .reindex(idx_completo, fill_value=0) #####Importante esto para asegurar que tenemos filas para todos los días y buckets, incluso si no hay contratos.
#     .reset_index()
# )

# grouped_tabla4 = grouped_105_183.groupby("moneyness_bucket").agg(
#     max_n_contracts = ("n_contracts",  "max"),
#     min_n_contracts = ("n_contracts",  "min"),
#     max_oi_dsum     = ("oi_dsum",      "max"),
#     min_oi_dsum     = ("oi_dsum",      "min"),
#     max_dolvol_dsum = ("dolvol_dsum",  "max"),
#     min_dolvol_dsum = ("dolvol_dsum",  "min"),
#     max_dbid_dmean  = ("dbid_dmean",   "max"),
#     min_dbid_dmean  = ("dbid_dmean",   "min")
# ).reset_index()

# grouped_tabla4[["max_dbid_dmean", "min_dbid_dmean"]] *= 100
# print(tabulate(grouped_tabla4, headers="keys", tablefmt="rounded_outline", floatfmt=".3f", showindex=False))

# # Y si quito OI<=1?


# # Y si quito Bid=0?



# #Veamos si hay suficiente continuidaad en el panel para cada uno de los buckets de moneyness:

# import matplotlib.pyplot as plt
# # Serie temporal del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros)

# data_105_183_v2 = data_105_183[data_105_183["Bid"] >=0]

# grouped_105_183 = (data_105_183_v2.groupby(["moneyness_bucket", "Date"]).agg(
#         n_contracts  = ("OptionID",      "count"),
#         oi_dsum      = ("OpenInterest",  "sum"),
#         dolvol_dsum  = ("DolarVolume",   "sum"),
#         dbid_dmean   = ("Dummy_Bid",     "mean")
#     ).reset_index(inplace=False)
# )

# # Serie temporal del rango de moneyness cubierto [m_min, m_max] por día (con y sin filtros)
# rango_moneyness = grouped_105_183.groupby("Date").agg(
#     m_min = ("moneyness_bucket", "min"),
#     m_max = ("moneyness_bucket", "max"),
#     n_buckets = ("moneyness_bucket", "count")
# ).reset_index()

# # Convertir Interval a extremo derecho
# rango_moneyness["m_min"] = rango_moneyness["m_min"].apply(lambda x: x.right)
# rango_moneyness["m_max"] = rango_moneyness["m_max"].apply(lambda x: x.right)


# fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

# axes[0].plot(rango_moneyness["Date"], rango_moneyness["m_min"], label="m_min")
# axes[1].plot(rango_moneyness["Date"], rango_moneyness["m_max"], label="m_max")
# axes[2].plot(rango_moneyness["Date"], rango_moneyness["n_buckets"], label="n_buckets", color="green")

# for ax, title in zip(axes, ["Bucket mínimo cubierto", "Bucket máximo cubierto", "Nº buckets con contratos"]):
#     ax.set_title(title)
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()


# In[]:

def analisis_vencimiento(opt_df, v_min, v_max):
    
    nombre = f"{v_min}_{v_max}"
    maturity_interval = pd.Interval(v_min, v_max, closed="right")
    data = opt_df[opt_df["maturity_bucket"] == maturity_interval].copy()
    
    if data.empty:
        print(f"[{nombre}] Sin datos.")
        return None, None, None, None

    # --- Tabla resumen con reindex ---
    todos_dias    = data["Date"].unique()
    todos_buckets = data["moneyness_bucket"].unique()
    idx_completo  = pd.MultiIndex.from_product(
        [todos_buckets, todos_dias],
        names=["moneyness_bucket", "Date"]
    )

    grouped = (data
        .groupby(["moneyness_bucket", "Date"]).agg(
            n_contracts = ("OptionID",     "count"),
            oi_dsum     = ("OpenInterest", "sum"),
            dolvol_dsum = ("DolarVolume",  "sum"),
            dbid_dmean  = ("Dummy_Bid",    "mean")
        )
        .reindex(idx_completo, fill_value=0)
        .reset_index()
    )

    tabla = grouped.groupby("moneyness_bucket").agg(
        max_n_contracts = ("n_contracts",  "max"),
        min_n_contracts = ("n_contracts",  "min"),
        max_oi_dsum     = ("oi_dsum",      "max"),
        min_oi_dsum     = ("oi_dsum",      "min"),
        max_dolvol_dsum = ("dolvol_dsum",  "max"),
        min_dolvol_dsum = ("dolvol_dsum",  "min"),
        max_dbid_dmean  = ("dbid_dmean",   "max"),
        min_dbid_dmean  = ("dbid_dmean",   "min")
    ).reset_index()
    tabla[["max_dbid_dmean", "min_dbid_dmean"]] *= 100

    # --- % de días con cobertura por bucket ---
    total_dias = len(todos_dias)
    cobertura = (grouped[grouped["n_contracts"] > 0]
        .groupby("moneyness_bucket")["Date"]
        .nunique()
        .reset_index()
        .rename(columns={"Date": "dias_con_datos"})
    )
    cobertura["pct_cobertura"] = cobertura["dias_con_datos"] / total_dias * 100
    tabla = tabla.merge(cobertura, on="moneyness_bucket", how="left").fillna(0)

    print(f"\n=== Vencimiento ({v_min}, {v_max}] días ===")
    print(tabulate(tabla, headers="keys", tablefmt="rounded_outline", floatfmt=".3f", showindex=False))

    # --- Continuidad por bucket ---
    fechas_ordenadas = pd.Series(sorted(todos_dias))
    resultados_cont = []
    for bucket, grupo in grouped.groupby("moneyness_bucket"):
        fechas_con_datos = set(grupo[grupo["n_contracts"] > 0]["Date"])
        serie = fechas_ordenadas.isin(fechas_con_datos).astype(int).values
        rachas = []
        racha_actual = 0
        for v in serie:
            if v == 1:
                racha_actual += 1
            else:
                if racha_actual > 0:
                    rachas.append(racha_actual)
                racha_actual = 0
        if racha_actual > 0:
            rachas.append(racha_actual)
        resultados_cont.append({
            "moneyness_bucket": bucket,
            "racha_max":        max(rachas) if rachas else 0,
            "racha_min":        min(rachas) if rachas else 0,
            "racha_media":      np.mean(rachas) if rachas else 0.0,
            "n_rachas":         len(rachas)
        })
    continuidad = pd.DataFrame(resultados_cont)
    print(f"\n=== Continuidad por bucket — ({v_min}, {v_max}] días ===")
    print(tabulate(continuidad, headers="keys", tablefmt="rounded_outline", floatfmt=".1f", showindex=False))

    # --- Gráfico de cobertura ---
    data_bid = data[data["Bid"] >= 0]
    rango = (data_bid
        .groupby(["moneyness_bucket", "Date"]).size()
        .reset_index(name="n")
        .groupby("Date")["moneyness_bucket"]
        .agg(m_min="min", m_max="max", n_buckets="count")
        .reset_index()
    )
    rango["m_min"] = rango["m_min"].apply(lambda x: x.right)
    rango["m_max"] = rango["m_max"].apply(lambda x: x.right)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    axes[0].plot(rango["Date"], rango["m_min"])
    axes[1].plot(rango["Date"], rango["m_max"])
    axes[2].plot(rango["Date"], rango["n_buckets"], color="green")
    for ax, title in zip(axes, ["Bucket mínimo cubierto", "Bucket máximo cubierto", "Nº buckets con contratos"]):
        ax.set_title(f"[{v_min},{v_max}] — {title}")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()

    return tabla, rango, cobertura, continuidad


# --- Ejecución ---
tramos = [(0, 15), (15, 45), (45, 105), (105, 183), (183, 365)]

resultados = {}
for v_min, v_max in tramos:
    tabla, rango, cobertura, continuidad = analisis_vencimiento(opt_df, v_min, v_max)
    resultados[f"{v_min}_{v_max}"] = {
        "tabla": tabla, "rango": rango, 
        "cobertura": cobertura, "continuidad": continuidad
    }

# In[]:
print("\n\n=== Resumen de resultados por tramo de vencimiento con filtro BID>0 ===")
# --- Ejecución ---

opt_df2 = opt_df[opt_df["Bid"] > 0]
resultados = {}
for v_min, v_max in tramos:
    tabla, rango, cobertura, continuidad = analisis_vencimiento(opt_df2, v_min, v_max)
    resultados[f"{v_min}_{v_max}"] = {
        "tabla": tabla, "rango": rango, 
        "cobertura": cobertura, "continuidad": continuidad
    }



# In[]:

# Para los datos filtrados con el Bid>0, y tramo 15-45 días.

data_15_45 = opt_df[(opt_df["maturity_bucket"] == pd.Interval(15, 45, closed="right")) & (opt_df["Bid"] > 0)]


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

def gamma_empirica_op3(delta_op2):
    df = delta_op2.copy()

    spot = data_15_45[["Date", "SpotPrice"]].drop_duplicates("Date")
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

def diagnostico_greek(df, greek_col, wrong_sign_col, outlier_col, nombre):
    # Renombra internamente para no depender del nombre de la columna
    df_valid = df[df[greek_col].notna()].copy()
    df_ws    = df_valid[df_valid[wrong_sign_col] | df_valid[outlier_col]]

    print(f"\n{'='*60}")
    print(f"DIAGNÓSTICO — {nombre}")
    print(f"{'='*60}")
    print(f"Observaciones con greek calculada : {len(df_valid):>10,}")
    print(f"Signo incorrecto / negativo       : {df_valid[wrong_sign_col].sum():>10,} ({df_valid[wrong_sign_col].mean()*100:.2f}%)")
    print(f"Outliers                          : {df_valid[outlier_col].sum():>10,} ({df_valid[outlier_col].mean()*100:.2f}%)")

    if df_ws.empty:
        print("Sin observaciones problemáticas.")
        return

    print(f"\n--- Por CallPut ---")
    print(tabulate(
        df_ws.groupby("CallPut").agg(
            n_problemas = (greek_col, "count"),
            greek_media = (greek_col, "mean"),
            greek_min   = (greek_col, "min"),
            greek_max   = (greek_col, "max"),
        ).reset_index(),
        headers="keys", tablefmt="rounded_outline", floatfmt=".3f", showindex=False
    ))

    print(f"\n--- Por bucket de moneyness ---")
    print(tabulate(
        df_ws.groupby(["CallPut", "moneyness_bucket"]).agg(
            n_problemas = (greek_col, "count"),
            greek_media = (greek_col, "mean"),
            oi_media    = ("OpenInterest", "mean"),
            oi_mediana  = ("OpenInterest", "median"),
        ).reset_index(),
        headers="keys", tablefmt="rounded_outline", floatfmt=".3f", showindex=False
    ))

    print(f"\n--- Por cuartil de OI (excluyendo OI=0) ---")
    df_ws_oi = df_ws[df_ws["OpenInterest"] > 0].copy()
    df_ws_oi["oi_cuartil"] = pd.qcut(df_ws_oi["OpenInterest"], q=4, labels=["Q1","Q2","Q3","Q4"])
    print(tabulate(
        df_ws_oi.groupby(["CallPut", "oi_cuartil"]).agg(
            n_problemas = (greek_col, "count"),
            greek_media = (greek_col, "mean"),
            oi_media    = ("OpenInterest", "mean"),
        ).reset_index(),
        headers="keys", tablefmt="rounded_outline", floatfmt=".3f", showindex=False
    ))

    print(f"\n--- Días sin ninguna greek válida ---")
    dias_con_alguna  = df_valid["Date"].nunique()
    dias_sin_ninguna = df["Date"].nunique() - dias_con_alguna
    print(f"Total días en el tramo            : {df['Date'].nunique():>10,}")
    print(f"Días con al menos una greek válida: {dias_con_alguna:>10,}")
    print(f"Días sin ninguna greek válida     : {dias_sin_ninguna:>10,} ({dias_sin_ninguna/df['Date'].nunique()*100:.2f}%)")

    print(f"\n--- Días sin greek válida por CallPut ---")
    resumen = []
    for cp, grupo in df.groupby("CallPut"):
        total_dias_cp     = grupo["Date"].nunique()
        dias_con_greek_cp = grupo[grupo[greek_col].notna()]["Date"].nunique()
        dias_sin_greek_cp = total_dias_cp - dias_con_greek_cp
        resumen.append({
            "CallPut":          cp,
            "dias_total":       total_dias_cp,
            "dias_con_greek":   dias_con_greek_cp,
            "dias_sin_greek":   dias_sin_greek_cp,
            "pct_sin_greek":    dias_sin_greek_cp / total_dias_cp * 100
        })
    print(tabulate(pd.DataFrame(resumen), headers="keys",
                   tablefmt="rounded_outline", floatfmt=".2f", showindex=False))

    print(f"\n--- Fechas sin ninguna greek válida ---")
    fechas_sin_greek = (df.groupby("Date")[greek_col]
        .apply(lambda x: x.notna().sum() == 0)
    )
    fechas_sin_greek = fechas_sin_greek[fechas_sin_greek].index.tolist()
    if fechas_sin_greek:
        print(pd.Series(fechas_sin_greek).to_string())
    else:
        print("Ningún día sin greek válida.")

    print(f"\n--- Días sin greek válida por bucket de moneyness ---")
    resumen_bucket = []
    for (cp, bucket), grupo in df.groupby(["CallPut", "moneyness_bucket"]):
        total_dias_b      = grupo["Date"].nunique()
        dias_con_greek_b  = grupo[grupo[greek_col].notna()]["Date"].nunique()
        dias_sin_greek_b  = total_dias_b - dias_con_greek_b
        resumen_bucket.append({
            "CallPut":          cp,
            "moneyness_bucket": bucket,
            "dias_total":       total_dias_b,
            "dias_con_greek":   dias_con_greek_b,
            "dias_sin_greek":   dias_sin_greek_b,
            "pct_sin_greek":    dias_sin_greek_b / total_dias_b * 100 if total_dias_b > 0 else np.nan
        })
    print(tabulate(
        pd.DataFrame(resumen_bucket).sort_values(["CallPut", "pct_sin_greek"], ascending=[True, False]),
        headers="keys", tablefmt="rounded_outline", floatfmt=".2f", showindex=False
    ))

# ============================================================
# EJECUCIÓN DELTA Y GAMMA EMPÍRICAS
# ============================================================

delta_1 = delta_empirica_op1(data_15_45)
delta_2 = delta_empirica_op2(delta_1)
delta_3 = delta_empirica_op3(data_15_45)

gamma_1 = gamma_empirica_op1(delta_1)
gamma_2 = gamma_empirica_op2(gamma_1)
gamma_3 = gamma_empirica_op3(delta_2)

# ============================================================
# MERGE EN data_15_45
# ============================================================

data_15_45 = data_15_45.merge(
    delta_1[["Date", "OptionID", "delta_emp"]],
    on=["Date", "OptionID"],
    how="left"
)

data_15_45 = data_15_45.merge(
    delta_2[["Date", "CallPut", "moneyness_bucket", "delta_emp_bucket"]].rename(columns={"delta_emp_bucket": "delta_emp_op2"}),
    on=["Date", "CallPut", "moneyness_bucket"],
    how="left"
)

data_15_45 = data_15_45.merge(
    delta_3[["Date", "CallPut", "moneyness_bucket", "delta_emp_bucket"]].rename(columns={"delta_emp_bucket": "delta_emp_op3"}),
    on=["Date", "CallPut", "moneyness_bucket"],
    how="left"
)

data_15_45 = data_15_45.merge(
    gamma_1[["Date", "OptionID", "gamma_emp"]],
    on=["Date", "OptionID"],
    how="left"
)

data_15_45 = data_15_45.merge(
    gamma_2[["Date", "CallPut", "moneyness_bucket", "gamma_emp_bucket"]].rename(columns={"gamma_emp_bucket": "gamma_emp_op2"}),
    on=["Date", "CallPut", "moneyness_bucket"],
    how="left"
)

data_15_45 = data_15_45.merge(
    gamma_3[["Date", "CallPut", "moneyness_bucket", "gamma_emp_bucket"]].rename(columns={"gamma_emp_bucket": "gamma_emp_op3"}),
    on=["Date", "CallPut", "moneyness_bucket"],
    how="left"
)



# ============================================================
# DIAGNÓSTICO DELTA
# ============================================================

df_diag = data_15_45.copy()

df_diag["delta_sign_teorico"] = df_diag["CallPut"].map({"C": 1, "P": -1})

for col, flag in [
    ("delta_emp",     "wrong_sign_op1"),
    ("delta_emp_op2", "wrong_sign_op2"),
    ("delta_emp_op3", "wrong_sign_op3"),
]:
    df_diag[flag] = (
        df_diag[col].notna() &
        (np.sign(df_diag[col]) != df_diag["delta_sign_teorico"])
    )

for col, flag in [
    ("delta_emp",     "outlier_op1"),
    ("delta_emp_op2", "outlier_op2"),
    ("delta_emp_op3", "outlier_op3"),
]:
    df_diag[flag] = df_diag[col].notna() & (df_diag[col].abs() > 1)
# EJECUCIÓN DELTA
diagnostico_greek(df_diag, "delta_emp",     "wrong_sign_op1", "outlier_op1", "Delta Opción 1 — contrato individual")
diagnostico_greek(df_diag, "delta_emp_op2", "wrong_sign_op2", "outlier_op2", "Delta Opción 2 — agregada por bucket")
diagnostico_greek(df_diag, "delta_emp_op3", "wrong_sign_op3", "outlier_op3", "Delta Opción 3 — precio agregado")

# ============================================================
# DIAGNÓSTICO GAMMA
# ============================================================

df_diag_gamma = data_15_45.copy()

for col, flag in [
    ("gamma_emp",     "wrong_sign_gop1"),
    ("gamma_emp_op2", "wrong_sign_gop2"),
    ("gamma_emp_op3", "wrong_sign_gop3"),
]:
    df_diag_gamma[flag] = (
        df_diag_gamma[col].notna() &
        (df_diag_gamma[col] < 0)
    )

for col, flag in [
    ("gamma_emp",     "outlier_gop1"),
    ("gamma_emp_op2", "outlier_gop2"),
    ("gamma_emp_op3", "outlier_gop3"),
]:
    umbral = df_diag_gamma[col].abs().quantile(0.99)
    df_diag_gamma[flag] = df_diag_gamma[col].notna() & (df_diag_gamma[col].abs() > umbral)


# EJECUCIÓN GAMMA
diagnostico_greek(df_diag_gamma, "gamma_emp",     "wrong_sign_gop1", "outlier_gop1", "Gamma Opción 1 — contrato individual")
diagnostico_greek(df_diag_gamma, "gamma_emp_op2", "wrong_sign_gop2", "outlier_gop2", "Gamma Opción 2 — agregada por bucket")
diagnostico_greek(df_diag_gamma, "gamma_emp_op3", "wrong_sign_gop3", "outlier_gop3", "Gamma Opción 3 — delta agregada")
# %%
