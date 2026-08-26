# In[]: "Importamos los datos"
import pandas as pd
import numpy as np
import sys
from   tabulate import tabulate
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates
import re

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

if os.name == 'nt':
    #PATH_DATA = r"Y:\OUTPUTS\opt_df_empirical_greeks_sinfiltro.parquet"
    PATH_DATA = r"Y:\OUTPUTS\opt_df_prueba.parquet"
else:
    PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_empirical_greeks.parquet"

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

#Añadimos algunas variables de interés:
opt_df["Dummy_Bid"] = opt_df["Bid"] > 0
opt_df["DolarVolume"] = opt_df["Volume"] * opt_df["MidPrice"]

opt_df = opt_df[opt_df["Bid"]>0]

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


# In[]: "Recuperamos el formato de intervalos:"

 
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

# In[]: "Análisis de vencimientos y continuidad 1"

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
        max_n_contracts  = ("n_contracts",  "max"),
        min_n_contracts  = ("n_contracts",  "min"),
        max_oi_dsum      = ("oi_dsum",      "max"),
        min_oi_dsum      = ("oi_dsum",      "min"),
        max_dolvol_dsum  = ("dolvol_dsum",  "max"),
        min_dolvol_dsum  = ("dolvol_dsum",  "min"),
        # max_dbid_dmean   = ("dbid_dmean",   "max"),
        min_dbid_dmean   = ("dbid_dmean",   "min"),
        n_contracts_dsum = ("n_contracts",  "sum")
    ).reset_index()
    tabla["min_dbid_dmean"] *= 100


    # --- % de días con cobertura por bucket ---
    total_dias = len(todos_dias)
    cobertura = (grouped[grouped["n_contracts"] > 0]
        .groupby("moneyness_bucket")["Date"]
        .nunique()
        .reset_index()
        .rename(columns={"Date": "dias_con_datos"})
    )
    cobertura["pct_cobertura"] = cobertura["dias_con_datos"] / total_dias * 100
    # tabla = tabla.merge(tabla, on="moneyness_bucket", how="left").fillna(0)

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

        })
    continuidad = pd.DataFrame(resultados_cont)
    continuidad = continuidad.merge(cobertura, on="moneyness_bucket", how="left").fillna(0)

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

    data_bid = data[data["Bid"] >= 0]
    total_dias = data_bid["Date"].nunique()
    dias_en_rango = rango["Date"].nunique()
    print(total_dias, dias_en_rango, total_dias - dias_en_rango)



    return tabla, rango, cobertura, continuidad


# --- Ejecución ---
tramos = [(0, 15), (15, 45), (45, 105), (105, 183), (183, 365)]
tramos = [(15, 45)]

resultados = {}
for v_min, v_max in tramos:
    tabla, rango, cobertura, continuidad = analisis_vencimiento(opt_df, v_min, v_max)
    resultados[f"{v_min}_{v_max}"] = {
        "tabla": tabla, "rango": rango, 
        "cobertura": cobertura, "continuidad": continuidad
    }

# In[]: "Análisis de vencimientos y continuidad 1: con filtro de BID >0"

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



# In[]: Análisis de venvimientos y continuidad 2: detallado

def analisis_detallado(opt_df, v_min, v_max, subperiodos=None):
    """
    Análisis detallado para un tramo de vencimiento dado.
    
    Parámetros:
    -----------
    opt_df : DataFrame con los datos de opciones
    v_min, v_max : límites del tramo de vencimiento
    subperiodos : dict opcional {"nombre": (fecha_ini, fecha_fin)}
                  Si None, usa la muestra completa
    """
    
    
    nombre = f"{v_min}_{v_max}"
    # Cambio: filtrar por Days en lugar de por maturity_bucket
    data = opt_df[(opt_df["Days"] > v_min) & (opt_df["Days"] <= v_max)].copy()
    
    if data.empty:
        print(f"[{nombre}] Sin datos.")
        return None

    if subperiodos is None:
        subperiodos = {"Muestra completa": (data["Date"].min(), data["Date"].max())}

    # ============================================================
    # MÉTRICA 2: Dummy coexistencia Bid=0 Y Bid>0 en mismo día/bucket
    # ============================================================
    
    def calcular_metrica2(df):
        def coexiste(group):
            tiene_bid0   = (group["Bid"] == 0).any()
            tiene_bidpos = (group["Bid"] > 0).any()
            return int(tiene_bid0 and tiene_bidpos)

        coex = (df.groupby(["Date", "moneyness_bucket"])
                .apply(coexiste)
                .reset_index()
                .rename(columns={0: "coexistencia"}))
        
        resumen = (coex.groupby("moneyness_bucket")
                   .agg(dias_coexistencia = ("coexistencia", "sum"),
                        total_dias        = ("coexistencia", "count"))
                   .reset_index())
        resumen["pct_coexistencia"] = resumen["dias_coexistencia"] / resumen["total_dias"] * 100
        return resumen

    # ============================================================
    # TABLA RESUMEN POR PERIODO
    # ============================================================

    def tabla_resumen_periodo(df, label):

        todos_dias    = df["Date"].unique()
        todos_buckets = df["moneyness_bucket"].unique()
        idx_completo  = pd.MultiIndex.from_product(
            [todos_buckets, todos_dias],
            names=["moneyness_bucket", "Date"]
        )

        # Sin filtro
        grouped_all = (df
        .groupby(["moneyness_bucket", "Date"]).agg(
        n_contracts  = ("OptionID",      "count"),
        oi_dsum      = ("OpenInterest",  "sum"),
        dolvol_dsum  = ("DolarVolume",   "sum"),
        dbid_dmean   = ("Dummy_Bid",     "mean")
    )
    .reset_index()  
    .set_index(["moneyness_bucket", "Date"])
    .reindex(idx_completo)
    .fillna({"n_contracts": 0, "oi_dsum": 0,
             "dolvol_dsum": 0, "dbid_dmean": 0})
    .reset_index()
)

        # Con filtro Bid>0
        df_bid = df[df["Bid"] > 0]
        if not df_bid.empty:
            grouped_bid = (df_bid
    .groupby(["moneyness_bucket", "Date"]).agg(
        n_contracts_bid = ("OptionID", "count"),
    )
    .reset_index()  # <- añadir esto
    .set_index(["moneyness_bucket", "Date"])
    .reindex(idx_completo)
    .fillna({"n_contracts_bid": 0})
    .reset_index()
)
        else:
            grouped_bid = None

        tabla_all = grouped_all.groupby("moneyness_bucket").agg(
            max_n_contracts = ("n_contracts",  "max"),
            min_n_contracts = ("n_contracts",  "min"),
            sum_n_contracts = ("n_contracts",  "sum"),
            max_oi_dsum     = ("oi_dsum",      "max"),
            min_oi_dsum     = ("oi_dsum",      "min"),
            max_dolvol_dsum = ("dolvol_dsum",  "max"),
            min_dolvol_dsum = ("dolvol_dsum",  "min"),
            min_dbid_dmean  = ("dbid_dmean",   "min"),
        ).reset_index()
        tabla_all["min_dbid_dmean"] *= 100

        if grouped_bid is not None:
            tabla_bid = grouped_bid.groupby("moneyness_bucket").agg(
                max_n_bid = ("n_contracts_bid", "max"),
                min_n_bid = ("n_contracts_bid", "min"),
            ).reset_index()
            tabla_all = tabla_all.merge(tabla_bid, on="moneyness_bucket", how="left")

        # Métrica 2
        m2 = calcular_metrica2(df)
        tabla_all = tabla_all.merge(
            m2[["moneyness_bucket", "dias_coexistencia", "pct_coexistencia"]],
            on="moneyness_bucket", how="left"
        )

        print(f"\n=== {label} — Vencimiento ({v_min}, {v_max}] ===")
        print(tabulate(tabla_all, headers="keys", tablefmt="rounded_outline",
                       floatfmt=".2f", showindex=False))
        
        # --- % de días con cobertura por bucket ---
        total_dias = len(todos_dias)
        cobertura = (grouped_all[grouped_all["n_contracts"] > 0]
            .groupby("moneyness_bucket")["Date"]
            .nunique()
            .reset_index()
            .rename(columns={"Date": "dias_con_datos"})
        )
        cobertura["pct_cobertura"] = cobertura["dias_con_datos"] / total_dias * 100
        
        # --- Continuidad por bucket ---
        fechas_ordenadas = pd.Series(sorted(todos_dias))
        resultados_cont = []
        for bucket, grupo in grouped_all.groupby("moneyness_bucket"):
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

            })
        continuidad = pd.DataFrame(resultados_cont)
        continuidad = continuidad.merge(cobertura, on="moneyness_bucket", how="left").fillna(0)

        print(f"\n=== Continuidad por bucket — ({v_min}, {v_max}] días ===")
        print(tabulate(continuidad, headers="keys", tablefmt="rounded_outline", floatfmt=".1f", showindex=False))




        return tabla_all, cobertura, continuidad

    # ============================================================
    # GRÁFICO 1: Spread m_min / m_max unificado por periodo
    # ============================================================

    def grafico_spread_cobertura():
        fig, ax = plt.subplots(figsize=(14, 5))
        # ax2 = ax.twinx()
        colors = ["steelblue", "darkorange", "green", "firebrick"]

        for (label, (f_ini, f_fin)), color in zip(subperiodos.items(), colors):
            df_p     = data[(data["Date"] >= f_ini) & (data["Date"] <= f_fin)]
            df_bid_p = df_p[df_p["Bid"] >= 0]
            if df_bid_p.empty:
                continue

            rango = (df_bid_p
                .groupby(["moneyness_bucket", "Date"]).size()
                .reset_index(name="n")
                .groupby("Date")["moneyness_bucket"]
                .agg(m_min="min", m_max="max")
                .reset_index()
            )
            rango["m_min"] = rango["m_min"].apply(lambda x: x.right).astype(float)
            rango["m_max"] = rango["m_max"].apply(lambda x: x.right).astype(float)

            ax.fill_between(rango["Date"], rango["m_min"], rango["m_max"],
                            alpha=0.25, color=color, label=label)
            ax.plot(rango["Date"], rango["m_min"], color=color, linewidth=0.6)
            ax.plot(rango["Date"], rango["m_max"], color=color, linewidth=0.6)



        # ax.set_title(f"Spread cobertura moneyness — [{v_min},{v_max}] días")
        ax.set_ylabel("Moneyness (right side of the interval)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def grafico_spread():
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ["steelblue", "darkorange", "green", "firebrick"]

        for (label, (f_ini, f_fin)), color in zip(subperiodos.items(), colors):
            df_p     = data[(data["Date"] >= f_ini) & (data["Date"] <= f_fin)]
            df_bid_p = df_p[df_p["Bid"] >= 0]
            if df_bid_p.empty:
                continue    

            rango = (df_bid_p
                .groupby(["moneyness_bucket", "Date"]).size()
                .reset_index(name="n")
                .groupby("Date")["moneyness_bucket"]
                .agg(m_min="min", m_max="max")
                .reset_index()
            )
            rango["m_min"] = rango["m_min"].apply(lambda x: x.right).astype(float)
            rango["m_max"] = rango["m_max"].apply(lambda x: x.right).astype(float)
            rango["amplitud"] = rango["m_max"] - rango["m_min"]


            ax.plot(rango["Date"], rango["amplitud"], color=color,
                     linewidth=1.2, linestyle="--", alpha=0.7)

        # ax.set_title(f"Spread cobertura moneyness — [{v_min},{v_max}] días")
        ax.set_ylabel("Spread (max − min moneyness)")
        #ax.tick_params(axis="y", labelcolor="dimgray")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


    # ============================================================
    # GRÁFICO 2: Estadísticos serie temporal por bucket
    # ============================================================

    # def grafico_series_bucket(df_p, label):
    #     serie_bucket = (df_p
    #         .groupby(["Date", "moneyness_bucket"]).agg(
    #             n_total = ("OptionID",  "count"),
    #             n_bid   = ("Dummy_Bid", "sum")
    #         )
    #         .reset_index()
    #     )

    #     buckets = sorted(serie_bucket["moneyness_bucket"].unique(),
    #                      key=lambda x: x.right)

    #     print(f"\n=== Estadísticos serie temporal por bucket — {label} ===")
    #     stats_rows = []
    #     for b in buckets:
    #         sub = serie_bucket[serie_bucket["moneyness_bucket"] == b]
    #         stats_rows.append({
    #             "bucket":        str(b),
    #             "n_total_media": sub["n_total"].mean(),
    #             "n_total_min":   sub["n_total"].min(),
    #             "n_total_max":   sub["n_total"].max(),
    #             "n_bid_media":   sub["n_bid"].mean(),
    #             "n_bid_min":     sub["n_bid"].min(),
    #             "n_bid_max":     sub["n_bid"].max(),
    #         })
    #     print(tabulate(pd.DataFrame(stats_rows), headers="keys",
    #                    tablefmt="rounded_outline", floatfmt=".1f", showindex=False))

    #     return serie_bucket, buckets

    # ============================================================
    # GRÁFICO 3: Ridgeline — n_contratos por bucket (rotado)
    # ============================================================

    def grafico_ridgeline(df_p, label, fill=0.88):
        serie_bucket = (df_p
            .groupby(["Date", "moneyness_bucket"])["OptionID"]
            .count()
            .reset_index()
            .rename(columns={"OptionID": "n_total"})
        )

        buckets = sorted(serie_bucket["moneyness_bucket"].unique(),
                        key=lambda x: x.right)
        fechas    = sorted(df_p["Date"].unique())
        fechas_dt = pd.to_datetime(fechas)
        x_vals    = np.arange(len(fechas))

        # Tres paneles según rango de moneyness (de mayor a menor)
        grupos  = [
            [b for b in buckets if b.right > 1.45], 
            [b for b in buckets if 0.75 < b.right <= 1.45],
            [b for b in buckets if b.right <= 0.75],
        ]
        titulos = ["Moneyness 1.5+", "Moneyness 0.8 – 1.4", "Moneyness 0.1 – 0.7"]

        # Índice global de color para mantener colores consistentes
        bucket_idx = {b: i for i, b in enumerate(buckets)}
        cmap = plt.colormaps.get_cmap("tab20").resampled(len(buckets))

        # Posiciones X con etiqueta anual
        tick_positions, tick_labels = [], []
        for año in sorted(fechas_dt.year.unique()):
            idx = np.where(fechas_dt.year == año)[0]
            if len(idx) > 0:
                tick_positions.append(idx[0])
                tick_labels.append(str(año))

        fig, axes = plt.subplots(3, 1, figsize=(16, 18), sharex=True)

        for ax, grupo_buckets, titulo in zip(axes, grupos, titulos):
            if not grupo_buckets:
                ax.set_visible(False)
                continue

            y_pos    = {b: i for i, b in enumerate(grupo_buckets)}
            max_vals = {}

            for b in grupo_buckets:
                sub = (serie_bucket[serie_bucket["moneyness_bucket"] == b]
                    .set_index("Date")[["n_total"]]
                    .reindex(fechas)
                    .fillna(0))

                max_val = sub["n_total"].max()
                max_vals[b] = int(max_val)
                n_norm = sub["n_total"] / max_val * fill if max_val > 0 else sub["n_total"] * 0

                y_base = y_pos[b]
                color  = cmap(bucket_idx[b])

                ax.fill_between(x_vals, y_base, y_base + n_norm.values,
                                alpha=0.4, color=color)
                ax.plot(x_vals, y_base + n_norm.values, color=color, linewidth=0.8)
                ax.axhline(y_base, color="gray", linewidth=0.4, linestyle="--")

            y_lim = (-0.3, len(grupo_buckets) - 1 + fill + 0.2)
            ax.set_ylim(y_lim)

            # Eje Y izquierdo — etiqueta de moneyness
            ax.set_yticks(list(y_pos.values()))
            ax.set_yticklabels([f"{b.right:.1f}" for b in grupo_buckets], fontsize=8)
            ax.set_ylabel("Moneyness", fontsize=9)

            # Eje Y derecho — 3 ticks por bucket: 0, 50%, máx
            ax2 = ax.twinx()
            ax2.set_ylim(y_lim)
            right_ticks, right_labels = [], []
            for b in grupo_buckets:
                m = max_vals[b]
                for frac in [0.0, 0.5, 1.0]:
                    right_ticks.append(y_pos[b] + frac * fill)
                    right_labels.append(f"{int(m * frac):,}" if frac > 0 else "0")
            ax2.set_yticks(right_ticks)
            ax2.set_yticklabels(right_labels, fontsize=6)
            ax2.set_ylabel("Nº Contracts", fontsize=9)

            ax.set_title(titulo, fontsize=9, loc="left", pad=3)
            ax.grid(True, alpha=0.2)

        axes[-1].set_xticks(tick_positions)
        axes[-1].set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
        axes[-1].set_xlabel("Time")

        # fig.suptitle(f"Nº contratos por bucket (ridgeline) — {label}", fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show()

    def grafico_contratos_por_bucket(df_p, label, v_min, v_max):
        """
        Serie temporal diaria de n_contratos por bucket de moneyness.
        Separado en dos paneles: buckets <= 1.0 y buckets > 1.0
        """
        serie_bucket = (df_p
            .groupby(["Date", "moneyness_bucket"])["OptionID"]
            .count()
            .reset_index()
            .rename(columns={"OptionID": "n_total"})
        )

        buckets = sorted(serie_bucket["moneyness_bucket"].unique(),
                        key=lambda x: x.right)

        buckets_bajo = [b for b in buckets if b.right <= 1.0]
        buckets_alto = [b for b in buckets if b.right > 1.0]

        cmap_bajo = plt.cm.get_cmap("Blues",  len(buckets_bajo) + 2)
        cmap_alto = plt.cm.get_cmap("Reds",   len(buckets_alto) + 2)

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # Panel inferior — buckets <= 1.0
        for i, b in enumerate(buckets_bajo):
            sub = serie_bucket[serie_bucket["moneyness_bucket"] == b]
            axes[0].plot(sub["Date"], sub["n_total"],
                        linewidth=0.7, color=cmap_bajo(i + 2),
                        label=f"{b.right:.1f}")
        axes[0].set_title(f"Nº contratos — buckets ≤ 1.0 | {label} [{v_min},{v_max}] días")
        axes[0].set_ylabel("Nº contratos")
        axes[0].legend(fontsize=7, ncol=5, loc="upper left")
        axes[0].grid(True, alpha=0.3)

        # Panel superior — buckets > 1.0
        for i, b in enumerate(buckets_alto):
            sub = serie_bucket[serie_bucket["moneyness_bucket"] == b]
            axes[1].plot(sub["Date"], sub["n_total"],
                        linewidth=0.7, color=cmap_alto(i + 2),
                        label=f"{b.right:.1f}")
        axes[1].set_title(f"Nº contratos — buckets > 1.0 | {label} [{v_min},{v_max}] días")
        axes[1].set_ylabel("Nº contratos")
        axes[1].legend(fontsize=7, ncol=5, loc="upper left")
        axes[1].grid(True, alpha=0.3)

        axes[1].xaxis.set(major_locator=mdates.YearLocator(2),
                        major_formatter=mdates.DateFormatter("%Y"))
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

        fig.suptitle(f"Series temporales de contratos por bucket — {label}", fontsize=13)
        plt.tight_layout()
        plt.show()
        # ============================================================
        # EJECUCIÓN COMPLETA
        # ============================================================

    tablas_periodo = {}

    for label, (f_ini, f_fin) in subperiodos.items():
        df_p = data[(data["Date"] >= f_ini) & (data["Date"] <= f_fin)].copy()
        if df_p.empty:
            print(f"[{label}] Sin datos.")
            continue
        t, g, h = tabla_resumen_periodo(df_p, label)
        tablas_periodo[label] = t

    grafico_spread_cobertura()
    grafico_spread()

    for label, (f_ini, f_fin) in subperiodos.items():
        df_p = data[(data["Date"] >= f_ini) & (data["Date"] <= f_fin)].copy()
        if df_p.empty:
            continue
             # grafico_series_bucket(df_p, label)
        grafico_ridgeline(df_p, label)
               # grafico_contratos_por_bucket(df_p, label, v_min, v_max)
 
    return tablas_periodo


# ============================================================
# EJECUCIÓN
# ============================================================

"""
subperiodos = {
    "2003-2008": (pd.Timestamp("2003-01-01"), pd.Timestamp("2008-12-31")),
    "2008-2015": (pd.Timestamp("2008-01-01"), pd.Timestamp("2015-12-31")),
    "2016-2024": (pd.Timestamp("2016-01-01"), pd.Timestamp("2024-12-31")),
    "Completo":  (pd.Timestamp("2003-01-01"), pd.Timestamp("2024-12-31")),
}

# Tramo completo 15-45
resultados_detallado = analisis_detallado(opt_df, 15, 45, subperiodos=subperiodos)

# # Subtramos de días
resultados_15_29 = analisis_detallado(opt_df, 15, 29, subperiodos=subperiodos)
resultados_30_45 = analisis_detallado(opt_df, 30, 45, subperiodos=subperiodos)
"""

print("========================= Resultados para CALLS =========================")

# opt_df_C = opt_df[opt_df["CallPut"] == "C"]
# resultados_15_45_C = analisis_detallado(opt_df_C, 15, 45, subperiodos=subperiodos)

print("========================= Resultados para PUTS =========================")
# %%


subperiodos = {
    "Completo":  (pd.Timestamp("2003-01-01"), pd.Timestamp("2024-12-31"))
    }

opt_df_filter = opt_df[opt_df["OpenInterest"] > 0]

# opt_df_P = opt_df[opt_df["CallPut"] == "P"]
resultados_15_45_P = analisis_detallado(opt_df_filter, 15, 45, subperiodos=subperiodos)




# In[]:     

"""
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
    
    # Total de puntos no-NaN por (CallPut, moneyness_bucket) en el universo válido
    total_validos = (df_valid
        .groupby(["CallPut", "moneyness_bucket"])[greek_col]
        .count()
        .reset_index()
        .rename(columns={greek_col: "n_validos"})
    )
    
    tabla_bucket = (df_ws
        .groupby(["CallPut", "moneyness_bucket"]).agg(
            n_problemas = (greek_col, "count"),
            greek_media = (greek_col, "mean"),
            oi_media    = ("OpenInterest", "mean"),
            oi_mediana  = ("OpenInterest", "median"),
        ).reset_index()
        .merge(total_validos, on=["CallPut", "moneyness_bucket"], how="left")
    )
    
    tabla_bucket["pct_problemas"] = tabla_bucket["n_problemas"] / tabla_bucket["n_validos"] * 100
    
    print(tabulate(
        tabla_bucket,
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
        df_diag[col].notna() &  (np.sign(df_diag[col]) != df_diag["delta_sign_teorico"])
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
"""

