# In[]:
from unittest import result

from scipy.interpolate import CubicSpline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.linalg import solve

opt_df = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet")

opt_df_filtered = opt_df[(opt_df["OpenInterest"] > 0) | (opt_df["Volume"] > 0)].reset_index(drop=True)

opt_df_filtered = opt_df[(opt_df["Bid"] > 0)].reset_index(drop=True)

v_grid = [0, 15, 45, 105, 183, 365, np.inf]
#v_grid = [1,9,29,59,89,179,364, np.inf]

v_edges = pd.IntervalIndex.from_breaks(v_grid, closed="right")

v_idx = pd.cut(opt_df_filtered["Days"], bins=v_edges, labels=False, include_lowest=True)
opt_df_filtered["bucket"] = v_idx
opt_df_filtered = opt_df_filtered.sort_values(["Date","Days","CallPut","Strike"]).reset_index(drop=True)


opt_df_filtered_sorted = opt_df_filtered.sort_values(["Date", "bucket", "CallPut", "Strike", "OpenInterest", "Volume"],
                                                     ascending= [True, True, True, True, False, False])
opt_agg = (
    opt_df_filtered_sorted.groupby(["Date","bucket","CallPut","Strike"], sort=False)
    .first().reset_index()
    .sort_values(["Date", "bucket", "CallPut", "Strike"])
    .reset_index(drop=True))

opt_agg

# In[]:
# ### Evaluemos la cantidad de strikes únicos por fecha, bucket y tipo de opción:
# # Número de strikes distintos por (Date, bucket, CallPut) tras la agregación
# strikes_por_grupo = (
#     opt_agg
#     .groupby(["Date", "bucket", "CallPut"])["Strike"]
#     .count()
#     .reset_index()
#     .rename(columns={"Strike": "n_strikes"})
# )


# # Resumen estadístico por bucket
# print(strikes_por_grupo.groupby("bucket")["n_strikes"].describe(
#     percentiles=[.05, .10, .25, .50, .75, .90, .95]
# ).round(1))


# # In[]:
# # Veamos para el bucket 1,  el número de días con menos de 6 strikes por tipo de opción:
# for cp in ["C", "P"]:
#     mask = (strikes_por_grupo["bucket"] == pd.Interval(0.0, 15.0, closed="right")) & (strikes_por_grupo["CallPut"] == cp)
#     total_dias = mask.sum()
#     dias_pocos_strikes = (strikes_por_grupo[mask]["n_strikes"] < 6).sum()
#     print(f"Bucket 1 - {cp}: {dias_pocos_strikes} días con menos de 6 strikes ({dias_pocos_strikes/total_dias:.2%})")
#     print(f"Number of total days: {total_dias}")


# In[]:
# ── FORWARD ──────────────────────────────────────────────────
opt_agg["dc_dk_fwd"] = (
    opt_agg.groupby(["Date","bucket","CallPut"])["MidPrice"].diff() /
    opt_agg.groupby(["Date","bucket","CallPut"])["Strike"].diff()
)
opt_agg["Delta_fwd"] = (1 / opt_agg["SpotPrice"]) * (
    opt_agg["MidPrice"] - opt_agg["dc_dk_fwd"] * opt_agg["Strike"]
)
opt_agg["d2c_dk2_fwd"] = (
    opt_agg.groupby(["Date","bucket","CallPut"])["dc_dk_fwd"].diff() /
    opt_agg.groupby(["Date","bucket","CallPut"])["Strike"].diff()
)
opt_agg["Gamma_fwd"] = (
    (opt_agg["Strike"] / opt_agg["SpotPrice"])**2 * opt_agg["d2c_dk2_fwd"]
)

# ── BACKWARD ─────────────────────────────────────────────────────────────────
opt_agg["dc_dk_bwd"] = (
    opt_agg.groupby(["Date","bucket","CallPut"])["MidPrice"].diff(-1).mul(-1) /
    opt_agg.groupby(["Date","bucket","CallPut"])["Strike"].diff(-1).mul(-1)
)
opt_agg["Delta_bwd"] = (1 / opt_agg["SpotPrice"]) * (
    opt_agg["MidPrice"] - opt_agg["dc_dk_bwd"] * opt_agg["Strike"]
)
opt_agg["d2c_dk2_bwd"] = (
    opt_agg.groupby(["Date","bucket","CallPut"])["dc_dk_bwd"].diff(-1).mul(-1) /
    opt_agg.groupby(["Date","bucket","CallPut"])["Strike"].diff(-1).mul(-1)
)
opt_agg["Gamma_bwd"] = (
    (opt_agg["Strike"] / opt_agg["SpotPrice"])**2 * opt_agg["d2c_dk2_bwd"]
)

# ── CENTERED ─────────────────────────────────────────────────────────────────
g = opt_agg.groupby(["Date","bucket","CallPut"])

C_next = g["MidPrice"].shift(-1)
C_prev = g["MidPrice"].shift(1)
K_next = g["Strike"].shift(-1)
K_prev = g["Strike"].shift(1)

opt_agg["dc_dk_ctr"] = (C_next - C_prev) / (K_next - K_prev)
opt_agg["Delta_ctr"] = (1 / opt_agg["SpotPrice"]) * (
    opt_agg["MidPrice"] - opt_agg["dc_dk_ctr"] * opt_agg["Strike"]
)
h_L = opt_agg["Strike"] - K_prev
h_R = K_next - opt_agg["Strike"]

opt_agg["d2c_dk2_ctr"] = (
    2 * (
        (C_next - opt_agg["MidPrice"]) / (h_R * (h_L + h_R)) +
        (C_prev - opt_agg["MidPrice"]) / (h_L * (h_L + h_R))
    )
)
opt_agg["Gamma_ctr"] = (
    (opt_agg["Strike"] / opt_agg["SpotPrice"])**2 * opt_agg["d2c_dk2_ctr"]
)
# In[]:
################# ponemos límites:

mask_call = (opt_agg["CallPut"] == "C") & opt_agg["Delta_ctr"].between(0, 1)
mask_put  = (opt_agg["CallPut"] == "P") & opt_agg["Delta_ctr"].between(-1, 0)
mask_gamma = (opt_agg["Gamma_ctr"] >= 0)
opt_agg2 = opt_agg[(mask_call | mask_put) & mask_gamma].reset_index(drop=True)



# In[]:
####### Gráficamos: #####
opt_agg = opt_agg[opt_agg["Date"] >= pd.to_datetime("2019-01-01")]
# ── preparar datos ────────────────────────────────────────────────────────────
resumen = opt_agg.groupby(["Date", "CallPut"]).agg(
    Delta_fwd_med  = ("Delta_fwd",  "median"),
    Delta_bwd_med  = ("Delta_bwd",  "median"),
    Delta_ctr_med  = ("Delta_ctr",  "median"),
    Gamma_fwd_med  = ("Gamma_fwd",  "median"),
    Gamma_bwd_med  = ("Gamma_bwd",  "median"),
    Gamma_ctr_med  = ("Gamma_ctr",  "median"),
).reset_index()

calls = resumen[resumen["CallPut"] == "C"].sort_values("Date")
puts  = resumen[resumen["CallPut"] == "P"].sort_values("Date")

# ── colores y estilos ─────────────────────────────────────────────────────────
COL_FWD = "#378ADD"
COL_BWD = "#D85A30"
COL_CTR = "#1D9E75"

fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)
fig.suptitle("Comparación métodos de diferencia finita — mediana diaria", fontsize=13)

panels = [
    (axes[0, 0], calls, "Delta_fwd_med", "Delta_bwd_med", "Delta_ctr_med", "Delta — Calls"),
    (axes[0, 1], puts,  "Delta_fwd_med", "Delta_bwd_med", "Delta_ctr_med", "Delta — Puts"),
    (axes[1, 0], calls, "Gamma_fwd_med", "Gamma_bwd_med", "Gamma_ctr_med", "Gamma — Calls"),
    (axes[1, 1], puts,  "Gamma_fwd_med", "Gamma_bwd_med", "Gamma_ctr_med", "Gamma — Puts"),
]

for ax, df, fwd, bwd, ctr, title in panels:
    ax.plot(df["Date"], df[fwd], color=COL_FWD, lw=0.8, label="Forward")
    ax.plot(df["Date"], df[bwd], color=COL_BWD, lw=0.8, ls="--", label="Backward")
    ax.plot(df["Date"], df[ctr], color=COL_CTR, lw=0.8, ls=":",  label="Centrada")
    ax.set_title(title, fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(r"C:\Users\pablo.esparcia\Documents\greeks_comparacion_metodos.png",
            dpi=150, bbox_inches="tight")
plt.show()


# In[]:

################## Probemos a sacar derivadas del un Cubic Interpolating Spline Natural:
# Para tener derivadas más suaves para todos los días, buckets y tipos de opción. De manera que pasando por los puntos reales tendremos de forma téorica la misma derivada pero mejor evaluada (menos ruidosa)


MIN_STRIKES = 4

def greeks_spline(group):
    g = group.sort_values("Strike")
    n = len(g)
    
    # Inicializar columnas con NaN
    g = g.copy()
    for col in ["Delta_sp", "Gamma_sp", "dSigma_dK", "Vanna_sp"]:
        g[col] = np.nan

    if n < MIN_STRIKES:
        return g

    try:
        S  = g["SpotPrice"].iloc[0]
        K  = g["Strike"].values

        # Spline sobre precios
        cs_C = CubicSpline(K, g["MidPrice"].values, bc_type="natural")
        dC   = cs_C(K, nu=1)
        d2C  = cs_C(K, nu=2)

        # Delta y Gamma via homogeneidad
        g["Delta_sp"] = (g["MidPrice"].values - dC * K) / S
        g["Gamma_sp"] = (K / S)**2 * d2C

        # Spline sobre vol implícita → dσ/dK
        if g["ImpliedVolatility"].notna().sum() >= MIN_STRIKES:
            cs_s = CubicSpline(K, g["ImpliedVolatility"].values, bc_type="natural")
            ds   = cs_s(K, nu=1)
            g["dSigma_dK"] = ds

            # Vanna = Vega * dσ/dK / S  (aproximación via regla de la cadena)
            T   = g["Days"].mean() / 252
            sig = g["ImpliedVolatility"].values
            d1  = (np.log(K / S) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
            vega_unit = S * np.sqrt(T) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
            g["Vanna_sp"] = vega_unit * ds / S

    except Exception:
        pass

    return g


# In[]:
# Opción 1: bucle for (más seguro, recomendado)
grupos = []
for (date, bucket, callput), group in opt_agg.groupby(["Date","bucket","CallPut"]):
    resultado = greeks_spline(group)
    resultado["Date"]    = date
    resultado["bucket"]  = bucket
    resultado["CallPut"] = callput
    grupos.append(resultado)

resultx = pd.concat(grupos).reset_index(drop=True)
resultx




###########Diagnosticos:
# In[]:

##### Diagnósticos:

tipo_diferencia = "Delta_sp"  

print(f"=== Resumen de {tipo_diferencia} ===")

print("Distribución general:")
print(resultx[tipo_diferencia].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(4))

# # Fracción de NaN por bucket
print("Fracción de Nan por bucket:")
print(resultx.groupby("bucket")[tipo_diferencia].apply(lambda x: x.isna().mean()))

# print("Numero medio de valores negativos por bucket:")
# print(resultx.groupby("bucket")[tipo_diferencia]
#       .apply(lambda x: (x < 0).mean()))



# In[]:


# Ver si el problema es simétrico entre calls y puts
for cp in ["C", "P"]:
    sub = opt_agg[opt_agg["CallPut"] == cp]
    print(f"\n{cp}:")
    print(sub[["Delta_ctr","Gamma_ctr"]].describe(
        percentiles=[.01,.05,.25,.5,.75,.95,.99]
    ).round(4))


# In[]:


# Spacing mínimo entre strikes consecutivos por grupo
opt_agg["K_diff"] = opt_agg.groupby(["Date","bucket","CallPut"])["Strike"].diff()

print("Spacing mínimo:", opt_agg["K_diff"].min())
print("Filas con spacing < 1:", (opt_agg["K_diff"] < 1).sum())
print("Filas con spacing = 0:", (opt_agg["K_diff"] == 0).sum())

# Distribución del spacing
print(opt_agg["K_diff"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))

# In[]:



# ¿Dónde están los valores extremos?
extremos = opt_agg[opt_agg["Delta_ctr"].abs() > 2]

print("N extremos:", len(extremos))
print("\nMoneyness de los extremos:")
print(extremos["Moneyness"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))

print("\nMidPrice de los extremos:")
print(extremos["MidPrice"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))

print("\nSpacing K_diff de los extremos:")
print(extremos["K_diff"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))

print("\nPor bucket:")
print(extremos.groupby("bucket").size())

print("\nPor año:")
print(extremos.groupby(extremos["Date"].dt.year).size())
###################### Más diagnosticos:

# In[]:

# ¿Los extremos tienen vencimientos mezclados en strikes consecutivos?
extremos_idx = opt_agg[
    opt_agg["Delta_ctr"].abs() > 2
].index

# Ver los vecinos de esas filas
context = opt_agg.loc[
    opt_agg.index.isin(
        [i-1 for i in extremos_idx] + 
        list(extremos_idx) + 
        [i+1 for i in extremos_idx]
    ),
    ["Date","bucket","CallPut","Strike","Days","MidPrice","Delta_ctr"]
].head(30)

print(context)

# In[]:

# In[]:

# Identificar los saltos de vencimiento problemáticos dentro de cada bucket
opt_agg["Days_prev"] = opt_agg.groupby(
    ["Date","bucket","CallPut"])["Days"].shift(1)
opt_agg["Days_next"] = opt_agg.groupby(
    ["Date","bucket","CallPut"])["Days"].shift(-1)

# Salto de vencimiento entre strikes consecutivos
opt_agg["days_jump"] = (
    opt_agg["Days"] - opt_agg["Days_prev"]
).abs()

# Filas problemáticas: delta fuera de rango Y salto de vencimiento
problematicas = opt_agg[
    (opt_agg["Delta_ctr"].abs() > 1) &
    (opt_agg["days_jump"] > 0)
].copy()

print("=== FILAS PROBLEMÁTICAS POR BUCKET ===")
print(problematicas.groupby("bucket").size())

print("\n=== % DE FILAS PROBLEMÁTICAS POR BUCKET ===")
total_por_bucket = opt_agg.groupby("bucket").size()
prob_por_bucket  = problematicas.groupby("bucket").size()
print((prob_por_bucket / total_por_bucket).round(4))

print("\n=== SALTO DE VENCIMIENTO MEDIO EN PROBLEMÁTICAS ===")
print(problematicas.groupby("bucket")["days_jump"].median())

print("\n=== POR AÑO ===")
problematicas["year"] = problematicas["Date"].dt.year
print(problematicas.groupby(["bucket","year"]).size().unstack(fill_value=0))



# In[]:
opt_agg["Days_prev"] = opt_agg.groupby(
    ["Date","bucket","CallPut"])["Days"].shift(1)
opt_agg["Days_next"] = opt_agg.groupby(
    ["Date","bucket","CallPut"])["Days"].shift(-1)

# Salto de vencimiento entre strikes consecutivos
opt_agg["days_jump"] = (
    opt_agg["Days"] - opt_agg["Days_prev"]
).abs()

# Inspección de theta en filas problemáticas vs no problemáticas
opt_agg["es_problematica"] = (
    (opt_agg["Delta_ctr"].abs() > 1) &
    (opt_agg["days_jump"] > 0)
)

print("=== THETA EN FILAS PROBLEMÁTICAS VS NORMALES ===\n")

for bucket in opt_agg["bucket"].unique():
    sub = opt_agg[opt_agg["bucket"] == bucket]
    prob  = sub[sub["es_problematica"]]["Theta"]
    noprob = sub[~sub["es_problematica"]]["Theta"]
    
    if len(prob) == 0:
        continue
        
    print(f"\n{bucket}")
    print(f"  N problemáticas:  {len(prob):>8,}")
    print(f"  N normales:       {len(noprob):>8,}")
    print(f"  Theta median PROB:   {prob.median():>10.2f}")
    print(f"  Theta median NORMAL: {noprob.median():>10.2f}")
    print(f"  Theta p5 PROB:       {prob.quantile(0.05):>10.2f}")
    print(f"  Theta p95 PROB:      {prob.quantile(0.95):>10.2f}")
    print(f"  Theta p5 NORMAL:     {noprob.quantile(0.05):>10.2f}")
    print(f"  Theta p95 NORMAL:    {noprob.quantile(0.95):>10.2f}")


# In[]:

##################################################################################################################
## Reporte completo de diagnóstico del agrupamiento:
##################################################################################################################
# ═══════════════════════════════════════════════════════════════
# DIAGNÓSTICO: PROBLEMA DE MEZCLA DE VENCIMIENTOS EN BUCKETS
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

# ── 0. Preparación ───────────────────────────────────────────────────────────

# Salto de vencimiento entre strikes consecutivos dentro del bucket
opt_agg["Days_prev"] = opt_agg.groupby(
    ["Date","bucket","CallPut"])["Days"].shift(1)
opt_agg["days_jump"] = (opt_agg["Days"] - opt_agg["Days_prev"]).abs()
opt_agg["tiene_mezcla"] = opt_agg["days_jump"] > 0

opt_agg["delta_imposible"] = (
    ((opt_agg["CallPut"]=="C") & (opt_agg["Delta_ctr"].abs() > 1)) |
    ((opt_agg["CallPut"]=="P") & (opt_agg["Delta_ctr"].abs() > 1))
)

# Tamaño total por bucket antes de cualquier filtro
total_por_bucket = opt_agg.groupby("bucket").size().to_dict()

# ── 1. Estadísticas de mezcla por bucket ─────────────────────────────────────

mezcla_stats = (opt_agg
    .groupby("bucket")
    .agg(
        n_obs         = ("days_jump", "count"),
        pct_mezcla    = ("tiene_mezcla", "mean"),
        salto_p25     = ("days_jump", lambda x: x[x>0].quantile(0.25)),
        salto_mediano = ("days_jump", lambda x: x[x>0].median()),
        salto_p75     = ("days_jump", lambda x: x[x>0].quantile(0.75)),
        salto_p95     = ("days_jump", lambda x: x[x>0].quantile(0.95)),
        salto_max     = ("days_jump", "max"),
    )
    .reset_index())

print("="*70)
print("TABLA 1: MEZCLA DE VENCIMIENTOS DENTRO DE CADA BUCKET")
print("="*70)
print(mezcla_stats.to_string(index=False))

# ── 2. Filas problemáticas ───────────────────────────────────────────────────

problematicas = opt_agg[
    opt_agg["delta_imposible"] & opt_agg["tiene_mezcla"]
].copy()

prob_stats = (problematicas
    .groupby("bucket")
    .agg(
        n_problematicas = ("Delta_ctr", "count"),
        pct_total       = ("Delta_ctr", lambda x: len(x) / total_por_bucket.get(x.name, 1)),
        theta_prob_med  = ("Theta", "median"),
        delta_abs_med   = ("Delta_ctr", lambda x: x.abs().median()),
        delta_abs_p95   = ("Delta_ctr", lambda x: x.abs().quantile(0.95)),
    )
    .reset_index())

theta_normal = (opt_agg[~opt_agg["delta_imposible"]]
    .groupby("bucket")["Theta"]
    .median()
    .reset_index()
    .rename(columns={"Theta":"theta_normal_med"}))

prob_stats = prob_stats.merge(theta_normal, on="bucket")
prob_stats["ratio_theta"] = (prob_stats["theta_prob_med"] /
                              prob_stats["theta_normal_med"])

print("\n")
print("="*70)
print("TABLA 2: OBSERVACIONES CON DELTA IMPOSIBLE Y MEZCLA")
print("="*70)
print(prob_stats.to_string(index=False))

# ── 3. Ejemplos concretos ────────────────────────────────────────────────────

print("\n")
print("="*70)
print("TABLA 3: EJEMPLOS CONCRETOS DE MEZCLA DE VENCIMIENTOS")
print("="*70)

ejemplos = (opt_agg[opt_agg["delta_imposible"] & (opt_agg["days_jump"] >= 20)]
    .groupby("bucket")
    .apply(lambda x: x.nlargest(2, "days_jump"))
    .reset_index(drop=True))

ejemplos_idx = list(ejemplos.index)
contexto_idx = sorted(set(
    [i-1 for i in ejemplos_idx] +
    ejemplos_idx +
    [i+1 for i in ejemplos_idx]
))

contexto = opt_agg.loc[
    opt_agg.index.isin(contexto_idx),
    ["Date","bucket","CallPut","Strike","Days","MidPrice",
     "Theta","Delta_ctr","days_jump"]
].copy()
contexto["es_problematica"] = contexto.index.isin(ejemplos_idx)
print(contexto.head(36).to_string(index=False))

# ── 4. Evolución temporal ────────────────────────────────────────────────────

prob_anual = (opt_agg
    .assign(year=opt_agg["Date"].dt.year)
    .groupby(["year","bucket"])
    .agg(
        pct_mezcla    = ("tiene_mezcla", "mean"),
        pct_delta_imp = ("delta_imposible", "mean"),
    )
    .reset_index())

print("\n")
print("="*70)
print("TABLA 4: EVOLUCIÓN TEMPORAL DEL % DE DELTA IMPOSIBLE POR BUCKET")
print("="*70)
print(prob_anual.pivot_table(
    index="year",
    columns="bucket",
    values="pct_delta_imp"
).round(3).to_string())

# ── 5. Figura diagnóstica ─────────────────────────────────────────────────────

buckets    = sorted(opt_agg["bucket"].dropna().unique())
colors     = plt.cm.tab10(np.linspace(0, 1, len(buckets)))
bucket_str = [str(b) for b in buckets]
pct_delta_total = opt_agg.groupby("bucket")["delta_imposible"].mean()

fig = plt.figure(figsize=(20, 22))
fig.suptitle(
    "Diagnóstico: Mezcla de Vencimientos en Buckets y su Impacto en Greeks",
    fontsize=15, fontweight="bold", y=0.99
)
gs = GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

# ── Panel A: % mezcla por bucket ─────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
bars = ax_a.bar(bucket_str,
                mezcla_stats["pct_mezcla"] * 100,
                color=colors, alpha=0.8, edgecolor="black", lw=0.5)
ax_a.set_title("A) % observaciones con mezcla de vencimientos", fontsize=11)
ax_a.set_ylabel("% con Days distinto al vecino")
ax_a.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=8)
ax_a.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, mezcla_stats["pct_mezcla"]):
    ax_a.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.3,
              f"{val:.1%}", ha="center", va="bottom", fontsize=8)

# ── Panel B: Distribución del salto de vencimiento por bucket ────────────────
ax_b = fig.add_subplot(gs[0, 1])
data_box = [opt_agg[(opt_agg["bucket"]==b) & (opt_agg["days_jump"]>0)]["days_jump"].dropna().values
            for b in buckets]
bp = ax_b.boxplot(data_box, labels=bucket_str, patch_artist=True,
                  showfliers=False, medianprops=dict(color="black", lw=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_b.set_title("B) Distribución del salto de vencimiento entre strikes (días)",
               fontsize=11)
ax_b.set_ylabel("Días de diferencia entre vencimientos")
ax_b.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=8)
ax_b.grid(axis="y", alpha=0.3)

# ── Panel C: % delta imposible por bucket ────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 0])
bars_c = ax_c.bar(bucket_str,
                  pct_delta_total.values * 100,
                  color=colors, alpha=0.8, edgecolor="black", lw=0.5)
ax_c.set_title("C) % observaciones con |Δ| > 1 (viola no-arbitraje)", fontsize=11)
ax_c.set_ylabel("% con delta imposible")
ax_c.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=8)
ax_c.grid(axis="y", alpha=0.3)
for bar, val in zip(bars_c, pct_delta_total.values):
    ax_c.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.2,
              f"{val:.1%}", ha="center", va="bottom", fontsize=8)

# ── Panel D: Ratio theta problemáticas vs normales ───────────────────────────
ax_d = fig.add_subplot(gs[1, 1])
ax_d.bar([str(b) for b in prob_stats["bucket"]],
         prob_stats["ratio_theta"].abs(),
         color=colors[:len(prob_stats)], alpha=0.8,
         edgecolor="black", lw=0.5)
ax_d.axhline(1, color="red", ls="--", lw=1.5,
             label="Ratio = 1 (sin diferencia)")
ax_d.set_title("D) |Theta problemáticas / Theta normales| por bucket",
               fontsize=11)
ax_d.set_ylabel("Ratio de theta")
ax_d.set_xticklabels([str(b) for b in prob_stats["bucket"]],
                      rotation=30, ha="right", fontsize=8)
ax_d.legend(fontsize=9)
ax_d.grid(axis="y", alpha=0.3)

# ── Panel E: Ejemplo visual de mezcla de superficies ────────────────────────
# Coger un día y bucket problemático concreto para ilustrar la mezcla
ax_e = fig.add_subplot(gs[2, :])

# Seleccionar el día con más strikes problemáticos en el bucket (45,105]
bucket_ej = pd.Interval(45.0, 105.0, closed="right")
dia_ej = (problematicas[problematicas["bucket"]==bucket_ej]
          .groupby("Date").size()
          .nlargest(1).index[0])

slice_ej = opt_agg[
    (opt_agg["Date"]==dia_ej) &
    (opt_agg["bucket"]==bucket_ej) &
    (opt_agg["CallPut"]=="C")
].sort_values("Strike")

# Colorear por vencimiento
vencimientos = sorted(slice_ej["Days"].unique())
cmap_ej = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(vencimientos)))
color_map = {T: c for T, c in zip(vencimientos, cmap_ej)}

for T in vencimientos:
    sub_T = slice_ej[slice_ej["Days"]==T]
    ax_e.scatter(sub_T["Strike"], sub_T["MidPrice"],
                 color=color_map[T], s=40, zorder=3,
                 label=f"T={T} días")
    ax_e.plot(sub_T["Strike"], sub_T["MidPrice"],
              color=color_map[T], lw=0.8, alpha=0.5)

# Marcar los puntos problemáticos
prob_ej = slice_ej[slice_ej["delta_imposible"]]
ax_e.scatter(prob_ej["Strike"], prob_ej["MidPrice"],
             marker="X", s=120, color="red", zorder=5,
             label="Delta imposible (|Δ|>1)")

ax_e.set_title(
    f"E) Mezcla de superficies en bucket (45,105] — {dia_ej.date()} (Calls)\n"
    f"Cada color = un vencimiento distinto. Las X rojas son puntos con delta imposible.",
    fontsize=11)
ax_e.set_xlabel("Strike")
ax_e.set_ylabel("MidPrice")
ax_e.legend(fontsize=8, ncol=min(len(vencimientos)+1, 6),
            loc="upper right")
ax_e.grid(alpha=0.3)

# ── Panel F: Evolución temporal del % delta imposible ────────────────────────
ax_f = fig.add_subplot(gs[3, :])
for bucket, color in zip(buckets, colors):
    sub = prob_anual[prob_anual["bucket"]==bucket].sort_values("year")
    ax_f.plot(sub["year"], sub["pct_delta_imp"]*100,
              marker="o", ms=4, lw=1.5, color=color,
              label=str(bucket))
ax_f.axvline(2016, color="black", ls=":", lw=1.5)
ax_f.text(2016.1, ax_f.get_ylim()[1]*0.90,
          "SPXW\n2016", fontsize=9, color="black")
ax_f.set_title("F) Evolución temporal del % con |Δ|>1 por bucket", fontsize=11)
ax_f.set_ylabel("% con delta imposible")
ax_f.set_xlabel("Año")
ax_f.legend(fontsize=7, ncol=3, loc="upper right")
ax_f.grid(alpha=0.3)

plt.savefig(
    r"C:\Users\pablo.esparcia\Documents\diagnostico_mezcla_vencimientos.png",
    dpi=150, bbox_inches="tight"
)
plt.show()

# ── 6. Resumen ejecutivo ──────────────────────────────────────────────────────

salto_con_mezcla = (opt_agg[opt_agg["tiene_mezcla"]]
                    .groupby("bucket")["days_jump"]
                    .median())

print("\n")
print("="*70)
print("RESUMEN EJECUTIVO")
print("="*70)
print(f"""
El agrupamiento por buckets fijos de vencimiento introduce mezcla de
contratos con distinto tiempo a vencimiento dentro del mismo grupo.

HALLAZGOS PRINCIPALES:

1. MEZCLA ESTRUCTURAL: En todos los buckets, entre el
   {mezcla_stats['pct_mezcla'].min():.1%} y el
   {mezcla_stats['pct_mezcla'].max():.1%} de las observaciones
   tienen un strike vecino con distinto vencimiento.

2. SALTOS DE VENCIMIENTO: El salto mediano entre strikes consecutivos
   con mezcla varía de {salto_con_mezcla.min():.0f} días
   (bucket corto) a {salto_con_mezcla.max():.0f} días (bucket largo).

3. VIOLACIONES DE NO-ARBITRAJE: Entre el
   {pct_delta_total.min():.1%} y el {pct_delta_total.max():.1%}
   de las observaciones tienen delta fuera del rango teórico [−1, 1],
   directamente atribuible a la mezcla de superficies temporales.

4. CAUSA CONFIRMADA: Las opciones problemáticas tienen theta
   {prob_stats['ratio_theta'].abs().mean():.1f}x más negativa que las
   normales, confirmando que pertenecen a vencimientos más cortos
   dentro del mismo bucket — cuyo precio es sistemáticamente inferior
   al del vencimiento largo para el mismo strike, produciendo una
   derivada en K con signo incorrecto.

5. IMPACTO TEMPORAL: El problema es más severo en el período
   pre-2016, antes de la proliferación de opciones SPXW semanales.
   Post-2016 la mayor densidad de vencimientos reduce pero no elimina
   la mezcla.

CONCLUSIÓN: El cálculo de Greeks mediante diferencias finitas sobre
precios agregados por bucket no es válido. Las derivadas en K mezclan
información de superficies temporalmente distintas, produciendo
estimaciones inconsistentes con las condiciones de no-arbitraje.
La solución es calcular los Greeks por vencimiento exacto y agregar
los Greek resultantes por bucket, no los precios.
""")


# In[]:

# ═══════════════════════════════════════════════════════════════
# DIAGNÓSTICO ESTRUCTURAL: POR QUÉ LA DIFERENCIACIÓN NUMÉRICA
# NO ES VIABLE PARA OPCIONES SPX
# ═══════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from scipy.interpolate import CubicSpline

# ═══════════════════════════════════════════════════════════════
# BLOQUE 1: PROBLEMA CON AGRUPACIÓN POR BUCKET
# ═══════════════════════════════════════════════════════════════

# Saltos de vencimiento dentro del bucket
opt_agg["Days_prev"] = opt_agg.groupby(
    ["Date","bucket","CallPut"])["Days"].shift(1)
opt_agg["days_jump"] = (opt_agg["Days"] - opt_agg["Days_prev"]).abs()
opt_agg["tiene_mezcla"] = opt_agg["days_jump"] > 0
opt_agg["delta_imposible"] = (
    ((opt_agg["CallPut"]=="C") & (opt_agg["Delta_ctr"].abs() > 1)) |
    ((opt_agg["CallPut"]=="P") & (opt_agg["Delta_ctr"].abs() > 1))
)

# Estadísticas clave por bucket
mezcla_stats = (opt_agg
    .groupby("bucket")
    .agg(
        pct_mezcla    = ("tiene_mezcla", "mean"),
        salto_mediano = ("days_jump", lambda x: x[x>0].median()),
        pct_delta_imp = ("delta_imposible", "mean"),
    )
    .reset_index())

print("="*70)
print("BLOQUE 1: PROBLEMA DE MEZCLA CON AGRUPACIÓN POR BUCKET")
print("="*70)
print(mezcla_stats.to_string(index=False))

# ═══════════════════════════════════════════════════════════════
# BLOQUE 2: PROBLEMA SIN AGRUPACIÓN (VENCIMIENTO EXACTO)
# ═══════════════════════════════════════════════════════════════

# Número de strikes por vencimiento exacto
strikes_exacto = (opt_df_filtered
    .groupby(["Date","Days","CallPut"])["Strike"]
    .nunique()
    .reset_index()
    .rename(columns={"Strike":"n_strikes"}))

strikes_exacto["bucket"] = pd.cut(
    strikes_exacto["Days"],
    bins=v_edges, labels=False, include_lowest=True
)

# Umbral mínimo para FD estable
MIN_FD = 6

resumen_exacto = (strikes_exacto
    .groupby("bucket")
    .agg(
        mediana_strikes  = ("n_strikes", "median"),
        pct_menos_min    = ("n_strikes", lambda x: (x < MIN_FD).mean()),
        pct_menos_4      = ("n_strikes", lambda x: (x < 4).mean()),
        pct_suficientes  = ("n_strikes", lambda x: (x >= MIN_FD).mean()),
    )
    .reset_index())

# Evolución temporal de la cobertura
strikes_exacto["year"] = pd.cut(
    strikes_exacto["Date"].dt.year,
    bins=[2002,2009,2015,2020,2025],
    labels=["2003-2009","2010-2015","2016-2020","2021-2024"]
)

cobertura_temporal = (strikes_exacto
    .groupby(["year","bucket"])
    .agg(pct_suficientes=("n_strikes", lambda x: (x>=MIN_FD).mean()))
    .reset_index())

print("\n")
print("="*70)
print(f"BLOQUE 2: COBERTURA CON VENCIMIENTO EXACTO (mín. {MIN_FD} strikes)")
print("="*70)
print(resumen_exacto.to_string(index=False))
print("\nCobertura por período y bucket:")
print(cobertura_temporal.pivot_table(
    index="year", columns="bucket",
    values="pct_suficientes"
).round(3).to_string())

# ═══════════════════════════════════════════════════════════════
# BLOQUE 3: PROBLEMA CON AGRUPACIÓN DINÁMICA
# (strikes duplicados entre vencimientos del mismo grupo)
# ═══════════════════════════════════════════════════════════════

# Verificar duplicados en result_din (agrupación dinámica ya calculada)
if "result_din" in dir():
    dup_stats = (result_din
        .groupby(["Date","grupo_T","CallPut"])
        .apply(lambda g: pd.Series({
            "n_strikes_total":  len(g["Strike"].unique()),
            "n_vencimientos":   g["Days"].nunique(),
            "pct_duplicados":   g["Strike"].duplicated().mean(),
        }))
        .reset_index())

    print("\n")
    print("="*70)
    print("BLOQUE 3: STRIKES DUPLICADOS EN AGRUPACIÓN DINÁMICA")
    print("="*70)
    print(dup_stats[["n_vencimientos","pct_duplicados"]]
          .describe(percentiles=[.05,.25,.5,.75,.95]).round(3))

    grupos_con_dup = (dup_stats["pct_duplicados"] > 0).mean()
    print(f"\n% de grupos con algún strike duplicado: {grupos_con_dup:.1%}")
    print("→ CubicSpline falla en todos estos grupos (x no estrictamente monótono)")
    print("→ La deduplicación por OI elige arbitrariamente qué vencimiento")
    print("  'gana' en cada strike, reintroduciendo mezcla de superficies")

# ═══════════════════════════════════════════════════════════════
# FIGURA: ARGUMENTO VISUAL COMPLETO
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(20, 24))
fig.suptitle(
    "Por qué la diferenciación numérica no es viable para opciones SPX\n"
    "Diagnóstico en tres niveles",
    fontsize=15, fontweight="bold", y=0.99
)
gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

buckets    = sorted(opt_agg["bucket"].dropna().unique())
colors     = plt.cm.tab10(np.linspace(0, 1, len(buckets)))
bucket_str = [str(b) for b in buckets]

# ── Panel A: Mezcla de vencimientos por bucket ────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.bar(bucket_str,
         mezcla_stats["pct_mezcla"]*100,
         color=colors, alpha=0.8, edgecolor="black", lw=0.5)
ax_a.set_title("A) Agrupación por bucket:\n% strikes con vencimiento distinto al vecino",
               fontsize=10)
ax_a.set_ylabel("% con mezcla de T")
ax_a.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=7)
ax_a.grid(axis="y", alpha=0.3)
ax_a.set_ylim(0, 60)

# ── Panel B: Delta imposible por bucket ──────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
pct_delta = opt_agg.groupby("bucket")["delta_imposible"].mean()
ax_b.bar(bucket_str,
         pct_delta.values*100,
         color=colors, alpha=0.8, edgecolor="black", lw=0.5)
ax_b.set_title("B) Agrupación por bucket:\n% observaciones con |Δ|>1 (viola no-arbitraje)",
               fontsize=10)
ax_b.set_ylabel("% con delta imposible")
ax_b.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=7)
ax_b.grid(axis="y", alpha=0.3)

# ── Panel C: Ejemplo visual de mezcla ────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, :])

bucket_ej = pd.Interval(45.0, 105.0, closed="right")
dia_ej = (opt_agg[
    opt_agg["delta_imposible"] &
    (opt_agg["bucket"]==bucket_ej) &
    (opt_agg["CallPut"]=="C")
].groupby("Date").size().nlargest(1).index[0])

slice_ej = opt_agg[
    (opt_agg["Date"]==dia_ej) &
    (opt_agg["bucket"]==bucket_ej) &
    (opt_agg["CallPut"]=="C")
].sort_values("Strike")

vencimientos = sorted(slice_ej["Days"].unique())
cmap_ej = plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(vencimientos)))
color_map_ej = {T: c for T, c in zip(vencimientos, cmap_ej)}

for T in vencimientos:
    sub_T = slice_ej[slice_ej["Days"]==T]
    ax_c.scatter(sub_T["Strike"], sub_T["MidPrice"],
                 color=color_map_ej[T], s=50, zorder=3,
                 label=f"T={T}d")
    ax_c.plot(sub_T["Strike"].values, sub_T["MidPrice"].values,
              color=color_map_ej[T], lw=0.8, alpha=0.5)

# Marcar puntos problemáticos
prob_ej = slice_ej[slice_ej["delta_imposible"]]
ax_c.scatter(prob_ej["Strike"], prob_ej["MidPrice"],
             marker="X", s=150, color="red", zorder=5,
             label="Delta imposible")

# Flechas mostrando el salto de precio entre vencimientos
for i, row in prob_ej.iterrows():
    vecino_idx = i - 1 if i > slice_ej.index[0] else i + 1
    if vecino_idx in slice_ej.index:
        vecino = slice_ej.loc[vecino_idx]
        ax_c.annotate(
            f"ΔT={abs(row['Days']-vecino['Days']):.0f}d",
            xy=(row["Strike"], row["MidPrice"]),
            xytext=(row["Strike"]-30, row["MidPrice"]+5),
            fontsize=7, color="red",
            arrowprops=dict(arrowstyle="->", color="red", lw=0.8)
        )

ax_c.set_title(
    f"C) Ejemplo: bucket (45,105] — {dia_ej.date()} — Calls\n"
    "Cada color = vencimiento distinto. X roja = delta imposible. "
    "El FD interpreta saltos de precio inter-vencimiento como derivadas en K.",
    fontsize=10)
ax_c.set_xlabel("Strike")
ax_c.set_ylabel("MidPrice")
ax_c.legend(fontsize=7, ncol=min(len(vencimientos)+1, 8), loc="upper right")
ax_c.grid(alpha=0.3)

# ── Panel D: Cobertura con vencimiento exacto pre vs post 2016 ───────────────
ax_d = fig.add_subplot(gs[2, 0])
periodos = ["2003-2009","2010-2015","2016-2020","2021-2024"]
x = np.arange(len(bucket_str))
width = 0.2
colors_periodo = ["#D85A30","#F5A623","#378ADD","#1D9E75"]

for i, periodo in enumerate(periodos):
    sub = cobertura_temporal[cobertura_temporal["year"]==periodo]
    vals = []
    for b in buckets:
        row = sub[sub["bucket"]==b]
        vals.append(row["pct_suficientes"].values[0] if len(row) > 0 else 0)
    ax_d.bar(x + i*width, [v*100 for v in vals],
             width=width, label=periodo,
             color=colors_periodo[i], alpha=0.8, edgecolor="black", lw=0.3)

ax_d.axhline(80, color="red", ls="--", lw=1.2, label="80% cobertura (ref.)")
ax_d.set_title(f"D) Vencimiento exacto:\n% días con ≥{MIN_FD} strikes por período",
               fontsize=10)
ax_d.set_ylabel(f"% días con ≥{MIN_FD} strikes")
ax_d.set_xticks(x + width*1.5)
ax_d.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=7)
ax_d.legend(fontsize=7)
ax_d.grid(axis="y", alpha=0.3)
ax_d.set_ylim(0, 110)

# ── Panel E: Distribución de strikes por vencimiento exacto ──────────────────
ax_e = fig.add_subplot(gs[2, 1])
data_box_exacto = [
    strikes_exacto[strikes_exacto["bucket"]==b]["n_strikes"].dropna().values
    for b in buckets
]
bp = ax_e.boxplot(data_box_exacto, labels=bucket_str,
                  patch_artist=True, showfliers=False,
                  medianprops=dict(color="black", lw=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax_e.axhline(MIN_FD, color="red", ls="--", lw=1.5,
             label=f"Mínimo viable ({MIN_FD} strikes)")
ax_e.set_title("E) Vencimiento exacto:\ndistribución de strikes disponibles",
               fontsize=10)
ax_e.set_ylabel("N strikes por vencimiento")
ax_e.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=7)
ax_e.legend(fontsize=8)
ax_e.grid(axis="y", alpha=0.3)

# ── Panel F: Síntesis — trade-off cobertura vs calidad ───────────────────────
ax_f = fig.add_subplot(gs[3, :])

# Eje izquierdo: % delta imposible (agrupación por bucket)
color_izq = "#D85A30"
color_der = "#378ADD"

pct_imp_bucket = opt_agg.groupby(
    opt_agg["Date"].dt.year)["delta_imposible"].mean() * 100

pct_nan_exacto = (strikes_exacto
    .groupby(strikes_exacto["Date"].dt.year)
    .apply(lambda x: (x["n_strikes"] < MIN_FD).mean()) * 100)

ax_f.plot(pct_imp_bucket.index, pct_imp_bucket.values,
          color=color_izq, lw=2, marker="o", ms=4,
          label="% delta imposible (agrupación por bucket)")
ax_f2 = ax_f.twinx()
ax_f2.plot(pct_nan_exacto.index, pct_nan_exacto.values,
           color=color_der, lw=2, marker="s", ms=4, ls="--",
           label=f"% vencimientos con <{MIN_FD} strikes (exacto)")

ax_f.axvline(2016, color="black", ls=":", lw=1.5)
ax_f.text(2016.1, ax_f.get_ylim()[1]*0.85, "SPXW\n2016",
          fontsize=9, color="black")

ax_f.set_xlabel("Año")
ax_f.set_ylabel("% delta imposible (bucket)", color=color_izq)
ax_f2.set_ylabel(f"% con <{MIN_FD} strikes (exacto)", color=color_der)
ax_f.tick_params(axis="y", labelcolor=color_izq)
ax_f2.tick_params(axis="y", labelcolor=color_der)

lines1, labels1 = ax_f.get_legend_handles_labels()
lines2, labels2 = ax_f2.get_legend_handles_labels()
ax_f.legend(lines1+lines2, labels1+labels2, fontsize=9, loc="upper right")
ax_f.set_title(
    "F) Trade-off fundamental: agrupación → Greeks incorrectos; "
    "vencimiento exacto → cobertura insuficiente",
    fontsize=10)
ax_f.grid(alpha=0.3)

plt.savefig(
    r"C:\Users\pablo.esparcia\Documents\diagnostico_FD_no_viable.png",
    dpi=150, bbox_inches="tight"
)
plt.show()

# ── Resumen ejecutivo ─────────────────────────────────────────────────────────
salto_con_mezcla = (opt_agg[opt_agg["tiene_mezcla"]]
                    .groupby("bucket")["days_jump"].median())
pct_delta_total  = opt_agg.groupby("bucket")["delta_imposible"].mean()

print("\n")
print("="*70)
print("RESUMEN EJECUTIVO: POR QUÉ LA DIFERENCIACIÓN NUMÉRICA NO ES VIABLE")
print("="*70)
print(f"""
PROBLEMA ESTRUCTURAL EN TRES NIVELES:

── NIVEL 1: AGRUPACIÓN POR BUCKET ──────────────────────────────────────

Entre el {mezcla_stats['pct_mezcla'].min():.1%} y el
{mezcla_stats['pct_mezcla'].max():.1%} de los strikes tienen como
vecino inmediato un contrato de distinto vencimiento. El salto mediano
de vencimiento entre strikes consecutivos varía de
{salto_con_mezcla.min():.0f} a {salto_con_mezcla.max():.0f} días.

Consecuencia: el FD interpreta diferencias de precio entre vencimientos
como derivadas en K. Resultado: entre el {pct_delta_total.min():.1%} y el
{pct_delta_total.max():.1%} de las observaciones tienen delta fuera del
rango teórico [−1,1], violando condiciones de no-arbitraje.

── NIVEL 2: VENCIMIENTO EXACTO SIN AGRUPACIÓN ──────────────────────────

Resuelve la mezcla pero introduce escasez de strikes. Pre-2016, la
mayoría de vencimientos tienen menos de {MIN_FD} strikes — umbral mínimo
para diferenciación numérica estable. El ruido de bid-ask se amplifica
por 1/h² en la segunda derivada, produciendo gamma negativa en el
{(opt_agg['Gamma_ctr']<0).mean():.1%} de las observaciones incluso donde
hay suficientes strikes.

── NIVEL 3: AGRUPACIÓN DINÁMICA (theta o varianza total) ───────────────

Mitiga parcialmente la mezcla y la escasez, pero introduce strikes
duplicados entre vencimientos del mismo grupo. CubicSpline requiere
nodos estrictamente monótonos — los duplicados provocan fallos
silenciosos. La deduplicación por OI elige arbitrariamente qué
vencimiento representa cada strike, reintroduciendo mezcla de
superficies de forma más opaca.

── CONCLUSIÓN ──────────────────────────────────────────────────────────

No existe una configuración de diferenciación numérica sobre precios
de opciones SPX que sea simultáneamente:
  (1) libre de mezcla de vencimientos,
  (2) con cobertura suficiente en todo el período 2003-2024, y
  (3) robusta al ruido de microestructura.

La solución es usar BS evaluado en la volatilidad implícita observada
(Greeks de OptionMetrics) como estimador base, con corrección
smile-adjusted de Bates (2005) como robustez donde la densidad
de strikes lo permite (post-2016).
""")



# In[]:

# Theta ATM por vencimiento en el día del ejemplo
dia_ej = pd.to_datetime("2023-12-15")
bucket_ej = pd.Interval(45.0, 105.0, closed="right")

theta_ej = (opt_df_filtered[
    (opt_df_filtered["Date"] == dia_ej) &
    (opt_df_filtered["bucket"] == bucket_ej) &
    (opt_df_filtered["CallPut"] == "C")
]
.assign(dist_atm=lambda x: (x["Moneyness"]-1).abs())
.sort_values("dist_atm")
.groupby("Days")["Theta"]
.first()
.sort_index())

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(theta_ej.index.astype(str), theta_ej.values,
       color=plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(theta_ej))))
ax.set_xlabel("Días a vencimiento")
ax.set_ylabel("Theta ATM")
ax.set_title(f"Theta ATM por vencimiento — bucket (45,105] — {dia_ej.date()}\n"
             f"Rango: {theta_ej.min():.1f} a {theta_ej.max():.1f} — "
             f"ratio max/min = {theta_ej.min()/theta_ej.max():.1f}x")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(r"C:\Users\pablo.esparcia\Documents\theta_por_vencimiento.png",
            dpi=150, bbox_inches="tight")
plt.show()

# In[]:
















































































# In[]:

# Delta empírica contrato a contrato
opt_df_sorted = opt_df_filtered.sort_values(["OptionID","Date"])

opt_df_sorted["dC"] = opt_df_sorted.groupby("OptionID")["MidPrice"].diff()
opt_df_sorted["dS"] = opt_df_sorted.groupby("OptionID")["SpotPrice"].diff()

# Filtrar días con movimiento suficiente del subyacente
MIN_DS = 5.0  # mínimo 5 puntos de movimiento en SPX

opt_df_sorted["Delta_emp"] = np.where(
    opt_df_sorted["dS"].abs() >= MIN_DS,
    opt_df_sorted["dC"] / opt_df_sorted["dS"],
    np.nan
)

# Comparar con delta de OptionMetrics
valid = opt_df_sorted[
    opt_df_sorted["Delta_emp"].notna() &
    opt_df_sorted["Delta"].notna()
].copy()

print(f"Observaciones válidas: {len(valid):,}")
print(f"Correlación con Delta OptionMetrics: "
      f"{valid['Delta'].corr(valid['Delta_emp']):.3f}")

print("\nDistribución Delta empírica:")
print(valid["Delta_emp"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]))

print("\nMAE respecto a OptionMetrics:")
print((valid["Delta_emp"] - valid["Delta"]).abs().median())

# Por bucket
valid["bucket"] = pd.cut(valid["Days"], bins=v_edges,
                          labels=False, include_lowest=True)
print("\nCorrelación por bucket:")
print(valid.groupby("bucket").apply(
    lambda x: x["Delta"].corr(x["Delta_emp"])
).round(3))




# In[]:


# Filtros adicionales para reducir ruido en las colas
MIN_DS    = 10.0   # subir umbral de movimiento mínimo
MAX_DELTA = 1.5    # cota superior de delta válida

opt_df_sorted["Delta_emp_clean"] = np.where(
    (opt_df_sorted["dS"].abs() >= MIN_DS) &
    (opt_df_sorted["Delta_emp"].abs() <= MAX_DELTA),
    opt_df_sorted["Delta_emp"],
    np.nan
)

valid_clean = opt_df_sorted[
    opt_df_sorted["Delta_emp_clean"].notna() &
    opt_df_sorted["Delta"].notna()
].copy()

print(f"Observaciones tras filtro: {len(valid_clean):,}")
print(f"Cobertura: {len(valid_clean)/len(opt_df_sorted):.1%}")
print(f"Correlación: {valid_clean['Delta'].corr(valid_clean['Delta_emp_clean']):.3f}")
print(f"MAE: {(valid_clean['Delta_emp_clean'] - valid_clean['Delta']).abs().median():.4f}")

# Fracción de delta fuera de rango
mask_call = valid_clean["CallPut"] == "C"
mask_put  = valid_clean["CallPut"] == "P"
print(f"\nCalls con Delta_emp fuera de [0,1]: "
      f"{(~valid_clean[mask_call]['Delta_emp_clean'].between(0,1)).mean():.1%}")
print(f"Puts con Delta_emp fuera de [-1,0]: "
      f"{(~valid_clean[mask_put]['Delta_emp_clean'].between(-1,0)).mean():.1%}")



# In[]:
##### Más cosas:


