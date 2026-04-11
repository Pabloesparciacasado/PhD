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

#opt_df_filtered = opt_df[(opt_df["Bid"] > 0)].reset_index(drop=True)

v_grid = [0, 15, 45, 105, 183, 365, np.inf]
v_grid = [1,9,29,59,89,179,364, np.inf]

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
### Evaluemos la cantidad de strikes únicos por fecha, bucket y tipo de opción:
# Número de strikes distintos por (Date, bucket, CallPut) tras la agregación
strikes_por_grupo = (
    opt_agg
    .groupby(["Date", "bucket", "CallPut"])["Strike"]
    .count()
    .reset_index()
    .rename(columns={"Strike": "n_strikes"})
)


# Resumen estadístico por bucket
print(strikes_por_grupo.groupby("bucket")["n_strikes"].describe(
    percentiles=[.05, .10, .25, .50, .75, .90, .95]
).round(1))


# In[]:
# Veamos para el bucket 1,  el número de días con menos de 6 strikes por tipo de opción:
for cp in ["C", "P"]:
    mask = (strikes_por_grupo["bucket"] == pd.Interval(0.0, 15.0, closed="right")) & (strikes_por_grupo["CallPut"] == cp)
    total_dias = mask.sum()
    dias_pocos_strikes = (strikes_por_grupo[mask]["n_strikes"] < 6).sum()
    print(f"Bucket 1 - {cp}: {dias_pocos_strikes} días con menos de 6 strikes ({dias_pocos_strikes/total_dias:.2%})")
    print(f"Number of total days: {total_dias}")

# In[]:

resumen = (strikes_por_grupo
           .groupby(["bucket", "CallPut"])
           .apply(lambda x: pd.Series({
               "total_dias":  len(x),
               "dias_pocos":  (x["n_strikes"] <= 5).sum(),
               "pct_pocos":   (x["n_strikes"] <= 5).mean(),
           }))
           .reset_index())

print(resumen.round(4))



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



###########Diagnosticos:
# In[]:

tipo_diferencia = "Gamma_ctr"
# Distribución de gamma x — si hay valores extremos hay problema
print(opt_agg2[tipo_diferencia].describe(percentiles=[.01,.05,.25,.75,.95,.99]))

# Fracción de NaN por bucket
print(opt_agg2.groupby("bucket")[tipo_diferencia].apply(lambda x: x.isna().mean()))

print(opt_agg2.groupby("bucket")[tipo_diferencia]
      .apply(lambda x: (x < 0).mean()))

# Ratio señal-ruido: mediana vs desviación estándar
print("Mediana:", opt_agg2[tipo_diferencia].median())
print("Std:",     opt_agg2[tipo_diferencia].std())

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





# %%

# Opción 1: bucle for (más seguro, recomendado)
grupos = []
for (date, bucket, callput), group in opt_agg2.groupby(["Date","bucket","CallPut"]):
    resultado = greeks_spline(group)
    resultado["Date"]    = date
    resultado["bucket"]  = bucket
    resultado["CallPut"] = callput
    grupos.append(resultado)

resultx = pd.concat(grupos).reset_index(drop=True)
resultx

# In[]:

######## Validaciónes y robsutez:

# Unir FD (opt_agg2) con spline (resultx) por las columnas clave
cols_spline = ["Date", "bucket", "CallPut", "Strike", 
               "Delta_sp", "Gamma_sp", "dSigma_dK", "Vanna_sp"]

result_combined = opt_agg.merge(
    resultx[cols_spline],
    on=["Date", "bucket", "CallPut", "Strike"],
    how="left"
)


def arbitrage_stats(df, method_suffix):
    """Estadísticas de violaciones por método."""
    delta_col = f"Delta_{method_suffix}"
    gamma_col = f"Gamma_{method_suffix}"
    
    mask_call = df["CallPut"] == "C"
    mask_put  = df["CallPut"] == "P"
    
    results = {}
    
    # Delta fuera de rango
    delta_valid = (
        (mask_call & df[delta_col].between(0, 1)) |
        (mask_put  & df[delta_col].between(-1, 0))
    )
    results["delta_viol_pct"] = (~delta_valid & df[delta_col].notna()).mean()
    
    # Gamma negativa
    results["gamma_neg_pct"] = (df[gamma_col] < 0).mean()
    
    # NaN rate
    results["delta_nan_pct"] = df[delta_col].isna().mean()
    results["gamma_nan_pct"] = df[gamma_col].isna().mean()
    
    return pd.Series(results)

methods = ["fwd", "bwd", "ctr", "sp"]
stats = pd.DataFrame({m: arbitrage_stats(resultx, m) for m in methods})
print("\n=== VIOLACIONES DE NO-ARBITRAJE ===")
print(stats.round(4))

# In[]:

def consistency_stats(df):
    """Correlación y diferencia entre métodos."""
    valid = df.dropna(subset=["Delta_fwd","Delta_bwd","Delta_ctr","Delta_sp",
                               "Gamma_fwd","Gamma_bwd","Gamma_ctr","Gamma_sp"])
    
    results = {}
    
    # Correlaciones entre métodos
    for greek in ["Delta", "Gamma"]:
        cols = [f"{greek}_{m}" for m in ["fwd","bwd","ctr","sp"]]
        corr = valid[cols].corr()
        # Correlación centrada vs spline (el par más relevante)
        results[f"{greek}_corr_ctr_sp"] = corr.loc[f"{greek}_ctr", f"{greek}_sp"]
        results[f"{greek}_corr_fwd_sp"] = corr.loc[f"{greek}_fwd", f"{greek}_sp"]
        
    # MAE entre métodos respecto al spline (benchmark)
    for greek in ["Delta", "Gamma"]:
        for m in ["fwd","bwd","ctr"]:
            diff = (valid[f"{greek}_{m}"] - valid[f"{greek}_sp"]).abs()
            results[f"{greek}_mae_{m}_vs_sp"] = diff.median()
    
    return pd.Series(results)

print("\n=== CONSISTENCIA ENTRE MÉTODOS ===")
print(consistency_stats(resultx).round(4))
# In[]:

def temporal_stability(df, method_suffix):
    """Volatilidad de la serie temporal de medianas diarias."""
    daily = (df.groupby(["Date","CallPut"])
               .agg(
                   delta_med=(f"Delta_{method_suffix}", "median"),
                   gamma_med=(f"Gamma_{method_suffix}", "median"),
               )
               .reset_index())
    
    results = {}
    for cp in ["C","P"]:
        sub = daily[daily["CallPut"]==cp].sort_values("Date")
        # Std de los cambios día a día (cuanto más pequeña, más suave)
        results[f"delta_daily_std_{cp}"] = sub["delta_med"].diff().std()
        results[f"gamma_daily_std_{cp}"] = sub["gamma_med"].diff().std()
    
    return pd.Series(results)

print("\n=== ESTABILIDAD TEMPORAL ===")
stab = pd.DataFrame({m: temporal_stability(resultx, m) for m in methods})
print(stab.round(6))
# In[]:


def temporal_stability(df, method_suffix):
    """Volatilidad de la serie temporal de medianas diarias."""
    daily = (df.groupby(["Date","CallPut"])
               .agg(
                   delta_med=(f"Delta_{method_suffix}", "median"),
                   gamma_med=(f"Gamma_{method_suffix}", "median"),
               )
               .reset_index())
    
    results = {}
    for cp in ["C","P"]:
        sub = daily[daily["CallPut"]==cp].sort_values("Date")
        # Std de los cambios día a día (cuanto más pequeña, más suave)
        results[f"delta_daily_std_{cp}"] = sub["delta_med"].diff().std()
        results[f"gamma_daily_std_{cp}"] = sub["gamma_med"].diff().std()
    
    return pd.Series(results)

print("\n=== ESTABILIDAD TEMPORAL ===")
stab = pd.DataFrame({m: temporal_stability(resultx, m) for m in methods})
print(stab.round(6))


print("\n=== TABLA COMPARATIVA FINAL ===")
summary = pd.DataFrame({
    "gamma_neg_%":    {m: (resultx[f"Gamma_{m}"] < 0).mean() for m in methods},
    "delta_viol_%":   {m: arbitrage_stats(resultx, m)["delta_viol_pct"] for m in methods},
    "gamma_nan_%":    {m: resultx[f"Gamma_{m}"].isna().mean() for m in methods},
    "corr_vs_spline": {m: resultx[[f"Gamma_{m}","Gamma_sp"]]
                         .dropna().corr().iloc[0,1] for m in methods},
}).T
print(summary.round(4))

# In[]:






# """

# from joblib import Parallel, delayed
# import numpy as np
# import pandas as pd


# def fornberg_weights(z, x, m):
#     n = len(x)
#     c = np.zeros((n, m+1))
#     c[0, 0] = 1.0
#     c1 = 1.0
#     c4 = x[0] - z
#     for i in range(1, n):
#         mn = min(i, m)
#         c2 = 1.0
#         c5 = c4
#         c4 = x[i] - z
#         for j in range(i):
#             c3 = x[i] - x[j]
#             if abs(c3) < 1e-10:
#                 return np.full(n, np.nan)
#             c2 *= c3
#             for k in range(mn, 0, -1):
#                 c[i, k] = c1 * (k * c[i-1, k-1] - c5 * c[i-1, k]) / c2
#             c[i, 0] = -c1 * c5 * c[i-1, 0] / c2
#             for k in range(mn, 0, -1):
#                 c[j, k] = (c4 * c[j, k] - k * c[j, k-1]) / c3
#             c[j, 0] = c4 * c[j, 0] / c3
#         c1 = c2
#     return c[:, m]


# def apply_fornberg_group(group, n_points=5):
#     date   = group["Date"].iloc[0]
#     days   = group["Days"].iloc[0]
#     cp     = group["CallPut"].iloc[0]

#     group_dedup = (group.groupby("Strike", as_index=False)
#                         .agg(MidPrice=("MidPrice", "mean"))
#                         .sort_values("Strike"))

#     strikes = group_dedup["Strike"].values.astype(np.float64)
#     prices  = group_dedup["MidPrice"].values.astype(np.float64)
#     n       = len(strikes)
#     half    = n_points // 2

#     dc_dk   = np.full(n, np.nan)
#     d2c_dk2 = np.full(n, np.nan)

#     for i in range(n):
#         lo = max(0, i - half)
#         hi = min(n, lo + n_points)
#         lo = max(0, hi - n_points)

#         x_win = strikes[lo:hi]
#         f_win = prices[lo:hi]

#         if len(x_win) < 3 or len(np.unique(x_win)) < len(x_win):
#             continue

#         w1 = fornberg_weights(strikes[i], x_win, 1)
#         w2 = fornberg_weights(strikes[i], x_win, 2)

#         if np.any(np.isnan(w1)) or np.any(np.isnan(w2)):
#             continue

#         dc_dk[i]   = w1 @ f_win
#         d2c_dk2[i] = w2 @ f_win

#     group_dedup["dc_dk_fn"]   = dc_dk
#     group_dedup["d2c_dk2_fn"] = d2c_dk2
#     group_dedup["Date"]       = date
#     group_dedup["Days"]       = days
#     group_dedup["CallPut"]    = cp

#     return group_dedup


# def process(key, grp):
#     return apply_fornberg_group(grp, n_points=5)


# groups  = [(key, grp) for key, grp in opt_df_filtered2.groupby(["Date", "Days", "CallPut"])]

# results = Parallel(n_jobs=-1, verbose=1)(
#     delayed(process)(k, g) for k, g in groups
# )

# fn_df = pd.concat(results)[["Date", "Days", "CallPut", "Strike", "dc_dk_fn", "d2c_dk2_fn"]]

# opt_df_filtered2 = opt_df_filtered2.merge(
#     fn_df, on=["Date", "Days", "CallPut", "Strike"], how="left"
# )

# S = opt_df_filtered2["SpotPrice"]
# K = opt_df_filtered2["Strike"]
# opt_df_filtered2["Delta_fn"] = (1/S) * (opt_df_filtered2["MidPrice"] - opt_df_filtered2["dc_dk_fn"] * K)
# opt_df_filtered2["Gamma_fn"] = opt_df_filtered2["d2c_dk2_fn"] * (K/S)**2




# # ── colores y estilos ─────────────────────────────────────────────────────────
# resumen = opt_df_filtered2.groupby(["Date", "CallPut"]).agg(
#     Delta_fwd_med = ("Delta_fwd", "median"),
#     Delta_bwd_med = ("Delta_bwd", "median"),
#     Delta_ctr_med = ("Delta_ctr", "median"),
#     Delta_fn_med  = ("Delta_fn",  "median"),
#     Gamma_fwd_med = ("Gamma_fwd", "median"),
#     Gamma_bwd_med = ("Gamma_bwd", "median"),
#     Gamma_ctr_med = ("Gamma_ctr", "median"),
#     Gamma_fn_med  = ("Gamma_fn",  "median"),
# ).reset_index()

# calls = resumen[resumen["CallPut"] == "C"].sort_values("Date")
# puts  = resumen[resumen["CallPut"] == "P"].sort_values("Date")

# COL_FWD = "#378ADD"
# COL_BWD = "#D85A30"
# COL_CTR = "#1D9E75"
# COL_FN  = "#000000"

# fig, axes = plt.subplots(2, 2, figsize=(16, 8), sharex=False)
# fig.suptitle("Comparación métodos de diferencia finita — mediana diaria", fontsize=13)

# panels = [
#     (axes[0, 0], calls, "Delta_fwd_med", "Delta_bwd_med", "Delta_ctr_med", "Delta_fn_med", "Delta — Calls"),
#     (axes[0, 1], puts,  "Delta_fwd_med", "Delta_bwd_med", "Delta_ctr_med", "Delta_fn_med", "Delta — Puts"),
#     (axes[1, 0], calls, "Gamma_fwd_med", "Gamma_bwd_med", "Gamma_ctr_med", "Gamma_fn_med", "Gamma — Calls"),
#     (axes[1, 1], puts,  "Gamma_fwd_med", "Gamma_bwd_med", "Gamma_ctr_med", "Gamma_fn_med", "Gamma — Puts"),
# ]

# for ax, df, fwd, bwd, ctr, fn, title in panels:
#     ax.plot(df["Date"], df[fwd], color=COL_FWD, lw=0.8, label="Forward")
#     ax.plot(df["Date"], df[bwd], color=COL_BWD, lw=0.8, ls="--", label="Backward")
#     ax.plot(df["Date"], df[ctr], color=COL_CTR, lw=0.8, ls=":",  label="Centrada")
#     ax.plot(df["Date"], df[fn],  color=COL_FN,  lw=0.8, ls="-.", label="Fornberg")
#     ax.set_title(title, fontsize=11)
#     ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
#     ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
#     ax.tick_params(axis="x", rotation=45, labelsize=8)
#     ax.tick_params(axis="y", labelsize=8)
#     ax.grid(alpha=0.2)
#     ax.legend(fontsize=8)

# plt.tight_layout()
# plt.savefig("greeks_comparacion_metodos.png", dpi=150, bbox_inches="tight")
# plt.show()

# """
# 



















