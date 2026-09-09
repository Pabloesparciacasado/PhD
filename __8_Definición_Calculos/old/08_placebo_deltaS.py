# In[]: Placebo de permutación de ΔS
"""
08_placebo_deltaS.py

Test placebo de permutación sobre los incrementos diarios del subyacente (ΔS).

Lógica (Fisher 1935; Lehmann & Romano 2005, cap. 15; Davison & Hinkley 1997):
  1. Permutamos los incrementos diarios de SpotPrice DENTRO de cada mes natural,
     manteniendo los cambios de precio de las opciones (ΔC) en su sitio.
  2. Reconstruimos la trayectoria placebo S* = S_0 + cumsum(ΔS permutado).
     Como la suma de incrementos dentro de cada mes se conserva, los niveles
     de fin de mes coinciden con los observados (la serie solo se "desordena"
     por dentro de cada mes).
  3. Recalculamos delta_emp y gamma_emp (op1, nivel contrato, idéntico a
     02_panel_construction) usando S*, y recomputamos los estadísticos
     agregados igual que en 05_gamma_spread:
        - w_gamma_OI / w_gamma_VD (WA diaria, media temporal) por CallPut
        - w_delta_OI / w_delta_VD por CallPut
        - spreads ATM − {near, deep, very_deep} (WA por OI) por CallPut
  4. Nula: "el estadístico toma este valor por la mecánica del cociente
     (denominador ΔS), sin correspondencia económica entre ΔC y ΔS".
     Si el valor observado cae dentro de la distribución placebo → artefacto.
  5. p-valor bilateral con corrección: p = (1 + #{|T*| ≥ |T_obs|}) / (B + 1).

DECISIÓN DE DISEÑO: condicionamos en la ESTRUCTURA de emparejamiento observada
(qué pares (t, t−1) de cada OptionID entran, y los buckets de moneyness
observados). Solo se aleatoriza el denominador. Así el placebo no mezcla el
efecto "selección de filas" con el efecto "mecánica del cociente". Los casos
con ΔS* = 0 se tratan como NaN y salen de las medias ponderadas.

Input : opt_df_empirical_greeks.parquet (solo se usan MidPrice, SpotPrice,
        buckets y pesos; las griegas guardadas solo sirven para verificar
        que la reconstrucción replica el pipeline).
Output: OUTPUTS/placebo_deltaS_draws.csv (una fila por réplica) + tabla
        resumen con p-valores + histogramas.
"""

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

# ------------------------- Configuración -------------------------

B          = 500          # nº de réplicas placebo (500–1000)
SEED       = 1234
V_MIN, V_MAX = 15, 45     # bucket de vencimiento (15, 45]
GUARDAR_DRAWS = True

if os.name == 'nt':
    PATH_DATA  = r"Y:\OUTPUTS\opt_df_empirical_greeks.parquet"
    PATH_DRAWS = r"Y:\OUTPUTS\placebo_deltaS_draws.csv"
else:
    PATH_DATA = r"/Users/pablo/codigos_github/opt_df_empirical_greeks.parquet"
    # PATH_DATA = r"/Volumes/data/OUTPUTS/opt_df_empirical_greeks.parquet"
    PATH_DRAWS = r"/Volumes/data/OUTPUTS/placebo_deltaS_draws.csv"

print("Cargando datos...")
opt_df = pd.read_parquet(
    PATH_DATA)

# Mismo filtro que 05_: vencimiento (15, 45]
df = opt_df[(opt_df["Days"] > V_MIN) & (opt_df["Days"] <= V_MAX)].copy()
del opt_df

df["DolarVolume"] = df["Volume"] * df["MidPrice"]
df = df.sort_values(["OptionID", "Date"]).reset_index(drop=True)
print(f"Panel (15,45]: {df.shape[0]:,} filas")

# In[]: Calendario del subyacente e incrementos ΔS por mes

cal = (df[["Date", "SpotPrice"]]
       .drop_duplicates("Date")
       .sort_values("Date")
       .reset_index(drop=True))

fechas   = cal["Date"].to_numpy()
S_obs    = cal["SpotPrice"].to_numpy(dtype=np.float64)
T        = len(S_obs)
dSpot    = np.diff(S_obs)                      # incremento t pertenece al mes de la fecha t
mes_inc  = cal["Date"].dt.to_period("M").to_numpy()[1:]

# Tramos contiguos de cada mes dentro del vector de incrementos
cambios     = np.r_[0, np.flatnonzero(mes_inc[1:] != mes_inc[:-1]) + 1, len(mes_inc)]
month_slices = [(cambios[i], cambios[i + 1]) for i in range(len(cambios) - 1)]

def path_placebo(rng):
    """Trayectoria S* con ΔS permutado dentro de cada mes (fin de mes anclado)."""
    inc = dSpot.copy()
    for s, e in month_slices:
        inc[s:e] = rng.permutation(inc[s:e])
    S = np.empty(T)
    S[0]  = S_obs[0]
    S[1:] = S_obs[0] + np.cumsum(inc)
    return S

print(f"Calendario: {T:,} días, {len(month_slices)} meses")

# In[]: Estructura de emparejamiento observada (fija en todas las réplicas)

n        = len(df)
idx      = np.arange(n)
opt_id   = df["OptionID"].to_numpy()
pos      = np.searchsorted(fechas, df["Date"].to_numpy())   # fila -> posición en calendario
midp     = df["MidPrice"].to_numpy(dtype=np.float64)
bucket   = df["moneyness_bucket"].to_numpy()

mismo_prev = np.r_[False, opt_id[1:] == opt_id[:-1]]

# --- Filas delta (réplica de delta_empirica_op1): prev obs del mismo OptionID,
#     mismo moneyness_bucket, y |ΔS| > 0 en los datos observados.
bucket_igual = np.r_[False, bucket[1:] == bucket[:-1]]
dS1_obs      = np.r_[np.nan, S_obs[pos[1:]] - S_obs[pos[:-1]]]
mask_delta   = mismo_prev & bucket_igual & (dS1_obs != 0) & ~np.isnan(dS1_obs)

drow  = idx[mask_delta]          # fila del panel de cada obs delta (t)
pT_d  = pos[drow]                # posición calendario de t
pL_d  = pos[drow - 1]            # posición calendario de t-1 (obs previa)
dC_d  = midp[drow] - midp[drow - 1]

# --- Filas gamma (réplica de gamma_empirica_op1): dos filas delta consecutivas
#     del mismo OptionID; dS2 = S(t) − S(fecha previa de la obs delta anterior).
d_opt        = opt_id[drow]
tiene_prev_d = np.r_[False, d_opt[1:] == d_opt[:-1]]
k_t = np.flatnonzero(tiene_prev_d)     # índice (dentro del vector delta) de t
k_l = k_t - 1                          # índice delta de t-1

pT_g  = pT_d[k_t]
pL2_g = pos[drow[k_l] - 1]             # SpotPrice_lag2: lag de la fila delta previa
grow  = drow[k_t]                      # fila del panel de cada obs gamma

def greeks_from_path(S):
    """delta y gamma empíricas op1 sobre una trayectoria S (observada o placebo)."""
    dS1   = S[pT_d] - S[pL_d]
    delta = np.divide(dC_d, dS1, out=np.full(dC_d.shape, np.nan), where=dS1 != 0)

    dS2    = S[pT_g] - S[pL2_g]
    ddelta = delta[k_t] - delta[k_l]
    gamma  = np.divide(2.0 * ddelta, dS2,
                       out=np.full(ddelta.shape, np.nan), where=dS2 != 0)
    return delta, gamma

print(f"Filas delta: {len(drow):,} | filas gamma: {len(grow):,}")

# In[]: Verificación — la reconstrucción replica el pipeline de 02_

delta_chk, gamma_chk = greeks_from_path(S_obs)

d_stored = df["delta_emp"].to_numpy(dtype=np.float64)[drow]
g_stored = df["gamma_emp"].to_numpy(dtype=np.float64)[grow]

for nombre, rec, sto in [("delta", delta_chk, d_stored), ("gamma", gamma_chk, g_stored)]:
    ok = np.isfinite(rec) & np.isfinite(sto)
    dif = np.abs(rec[ok] - sto[ok])
    print(f"[verificación {nombre}] n={ok.sum():,} | "
          f"max|dif|={dif.max():.3e} | p99|dif|={np.quantile(dif, 0.99):.3e} | "
          f"corr={np.corrcoef(rec[ok], sto[ok])[0,1]:.6f}")
# Si corr < 0.999 o max|dif| grande: revisar (float32 en 01_ explica difs pequeñas).

# In[]: Maquinaria de estadísticos (grupos fijos, bincount por réplica)

cp_d = (df["CallPut"].to_numpy()[drow] == "P").astype(np.int64)
cp_g = (df["CallPut"].to_numpy()[grow] == "P").astype(np.int64)
OI_d = df["OpenInterest"].to_numpy(dtype=np.float64)[drow]
OI_g = df["OpenInterest"].to_numpy(dtype=np.float64)[grow]
DV_d = df["DolarVolume"].to_numpy(dtype=np.float64)[drow]
DV_g = df["DolarVolume"].to_numpy(dtype=np.float64)[grow]

# Zonas de moneyness (idénticas a gamma_spread_left): 0=very_deep 1=deep 2=near 3=ATM
zonas_bins = np.array([0.0, 0.5, 0.7, 0.9, 1.1])
mon_d = df["Moneyness"].to_numpy(dtype=np.float64)[drow]
mon_g = df["Moneyness"].to_numpy(dtype=np.float64)[grow]
z_d = np.digitize(mon_d, zonas_bins[1:], right=True)   # 4 = fuera (>1.1)
z_g = np.digitize(mon_g, zonas_bins[1:], right=True)
ZONAS = ["very_deep", "deep", "near", "ATM"]

def make_groups(pos_rows, cp_rows, z_rows=None):
    """Ids compactos de grupo (Date,CallPut[,zona]) + metadatos por grupo."""
    if z_rows is None:
        key = pos_rows * 2 + cp_rows
    else:
        key = (pos_rows * 2 + cp_rows) * 4 + z_rows
    uniq, gid = np.unique(key, return_inverse=True)
    return gid, uniq, len(uniq)

# Niveles WA diaria por (Date, CallPut)
gid_lvl_g, uq_lvl_g, G_lvl_g = make_groups(pT_g, cp_g)
gid_lvl_d, uq_lvl_d, G_lvl_d = make_groups(pT_d, cp_d)
cp_of_lvl_g = uq_lvl_g % 2
cp_of_lvl_d = uq_lvl_d % 2

# Spreads: (Date, CallPut, zona), con OI > 0 y zona válida (como en 05_)
def groups_spread(pos_rows, cp_rows, z_rows, w_rows):
    m = (z_rows < 4) & (w_rows > 0)
    gid, uniq, G = make_groups(pos_rows[m], cp_rows[m], z_rows[m])
    daycp  = uniq // 4
    zona   = uniq % 4
    # id compacto de (Date, CallPut) para montar la matriz día × zona
    uq_daycp, daycp_id = np.unique(daycp, return_inverse=True)
    return m, gid, G, daycp_id, zona, len(uq_daycp), uq_daycp % 2

msp_g, gid_sp_g, G_sp_g, daycp_g, zona_g, D_g, cp_day_g = groups_spread(pT_g, cp_g, z_g, OI_g)
msp_d, gid_sp_d, G_sp_d, daycp_d, zona_d, D_d, cp_day_d = groups_spread(pT_d, cp_d, z_d, OI_d)

def wa_por_grupo(gid, G, w, x):
    """Media ponderada de x por grupo (NaN se excluye vía peso 0)."""
    ok  = np.isfinite(x)
    wv  = np.where(ok, w, 0.0)
    num = np.bincount(gid, wv * np.where(ok, x, 0.0), minlength=G)
    den = np.bincount(gid, wv, minlength=G)
    return np.divide(num, den, out=np.full(G, np.nan), where=den > 0)

def estadisticos(delta, gamma):
    """Réplica de los agregados de 05_: media temporal de las series diarias."""
    out = {}
    # --- niveles WA diaria
    for greek, x, gid, G, cp_of, w_oi, w_dv in [
        ("gamma", gamma, gid_lvl_g, G_lvl_g, cp_of_lvl_g, OI_g, DV_g),
        ("delta", delta, gid_lvl_d, G_lvl_d, cp_of_lvl_d, OI_d, DV_d),
    ]:
        for wname, w in [("OI", w_oi), ("VD", w_dv)]:
            wa = wa_por_grupo(gid, G, w, x)
            for cp_val, cp_name in [(0, "C"), (1, "P")]:
                sel = (cp_of == cp_val)
                out[f"w_{greek}_{wname}_{cp_name}"] = np.nanmean(wa[sel])
    # --- spreads ATM − zona (WA por OI)
    for greek, x, m, gid, G, daycp, zona, D, cp_day, w in [
        ("gamma", gamma, msp_g, gid_sp_g, G_sp_g, daycp_g, zona_g, D_g, cp_day_g, OI_g),
        ("delta", delta, msp_d, gid_sp_d, G_sp_d, daycp_d, zona_d, D_d, cp_day_d, OI_d),
    ]:
        wa = wa_por_grupo(gid, G, w[m], x[m])
        M  = np.full((D, 4), np.nan)
        M[daycp, zona] = wa
        for z_val in (0, 1, 2):  # very_deep, deep, near
            spread = M[:, 3] - M[:, z_val]
            for cp_val, cp_name in [(0, "C"), (1, "P")]:
                sel = (cp_day == cp_val)
                out[f"spread_{greek}_ATM_minus_{ZONAS[z_val]}_{cp_name}"] = np.nanmean(spread[sel])
    return out

# In[]: Estadísticos observados (mismo framework -> comparación interna coherente)

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

stats_obs = estadisticos(delta_chk, gamma_chk)
print(tabulate(sorted(stats_obs.items()), headers=["estadístico", "observado"],
               floatfmt=".6f", tablefmt="github"))

# In[]: Bucle placebo

rng = np.random.default_rng(SEED)
draws = []
for b in range(B):
    S_star = path_placebo(rng)
    delta_s, gamma_s = greeks_from_path(S_star)
    draws.append(estadisticos(delta_s, gamma_s))
    if (b + 1) % 50 == 0:
        print(f"réplica {b + 1}/{B}")

draws = pd.DataFrame(draws)
if GUARDAR_DRAWS:
    draws.to_csv(PATH_DRAWS, index=False)
    print(f"Réplicas guardadas en: {PATH_DRAWS}")

# In[]: Resultados: p-valores bilaterales y resumen

filas = []
for k in draws.columns:
    t_obs = stats_obs[k]
    t_pl  = draws[k].to_numpy()
    t_pl  = t_pl[np.isfinite(t_pl)]
    if not np.isfinite(t_obs) or len(t_pl) == 0:
        continue
    # bilateral centrado en la media placebo (la nula no está centrada en 0:
    # la mecánica del cociente puede generar sesgo, y eso ES parte de la nula)
    mu, sd = t_pl.mean(), t_pl.std(ddof=1)
    p_two  = (1 + np.sum(np.abs(t_pl - mu) >= np.abs(t_obs - mu))) / (len(t_pl) + 1)
    filas.append({
        "estadístico": k,
        "observado":   t_obs,
        "placebo_mean": mu,
        "placebo_sd":   sd,
        "z": (t_obs - mu) / sd if sd > 0 else np.nan,
        "p_two": p_two,
        "q2.5": np.quantile(t_pl, 0.025),
        "q97.5": np.quantile(t_pl, 0.975),
    })

resumen = pd.DataFrame(filas)
print(tabulate(resumen, headers="keys", floatfmt=".6f", tablefmt="github", showindex=False))

# In[]: Histogramas (gamma: niveles y spreads)

cols_plot = [c for c in draws.columns if "gamma" in c]
ncol = 4
nrow = int(np.ceil(len(cols_plot) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow))
for ax, c in zip(np.ravel(axes), cols_plot):
    ax.hist(draws[c].dropna(), bins=40, color="#9ecae1", edgecolor="white")
    ax.axvline(stats_obs[c], color="crimson", lw=2, label="observado")
    ax.set_title(c, fontsize=9)
    ax.legend(fontsize=7)
for ax in np.ravel(axes)[len(cols_plot):]:
    ax.axis("off")
fig.suptitle("Placebo ΔS: distribución bajo la nula vs valor observado", y=1.02)
fig.tight_layout()
plt.show()

# %%
