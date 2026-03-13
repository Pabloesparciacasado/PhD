# In[]
#Time series analysis for NTM srikes:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm

import duckdb



# ============================================================
# 1. CARGAR SUPERFICIES
# ============================================================
atm_band = 0.015 # ±1% alrededor de ATM


raw_df_clean = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\superficie_con_greeks_shimko_2.parquet")
raw_df_clean["Date"] = pd.to_datetime(raw_df_clean["Date"])


raw_df_clean["Date"] = pd.to_datetime(raw_df_clean["Date"])
raw_df_clean = raw_df_clean[['delta','delta_bs','Date', 'Days', 'T', 'rate', 'moneyness',
       'log_moneyness', 'CallPut', 'implied_vol', 'Precio_Modelo', 'vega', 'gamma_bs', 'vanna_K',
       'volga', 'dsigma_dK', 'd2sigma_dK2', 'gamma']]

# raw_df_NTM = raw_df_clean[
#         (raw_df_clean["moneyness"] >= 1 - atm_band) &
#         (raw_df_clean["moneyness"] <= 1 + atm_band)]
# In[]
# ============================================================
# 2. GRÁFICO 3D: Delta por Fecha y Moneyness
# ============================================================

# Filtrar calls y un vencimiento representativo (~30 días)
target_days = 30
tol_days = 10

df_plot = raw_df_clean[
    (raw_df_clean["CallPut"] == "C") 
].copy()

# Pivot: filas = fecha, columnas = moneyness discretizado
df_plot["moneyness_round"] = df_plot["moneyness"].round(2)
pivot = df_plot.groupby(["Date", "moneyness_round"])["delta"].mean().unstack("moneyness_round")
pivot = pivot.dropna(thresh=int(pivot.shape[1] * 0.5))  # quitar fechas con demasiados NaN
pivot = pivot.interpolate(axis=1)                        # interpolar NaN restantes

# Meshgrid
dates_num = np.arange(len(pivot.index))
moneyness_vals = pivot.columns.values.astype(float)
X, Y = np.meshgrid(dates_num, moneyness_vals, indexing="ij")
Z = pivot.values

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(X, Y, Z, cmap=cm.RdYlGn, alpha=0.85, linewidth=0, antialiased=True)

# Etiquetas del eje X con fechas legibles
tick_step = max(1, len(pivot.index) // 8)
ax.set_xticks(dates_num[::tick_step])
ax.set_xticklabels(
    [d.strftime("%Y-%m") for d in pivot.index[::tick_step]],
    rotation=30, ha="right", fontsize=7
)
ax.set_xlabel("Fecha", labelpad=10)
ax.set_ylabel("Moneyness (S/K)", labelpad=10)
ax.set_zlabel("delta", labelpad=10)
ax.set_title(f"Delta de calls (~{target_days}d) por Fecha y Moneyness", pad=15)

fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, label="delta")
plt.tight_layout()
plt.show()


# ============================================================
# 3. GRÁFICO 3D: Calls vs Puts — Delta por Fecha y Moneyness
# ============================================================

def build_pivot(cp_flag):
    df = raw_df_clean[raw_df_clean["CallPut"] == cp_flag].copy()
    df["moneyness_round"] = df["moneyness"].round(2)
    piv = df.groupby(["Date", "moneyness_round"])["delta_bs"].mean().unstack("moneyness_round")
    piv = piv.dropna(thresh=int(piv.shape[1] * 0.5))
    piv = piv.interpolate(axis=1)
    return piv

pivot_c = build_pivot("C")
pivot_p = build_pivot("P")

# Alinear fechas comunes
common_dates = pivot_c.index.intersection(pivot_p.index)
pivot_c = pivot_c.loc[common_dates]
pivot_p = pivot_p.loc[common_dates]

fig = plt.figure(figsize=(18, 7))

for i, (piv, label, cmap) in enumerate(
    [(pivot_c, "Calls", cm.Blues), (pivot_p, "Puts", cm.Reds)], start=1
):
    ax = fig.add_subplot(1, 2, i, projection="3d")

    dates_num = np.arange(len(piv.index))
    moneyness_vals = piv.columns.values.astype(float)
    X, Y = np.meshgrid(dates_num, moneyness_vals, indexing="ij")
    Z = piv.values

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, linewidth=0, antialiased=True)

    tick_step = max(1, len(piv.index) // 8)
    ax.set_xticks(dates_num[::tick_step])
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in piv.index[::tick_step]],
        rotation=30, ha="right", fontsize=7
    )
    ax.set_xlabel("Fecha", labelpad=10)
    ax.set_ylabel("Moneyness (S/K)", labelpad=10)
    ax.set_zlabel("delta_bs", labelpad=10)
    ax.set_title(f"delta_bs — {label}", pad=12)
    fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, label="delta_bs")

plt.suptitle("Comparación delta_bs: Calls vs Puts por Fecha y Moneyness", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# In[]
# ============================================================
# 4. Análisis de series temporales (delta)
# ============================================================

raw_quantile_C = raw_df_clean[raw_df_clean["CallPut"]=="C"]["moneyness"]
raw_quantile_P = raw_df_clean[raw_df_clean["CallPut"]=="P"]["moneyness"]
q = 10
quantiles = list(np.round(np.linspace(0.1, 0.9, q-1), 1))


vector_quantile_C = np.quantile(raw_quantile_C,quantiles)
vector_quantile_P = np.quantile(raw_quantile_P,quantiles)

targets_C = dict(zip(
    [f"c{int(c*100)}" for c in quantiles],
    np.log(vector_quantile_C)  # en log_moneyness para usar con np.interp
))

targets_P = dict(zip(
    [f"p{int(p*100)}" for p in quantiles],
    np.log(vector_quantile_P)  # en log_moneyness para usar con np.interp
))

df_targets = pd.DataFrame({
    "quantile": quantiles * 2,
    "type": ["C"]*len(quantiles) + ["P"]*len(quantiles),
    "log_moneyness": list(targets_C.values()) + list(targets_P.values())
})
df_targets["label"] = df_targets["quantile"].astype(str) + df_targets["type"]

def interp_quantile(group, cp):
    group = group.sort_values("log_moneyness")
    rows = df_targets[df_targets["type"] == cp]
    return pd.Series({
        row.label: np.interp(row.log_moneyness, group["log_moneyness"], group["delta"])
        for row in rows.itertuples()
    })

qg_C = (
    raw_df_clean[raw_df_clean["CallPut"] == "C"]
    .groupby("Date")
    .apply(lambda g: interp_quantile(g, "C"))
    .reset_index()
)
qg_P = (
    raw_df_clean[raw_df_clean["CallPut"] == "P"]
    .groupby("Date")
    .apply(lambda g: interp_quantile(g, "P"))
    .reset_index()
)
quantile_greeks = qg_C.merge(qg_P, on="Date", how="inner")
print(pd.DataFrame(quantile_greeks))


# %%
#ploteamos las series para distintos puntos de la moneyness:


from matplotlib.widgets import Button

cols = [c for c in quantile_greeks.columns if c != "Date"]
idx = [0]

fig, ax = plt.subplots(figsize=(14, 4))
plt.subplots_adjust(bottom=0.15)

def update():
    col = cols[idx[0]]
    ax.cla()
    ax.plot(quantile_greeks["Date"], quantile_greeks[col], linewidth=0.8)
    # ax.set_ylim(quantile_greeks[col].quantile(0.01), quantile_greeks[col].quantile(0.99))
    ax.set_title(f"{col}  ({idx[0]+1}/{len(cols)})")
    ax.set_ylabel("delta")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    fig.canvas.draw()

ax_prev = plt.axes([0.35, 0.03, 0.1, 0.05])
ax_next = plt.axes([0.55, 0.03, 0.1, 0.05])
btn_prev = Button(ax_prev, '◀ Anterior')
btn_next = Button(ax_next, 'Siguiente ▶')

btn_prev.on_clicked(lambda e: [idx.__setitem__(0, (idx[0]-1) % len(cols)), update()])
btn_next.on_clicked(lambda e: [idx.__setitem__(0, (idx[0]+1) % len(cols)), update()])

update()
plt.show()

# %%

PARQUET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\time_series\quantile_delta.parquet"
duckdb.from_df(quantile_greeks).write_parquet(PARQUET_OUTPUT, compression='snappy')
print("="*100)
print(f"Fichero guardado correctamente en: {PARQUET_OUTPUT}")
