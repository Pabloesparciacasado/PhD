# In[]
# Time series analysis for NTM strikes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import cm
from matplotlib.widgets import Button
import duckdb


# ============================================================
# 1. CARGAR Y CONSTRUIR PIVOTS (fuente única de datos)
# ============================================================

raw_df_clean = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\superficie_con_greeks_shimko_2.parquet")
raw_df_clean["Date"] = pd.to_datetime(raw_df_clean["Date"])
raw_df_clean = raw_df_clean[['delta', 'delta_bs', 'Date', 'Days', 'T', 'rate', 'moneyness',
       'log_moneyness', 'CallPut', 'implied_vol', 'Precio_Modelo', 'vega', 'gamma_bs', 'vanna_K',
       'volga', 'dsigma_dK', 'd2sigma_dK2', 'gamma']]
raw_df_clean["moneyness_round"] = raw_df_clean["moneyness"].round(2)


def build_pivot(cp_flag, value_col="delta"):
    df = raw_df_clean[raw_df_clean["CallPut"] == cp_flag]
    piv = df.groupby(["Date", "moneyness_round"])[value_col].mean().unstack("moneyness_round")
    piv = piv.dropna(thresh=int(piv.shape[1] * 0.5))
    # Cutoff: columna más baja desde la que TODAS las fechas tienen dato
    cutoff = piv.apply(lambda row: row.first_valid_index(), axis=1).max()
    piv = piv.loc[:, piv.columns >= cutoff]
    piv = piv.interpolate(axis=1)
    print(f"Pivot {cp_flag} [{value_col}]: {piv.shape} | cutoff moneyness = {cutoff:.2f}")
    return piv, cutoff


pivot_c,    cutoff_C    = build_pivot("C", "delta")
pivot_p,    cutoff_P    = build_pivot("P", "delta")
pivot_c_bs, _           = build_pivot("C", "delta_bs")

# Aplicar cutoff al raw para las series temporales
raw_df_clean = raw_df_clean[
    ((raw_df_clean["CallPut"] == "C") & (raw_df_clean["moneyness_round"] >= cutoff_C)) |
    ((raw_df_clean["CallPut"] == "P") & (raw_df_clean["moneyness_round"] >= cutoff_P))
]

# Alinear fechas comunes entre calls y puts
common_dates = pivot_c.index.intersection(pivot_p.index)
pivot_c    = pivot_c.loc[common_dates]
pivot_p    = pivot_p.loc[common_dates]
pivot_c_bs = pivot_c_bs.loc[pivot_c_bs.index.intersection(common_dates)]
print(pivot_c)

# In[]
# ============================================================
# 2. GRÁFICO 3D: Delta BS de Calls por Fecha y Moneyness
# ============================================================

def plot_3d(piv, title, zlabel, cmap=cm.RdYlGn):
    dates_num     = np.arange(len(piv.index))
    moneyness_vals = piv.columns.values.astype(float)
    X, Y = np.meshgrid(dates_num, moneyness_vals, indexing="ij")
    Z    = piv.values

    fig = plt.figure(figsize=(14, 7))
    ax  = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, linewidth=0, antialiased=True)

    tick_step = max(1, len(piv.index) // 8)
    ax.set_xticks(dates_num[::tick_step])
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in piv.index[::tick_step]],
        rotation=30, ha="right", fontsize=7
    )
    ax.set_xlabel("Fecha", labelpad=10)
    ax.set_ylabel("Moneyness (S/K)", labelpad=10)
    ax.set_zlabel(zlabel, labelpad=10)
    ax.set_title(title, pad=15)
    fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, label=zlabel)
    plt.tight_layout()
    plt.show()


plot_3d(pivot_c_bs, "Delta BS de Calls por Fecha y Moneyness", "delta_bs")


# ============================================================
# 3. GRÁFICO 3D: Calls vs Puts — Delta por Fecha y Moneyness
# ============================================================

fig = plt.figure(figsize=(18, 7))

for i, (piv, label, cmap) in enumerate(
    [(pivot_c, "Calls", cm.Blues), (pivot_p, "Puts", cm.Reds)], start=1
):
    ax = fig.add_subplot(1, 2, i, projection="3d")

    dates_num      = np.arange(len(piv.index))
    moneyness_vals = piv.columns.values.astype(float)
    X, Y = np.meshgrid(dates_num, moneyness_vals, indexing="ij")
    Z    = piv.values

    surf = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=0.85, linewidth=0, antialiased=True)

    tick_step = max(1, len(piv.index) // 8)
    ax.set_xticks(dates_num[::tick_step])
    ax.set_xticklabels(
        [d.strftime("%Y-%m") for d in piv.index[::tick_step]],
        rotation=30, ha="right", fontsize=7
    )
    ax.set_xlabel("Fecha", labelpad=10)
    ax.set_ylabel("Moneyness (S/K)", labelpad=10)
    ax.set_zlabel("delta", labelpad=10)
    ax.set_title(f"delta — {label}", pad=12)
    fig.colorbar(surf, ax=ax, shrink=0.4, aspect=10, label="delta")

plt.suptitle("Comparación delta: Calls vs Puts por Fecha y Moneyness", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()


# In[]
# ============================================================
# 4. Series temporales: delta interpolado en cuantiles de moneyness
# ============================================================

q = 10
quantiles = list(np.round(np.linspace(0.1, 1, q), 1))

# Targets en log_moneyness: cuantiles de la distribución real de cada tipo
targets_C = dict(zip(
    [f"{int(c*100)}C" for c in quantiles],
    np.log(np.quantile(raw_df_clean[raw_df_clean["CallPut"] == "C"]["moneyness"], quantiles))
))
targets_P = dict(zip(
    [f"{int(p*100)}P" for p in quantiles],
    np.log(np.quantile(raw_df_clean[raw_df_clean["CallPut"] == "P"]["moneyness"], quantiles))
))

df_targets = pd.DataFrame({
    "label":         list(targets_C.keys())    + list(targets_P.keys()),
    "type":          ["C"] * len(quantiles)    + ["P"] * len(quantiles),
    "log_moneyness": list(targets_C.values())  + list(targets_P.values()),
})


def interp_group(group, cp):
    group = group.sort_values("log_moneyness")
    rows  = df_targets[df_targets["type"] == cp]
    return pd.Series({
        row.label: np.interp(row.log_moneyness, group["log_moneyness"], group["delta"])
        for row in rows.itertuples()
    })


qg_C = (raw_df_clean[raw_df_clean["CallPut"] == "C"]
        .groupby("Date").apply(lambda g: interp_group(g, "C")).reset_index())
qg_P = (raw_df_clean[raw_df_clean["CallPut"] == "P"]
        .groupby("Date").apply(lambda g: interp_group(g, "P")).reset_index())

quantile_greeks = qg_C.merge(qg_P, on="Date", how="inner")
print(quantile_greeks)


# %%
# Navegador interactivo de series temporales

cols = [c for c in quantile_greeks.columns if c != "Date"]
idx  = [0]

fig, ax = plt.subplots(figsize=(14, 4))
plt.subplots_adjust(bottom=0.15)


def update():
    col = cols[idx[0]]
    ax.cla()
    ax.plot(quantile_greeks["Date"], quantile_greeks[col], linewidth=0.8)
    ax.set_title(f"{col}  ({idx[0]+1}/{len(cols)})")
    ax.set_ylabel("delta")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45)
    fig.canvas.draw()


ax_prev  = plt.axes([0.35, 0.03, 0.1, 0.05])
ax_next  = plt.axes([0.55, 0.03, 0.1, 0.05])
btn_prev = Button(ax_prev, '◀ Anterior')
btn_next = Button(ax_next, 'Siguiente ▶')

btn_prev.on_clicked(lambda _: [idx.__setitem__(0, (idx[0] - 1) % len(cols)), update()])
btn_next.on_clicked(lambda _: [idx.__setitem__(0, (idx[0] + 1) % len(cols)), update()])

update()
plt.show()


# %%

PARQUET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\time_series\quantile_delta.parquet"
duckdb.from_df(quantile_greeks).write_parquet(PARQUET_OUTPUT, compression='snappy')
print("=" * 100)
print(f"Fichero guardado correctamente en: {PARQUET_OUTPUT}")
