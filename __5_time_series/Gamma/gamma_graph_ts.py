# In[]
#Time series analysis for NTM srikes:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from matplotlib.widgets import Button




quantile_gammas = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\time_series\quantile_gamma.parquet").set_index("Date")

print(quantile_gammas.describe())


###############################################################################
#   Análisis de las series temporales
##############################################################################

#1) Hacemos un test de raices unitarias para comprobar la estacionariedad de las series
    #con constant, linear, y término cuadrático (ctt):

print("================= ADF I(0) =================")

for quantile in quantile_gammas.columns:
    result = adfuller(quantile_gammas[quantile], regression="ctt")
    print(f"{quantile}: p-value={100*result[1]:.4f} %")



#2) Calculamos primeras diferencia y pruebo de nuevo el ADF-test:

quantile_diff = quantile_gammas.pct_change().iloc[1:-1]
print("================= ADF I(1) =================")
for quantile in quantile_diff.columns:
    result = adfuller(quantile_diff[quantile], regression="ctt")
    print(f"{quantile}: p-value={100*result[1]:.4f} %")


#3) Ploteo de nuevo como en gamma_graph_1 pero para la diferencia:


cols = [c for c in quantile_diff.columns if c != "Date"]
idx  = [0]

fig, ax = plt.subplots(figsize=(14, 4))
plt.subplots_adjust(bottom=0.15)


def update():
    col = cols[idx[0]]
    ax.cla()
    ax.plot(quantile_diff.index, quantile_diff[col], linewidth=0.8)
    ax.set_title(f"{col}  ({idx[0]+1}/{len(cols)})")
    ax.set_ylabel("gamma")
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



