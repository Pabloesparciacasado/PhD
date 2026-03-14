# In[]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\volatility_surface_30_shimko_2_prueba.parquet")
df["Date"] = pd.to_datetime(df["Date"])

atm_iv = (df.groupby(["Date", "Days","CallPut"])
    .apply(lambda g: np.interp(0.0, g.sort_values("log_moneyness")["log_moneyness"], 
                                    g.sort_values("log_moneyness")["implied_vol"]))
    .unstack("CallPut")
)
print((atm_iv["C"] - atm_iv["P"]).describe())



# In[]
print(df["moneyness"].min())

print(df["moneyness"].max())

# In[]

##### Visualización Gráfica implied Vol (continuidad, Put-Call)
# Columna izquierda : smile C (azul) y P (rojo) superpuestos
####################################################################################################
N_PLOTS = 10
dates_all = df["Date"].unique()
random_dates = sorted(pd.to_datetime(
    np.random.choice(dates_all, size=N_PLOTS, replace=False)
))

fig, axes = plt.subplots(N_PLOTS // 2, 2, figsize=(14, 7 * (N_PLOTS // 2)),
                         constrained_layout=True)

for i, date in enumerate(random_dates):
    snap = df[df["Date"] == date].copy()

    calls = snap[snap["CallPut"] == "C"].sort_values("moneyness")
    puts  = snap[snap["CallPut"] == "P"].sort_values("moneyness")

    # ---- smile C vs P ----
    ax_l = axes[i // 2, i % 2]
    ax_l.plot(calls["moneyness"], calls["implied_vol"],
              color="steelblue", lw=1.5, label="Call")
    ax_l.plot(puts["moneyness"],  puts["implied_vol"],
              color="tomato", ls="--", lw=1.5, label="Put")

    # puntos coloreados por flag_inside_observed_range
    for sub, color in [(calls, "steelblue"), (puts, "tomato")]:
        inside  = sub[sub["flag_inside_observed_range"]]
        outside = sub[~sub["flag_inside_observed_range"]]
        ax_l.scatter(inside["moneyness"],  inside["implied_vol"],
                     color=color, s=18, zorder=3)
        ax_l.scatter(outside["moneyness"], outside["implied_vol"],
                     color="lightgray", edgecolors=color, lw=0.6, s=18, zorder=3,
                     label="extrapolado" if color == "steelblue" else "")

    ax_l.axvline(1.0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax_l.set_title(date.strftime("%Y-%m-%d") + " — Smile C vs P",
                   fontsize=9, fontweight="bold")
    ax_l.set_xlabel("Moneyness (K/F)")
    ax_l.set_ylabel("IV")
    ax_l.legend(fontsize=8)
    ax_l.grid(alpha=0.3)

  
plt.suptitle("Verificación PCP — IV smile",
             fontsize=13, fontweight="bold")
plt.show()
# In[]
