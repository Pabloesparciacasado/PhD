# ═══════════════════════════════════════════════════════════════
# ANÁLISIS RIGUROSO: GREEKS POR VENCIMIENTO EXACTO
# Sin filtros — datos crudos
# ═══════════════════════════════════════════════════════════════

from scipy.interpolate import CubicSpline
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

MIN_STRIKES = 4


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

# In[]:

# ── 1. Cobertura por vencimiento exacto ──────────────────────────────────────

strikes_exacto = (opt_df_filtered
    .groupby(["Date","Days","CallPut"])["Strike"]
    .nunique()
    .reset_index()
    .rename(columns={"Strike":"n_strikes"}))

strikes_exacto["bucket"] = pd.cut(
    strikes_exacto["Days"],
    bins=v_edges, labels=False, include_lowest=True
)

print("="*70)
print("BLOQUE 1: COBERTURA POR VENCIMIENTO EXACTO")
print("="*70)

cobertura = (strikes_exacto
    .groupby("bucket")
    .agg(
        n_slices        = ("n_strikes", "count"),
        mediana_strikes = ("n_strikes", "median"),
        p5_strikes      = ("n_strikes", lambda x: x.quantile(0.05)),
        p10_strikes     = ("n_strikes", lambda x: x.quantile(0.10)),
        p25_strikes     = ("n_strikes", lambda x: x.quantile(0.25)),
        p75_strikes     = ("n_strikes", lambda x: x.quantile(0.75)),
        pct_menos4      = ("n_strikes", lambda x: (x < 4).mean()),
        pct_menos6      = ("n_strikes", lambda x: (x < 6).mean()),
        pct_menos10     = ("n_strikes", lambda x: (x < 10).mean()),
    )
    .reset_index())

print(cobertura.to_string(index=False))

# Por año
cobertura_anual = (strikes_exacto
    .assign(year=strikes_exacto["Date"].dt.year)
    .groupby(["year","bucket"])
    .agg(
        mediana_strikes = ("n_strikes", "median"),
        pct_menos4      = ("n_strikes", lambda x: (x < 4).mean()),
    )
    .reset_index())

print("\nMediana de strikes por vencimiento exacto — por año y bucket:")
print(cobertura_anual.pivot_table(
    index="year", columns="bucket",
    values="mediana_strikes"
).round(0).to_string())


# ── 2. FD centrado por vencimiento exacto ────────────────────────────────────

print("\n")
print("="*70)
print("BLOQUE 2: FD CENTRADO POR VENCIMIENTO EXACTO")
print("="*70)

# Calcular FD centrado sobre opt_df_filtered agrupando por vencimiento exacto
opt_exact = opt_df_filtered.sort_values(
    ["Date","Days","CallPut","Strike"]).copy()

g_exact = opt_exact.groupby(["Date","Days","CallPut"])

C_next = g_exact["MidPrice"].shift(-1)
C_prev = g_exact["MidPrice"].shift(1)
K_next = g_exact["Strike"].shift(-1)
K_prev = g_exact["Strike"].shift(1)

opt_exact["dc_dk_ctr"] = (C_next - C_prev) / (K_next - K_prev)
opt_exact["Delta_ctr_exact"] = (
    (opt_exact["MidPrice"] - opt_exact["dc_dk_ctr"] * opt_exact["Strike"])
    / opt_exact["SpotPrice"]
)

h_L = opt_exact["Strike"] - K_prev
h_R = K_next - opt_exact["Strike"]

opt_exact["d2c_dk2_ctr"] = 2 * (
    (C_next - opt_exact["MidPrice"]) / (h_R * (h_L + h_R)) +
    (C_prev - opt_exact["MidPrice"]) / (h_L * (h_L + h_R))
)
opt_exact["Gamma_ctr_exact"] = (
    (opt_exact["Strike"] / opt_exact["SpotPrice"])**2
    * opt_exact["d2c_dk2_ctr"]
)

opt_exact["bucket"] = pd.cut(
    opt_exact["Days"], bins=v_edges,
    labels=False, include_lowest=True
)

# Descriptivos
for greek in ["Delta_ctr_exact","Gamma_ctr_exact"]:
    print(f"\n--- {greek} ---")
    print(opt_exact[greek].describe(
        percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(4))
    print("\nNaN por bucket:")
    print(opt_exact.groupby("bucket")[greek]
          .apply(lambda x: x.isna().mean()).round(4))

# Violaciones
delta_viol_call = (
    (opt_exact["CallPut"]=="C") &
    (~opt_exact["Delta_ctr_exact"].between(0,1))
).mean()
delta_viol_put = (
    (opt_exact["CallPut"]=="P") &
    (~opt_exact["Delta_ctr_exact"].between(-1,0))
).mean()
gamma_neg = (opt_exact["Gamma_ctr_exact"] < 0).mean()

print(f"\nDelta imposible calls: {delta_viol_call:.1%}")
print(f"Delta imposible puts:  {delta_viol_put:.1%}")
print(f"Gamma negativa:        {gamma_neg:.1%}")

print("\nGamma negativa por bucket:")
print(opt_exact.groupby("bucket")["Gamma_ctr_exact"]
      .apply(lambda x: (x<0).mean()).round(4))


# ── 3. Spline sobre vol implícita + Bates por vencimiento exacto ─────────────

print("\n")
print("="*70)
print("BLOQUE 3: BATES SOBRE VOL IMPLÍCITA POR VENCIMIENTO EXACTO")
print("="*70)

def bates_greeks_exacto(group):
    g = group.sort_values("Strike").copy()
    for col in ["Delta_bates","Gamma_bates","dSigma_dK"]:
        g[col] = np.nan

    if len(g) < MIN_STRIKES:
        return g

    try:
        S   = g["SpotPrice"].iloc[0]
        K   = g["Strike"].values
        sig = g["ImpliedVolatility"].values

        valid = ~np.isnan(sig)
        if valid.sum() < MIN_STRIKES:
            return g

        K_v   = K[valid]
        sig_v = sig[valid]

        if np.any(np.diff(K_v) <= 0):
            return g

        cs     = CubicSpline(K_v, sig_v, bc_type="natural")
        dsdK   = cs(K_v, nu=1)
        d2sdK2 = cs(K_v, nu=2)

        dsdS   = -(K_v / S) * dsdK
        d2sdS2 =  (K_v / S)**2 * d2sdK2

        Delta_BS = g["Delta"].values[valid]
        Gamma_BS = g["Gamma"].values[valid]
        Vega_BS  = g["Vega"].values[valid]

        T    = g["Days"].iloc[0] / 252.0
        d1   = (np.log(S/K_v) + 0.5*sig_v**2*T) / (sig_v*np.sqrt(T))
        d2_  = d1 - sig_v*np.sqrt(T)
        Vanna_BS = -norm.pdf(d1) * d2_ / sig_v

        g.loc[g.index[valid], "Delta_bates"] = (
            Delta_BS + Vega_BS * dsdS)
        g.loc[g.index[valid], "Gamma_bates"] = (
            Gamma_BS
            + 2 * Vega_BS  * d2sdS2
            + Vanna_BS     * dsdS**2)
        g.loc[g.index[valid], "dSigma_dK"] = dsdK

    except Exception:
        pass

    return g

# Aplicar por vencimiento exacto
print("Calculando Bates por vencimiento exacto...")
result_bates = []
for (date, days, cp), grp in opt_df_filtered.groupby(
        ["Date","Days","CallPut"]):
    res = bates_greeks_exacto(grp)
    res["Date"]    = date
    res["Days"]    = days
    res["CallPut"] = cp
    result_bates.append(res)

result_bates = pd.concat(result_bates).reset_index(drop=True)
result_bates["bucket"] = pd.cut(
    result_bates["Days"], bins=v_edges,
    labels=False, include_lowest=True
)

# Descriptivos Bates
for greek in ["Delta_bates","Gamma_bates"]:
    print(f"\n--- {greek} ---")
    print(result_bates[greek].describe(
        percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(4))
    print("\nNaN por bucket:")
    print(result_bates.groupby("bucket")[greek]
          .apply(lambda x: x.isna().mean()).round(4))

# Violaciones Bates
db_viol_call = (
    (result_bates["CallPut"]=="C") &
    (~result_bates["Delta_bates"].between(0,1)) &
    result_bates["Delta_bates"].notna()
).mean()
db_viol_put = (
    (result_bates["CallPut"]=="P") &
    (~result_bates["Delta_bates"].between(-1,0)) &
    result_bates["Delta_bates"].notna()
).mean()
gb_neg = (result_bates["Gamma_bates"] < 0).mean()

print(f"\nDelta imposible calls (Bates): {db_viol_call:.1%}")
print(f"Delta imposible puts  (Bates): {db_viol_put:.1%}")
print(f"Gamma negativa        (Bates): {gb_neg:.1%}")

print("\nGamma negativa por bucket (Bates):")
print(result_bates.groupby("bucket")["Gamma_bates"]
      .apply(lambda x: (x<0).mean()).round(4))


# ── 4. Comparación: bucket vs exacto ─────────────────────────────────────────

print("\n")
print("="*70)
print("BLOQUE 4: COMPARACIÓN BUCKET vs VENCIMIENTO EXACTO")
print("="*70)

comparacion = pd.DataFrame({
    "Método": [
        "FD bucket (centrado)",
        "FD exacto (centrado)",
        "Bates exacto (vol impl.)",
        "OptionMetrics (BS)",
    ],
    "Gamma_neg_%": [
        (opt_agg["Gamma_ctr"] < 0).mean(),
        (opt_exact["Gamma_ctr_exact"] < 0).mean(),
        (result_bates["Gamma_bates"] < 0).mean(),
        0.0,  # por construcción BS gamma >= 0
    ],
    "Delta_viol_%": [
        (opt_agg["delta_imposible"]).mean(),
        (delta_viol_call + delta_viol_put),
        (db_viol_call + db_viol_put),
        0.0,
    ],
    "NaN_%": [
        opt_agg["Gamma_ctr"].isna().mean(),
        opt_exact["Gamma_ctr_exact"].isna().mean(),
        result_bates["Gamma_bates"].isna().mean(),
        0.0,
    ],
})

print(comparacion.to_string(index=False))


# ── 5. Estabilidad temporal ───────────────────────────────────────────────────

print("\n")
print("="*70)
print("BLOQUE 5: ESTABILIDAD TEMPORAL")
print("="*70)

for nombre, df, col in [
    ("FD bucket",      opt_agg,       "Gamma_ctr"),
    ("FD exacto",      opt_exact,     "Gamma_ctr_exact"),
    ("Bates exacto",   result_bates,  "Gamma_bates"),
    ("OptionMetrics",  opt_df_filtered, "Gamma"),
]:
    stds = []
    for cp in ["C","P"]:
        std = (df[df["CallPut"]==cp]
               .groupby("Date")[col].median()
               .diff().std())
        stds.append(std)
    print(f"{nombre:<25} Calls={stds[0]:.6f}  Puts={stds[1]:.6f}")


# ── 6. Figura diagnóstica completa ────────────────────────────────────────────

fig = plt.figure(figsize=(20, 24))
fig.suptitle(
    "Análisis riguroso: Greeks por vencimiento exacto vs bucket\n"
    "Sin filtros — datos crudos",
    fontsize=14, fontweight="bold", y=0.99
)
gs = GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)

buckets    = sorted(opt_df_filtered["bucket"].dropna().unique())
colors     = plt.cm.tab10(np.linspace(0, 1, len(buckets)))
bucket_str = [str(b) for b in buckets]

# Panel A: Mediana de strikes por vencimiento exacto
ax_a = fig.add_subplot(gs[0, 0])
med_strikes = strikes_exacto.groupby("bucket")["n_strikes"].median()
ax_a.bar(bucket_str, med_strikes.values,
         color=colors, alpha=0.8, edgecolor="black", lw=0.5)
ax_a.axhline(MIN_STRIKES, color="red", ls="--", lw=1.5,
             label=f"Mínimo viable ({MIN_STRIKES})")
ax_a.set_title("A) Mediana de strikes por vencimiento exacto", fontsize=11)
ax_a.set_ylabel("N strikes")
ax_a.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=8)
ax_a.legend(fontsize=8)
ax_a.grid(axis="y", alpha=0.3)

# Panel B: % slices con menos de 4 strikes
ax_b = fig.add_subplot(gs[0, 1])
pct_menos4 = strikes_exacto.groupby("bucket")["n_strikes"].apply(
    lambda x: (x<4).mean())
ax_b.bar(bucket_str, pct_menos4.values * 100,
         color=colors, alpha=0.8, edgecolor="black", lw=0.5)
ax_b.set_title("B) % vencimientos con < 4 strikes (inviables)", fontsize=11)
ax_b.set_ylabel("% inviables")
ax_b.set_xticklabels(bucket_str, rotation=30, ha="right", fontsize=8)
ax_b.grid(axis="y", alpha=0.3)

# Panel C: Gamma negativa — comparación de métodos
ax_c = fig.add_subplot(gs[1, 0])
metodos = ["FD bucket", "FD exacto", "Bates exacto", "OptionMetrics"]
gamma_neg_vals = [
    (opt_agg["Gamma_ctr"] < 0).mean() * 100,
    (opt_exact["Gamma_ctr_exact"] < 0).mean() * 100,
    (result_bates["Gamma_bates"] < 0).mean() * 100,
    0.0,
]
colors_met = ["#D85A30","#F5A623","#378ADD","#1D9E75"]
bars = ax_c.bar(metodos, gamma_neg_vals,
                color=colors_met, alpha=0.8, edgecolor="black", lw=0.5)
ax_c.set_title("C) % gamma negativa por método", fontsize=11)
ax_c.set_ylabel("% gamma negativa")
ax_c.set_xticklabels(metodos, rotation=20, ha="right", fontsize=9)
ax_c.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, gamma_neg_vals):
    ax_c.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.3,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# Panel D: Delta imposible — comparación de métodos
ax_d = fig.add_subplot(gs[1, 1])
delta_viol_vals = [
    opt_agg["delta_imposible"].mean() * 100,
    (delta_viol_call + delta_viol_put) * 100,
    (db_viol_call + db_viol_put) * 100,
    0.0,
]
bars_d = ax_d.bar(metodos, delta_viol_vals,
                  color=colors_met, alpha=0.8, edgecolor="black", lw=0.5)
ax_d.set_title("D) % delta fuera de rango por método", fontsize=11)
ax_d.set_ylabel("% delta imposible")
ax_d.set_xticklabels(metodos, rotation=20, ha="right", fontsize=9)
ax_d.grid(axis="y", alpha=0.3)
for bar, val in zip(bars_d, delta_viol_vals):
    ax_d.text(bar.get_x() + bar.get_width()/2,
              bar.get_height() + 0.1,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# Panel E: Distribución Gamma — FD exacto vs Bates exacto
ax_e = fig.add_subplot(gs[2, :])
p1_fd  = opt_exact["Gamma_ctr_exact"].quantile(0.01)
p99_fd = opt_exact["Gamma_ctr_exact"].quantile(0.99)
p1_bt  = result_bates["Gamma_bates"].quantile(0.01)
p99_bt = result_bates["Gamma_bates"].quantile(0.99)
p_low  = min(p1_fd, p1_bt)
p_high = max(p99_fd, p99_bt)

ax_e.hist(opt_exact["Gamma_ctr_exact"].clip(p_low, p_high).dropna(),
          bins=300, alpha=0.5, density=True, color="#D85A30",
          label="FD centrado exacto")
ax_e.hist(result_bates["Gamma_bates"].clip(p_low, p_high).dropna(),
          bins=300, alpha=0.5, density=True, color="#378ADD",
          label="Bates exacto (vol impl.)")
ax_e.axvline(0, color="black", lw=1.5, ls="--", label="Gamma=0")
ax_e.set_title("E) Distribución de Gamma — FD exacto vs Bates exacto (p1-p99)",
               fontsize=11)
ax_e.set_xlabel("Gamma")
ax_e.set_ylabel("Densidad")
ax_e.legend(fontsize=9)
ax_e.grid(alpha=0.3)

# Panel F: Estabilidad temporal — mediana diaria de Gamma
ax_f = fig.add_subplot(gs[3, :])

for nombre, df, col, color, ls in [
    ("FD bucket",    opt_agg,        "Gamma_ctr",       "#D85A30", "-"),
    ("FD exacto",    opt_exact,      "Gamma_ctr_exact", "#F5A623", "--"),
    ("Bates exacto", result_bates,   "Gamma_bates",     "#378ADD", "-."),
    ("OptionMetrics",opt_df_filtered,"Gamma",            "#1D9E75", ":"),
]:
    serie = (df[df["CallPut"]=="C"]
             .groupby("Date")[col]
             .median())
    ax_f.plot(serie.index, serie.values,
              lw=0.8, color=color, ls=ls, label=nombre, alpha=0.85)

ax_f.axvline(pd.to_datetime("2016-01-01"), color="black",
             ls=":", lw=1.2)
ax_f.text(pd.to_datetime("2016-03-01"),
          ax_f.get_ylim()[1]*0.85,
          "SPXW\n2016", fontsize=8)
ax_f.set_title("F) Mediana diaria de Gamma (calls) — todos los métodos",
               fontsize=11)
ax_f.set_ylabel("Gamma mediana")
ax_f.xaxis.set_major_locator(mdates.YearLocator(2))
ax_f.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax_f.tick_params(axis="x", rotation=45)
ax_f.legend(fontsize=8, ncol=4)
ax_f.grid(alpha=0.3)

plt.savefig(
    r"C:\Users\pablo.esparcia\Documents\analisis_vencimiento_exacto.png",
    dpi=150, bbox_inches="tight"
)
plt.show()

# ── 7. Resumen ejecutivo ──────────────────────────────────────────────────────

print("\n")
print("="*70)
print("RESUMEN EJECUTIVO: VENCIMIENTO EXACTO")
print("="*70)
print(f"""
COBERTURA:
- Mediana de strikes por vencimiento: {strikes_exacto['n_strikes'].median():.0f}
- % vencimientos con < 4 strikes: {(strikes_exacto['n_strikes']<4).mean():.1%}
- % vencimientos con < 6 strikes: {(strikes_exacto['n_strikes']<6).mean():.1%}

CALIDAD DE GREEKS (sin filtros):

  FD centrado exacto:
  - Gamma negativa:    {(opt_exact['Gamma_ctr_exact']<0).mean():.1%}
  - Delta imposible:   {(delta_viol_call+delta_viol_put):.1%}
  - NaN Gamma:         {opt_exact['Gamma_ctr_exact'].isna().mean():.1%}

  Bates exacto (vol implícita):
  - Gamma negativa:    {(result_bates['Gamma_bates']<0).mean():.1%}
  - Delta imposible:   {(db_viol_call+db_viol_put):.1%}
  - NaN Gamma:         {result_bates['Gamma_bates'].isna().mean():.1%}

  OptionMetrics (BS):
  - Gamma negativa:    0.0%
  - Delta imposible:   0.0%
  - NaN Gamma:         {opt_df_filtered['Gamma'].isna().mean():.1%}
""")
# %%

####################################################################
# Analisis de duplicados
####################################################################


# ¿Hay duplicados de strike por vencimiento exacto?
dup_exacto = (opt_df_filtered
    .groupby(["Date","Days","CallPut","Strike"])
    .size()
    .reset_index(name="n"))

print(f"Strikes duplicados: {(dup_exacto['n']>1).sum():,}")
print(f"% sobre total:      {(dup_exacto['n']>1).mean():.4%}")

# Ver ejemplos
print("\nEjemplos de duplicados:")
print(dup_exacto[dup_exacto['n']>1].head(10))
# %%
##########################################################################
#Recalculamos quitando duplicados. mantenemos los puntos mas liquidos.
##########################################################################

# Deduplicar por vencimiento exacto
opt_df_filtered_dedup = (opt_df_filtered
    .sort_values(["Date","Days","CallPut","Strike",
                  "OpenInterest","Volume"],
                 ascending=[True,True,True,True,False,False])
    .drop_duplicates(subset=["Date","Days","CallPut","Strike"])
    .sort_values(["Date","Days","CallPut","Strike"])
    .reset_index(drop=True))

print(f"Filas originales:      {len(opt_df_filtered):,}")
print(f"Filas deduplicadas:    {len(opt_df_filtered_dedup):,}")
print(f"Duplicados eliminados: {len(opt_df_filtered)-len(opt_df_filtered_dedup):,}")

# Recalcular FD centrado exacto sobre datos deduplicados
opt_exact_dd = opt_df_filtered_dedup.sort_values(
    ["Date","Days","CallPut","Strike"]).copy()

g_dd = opt_exact_dd.groupby(["Date","Days","CallPut"])

C_next = g_dd["MidPrice"].shift(-1)
C_prev = g_dd["MidPrice"].shift(1)
K_next = g_dd["Strike"].shift(-1)
K_prev = g_dd["Strike"].shift(1)

opt_exact_dd["dc_dk_ctr"] = (C_next - C_prev) / (K_next - K_prev)
opt_exact_dd["Delta_ctr_exact"] = (
    (opt_exact_dd["MidPrice"] - opt_exact_dd["dc_dk_ctr"] * opt_exact_dd["Strike"])
    / opt_exact_dd["SpotPrice"]
)

h_L = opt_exact_dd["Strike"] - K_prev
h_R = K_next - opt_exact_dd["Strike"]

opt_exact_dd["d2c_dk2_ctr"] = 2 * (
    (C_next - opt_exact_dd["MidPrice"]) / (h_R * (h_L + h_R)) +
    (C_prev - opt_exact_dd["MidPrice"]) / (h_L * (h_L + h_R))
)
opt_exact_dd["Gamma_ctr_exact"] = (
    (opt_exact_dd["Strike"] / opt_exact_dd["SpotPrice"])**2
    * opt_exact_dd["d2c_dk2_ctr"]
)

opt_exact_dd["bucket"] = pd.cut(
    opt_exact_dd["Days"], bins=v_edges,
    labels=False, include_lowest=True
)

# Diagnóstico post-deduplicación
inf_mask = np.isinf(opt_exact_dd["Gamma_ctr_exact"])
print(f"\nValores infinitos tras dedup: {inf_mask.sum():,}")

print("\nDelta imposible calls:", 
      ((opt_exact_dd["CallPut"]=="C") & 
       (~opt_exact_dd["Delta_ctr_exact"].between(0,1))).mean())
print("Delta imposible puts:", 
      ((opt_exact_dd["CallPut"]=="P") & 
       (~opt_exact_dd["Delta_ctr_exact"].between(-1,0))).mean())
print("Gamma negativa:", 
      (opt_exact_dd["Gamma_ctr_exact"] < 0).mean())

print("\nGamma negativa por bucket:")
print(opt_exact_dd.groupby("bucket")["Gamma_ctr_exact"]
      .apply(lambda x: (x<0).mean()).round(4))

print("\nGamma — descriptivos:")
print(opt_exact_dd["Gamma_ctr_exact"].describe(
    percentiles=[.01,.05,.25,.5,.75,.95,.99]))
# %%

##################################################################
# Verificación:
##################################################################

neg_gamma = opt_exact_dd[opt_exact_dd["Gamma_ctr_exact"] < 0]

print("=== DIAGNÓSTICO GAMMA NEGATIVA ===")
print(f"\nTotal negativas: {len(neg_gamma):,} ({len(neg_gamma)/len(opt_exact_dd):.1%})")

print("\nDistribución de las gammas negativas:")
print(neg_gamma["Gamma_ctr_exact"].describe(
    percentiles=[.05,.25,.5,.75,.95,.99]))

# ¿Cuántas son casi cero?
umbrales = [0.0001, 0.001, 0.005, 0.01]
for u in umbrales:
    pct = (neg_gamma["Gamma_ctr_exact"].abs() < u).mean()
    print(f"  |Gamma| < {u}: {pct:.1%} de las negativas")

# ¿Dónde están en moneyness?
print("\nMoneyness de las negativas:")
print(neg_gamma["Moneyness"].describe(
    percentiles=[.05,.25,.5,.75,.95]))

# ¿Concentradas en opciones OTM?
print("\n% negativas que son deep OTM (moneyness < 0.85 o > 1.15):")
deep_otm = (
    (neg_gamma["Moneyness"] < 0.4) |
    (neg_gamma["Moneyness"] > 1.4)
).mean()
print(f"{deep_otm:.1%}")


# %%

##################################################################
# Diagnostico 2
###################################################################

# Filtro mínimo: eliminar gammas negativas pequeñas
# manteniendo solo las positivas significativas

GAMMA_MIN = 0.0  # no negativa
GAMMA_MAX = opt_exact_dd["Gamma_ctr_exact"].quantile(0.995)  # sin extremos

opt_exact_dd["Gamma_ctr_clean"] = opt_exact_dd["Gamma_ctr_exact"].where(
    (opt_exact_dd["Gamma_ctr_exact"] >= GAMMA_MIN) &
    (opt_exact_dd["Gamma_ctr_exact"] <= GAMMA_MAX)
)

# ¿Cuánto perdemos?
nan_pct = opt_exact_dd["Gamma_ctr_clean"].isna().mean()
print(f"NaN tras filtro: {nan_pct:.1%}")
print(f"Observaciones válidas: {opt_exact_dd['Gamma_ctr_clean'].notna().sum():,}")

# Descriptivos post-filtro
print("\nGamma limpia:")
print(opt_exact_dd["Gamma_ctr_clean"].describe(
    percentiles=[.01,.05,.25,.5,.75,.95,.99]))

# Estabilidad temporal post-filtro
std_daily = (opt_exact_dd[opt_exact_dd["CallPut"]=="C"]
             .groupby("Date")["Gamma_ctr_clean"]
             .median()
             .diff()
             .std())
print(f"\nEstabilidad temporal (std cambios diarios): {std_daily:.6f}")

# Comparación con OptionMetrics
corr_om = (opt_exact_dd
           .dropna(subset=["Gamma_ctr_clean","Gamma"])
           [["Gamma_ctr_clean","Gamma"]]
           .corr()
           .iloc[0,1])
print(f"Correlación con Gamma OptionMetrics: {corr_om:.3f}")

mae_om = (opt_exact_dd
          .dropna(subset=["Gamma_ctr_clean","Gamma"])
          .eval("abs(Gamma_ctr_clean - Gamma)")
          .median())
print(f"MAE respecto a OptionMetrics: {mae_om:.6f}")
# %%

######################################################################
## Pruebas de agrupamiento pequeño
######################################################################
def agrupar_ventana_dias(day_data, ventana=5, min_strikes=MIN_STRIKES):
    """
    Agrupa vencimientos dentro de una ventana de ±ventana días.
    El vencimiento ancla es el más líquido (mayor OI total).
    No mezcla vencimientos separados por más de ventana días.
    """
    day_data = day_data.copy()
    day_data["grupo_T"] = np.nan

    for cp in ["C", "P"]:
        sub = day_data[day_data["CallPut"] == cp]
        if sub.empty:
            continue

        T_vals = sorted(sub["Days"].unique())
        usado  = set()

        # OI total por vencimiento para elegir el ancla
        oi_por_T = (sub.groupby("Days")["OpenInterest"]
                       .sum()
                       .to_dict())

        for T_ancla in sorted(T_vals, 
                               key=lambda t: oi_por_T.get(t, 0),
                               reverse=True):
            if T_ancla in usado:
                continue

            # Añadir vencimientos dentro de la ventana
            grupo = [T for T in T_vals
                     if abs(T - T_ancla) <= ventana
                     and T not in usado]

            if not grupo:
                continue

            for T in grupo:
                usado.add(T)

            mask = (
                day_data["Days"].isin(grupo) &
                (day_data["CallPut"] == cp)
            )
            day_data.loc[mask, "grupo_T"] = T_ancla

    return day_data


# ── Aplicar ───────────────────────────────────────────────────────────────────
print("Aplicando agrupación por ventana ±5 días...")
dias = []
for date, day_data in opt_df_filtered_dedup.groupby("Date"):
    res = agrupar_ventana_dias(day_data, ventana=5)
    res["Date"] = date
    dias.append(res)

result_v5 = pd.concat(dias).reset_index(drop=True)
print(f"Cobertura: {result_v5['grupo_T'].notna().mean():.1%}")

# Diagnóstico de grupos
n_grupos = (result_v5.dropna(subset=["grupo_T"])
            .groupby(["Date","grupo_T","CallPut"])["Strike"]
            .nunique()
            .reset_index()
            .rename(columns={"Strike":"n_K"}))

print(f"\nGrupos totales: {len(n_grupos):,}")
print(f"Mediana strikes/grupo: {n_grupos['n_K'].median():.0f}")
print(f"% grupos >= 4K: {(n_grupos['n_K']>=4).mean():.1%}")
print(f"% grupos >= 10K: {(n_grupos['n_K']>=10).mean():.1%}")

print("\nDistribución strikes por grupo:")
print(n_grupos["n_K"].describe(
    percentiles=[.05,.10,.25,.5,.75,.90,.95]))

# Verificar que la mezcla de theta es mínima
# Calcular ratio de theta entre vencimientos del mismo grupo
print("\n=== HOMOGENEIDAD DE THETA DENTRO DE GRUPOS ±5 DÍAS ===")
theta_check = (result_v5.dropna(subset=["grupo_T"])
    .assign(dist_atm=(result_v5["Moneyness"]-1).abs())
    .sort_values("dist_atm")
    .groupby(["Date","grupo_T","CallPut","Days"])["Theta"]
    .first()
    .reset_index())

ratio_theta = (theta_check
    .groupby(["Date","grupo_T","CallPut"])
    .apply(lambda x: x["Theta"].max() / x["Theta"].min()
           if len(x) > 1 and x["Theta"].min() != 0
           else 1.0)
    .reset_index(name="ratio_theta"))

print(f"Ratio theta max/min dentro del grupo:")
print(ratio_theta["ratio_theta"].describe(
    percentiles=[.05,.25,.5,.75,.95,.99]))
print(f"\n% grupos con ratio < 1.10 (theta casi idéntica): "
      f"{(ratio_theta['ratio_theta'] < 1.10).mean():.1%}")
print(f"% grupos con ratio < 1.20: "
      f"{(ratio_theta['ratio_theta'] < 1.20).mean():.1%}")
# %%
