# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS COMPARATIVO EXHAUSTIVO DE MÉTODOS DE CÁLCULO DE GREEKS
# Punto de partida: opt_df_filtered
# Métodos: (1) Bucket, (2) Vencimiento exacto, (3) Delta empírica temporal
# Métricas: NaN, gamma negativa, delta fuera de rango, cobertura de moneyness
# ═══════════════════════════════════════════════════════════════════════════════

from scipy.interpolate import CubicSpline
from scipy.stats import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

opt_df = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet")

opt_df_filtered = opt_df[(opt_df["OpenInterest"] > 0) | (opt_df["Volume"] > 0)].reset_index(drop=True)

opt_df_filtered = opt_df[(opt_df["Bid"] > 0)].reset_index(drop=True)


# ── Parámetros globales ───────────────────────────────────────────────────────
MIN_DS       = 1  # movimiento mínimo del subyacente para delta empírica
v_grid       = [0, 15, 45, 105, 183, 365, np.inf]
v_edges      = pd.IntervalIndex.from_breaks(v_grid, closed="right")

MONEYNESS_BINS = [0, 0.70, 0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 1.30, np.inf]
MONEYNESS_LABELS = [
    "<0.70","0.70-0.80","0.80-0.90","0.90-0.95",
    "0.95-1.05","1.05-1.10","1.10-1.20","1.20-1.30",">1.30"
]

print("="*80)
print("ANÁLISIS COMPARATIVO DE MÉTODOS DE CÁLCULO DE GREEKS")
print("="*80)

# ══════════════════════════════════════════════════════════════════════════════
# MÉTODO 1: AGRUPACIÓN POR BUCKET DE VENCIMIENTO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*80)
print("MÉTODO 1: AGRUPACIÓN POR BUCKET")
print("─"*80)

# Preparar datos
opt_m1 = opt_df_filtered.copy()
opt_m1["bucket"] = pd.cut(opt_m1["Days"], bins=v_edges,
                           labels=False, include_lowest=True)

# Deduplicar por bucket (strike más líquido)
opt_m1 = (opt_m1
    .sort_values(["Date","bucket","CallPut","Strike","OpenInterest","Volume"],
                 ascending=[True,True,True,True,False,False])
    .drop_duplicates(subset=["Date","bucket","CallPut","Strike"])
    .sort_values(["Date","bucket","CallPut","Strike"])
    .reset_index(drop=True))

# FD centrado por bucket
g1 = opt_m1.groupby(["Date","bucket","CallPut"])
C_next = g1["MidPrice"].shift(-1)
C_prev = g1["MidPrice"].shift(1)
K_next = g1["Strike"].shift(-1)
K_prev = g1["Strike"].shift(1)

opt_m1["dc_dk"] = (C_next - C_prev) / (K_next - K_prev)
opt_m1["Delta"] = (opt_m1["MidPrice"] - opt_m1["dc_dk"]*opt_m1["Strike"]) / opt_m1["SpotPrice"]

h_L = opt_m1["Strike"] - K_prev
h_R = K_next - opt_m1["Strike"]
opt_m1["d2c_dk2"] = 2*(
    (C_next - opt_m1["MidPrice"])/(h_R*(h_L+h_R)) +
    (C_prev - opt_m1["MidPrice"])/(h_L*(h_L+h_R))
)
opt_m1["Gamma"] = (opt_m1["Strike"]/opt_m1["SpotPrice"])**2 * opt_m1["d2c_dk2"]
opt_m1["Moneyness_bin"] = pd.cut(opt_m1["Moneyness"],
                                  bins=MONEYNESS_BINS, labels=MONEYNESS_LABELS)
print(f"Filas totales M1: {len(opt_m1):,}")

# ══════════════════════════════════════════════════════════════════════════════
# MÉTODO 2: VENCIMIENTO EXACTO
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*80)
print("MÉTODO 2: VENCIMIENTO EXACTO")
print("─"*80)

# Deduplicar por vencimiento exacto
opt_m2 = (opt_df_filtered
    .sort_values(["Date","Days","CallPut","Strike","OpenInterest","Volume"],
                 ascending=[True,True,True,True,False,False])
    .drop_duplicates(subset=["Date","Days","CallPut","Strike"])
    .sort_values(["Date","Days","CallPut","Strike"])
    .reset_index(drop=True))

opt_m2["bucket"] = pd.cut(opt_m2["Days"], bins=v_edges,
                           labels=False, include_lowest=True)

# FD centrado por vencimiento exacto
g2 = opt_m2.groupby(["Date","Days","CallPut"])
C_next2 = g2["MidPrice"].shift(-1)
C_prev2 = g2["MidPrice"].shift(1)
K_next2 = g2["Strike"].shift(-1)
K_prev2 = g2["Strike"].shift(1)

opt_m2["dc_dk"] = (C_next2 - C_prev2) / (K_next2 - K_prev2)
opt_m2["Delta"] = (opt_m2["MidPrice"] - opt_m2["dc_dk"]*opt_m2["Strike"]) / opt_m2["SpotPrice"]

h_L2 = opt_m2["Strike"] - K_prev2
h_R2 = K_next2 - opt_m2["Strike"]
opt_m2["d2c_dk2"] = 2*(
    (C_next2 - opt_m2["MidPrice"])/(h_R2*(h_L2+h_R2)) +
    (C_prev2 - opt_m2["MidPrice"])/(h_L2*(h_L2+h_R2))
)
opt_m2["Gamma"] = (opt_m2["Strike"]/opt_m2["SpotPrice"])**2 * opt_m2["d2c_dk2"]
opt_m2["Moneyness_bin"] = pd.cut(opt_m2["Moneyness"],
                                  bins=MONEYNESS_BINS, labels=MONEYNESS_LABELS)
print(f"Filas totales M2: {len(opt_m2):,}")

# ══════════════════════════════════════════════════════════════════════════════
# MÉTODO 3: DELTA EMPÍRICA TEMPORAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─"*80)
print("MÉTODO 3: DELTA EMPÍRICA TEMPORAL")
print("─"*80)

opt_m3 = (opt_df_filtered
    .sort_values(["OptionID","Date"])
    .copy())

opt_m3["bucket"] = pd.cut(opt_m3["Days"], bins=v_edges,
                           labels=False, include_lowest=True)

# Primera diferencia temporal
opt_m3["dC"] = opt_m3.groupby("OptionID")["MidPrice"].diff()
opt_m3["dS"] = opt_m3.groupby("OptionID")["SpotPrice"].diff()

# Delta empírica
opt_m3["Delta"] = np.where(
    opt_m3["dS"].abs() >= MIN_DS,
    opt_m3["dC"] / opt_m3["dS"],
    np.nan
)

# Gamma empírica via segunda diferencia
opt_m3["d2C"]    = opt_m3.groupby("OptionID")["MidPrice"].diff().diff()
opt_m3["dS_lag"] = opt_m3.groupby("OptionID")["SpotPrice"].diff().shift(1)
opt_m3["Gamma"]  = np.where(
    (opt_m3["dS"].abs() >= MIN_DS) &
    (opt_m3["dS_lag"].abs() >= MIN_DS) &
    (opt_m3["dS"] * opt_m3["dS_lag"] > 0),
    opt_m3["d2C"] / (opt_m3["dS"] * opt_m3["dS_lag"]),
    np.nan
)

opt_m3["Moneyness_bin"] = pd.cut(opt_m3["Moneyness"],
                                  bins=MONEYNESS_BINS, labels=MONEYNESS_LABELS)
print(f"Filas totales M3: {len(opt_m3):,}")

# ══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════════════════

def diagnostico_metodo(df, nombre, delta_col="Delta", gamma_col="Gamma"):
    """
    Calcula todas las métricas de diagnóstico para un método dado.
    Excluye NaN y puntos donde no fue posible calcular antes de
    calcular violaciones y rangos de moneyness.
    """
    n_total = len(df)
    
    # ── NaN ──────────────────────────────────────────────────────────────────
    nan_delta = df[delta_col].isna().sum()
    nan_gamma = df[gamma_col].isna().sum()
    nan_ambos = (df[delta_col].isna() & df[gamma_col].isna()).sum()
    
    # ── Subsets válidos (excluyendo NaN e infinitos) ──────────────────────────
    valid_delta = df[
        df[delta_col].notna() & 
        np.isfinite(df[delta_col])
    ].copy()
    
    valid_gamma = df[
        df[gamma_col].notna() & 
        np.isfinite(df[gamma_col])
    ].copy()
    
    # ── Violaciones de delta ──────────────────────────────────────────────────
    viol_call = (
        (valid_delta["CallPut"]=="C") & 
        (~valid_delta[delta_col].between(0, 1))
    ).sum()
    viol_put = (
        (valid_delta["CallPut"]=="P") & 
        (~valid_delta[delta_col].between(-1, 0))
    ).sum()
    n_valid_delta = len(valid_delta)
    
    # ── Violaciones de gamma ──────────────────────────────────────────────────
    gamma_neg = (valid_gamma[gamma_col] < 0).sum()
    n_valid_gamma = len(valid_gamma)
    
    # ── Cobertura por zona de moneyness ──────────────────────────────────────
    # Para cada zona: % de observaciones con Greek válido
    mon_cob_delta = (valid_delta
        .groupby("Moneyness_bin", observed=True)
        .size()
        .reindex(MONEYNESS_LABELS, fill_value=0))
    
    mon_total = (df
        .groupby("Moneyness_bin", observed=True)
        .size()
        .reindex(MONEYNESS_LABELS, fill_value=0))
    
    mon_cob_pct = (mon_cob_delta / mon_total.replace(0, np.nan)).fillna(0)
    
    # ── Cobertura por bucket ──────────────────────────────────────────────────
    cob_bucket_delta = (valid_delta
        .groupby("bucket")
        .size()
        .div(df.groupby("bucket").size())
        .fillna(0))
    
    cob_bucket_gamma = (valid_gamma
        .groupby("bucket")
        .size()
        .div(df.groupby("bucket").size())
        .fillna(0))
    
    # ── Distribución de gamma válida ──────────────────────────────────────────
    gamma_pos_valid = valid_gamma[valid_gamma[gamma_col] >= 0][gamma_col]
    
    return {
        "nombre":           nombre,
        "n_total":          n_total,
        # NaN
        "nan_delta_n":      nan_delta,
        "nan_delta_pct":    nan_delta / n_total,
        "nan_gamma_n":      nan_gamma,
        "nan_gamma_pct":    nan_gamma / n_total,
        "nan_ambos_pct":    nan_ambos / n_total,
        # Válidos
        "n_valid_delta":    n_valid_delta,
        "n_valid_gamma":    n_valid_gamma,
        # Violaciones delta
        "viol_call_n":      viol_call,
        "viol_call_pct":    viol_call / n_valid_delta if n_valid_delta > 0 else np.nan,
        "viol_put_n":       viol_put,
        "viol_put_pct":     viol_put / n_valid_delta if n_valid_delta > 0 else np.nan,
        "viol_delta_total": (viol_call + viol_put) / n_valid_delta if n_valid_delta > 0 else np.nan,
        # Gamma negativa
        "gamma_neg_n":      gamma_neg,
        "gamma_neg_pct":    gamma_neg / n_valid_gamma if n_valid_gamma > 0 else np.nan,
        # Gamma positiva: distribución
        "gamma_p50":        gamma_pos_valid.median() if len(gamma_pos_valid) > 0 else np.nan,
        "gamma_p95":        gamma_pos_valid.quantile(0.95) if len(gamma_pos_valid) > 0 else np.nan,
        "gamma_std":        gamma_pos_valid.std() if len(gamma_pos_valid) > 0 else np.nan,
        # Cobertura por moneyness
        "cob_moneyness":    mon_cob_pct,
        # Cobertura por bucket
        "cob_bucket_delta": cob_bucket_delta,
        "cob_bucket_gamma": cob_bucket_gamma,
    }


# ── Calcular diagnósticos para los tres métodos ───────────────────────────────
print("\nCalculando diagnósticos...")
diag_m1 = diagnostico_metodo(opt_m1, "Bucket")
diag_m2 = diagnostico_metodo(opt_m2, "Venc. Exacto")
diag_m3 = diagnostico_metodo(opt_m3, "Delta Empírica")
diags   = [diag_m1, diag_m2, diag_m3]

# ══════════════════════════════════════════════════════════════════════════════
# TABLAS COMPARATIVAS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("TABLA 1: RESUMEN DE NaN POR MÉTODO")
print("="*80)
t1 = pd.DataFrame([{
    "Método":           d["nombre"],
    "N total":          f"{d['n_total']:,}",
    "NaN Delta (N)":    f"{d['nan_delta_n']:,}",
    "NaN Delta (%)":    f"{d['nan_delta_pct']:.1%}",
    "NaN Gamma (N)":    f"{d['nan_gamma_n']:,}",
    "NaN Gamma (%)":    f"{d['nan_gamma_pct']:.1%}",
    "NaN Ambos (%)":    f"{d['nan_ambos_pct']:.1%}",
} for d in diags])
print(t1.to_string(index=False))

print("\n" + "="*80)
print("TABLA 2: VIOLACIONES DE DELTA (sobre observaciones válidas, |Δ| finito)")
print("="*80)
t2 = pd.DataFrame([{
    "Método":           d["nombre"],
    "N válidas Delta":  f"{d['n_valid_delta']:,}",
    "Viol. Calls (N)":  f"{d['viol_call_n']:,}",
    "Viol. Calls (%)":  f"{d['viol_call_pct']:.1%}",
    "Viol. Puts (N)":   f"{d['viol_put_n']:,}",
    "Viol. Puts (%)":   f"{d['viol_put_pct']:.1%}",
    "Viol. Total (%)":  f"{d['viol_delta_total']:.1%}",
} for d in diags])
print(t2.to_string(index=False))

print("\n" + "="*80)
print("TABLA 3: GAMMA NEGATIVA (sobre observaciones válidas, Γ finita)")
print("="*80)
t3 = pd.DataFrame([{
    "Método":           d["nombre"],
    "N válidas Gamma":  f"{d['n_valid_gamma']:,}",
    "Gamma neg. (N)":   f"{d['gamma_neg_n']:,}",
    "Gamma neg. (%)":   f"{d['gamma_neg_pct']:.1%}",
    "Gamma p50 (pos)":  f"{d['gamma_p50']:.6f}" if pd.notna(d['gamma_p50']) else "N/A",
    "Gamma p95 (pos)":  f"{d['gamma_p95']:.6f}" if pd.notna(d['gamma_p95']) else "N/A",
    "Gamma std (pos)":  f"{d['gamma_std']:.6f}"  if pd.notna(d['gamma_std'])  else "N/A",
} for d in diags])
print(t3.to_string(index=False))

print("\n" + "="*80)
print("TABLA 4: COBERTURA POR ZONA DE MONEYNESS (% observaciones con Greek válido)")
print("="*80)
t4 = pd.DataFrame({
    "Moneyness":    MONEYNESS_LABELS,
    "Bucket":       [f"{v:.1%}" for v in diag_m1["cob_moneyness"].values],
    "Venc. Exacto": [f"{v:.1%}" for v in diag_m2["cob_moneyness"].values],
    "Delta Empír.": [f"{v:.1%}" for v in diag_m3["cob_moneyness"].values],
})
print(t4.to_string(index=False))

print("\n" + "="*80)
print("TABLA 5: COBERTURA DE DELTA POR BUCKET (% observaciones con Delta válida)")
print("="*80)
buckets_str = [str(b) for b in sorted(opt_m1["bucket"].dropna().unique())]
t5_data = {"Bucket": buckets_str}
for d in diags:
    cob = d["cob_bucket_delta"].reindex(
        sorted(opt_m1["bucket"].dropna().unique())
    ).fillna(0)
    t5_data[d["nombre"]] = [f"{v:.1%}" for v in cob.values]
t5 = pd.DataFrame(t5_data)
print(t5.to_string(index=False))

print("\n" + "="*80)
print("TABLA 6: COBERTURA DE GAMMA POR BUCKET (% observaciones con Gamma válida)")
print("="*80)
t6_data = {"Bucket": buckets_str}
for d in diags:
    cob = d["cob_bucket_gamma"].reindex(
        sorted(opt_m1["bucket"].dropna().unique())
    ).fillna(0)
    t6_data[d["nombre"]] = [f"{v:.1%}" for v in cob.values]
t6 = pd.DataFrame(t6_data)
print(t6.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# FIGURA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 26))
fig.suptitle(
    "Comparación exhaustiva de métodos de cálculo de Greeks\n"
    "Bucket vs Vencimiento Exacto vs Delta Empírica",
    fontsize=15, fontweight="bold", y=0.99
)
gs = GridSpec(4, 3, figure=fig, hspace=0.55, wspace=0.40)

nombres  = ["Bucket", "Venc. Exacto", "Delta Empírica"]
colors_m = ["#D85A30", "#378ADD", "#1D9E75"]

# ── Panel A: NaN Delta por método ────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
nan_d_pct = [d["nan_delta_pct"]*100 for d in diags]
bars = ax_a.bar(nombres, nan_d_pct, color=colors_m,
                alpha=0.8, edgecolor="black", lw=0.5)
ax_a.set_title("A) % NaN en Delta", fontsize=11)
ax_a.set_ylabel("%")
ax_a.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, nan_d_pct):
    ax_a.text(bar.get_x()+bar.get_width()/2,
              bar.get_height()+0.2,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# ── Panel B: NaN Gamma por método ────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
nan_g_pct = [d["nan_gamma_pct"]*100 for d in diags]
bars = ax_b.bar(nombres, nan_g_pct, color=colors_m,
                alpha=0.8, edgecolor="black", lw=0.5)
ax_b.set_title("B) % NaN en Gamma", fontsize=11)
ax_b.set_ylabel("%")
ax_b.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, nan_g_pct):
    ax_b.text(bar.get_x()+bar.get_width()/2,
              bar.get_height()+0.2,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# ── Panel C: Violaciones delta por método ────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
viol_d = [d["viol_delta_total"]*100 for d in diags]
bars = ax_c.bar(nombres, viol_d, color=colors_m,
                alpha=0.8, edgecolor="black", lw=0.5)
ax_c.set_title("C) % Delta fuera de [−1,1]\n(sobre válidas)", fontsize=11)
ax_c.set_ylabel("%")
ax_c.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, viol_d):
    ax_c.text(bar.get_x()+bar.get_width()/2,
              bar.get_height()+0.1,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# ── Panel D: Gamma negativa por método ───────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
gamma_n = [d["gamma_neg_pct"]*100 for d in diags]
bars = ax_d.bar(nombres, gamma_n, color=colors_m,
                alpha=0.8, edgecolor="black", lw=0.5)
ax_d.set_title("D) % Gamma negativa\n(sobre válidas)", fontsize=11)
ax_d.set_ylabel("%")
ax_d.grid(axis="y", alpha=0.3)
for bar, val in zip(bars, gamma_n):
    ax_d.text(bar.get_x()+bar.get_width()/2,
              bar.get_height()+0.2,
              f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

# ── Panel E: Gamma negativa por bucket — los 3 métodos ───────────────────────
ax_e = fig.add_subplot(gs[1, 1:])
x = np.arange(len(buckets_str))
width = 0.25
for i, (d, df_m, color) in enumerate(zip(
        diags,
        [opt_m1, opt_m2, opt_m3],
        colors_m)):
    vals = []
    for b in sorted(opt_m1["bucket"].dropna().unique()):
        sub = df_m[df_m["bucket"]==b]["Gamma"]
        valid_sub = sub[sub.notna() & np.isfinite(sub)]
        vals.append((valid_sub < 0).mean()*100 if len(valid_sub) > 0 else 0)
    ax_e.bar(x + i*width, vals, width=width,
             label=d["nombre"], color=color,
             alpha=0.8, edgecolor="black", lw=0.3)

ax_e.set_title("E) % Gamma negativa por bucket y método", fontsize=11)
ax_e.set_ylabel("%")
ax_e.set_xticks(x + width)
ax_e.set_xticklabels(buckets_str, rotation=25, ha="right", fontsize=8)
ax_e.legend(fontsize=8)
ax_e.grid(axis="y", alpha=0.3)

# ── Panel F: Cobertura de Delta por zona de moneyness ────────────────────────
ax_f = fig.add_subplot(gs[2, :])
x_m = np.arange(len(MONEYNESS_LABELS))
width_m = 0.25
for i, (d, color) in enumerate(zip(diags, colors_m)):
    vals = [d["cob_moneyness"].get(lab, 0)*100
            for lab in MONEYNESS_LABELS]
    ax_f.bar(x_m + i*width_m, vals, width=width_m,
             label=d["nombre"], color=color,
             alpha=0.8, edgecolor="black", lw=0.3)

ax_f.axhline(80, color="red", ls="--", lw=1.2,
             label="80% referencia")
ax_f.set_title("F) % cobertura de Delta por zona de moneyness", fontsize=11)
ax_f.set_ylabel("% con Delta válida")
ax_f.set_xticks(x_m + width_m)
ax_f.set_xticklabels(MONEYNESS_LABELS, rotation=30, ha="right", fontsize=9)
ax_f.legend(fontsize=8, ncol=4)
ax_f.grid(axis="y", alpha=0.3)
ax_f.set_ylim(0, 115)

# ── Panel G: Evolución temporal de gamma negativa ────────────────────────────
ax_g = fig.add_subplot(gs[3, :])
for df_m, d, color, ls in zip(
        [opt_m1, opt_m2, opt_m3],
        diags, colors_m,
        ["-","--","-."]):
    serie = (df_m
        .assign(year=df_m["Date"].dt.year)
        .groupby("year")
        .apply(lambda x: (
            x["Gamma"][x["Gamma"].notna() & np.isfinite(x["Gamma"])] < 0
        ).mean()) * 100)
    ax_g.plot(serie.index, serie.values,
              color=color, lw=2, ls=ls, marker="o", ms=4,
              label=d["nombre"])

ax_g.axvline(2016, color="black", ls=":", lw=1.5)
ax_g.text(2016.1, ax_g.get_ylim()[1]*0.85 if ax_g.get_ylim()[1] > 0 else 5,
          "SPXW\n2016", fontsize=9)
ax_g.set_title("G) Evolución temporal de % Gamma negativa por método",
               fontsize=11)
ax_g.set_ylabel("% Gamma negativa")
ax_g.set_xlabel("Año")
ax_g.legend(fontsize=9, ncol=3)
ax_g.grid(alpha=0.3)

plt.savefig(
    r"C:\Users\pablo.esparcia\Documents\comparacion_exhaustiva_metodos.png",
    dpi=150, bbox_inches="tight"
)
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# RESUMEN EJECUTIVO FINAL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("RESUMEN EJECUTIVO COMPARATIVO")
print("="*80)
for d in diags:
    print(f"""
── {d['nombre'].upper()} ──
  Observaciones totales:     {d['n_total']:>12,}
  NaN Delta:                 {d['nan_delta_pct']:>11.1%}
  NaN Gamma:                 {d['nan_gamma_pct']:>11.1%}
  Delta fuera de rango:      {d['viol_delta_total']:>11.1%}  (sobre válidas)
  Gamma negativa:            {d['gamma_neg_pct']:>11.1%}  (sobre válidas)
  Cobertura moneyness ATM:   {d['cob_moneyness'].get('0.95-1.05', 0):>11.1%}
  Cobertura moneyness OTM:   {d['cob_moneyness'].get('0.80-0.90', 0):>11.1%}
""")