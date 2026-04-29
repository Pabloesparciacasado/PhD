"""
LPR Bivariado — Modo B Corregido
================================
Evalúa el estimador en cada contrato real con OI > 0.

Correcciones clave:
1) El LPR estima Y = MidPrice / SpotPrice = P/S
2) La variable de estado es m = K/S
3) Los coeficientes locales NO son directamente delta/gamma spot
4) Se transforman correctamente a:
      Delta = y - m * y_m
      Gamma = (m^2 / S) * y_mm
   donde:
      y    = P/S
      y_m  = d(P/S)/dm
      y_mm = d²(P/S)/dm²

Resultado:
- Greeks reales por contrato
- GEX / DEX agregados por OI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. PARÁMETROS
# =============================================================================

H_M         = 0.05    # bandwidth moneyness (m = K/S)
H_TAU       = 45.0    # bandwidth tau (días)
OPTION_TYPE = "P"     # "C" o "P"
MIN_OBS     = 30      # mínimo obs efectivas por punto

PATH_DATA   = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"
PATH_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output"

# =============================================================================
# 2. FUNCIONES
# =============================================================================

def epan(u):
    """Kernel Epanechnikov univariado."""
    return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)


def lpr_bivariate_fast(m0, tau0, m_sub, tau_sub, Y_sub, w_oi_sub, min_obs=30):
    """
    LPR bivariado grado 2 en (m0, tau0) sobre subconjunto pre-filtrado.

    Estima:
        Y = P/S = y(m, tau)

    Aproximación local:
        y(m, tau) ≈ β0 + β1 dm + β2 dtau + β3 dm² + β4 dtau² + β5 dm*dtau

    Devuelve:
        y_hat     = y(m0, tau0)
        dy_dm     = ∂y/∂m
        dY_dtau   = ∂y/∂tau
        d2y_dm2   = ∂²y/∂m²
        n_eff     = nº obs con peso efectivo
    """
    n = len(m_sub)
    if n < min_obs:
        return None

    # Pesos kernel Epanechnikov bivariado separable
    u_m   = (m_sub   - m0) / H_M
    u_tau = (tau_sub - tau0) / H_TAU
    w_kernel = epan(u_m) * epan(u_tau)

    # Peso combinado kernel × actividad
    w = w_kernel * w_oi_sub

    # Observaciones con peso efectivo
    n_eff = int(np.sum(w > 0.01))
    if n_eff < min_obs:
        return None

    # Desviaciones locales
    dm   = m_sub   - m0
    dtau = tau_sub - tau0

    # Matriz de diseño (grado 2)
    X = np.column_stack([
        np.ones(n),
        dm,
        dtau,
        dm**2,
        dtau**2,
        dm * dtau
    ])

    # WLS
    Xw   = X * w[:, np.newaxis]
    XtWX = Xw.T @ X
    XtWY = Xw.T @ Y_sub

    try:
        beta = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        return None

    return {
        "y_hat":    beta[0],          # y = P/S
        "dy_dm":    beta[1],          # dy/dm
        "dY_dtau":  beta[2],          # dy/dtau  (NO es vega)
        "d2y_dm2":  2.0 * beta[3],    # d²y/dm²
        "n_eff":    n_eff
    }


def transform_greeks_ks(y_hat, dy_dm, d2y_dm2, m, S):
    """
    Transformación correcta a greeks spot cuando:

        y = P/S
        m = K/S

    Entonces:
        P = S * y(m, tau)

    Manteniendo K fijo:
        Delta = ∂P/∂S = y - m * y_m
        Gamma = ∂²P/∂S² = (m² / S) * y_mm

    Devuelve:
        price_hat, delta_true, gamma_true
    """
    price_hat = S * y_hat
    delta_true = y_hat - m * dy_dm
    gamma_true = (m**2 / S) * d2y_dm2
    return price_hat, delta_true, gamma_true


# =============================================================================
# 3. CARGA Y PREPARACIÓN
# =============================================================================

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

# Filtros base
opt_df_filtered = opt_df[
    (opt_df["Bid"] > 0) &
    (opt_df["OpenInterest"] > 0)
].reset_index(drop=True)

opt_df_filtered["YearMonth"] = opt_df_filtered["Date"].dt.to_period("M")
sample_month = opt_df_filtered["YearMonth"].value_counts().idxmax()
print(f"Mes de prueba: {sample_month}")

df = opt_df_filtered[
    (opt_df_filtered["YearMonth"] == sample_month) &
    (opt_df_filtered["CallPut"] == OPTION_TYPE)
].copy()

print(f"Observaciones ({OPTION_TYPE}, OI>0): {len(df):,}")

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

df["Y"]   = df["MidPrice"] / df["SpotPrice"]
df["m"]   = df["Moneyness"].astype(float)
df["tau"] = df["Days"].astype(float)

# Pesos OI  normalizados
df["w_oi_vol"] = (df["OpenInterest"] + 0*df["Volume"]).clip(lower=0)
max_w = df["w_oi_vol"].max()
if max_w <= 0:
    df["w_oi_vol"] = 1.0
else:
    df["w_oi_vol"] = (df["w_oi_vol"] / max_w).clip(lower=1e-6)

# Arrays del pool de estimación
m_all    = df["m"].values
tau_all  = df["tau"].values
Y_all    = df["Y"].values
w_all    = df["w_oi_vol"].values

print(f"Pool de estimación: {len(m_all):,} observaciones")

# =============================================================================
# 4. PUNTOS DE EVALUACIÓN — contratos únicos con OI > 0
# =============================================================================

# Redondeo para evitar recalcular puntos casi idénticos
df["m_eval"]   = df["m"].round(3)
df["tau_eval"] = df["tau"].round(0).astype(int)

eval_points = (
    df.groupby(["m_eval", "tau_eval"])
      .agg(
          OI_total=("OpenInterest", "sum"),
          n_contracts=("OptionID", "nunique")
      )
      .reset_index()
)

# Filtro para evitar puntos anecdóticos
eval_points = eval_points[eval_points["OI_total"] >= 5].copy()

print(f"Puntos únicos de evaluación (OI_total >= 5): {len(eval_points):,}")

# =============================================================================
# 5. ESTIMACIÓN LPR
# =============================================================================

print(f"\nEstimando LPR en {len(eval_points):,} puntos únicos...")
print(f"Kernel: Epanechnikov, h_m={H_M}, h_tau={H_TAU}")
print("Pre-filtrado activo: solo obs dentro del soporte del kernel por punto")

m_eval_arr   = eval_points["m_eval"].values
tau_eval_arr = eval_points["tau_eval"].values

# Arrays de resultados
y_hats    = np.full(len(eval_points), np.nan)
dy_dms    = np.full(len(eval_points), np.nan)
dY_dtaus  = np.full(len(eval_points), np.nan)
d2y_dm2s  = np.full(len(eval_points), np.nan)
n_effs    = np.zeros(len(eval_points), dtype=int)

log_every = max(1, len(eval_points) // 20)  # log cada 5%

for i in range(len(eval_points)):

    if i % log_every == 0:
        pct = 100 * i / len(eval_points)
        print(f"  {pct:5.1f}%  ({i:,} / {len(eval_points):,})", end="\r")

    m0   = m_eval_arr[i]
    tau0 = tau_eval_arr[i]

    # Pre-filtro soporte kernel
    mask = (
        (m_all   >= m0   - H_M)   & (m_all   <= m0   + H_M) &
        (tau_all >= tau0 - H_TAU) & (tau_all <= tau0 + H_TAU)
    )

    n_local = mask.sum()
    if n_local < MIN_OBS:
        n_effs[i] = n_local
        continue

    res = lpr_bivariate_fast(
        m0, tau0,
        m_all[mask], tau_all[mask], Y_all[mask], w_all[mask],
        min_obs=MIN_OBS
    )

    if res is not None:
        y_hats[i]    = res["y_hat"]
        dy_dms[i]    = res["dy_dm"]
        dY_dtaus[i]  = res["dY_dtau"]
        d2y_dm2s[i]  = res["d2y_dm2"]
        n_effs[i]    = res["n_eff"]

print(f"\n  100.0%  ({len(eval_points):,} / {len(eval_points):,})")

# Añadir resultados a eval_points
eval_points["y_hat"]    = y_hats
eval_points["dy_dm"]    = dy_dms
eval_points["dY_dtau"]  = dY_dtaus
eval_points["d2y_dm2"]  = d2y_dm2s
eval_points["n_eff"]    = n_effs
eval_points["valid"]    = ~np.isnan(eval_points["d2y_dm2"])

valid_pct = eval_points["valid"].mean() * 100
print(f"\nPuntos válidos: {eval_points['valid'].sum():,} / {len(eval_points):,} ({valid_pct:.1f}%)")

# =============================================================================
# 6. MERGE CON DATAFRAME ORIGINAL
# =============================================================================

df = df.merge(
    eval_points[["m_eval", "tau_eval", "y_hat", "dy_dm", "d2y_dm2", "dY_dtau", "n_eff"]],
    on=["m_eval", "tau_eval"],
    how="left"
)

# =============================================================================
# 7. TRANSFORMACIÓN A GREEKS REALES
# =============================================================================

# Precio estimado, delta y gamma REALES respecto a Spot
transformed = df.apply(
    lambda row: transform_greeks_ks(
        y_hat=row["y_hat"],
        dy_dm=row["dy_dm"],
        d2y_dm2=row["d2y_dm2"],
        m=row["m_eval"],
        S=row["SpotPrice"]
    ) if pd.notna(row["y_hat"]) and row["SpotPrice"] > 0 else (np.nan, np.nan, np.nan),
    axis=1,
    result_type="expand"
)

transformed.columns = ["price_hat", "delta_true", "gamma_true"]
df[["price_hat", "delta_true", "gamma_true"]] = transformed

# Contribuciones
df["gex_contribution"] = df["gamma_true"] * df["OpenInterest"]
df["dex_contribution"] = df["delta_true"] * df["OpenInterest"]

# =============================================================================
# 8. AGREGACIÓN GEX / DEX
# =============================================================================

print("\n=== AGREGACIÓN GEX/DEX ===")

gex_total = df["gex_contribution"].sum()
dex_total = df["dex_contribution"].sum()

label_type = "puts" if OPTION_TYPE == "P" else "calls"
print(f"GEX total ({label_type}): {gex_total:>15,.6f}")
print(f"DEX total ({label_type}): {dex_total:>15,.6f}")

# Buckets de moneyness
bins_m = [0.0, 0.80, 0.90, 0.95, 1.05, 1.10, 1.20, 99]
labels_m = ["<0.80","0.80-0.90","0.90-0.95","ATM[0.95-1.05]",
            "1.05-1.10","1.10-1.20",">1.20"]
df["m_bucket"] = pd.cut(df["m"], bins=bins_m, labels=labels_m)

gex_by_m = df.groupby("m_bucket", observed=True).agg(
    GEX=("gex_contribution","sum"),
    DEX=("dex_contribution","sum"),
    OI_total=("OpenInterest","sum"),
    n_contracts=("OptionID","nunique")
).round(6)

print("\nGEX/DEX por bucket de moneyness:")
print(gex_by_m.to_string())

# Buckets de madurez
bins_tau = [0, 15, 45, 105, 183, 365, 9999]
labels_tau = ["0-15d","15-45d","45-105d","105-183d","183-365d",">365d"]
df["tau_bucket"] = pd.cut(df["tau"], bins=bins_tau, labels=labels_tau)

gex_by_tau = df.groupby("tau_bucket", observed=True).agg(
    GEX=("gex_contribution","sum"),
    DEX=("dex_contribution","sum"),
    OI_total=("OpenInterest","sum"),
    n_contracts=("OptionID","nunique")
).round(6)

print("\nGEX/DEX por bucket de madurez:")
print(gex_by_tau.to_string())

# =============================================================================
# 9. DIAGNÓSTICO DE VIOLACIONES
# =============================================================================

print("\n=== DIAGNÓSTICO DE VIOLACIONES ===")

v = df[pd.notna(df["gamma_true"])].copy()

# Violaciones según tipo de opción
if OPTION_TYPE == "C":
    delta_violation_pct = (v["delta_true"] < 0).mean() * 100
    print(f"Delta < 0 (violación en calls): {delta_violation_pct:.1f}%")
else:
    delta_violation_pct = (v["delta_true"] > 0).mean() * 100
    print(f"Delta > 0 (violación en puts):  {delta_violation_pct:.1f}%")

gamma_violation_pct = (v["gamma_true"] < 0).mean() * 100
price_violation_pct = (v["price_hat"] < 0).mean() * 100

print(f"Gamma < 0:                    {gamma_violation_pct:.1f}%")
print(f"Precio estimado < 0:          {price_violation_pct:.1f}%")

# Zonas de moneyness
v["m_zone"] = pd.cut(
    v["m_eval"],
    bins=[0, 0.85, 0.95, 1.05, 1.15, 99],
    labels=["OTM deep", "OTM", "ATM", "ITM", "ITM deep"]
)

print("\nGamma < 0 por zona de moneyness:")
print(
    v.groupby("m_zone", observed=True)["gamma_true"]
     .apply(lambda x: f"{(x < 0).mean() * 100:.1f}%")
     .to_string()
)

if OPTION_TYPE == "C":
    print("\nDelta < 0 por zona de moneyness (calls):")
    print(
        v.groupby("m_zone", observed=True)["delta_true"]
         .apply(lambda x: f"{(x < 0).mean() * 100:.1f}%")
         .to_string()
    )
else:
    print("\nDelta > 0 por zona de moneyness (puts):")
    print(
        v.groupby("m_zone", observed=True)["delta_true"]
         .apply(lambda x: f"{(x > 0).mean() * 100:.1f}%")
         .to_string()
    )

# =============================================================================
# 10. GRÁFICOS
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    f"LPR Bivariado Modo B Corregido — {OPTION_TYPE} — {sample_month}\n"
    f"Y=P/S, m=K/S, h_m={H_M}, h_τ={H_TAU}, Epanechnikov, sin restricciones",
    fontsize=12
)

valid_plot = df[pd.notna(df["gamma_true"])].copy()

# Gamma real vs moneyness
ax = axes[0, 0]
sc = ax.scatter(
    valid_plot["m_eval"],
    valid_plot["gamma_true"],
    c=valid_plot["tau_eval"],
    cmap="viridis",
    s=3,
    alpha=0.5
)
plt.colorbar(sc, ax=ax, label="tau (días)")
ax.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("Moneyness K/S")
ax.set_ylabel("Gamma real")
ax.set_title("Gamma real vs Moneyness (color = tau)")
ax.set_xlim(0.70, 1.30)

# Delta real vs moneyness
ax = axes[0, 1]
ax.scatter(
    valid_plot["m_eval"],
    valid_plot["delta_true"],
    c=valid_plot["tau_eval"],
    cmap="viridis",
    s=3,
    alpha=0.5
)
ax.axhline(0, color="red", ls="--", lw=1.5, alpha=0.7)
ax.set_xlabel("Moneyness K/S")
ax.set_ylabel("Delta real")
ax.set_title("Delta real vs Moneyness (color = tau)")
ax.set_xlim(0.70, 1.30)

# GEX por bucket moneyness
ax = axes[1, 0]
gex_plot = gex_by_m.reset_index()
colors = ["red" if g < 0 else "steelblue" for g in gex_plot["GEX"]]
ax.bar(range(len(gex_plot)), gex_plot["GEX"], color=colors, alpha=0.8)
ax.set_xticks(range(len(gex_plot)))
ax.set_xticklabels(gex_plot["m_bucket"], rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("GEX por bucket moneyness")
ax.set_ylabel("GEX (gamma real × OI)")

# GEX por bucket madurez
ax = axes[1, 1]
gex_tau_plot = gex_by_tau.reset_index()
colors = ["red" if g < 0 else "steelblue" for g in gex_tau_plot["GEX"]]
ax.bar(range(len(gex_tau_plot)), gex_tau_plot["GEX"], color=colors, alpha=0.8)
ax.set_xticks(range(len(gex_tau_plot)))
ax.set_xticklabels(gex_tau_plot["tau_bucket"], rotation=45, ha="right", fontsize=8)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("GEX por bucket madurez")
ax.set_ylabel("GEX (gamma real × OI)")

plt.tight_layout()
fname_plot = f"{PATH_OUTPUT}\\lpr_modoB_corregido_{sample_month}_{OPTION_TYPE}.png"
plt.savefig(fname_plot, dpi=150, bbox_inches="tight")
plt.show()

print(f"\nGráfico guardado: {fname_plot}")

# =============================================================================
# 11. OUTPUTS
# =============================================================================

# Puntos de evaluación
fname_eval = f"{PATH_OUTPUT}\\lpr_eval_points_corregido_{sample_month}_{OPTION_TYPE}.csv"
eval_points.to_csv(fname_eval, index=False)

# DataFrame completo con greeks reales
fname_df = f"{PATH_OUTPUT}\\lpr_contracts_corregido_{sample_month}_{OPTION_TYPE}.parquet"
df.to_parquet(fname_df, index=False)

print("\nResultados guardados:")
print(f"  {fname_eval}")
print(f"  {fname_df}")
print("\n=== FIN ===")