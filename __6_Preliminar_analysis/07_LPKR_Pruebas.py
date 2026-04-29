
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =============================================================================
# 1. PARÁMETROS
# =============================================================================


# --- Bandwidths ---
H_M = 0.05          # Bandwith global. Parametrización según Benlo et al. (2007)
H_TAU = 0.2         # Bandwith global para tau. La arametrización según Benlo et al. (2007) es una función que se incrementa dependeiendo del tiempo a vencimiento. De momento dejo fijo el valor. Se correspondería con 73 días

# --- Kernel ---
 # "epanechnikov" (Benko). en Teoría debería elegir el kernel (dado el bandwith) que minimice MSE y MISE, que en teoría vendrá de minimzar la varianza. Según el teorema 3.4 de Fan-Gijbels (1996) Epanechnikov es el optimo (bajo ciertas condiciones).
KERNEL = "epanechnikov"   #"gaussian"


# --- Filtros ---
OPTION_TYPE = "P"       # "P" puts, "C" calls

# --- Rutas ---
# PATH_DATA   = r"/Users/pablo/Library/CloudStorage/GoogleDrive-pabloesparcia.pe@gmail.com/My Drive/PhD/B_Coding/Datos_OM/preliminares/opt_df_prueba.parquet"
PATH_DATA   =r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"

# PATH_OUTPUT = r"/Users/pablo/Library/CloudStorage/GoogleDrive-pabloesparcia.pe@gmail.com/My Drive/PhD/B_Coding/Datos_OM/preliminares/output"
PATH_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares"


# =============================================================================
# 1. PARÁMETROS
# =============================================================================

# --- Bandwidths ---
H_M   = 0.05    # bandwidth en moneyness  — empieza aquí, prueba 0.08 y 0.10
H_TAU = 45.0    # bandwidth en tau (días) — empieza aquí, prueba 30 y 60

# --- Kernel ---
KERNEL = "epanechnikov"   # "epanechnikov" (Benko) o "gaussian"

# --- Grid de contratos dummy (modo A) ---
# Puntos representativos del smile donde evaluar mensualmente
M0_GRID   = np.array([0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15])
TAU0_GRID = np.array([15, 30, 60, 90, 180])

# --- Filtros ---
OPTION_TYPE = "C"       # "P" puts, "C" calls
MIN_OBS     = 5        # mínimo observaciones con peso > umbral por punto
W_THRESHOLD = 0.001      # umbral de peso para contar observaciones efectivas




# =============================================================================
# 2. KERNEL
# =============================================================================

def kernel_1d(u, kernel="epanechnikov"):
    """
    Kernel univariado evaluado en u = (x - x0) / h.
    Epanechnikov: K(u) = 0.75*(1 - u²) si |u| <= 1, else 0
    Gaussian:     K(u) = exp(-0.5*u²)
    """
    if kernel == "epanechnikov":
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)
    elif kernel == "gaussian":
        return np.exp(-0.5 * u**2)
    else:
        raise ValueError(f"Kernel desconocido: {kernel}")


def kernel_2d(u_m, u_tau, kernel="epanechnikov"):
    """Kernel bivariado separable: K(u_m, u_tau) = K(u_m) * K(u_tau)"""
    return kernel_1d(u_m, kernel) * kernel_1d(u_tau, kernel)


# =============================================================================
# 3. ESTIMADOR LPR BIVARIADO GRADO 2
# =============================================================================

def lpr_bivariate(m0, tau0,
                  m_arr, tau_arr, Y_arr, w_oi_arr,
                  h_m, h_tau,
                  kernel=KERNEL,
                  use_oi_vol=True,
                  min_obs=MIN_OBS,
                  w_threshold=W_THRESHOLD):
    """
    LPR local cuadrático bivariado en (m0, tau0) sobre P/S.

    Modelo local:
        P/S ≈ α0
              + α1*(m - m0)
              + α2*(tau - tau0)
              + α3*(m - m0)²
              + α4*(tau - tau0)²
              + α5*(m - m0)*(tau - tau0)

    Coeficientes de interés:
        α0  → precio normalizado P/S en (m0, tau0)
        α1  → delta    = d(P/S)/dm
        α2  → vega     = d(P/S)/dtau
        2*α3 → gamma   = d²(P/S)/dm²

    Parámetros
    ----------
    m0, tau0 : punto de evaluación
    m_arr, tau_arr, Y_arr, w_oi_arr : arrays numpy del mes
    h_m, h_tau : bandwidths
    kernel : "epanechnikov" o "gaussian"
    use_oi_vol : ponderar por OI/Vol además del kernel
    min_obs : mínimo de observaciones efectivas para estimar
    w_threshold : umbral para contar observaciones efectivas

    Retorna
    -------
    dict con: price, delta, vega, gamma, n_eff, cond_number
    None si no hay suficientes observaciones
    """
    # --- Pesos kernel ---
    u_m   = (m_arr   - m0)   / h_m
    u_tau = (tau_arr - tau0) / h_tau
    w_kernel = kernel_2d(u_m, u_tau, kernel)

    # --- Peso combinado kernel × OI/Vol ---
    w_oi = w_oi_arr if use_oi_vol else np.ones(len(m_arr))
    w = w_kernel * w_oi

    # --- Observaciones efectivas ---
    mask = w > w_threshold
    n_eff = int(np.sum(mask))

    if n_eff < min_obs:
        return None

    # --- Matriz de diseño local (solo obs con peso suficiente para eficiencia) ---
    # Nota: usamos todas las obs pero las de peso ~0 no contribuyen numéricamente
    dm_std   = (m_arr   - m0) / H_M      # escala a [-1, 1]
    dtau_std = (tau_arr - tau0) / H_TAU  # escala a [-1, 1]

    X = np.column_stack([
        np.ones(len(m_arr)),  # α0: precio en (m0, tau0)
        dm_std,                    # α1: delta
        dtau_std,                  # α2: vega proxy
        dm_std**2,                 # α3: gamma/2
        dtau_std**2,               # α4: curvatura en tau
        dm_std * dtau_std          # α5: término cruzado
    ])

    # --- WLS eficiente: X'WX sin matriz NxN ---
    Xw   = X * w[:, np.newaxis]
    XtWX = Xw.T @ X       # (6, 6)
    XtWY = Xw.T @ Y_arr   # (6,)

    # --- Número de condición para diagnóstico de multicolinealidad ---
    try:
        cond = np.linalg.cond(XtWX)
    except Exception:
        cond = np.nan

    # --- Resolución del sistema ---
    try:
        beta = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        return None

    return {
        "price":      beta[0],
        "delta":      beta[1] / H_M,
        "vega":       beta[2] / H_TAU,
        "gamma":      2.0 * beta[3] / (H_M**2),   # factor 2: segunda derivada de dm²
        "curv_tau":   2.0 * beta[4] / (H_TAU**2),
        "cross":      beta[5] / (H_M * H_TAU),
        "n_eff":      n_eff,
        "cond":       cond
    }


# =============================================================================
# 4. CARGA Y PREPARACIÓN DE DATOS
# =============================================================================

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

# Filtros básicos
opt_df_filtered = opt_df[
    (opt_df["Bid"] > 0) &
    ((opt_df["OpenInterest"] > 0))
].reset_index(drop=True)

opt_df_filtered["YearDay"] = opt_df_filtered["Date"].dt.to_period("D").dt.strftime("%Y-%m-%d")

# Día de prueba — el más denso
sample_day = opt_df_filtered["YearDay"].value_counts().idxmin()
print(f"Día de prueba: {sample_day} con {opt_df_filtered['YearDay'].value_counts().max():,} observaciones")

df = opt_df_filtered[
    (opt_df_filtered["YearDay"] == sample_day) &
    (opt_df_filtered["CallPut"] == OPTION_TYPE)
].copy()

print(f"Observaciones en el día ({OPTION_TYPE}): {len(df):,}")





# # Filtros básicos
# opt_df_filtered = opt_df[
#     (opt_df["Bid"] > 0) &
#     ((opt_df["OpenInterest"] > 0))
# ].reset_index(drop=True)

# opt_df_filtered["YearMonth"] = opt_df_filtered["Date"].dt.to_period("M")

# # Mes de prueba — el más denso
# sample_month = opt_df_filtered["YearMonth"].value_counts().idxmin()
# print(f"Mes de prueba: {sample_month}")

# df = opt_df_filtered[
#     (opt_df_filtered["YearMonth"] == sample_month) &
#     (opt_df_filtered["CallPut"] == OPTION_TYPE)
# ].copy()






# --- Variables ---
df["Y"]   = df["MidPrice"] / df["SpotPrice"]
df["m"]   = df["Moneyness"]
df["tau"] = df["Days"].astype(float)

# --- Pesos OI/Vol ---
# Suma OI + Volume, normaliza a [0,1], clip para evitar ceros
df["w_oi_vol"] = (df["OpenInterest"]).clip(lower=0)
df["w_oi_vol"] = (df["w_oi_vol"] / df["w_oi_vol"].max()).clip(lower=1e-6)

# --- Arrays numpy (extraer una vez para eficiencia) ---
m_arr    = df["m"].values
tau_arr  = df["tau"].values
Y_arr    = df["Y"].values
w_oi_arr = df["w_oi_vol"].values

print(f"\nRango moneyness: [{m_arr.min():.3f}, {m_arr.max():.3f}]")
print(f"Rango tau:       [{tau_arr.min():.0f}, {tau_arr.max():.0f}] días")
print(f"Rango Y (P/S):   [{Y_arr.min():.6f}, {Y_arr.max():.6f}]")

# =============================================================================
# 5. MODO A — ESTIMACIÓN EN GRID DE CONTRATOS DUMMY
# =============================================================================

print(f"\nEstimando LPR en grid {len(M0_GRID)}×{len(TAU0_GRID)} = {len(M0_GRID)*len(TAU0_GRID)} puntos...")
print(f"Kernel: {KERNEL}, h_m={H_M}, h_tau={H_TAU}")

results = []

for tau0 in TAU0_GRID:
    for m0 in M0_GRID:
        res = lpr_bivariate(
            m0, tau0,
            m_arr, tau_arr, Y_arr, w_oi_arr,
            h_m=H_M, h_tau=H_TAU
        )
        row = {
            "tau0":  tau0,
            "m0":    round(m0, 4),
            "valid": res is not None
        }
        if res is not None:
            row.update(res)
        else:
            row.update({k: np.nan for k in
                        ["price","delta","vega","gamma","curv_tau","cross","n_eff","cond"]})
        results.append(row)

results_df = pd.DataFrame(results)

# Resumen
valid_pct = results_df["valid"].mean() * 100
print(f"\nPuntos válidos: {results_df['valid'].sum()} / {len(results_df)} ({valid_pct:.1f}%)")
print("\nResultados LPR bivariado (grid dummy):")
cols_show = ["tau0","m0","price","delta","vega","gamma","n_eff","valid"]
print(results_df[cols_show].to_string(index=False, float_format="{:.6f}".format))

# =============================================================================
# 6. MODO B — ESTIMACIÓN EN CADA OBSERVACIÓN REAL
# =============================================================================

print("\nEstimando LPR en cada observación real (para agregación por OI)...")
print("(Esto puede tardar varios minutos con 200k+ observaciones)")

# Para eficiencia: solo estimamos en observaciones únicas de (m, tau) redondeadas
# El objetivo es asignar greeks a cada contrato para GEX
df_unique = df[["m","tau"]].drop_duplicates().copy()
df_unique["m_round"]   = df_unique["m"].round(3)
df_unique["tau_round"] = df_unique["tau"].round(0)
df_unique = df_unique.drop_duplicates(subset=["m_round","tau_round"])

print(f"Puntos únicos (m, tau) redondeados: {len(df_unique):,}")

res_real = []
for _, row in df_unique.iterrows():
    m0   = row["m_round"]
    tau0 = row["tau_round"]
    res = lpr_bivariate(
        m0, tau0,
        m_arr, tau_arr, Y_arr, w_oi_arr,
        h_m=H_M, h_tau=H_TAU
    )
    entry = {"m_round": m0, "tau_round": tau0}
    if res is not None:
        entry.update(res)
    else:
        entry.update({k: np.nan for k in
                      ["price","delta","vega","gamma","curv_tau","cross","n_eff","cond"]})
    res_real.append(entry)

res_real_df = pd.DataFrame(res_real)

# Merge con dataframe original
df["m_round"]   = df["m"].round(3)
df["tau_round"] = df["tau"].round(0)
df = df.merge(
    res_real_df[["m_round","tau_round","delta","gamma","vega","n_eff"]],
    on=["m_round","tau_round"],
    how="left"
)

# GEX contribution por contrato
df["gex_contribution"] = df["gamma"] * df["OpenInterest"]
df["dex_contribution"] = df["delta"] * df["OpenInterest"]

# Agregación mensual total
gex_total = df["gex_contribution"].sum()
dex_total = df["dex_contribution"].sum()
print(f"\nGEX mensual total: {gex_total:,.2f}")
print(f"DEX mensual total: {dex_total:,.2f}")

# Agregación por bucket de moneyness
df["m_bucket"] = pd.cut(df["m"], bins=[0.7,0.8,0.9,0.95,1.05,1.1,1.2,1.3,99],
                         labels=["<0.8","0.8-0.9","0.9-0.95","0.95-1.05",
                                 "1.05-1.1","1.1-1.2","1.2-1.3",">1.3"])
gex_by_m = df.groupby("m_bucket")["gex_contribution"].sum()
print("\nGEX por bucket de moneyness:")
print(gex_by_m.to_string())

# =============================================================================
# 7. GRÁFICOS
# =============================================================================

valid_taus = [t for t in TAU0_GRID
              if results_df[(results_df["tau0"]==t) & results_df["valid"]].shape[0] >= 3]

if valid_taus:
    n_rows = len(valid_taus)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"LPR Bivariado ({KERNEL}) — {OPTION_TYPE}uts — {sample_day}\n"
        f"h_m={H_M}, h_τ={H_TAU}, sin restricciones",
        fontsize=13, y=1.01
    )

    for row_idx, tau0 in enumerate(valid_taus):
        sub = results_df[(results_df["tau0"]==tau0) & results_df["valid"]].copy()

        # Precio
        ax = axes[row_idx, 0]
        ax.plot(sub["m0"], sub["price"], "o-", color="steelblue", lw=2)
        ax.set_title(f"P/S  |  τ={tau0}d")
        ax.set_xlabel("Moneyness K/S")
        ax.axvline(1.0, color="gray", ls="--", alpha=0.4)
        ax.set_xlim(M0_GRID.min()-0.02, M0_GRID.max()+0.02)

        # Delta
        ax = axes[row_idx, 1]
        ax.plot(sub["m0"], sub["delta"], "o-", color="darkorange", lw=2)
        ax.set_title(f"Delta ∂(P/S)/∂m  |  τ={tau0}d")
        ax.set_xlabel("Moneyness K/S")
        ax.axvline(1.0, color="gray", ls="--", alpha=0.4)
        ax.axhline(0.0, color="gray", ls="--", alpha=0.4)

        # Gamma
        ax = axes[row_idx, 2]
        colors_gamma = ["red" if g < 0 else "seagreen" for g in sub["gamma"]]
        ax.scatter(sub["m0"], sub["gamma"], color=colors_gamma, s=60, zorder=3)
        ax.plot(sub["m0"], sub["gamma"], color="seagreen", lw=1.5, alpha=0.5)
        ax.set_title(f"Gamma ∂²(P/S)/∂m²  |  τ={tau0}d")
        ax.set_xlabel("Moneyness K/S")
        ax.axvline(1.0, color="gray", ls="--", alpha=0.4)
        ax.axhline(0.0, color="red", ls="--", alpha=0.6)

        # n_eff — diagnóstico de densidad
        ax = axes[row_idx, 3]
        ax.bar(sub["m0"], sub["n_eff"], width=0.03, color="slategray", alpha=0.7)
        ax.set_title(f"Obs. efectivas  |  τ={tau0}d")
        ax.set_xlabel("Moneyness K/S")
        ax.axhline(MIN_OBS, color="red", ls="--", alpha=0.6, label=f"MIN_OBS={MIN_OBS}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fname = f"{PATH_OUTPUT}\\lpr_bivariate_{KERNEL}_hm{H_M}_htau{H_TAU}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGráfico guardado: {fname}")

# =============================================================================
# 8. DIAGNÓSTICO — VIOLACIONES SIN RESTRICCIONES
# =============================================================================

print("\n=== DIAGNÓSTICO DE VIOLACIONES (sin restricciones) ===")

valid_res = results_df[results_df["valid"]].copy()

gamma_neg_pct  = (valid_res["gamma"] < 0).mean() * 100
price_neg_pct  = (valid_res["price"] < 0).mean() * 100
high_cond_pct  = (valid_res["cond"] > 1e10).mean() * 100

print(f"Gamma < 0:            {gamma_neg_pct:.1f}%  (viola convexidad)")
if OPTION_TYPE == "P":
    delta_violation = (valid_res["delta"] < 0).mean() * 100
    print(f"Delta < 0: {delta_violation:.1f}%")
elif OPTION_TYPE == "C":
    delta_violation = (valid_res["delta"] > 0).mean() * 100
    print(f"Delta > 0: {delta_violation:.1f}%")
print(f"Precio < 0:           {price_neg_pct:.1f}%  (viola no-negatividad)")
print(f"Cond. number > 1e10:  {high_cond_pct:.1f}%  (multicolinealidad local)")

print("\nDistribución de gamma por bucket tau0:")
print(valid_res.groupby("tau0")["gamma"].describe().round(4).to_string())

print("\nDistribución de delta por bucket tau0:")
print(valid_res.groupby("tau0")["delta"].describe().round(4).to_string())

# =============================================================================
# 9. GUARDAR RESULTADOS
# =============================================================================

# Grid dummy
fname_grid = f"{PATH_OUTPUT}\\lpr_grid_{sample_day}_{OPTION_TYPE}.csv"
results_df.to_csv(fname_grid, index=False)
print(f"\nResultados grid guardados: {fname_grid}")

# Observaciones reales con greeks
fname_real = f"{PATH_OUTPUT}\\lpr_real_{sample_day}_{OPTION_TYPE}.parquet"
df.to_parquet(fname_real, index=False)
print(f"Resultados reales guardados: {fname_real}")

print("\n=== FIN ===")