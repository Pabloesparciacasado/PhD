import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARÁMETROS
# =============================================================================

BANDWIDTH = 0.05          # bandwidth kernel Gaussiano en moneyness
M0_GRID = np.arange(0.70, 1.31, 0.05)  # grid de evaluación
OPTION_TYPE = "P"         # "P" o "C"
MIN_OBS = 30              # mínimo de observaciones con peso > 0.01 para estimar

# =============================================================================
# CARGA Y FILTRADO
# =============================================================================

opt_df = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet")

opt_df_filtered = opt_df[
    (opt_df["Bid"] > 0)].reset_index(drop=True)

# =============================================================================
# SELECCIÓN DE UN MES DE PRUEBA
# =============================================================================

# Selecciona un mes concreto — ajusta según tus datos
opt_df_filtered["YearMonth"] = opt_df_filtered["Date"].dt.to_period("M")
sample_month = opt_df_filtered["YearMonth"].value_counts().idxmax()  # mes con más datos
print(f"Mes de prueba: {sample_month}")

df = opt_df_filtered[
    (opt_df_filtered["YearMonth"] == sample_month) &
    (opt_df_filtered["CallPut"] == OPTION_TYPE)
].copy()

print(f"Observaciones: {len(df)}")

# =============================================================================
# VARIABLES PARA LPR
# =============================================================================

df["Y"] = df["MidPrice"] / df["SpotPrice"]   # precio normalizado P/S
df["m"] = df["Moneyness"]                     # K/S ya calculado

# Pesos OI/Vol — normalización para que no haya dominancia de fin de mes
df["weight_oi"] = df["OpenInterest"].clip(lower=0)
df["weight_vol"] = df["Volume"].clip(lower=0)

# Combina OI y volumen — usa OI como base, volumen como complemento
df["w_oi_vol"] = df["weight_oi"] + df["weight_vol"]
df["w_oi_vol"] = df["w_oi_vol"] / df["w_oi_vol"].max()  # normaliza a [0,1]
df["w_oi_vol"] = df["w_oi_vol"].clip(lower=1e-6)         # evita ceros exactos

# =============================================================================
# FUNCIÓN LPR LOCAL CUADRÁTICA
# =============================================================================

def lpr_local(df, m0, h, use_oi_vol=True):
    """
    Estimador local polinomial de grado 2 en m0.
    
    Retorna:
        beta0: P/S estimado en m0
        beta1: delta normalizado = d(P/S)/dm en m0
        beta2: gamma normalizado = d²(P/S)/dm² en m0
        n_eff: número de observaciones con peso > 0.01
    """
    m = df["m"].values
    Y = df["Y"].values
    w_oi = df["w_oi_vol"].values if use_oi_vol else np.ones(len(df))

    # Kernel Gaussiano
    u = (m - m0) / h
    w_kernel = np.exp(-0.5 * u**2)

    # Peso combinado: kernel × OI/Vol
    w = w_kernel * w_oi

    # Observaciones efectivas
    n_eff = np.sum(w > 0.01)
    if n_eff < MIN_OBS:
        return None, None, None, n_eff

    # Matriz de diseño local: [1, (m-m0), (m-m0)^2]
    dm = m - m0
    X = np.column_stack([
        np.ones(len(m)),
        dm,
        dm**2
    ])

    # WLS sin matriz diagonal explícita — evita allocar N×N
    # Equivalente a X'WX pero multiplicando columna a columna
    Xw   = X * w[:, np.newaxis]   # shape (N, 3): cada fila de X multiplicada por w_i
    XtWX = Xw.T @ X               # shape (3, 3)
    XtWY = Xw.T @ Y               # shape (3,)

    try:
        beta = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        return None, None, None, n_eff

    beta0 = beta[0]   # P/S en m0
    beta1 = beta[1]   # d(P/S)/dm  → delta normalizado
    beta2 = beta[2]   # d²(P/S)/dm² / 2 → necesita ×2 para gamma

    return beta0, beta1, 2 * beta2, n_eff

# =============================================================================
# ESTIMACIÓN EN EL GRID
# =============================================================================

results = []

for m0 in M0_GRID:
    b0, b1, b2, n_eff = lpr_local(df, m0, h=BANDWIDTH)
    results.append({
        "m0":     round(m0, 4),
        "price":  b0,
        "delta":  b1,
        "gamma":  b2,
        "n_eff":  n_eff,
        "valid":  b0 is not None
    })

results_df = pd.DataFrame(results)
print("\nResultados LPR:")
print(results_df.to_string(index=False))

# =============================================================================
# GRÁFICOS
# =============================================================================

valid = results_df[results_df["valid"]]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"LPR — {OPTION_TYPE}uts — {sample_month} — h={BANDWIDTH}", fontsize=13)

# Precio normalizado
axes[0].plot(valid["m0"], valid["price"], "o-", color="steelblue")
axes[0].set_title("Precio normalizado P/S")
axes[0].set_xlabel("Moneyness K/S")
axes[0].axvline(1.0, color="gray", linestyle="--", alpha=0.5)

# Delta
axes[1].plot(valid["m0"], valid["delta"], "o-", color="darkorange")
axes[1].set_title("Delta  d(P/S)/dm")
axes[1].set_xlabel("Moneyness K/S")
axes[1].axvline(1.0, color="gray", linestyle="--", alpha=0.5)
axes[1].axhline(0.0, color="gray", linestyle="--", alpha=0.5)

# Gamma
axes[2].plot(valid["m0"], valid["gamma"], "o-", color="seagreen")
axes[2].set_title("Gamma  d²(P/S)/dm²")
axes[2].set_xlabel("Moneyness K/S")
axes[2].axvline(1.0, color="gray", linestyle="--", alpha=0.5)
axes[2].axhline(0.0, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\lpr_test.png", dpi=150)
plt.show()

print("\nHecho. Gráfico guardado.")