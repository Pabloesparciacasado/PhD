"""
LPR Bivariado — Modo B Diario (versión original, sin pre-filtrado)
===================================================================
Igual que lpr_diario_fast.py pero usando lpr_bivariate completo
(sin pre-filtrado por soporte) — más lento pero más explícito.
Incluye delta empírica agrupada por bucket (Opción 3).

Autor: Pablo Esparcia Casado
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. PARÁMETROS
# =============================================================================

H_M         = 0.05
H_TAU       = 45.0
KERNEL      = "epanechnikov"
OPTION_TYPE = "P"
MIN_OBS     = 5          # más alto que la versión fast para evitar outliers
W_THRESHOLD = 0.001

# Delta empírica
MIN_RET_S   = 0.001       # movimiento mínimo del subyacente (0.1%)
MIN_OI_CELL = 10          # OI mínimo en celda para estimar delta empírica

# Buckets para delta empírica
M_BINS   = [0.0, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 99]
TAU_BINS = [0, 15, 45, 105, 183, 365, 9999]
M_LABELS   = ["<0.80","0.80-0.85","0.85-0.90","0.90-0.95","0.95-1.00",
               "1.00-1.05","1.05-1.10","1.10-1.15","1.15-1.20",">1.20"]
TAU_LABELS = ["0-15d","15-45d","45-105d","105-183d","183-365d",">365d"]

PATH_DATA   = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet"
PATH_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares"

# =============================================================================
# 2. KERNEL Y ESTIMADOR
# =============================================================================

def kernel_1d(u, kernel="epanechnikov"):
    if kernel == "epanechnikov":
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u**2), 0.0)
    elif kernel == "gaussian":
        return np.exp(-0.5 * u**2)
    else:
        raise ValueError(f"Kernel desconocido: {kernel}")


def kernel_2d(u_m, u_tau, kernel="epanechnikov"):
    return kernel_1d(u_m, kernel) * kernel_1d(u_tau, kernel)


def lpr_bivariate(m0, tau0,
                  m_arr, tau_arr, Y_arr, w_oi_arr,
                  h_m=H_M, h_tau=H_TAU,
                  kernel=KERNEL,
                  min_obs=MIN_OBS,
                  w_threshold=W_THRESHOLD):
    """
    LPR local cuadrático bivariado en (m0, tau0) sobre P/S.
    Versión original — recibe el pool completo del día sin pre-filtrar.
    Variables estandarizadas por h para buen condicionamiento.

    Retorna dict con price, delta, vega, gamma, n_eff, cond — o None.
    """
    # Pesos kernel sobre pool completo
    u_m   = (m_arr   - m0) / h_m
    u_tau = (tau_arr - tau0) / h_tau
    w_kernel = kernel_2d(u_m, u_tau, kernel)
    w = w_kernel * w_oi_arr

    n_eff = int(np.sum(w > w_threshold))
    if n_eff < min_obs:
        return None

    # Variables estandarizadas — evita mal condicionamiento de XtWX
    dm_std   = (m_arr   - m0) / h_m
    dtau_std = (tau_arr - tau0) / h_tau

    X = np.column_stack([
        np.ones(len(m_arr)),
        dm_std,
        dtau_std,
        dm_std**2,
        dtau_std**2,
        dm_std * dtau_std
    ])

    # WLS eficiente sin matriz NxN
    Xw   = X * w[:, np.newaxis]
    XtWX = Xw.T @ X
    XtWY = Xw.T @ Y_arr

    try:
        cond = np.linalg.cond(XtWX)
    except Exception:
        cond = np.nan

    try:
        beta = np.linalg.solve(XtWX, XtWY)
    except np.linalg.LinAlgError:
        return None

    return {
        "price":    beta[0],
        "delta":    beta[1] / h_m,
        "vega":     beta[2] / h_tau,
        "gamma":    2.0 * beta[3] / (h_m**2),
        "curv_tau": 2.0 * beta[4] / (h_tau**2),
        "cross":    beta[5] / (h_m * h_tau),
        "n_eff":    n_eff,
        "cond":     cond
    }

# =============================================================================
# 3. CARGA Y PREPARACIÓN
# =============================================================================

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

opt_df_filtered = opt_df[
    (opt_df["Bid"] > 0) &
    (opt_df["OpenInterest"] > 0) &
    (opt_df["CallPut"] == OPTION_TYPE)
].reset_index(drop=True)

# Año de prueba
df = opt_df_filtered[opt_df_filtered["Date"].dt.year == 2023].copy()
print(f"Año de prueba: {df['Date'].dt.year.iloc[0]}")
print(f"Observaciones ({OPTION_TYPE}): {len(df):,}")
print(f"Días únicos: {df['Date'].nunique():,}")

# Variables base
df["Y"]         = df["MidPrice"] / df["SpotPrice"]
df["m"]         = df["Moneyness"]
df["tau"]       = df["Days"].astype(float)
df["m_round"]   = df["m"].round(3)
df["tau_round"] = df["tau"].round(0).astype(int)

# =============================================================================
# 4. MODO B — LPR DIARIO (versión original)
# =============================================================================

print("\nEstimando LPR diario (versión original, sin pre-filtrado)...")
print(f"Kernel: {KERNEL}, h_m={H_M}, h_tau={H_TAU}")
print("NOTA: más lento que la versión fast — pool completo por punto")

all_results = []
dates   = sorted(df["Date"].unique())
n_dates = len(dates)

for idx_d, date in enumerate(dates):

    df_day = df[df["Date"] == date].copy()

    # Pesos OI normalizados al día
    df_day["w_oi_vol"] = df_day["OpenInterest"].clip(lower=0)
    df_day["w_oi_vol"] = (
        df_day["w_oi_vol"] / df_day["w_oi_vol"].max()
    ).clip(lower=1e-6)

    # Arrays del pool completo del día
    m_day   = df_day["m"].values
    tau_day = df_day["tau"].values
    Y_day   = df_day["Y"].values
    w_day   = df_day["w_oi_vol"].values

    # Puntos únicos del día con OI suficiente
    eval_day = (df_day.groupby(["m_round","tau_round"])
                      .agg(OI_total=("OpenInterest","sum"))
                      .reset_index())
    eval_day = eval_day[eval_day["OI_total"] >= 5].copy()

    if len(eval_day) == 0:
        continue

    pct = 100 * (idx_d + 1) / n_dates
    print(f"  {pct:5.1f}%  {date.date()} — {len(eval_day):,} puntos únicos", end="\r")

    for i in range(len(eval_day)):
        m0   = eval_day["m_round"].iloc[i]
        tau0 = eval_day["tau_round"].iloc[i]

        # Sin pre-filtrado — pasa el pool completo del día
        res = lpr_bivariate(
            m0, tau0,
            m_day, tau_day,
            Y_day, w_day,
            h_m=H_M, h_tau=H_TAU
        )

        entry = {"Date": date, "m_round": m0, "tau_round": tau0}
        entry.update(res if res is not None else
                     {k: np.nan for k in
                      ["price","delta","vega","gamma","curv_tau",
                       "cross","n_eff","cond"]})
        all_results.append(entry)

print(f"\n  100.0%  Estimación LPR completada — {len(all_results):,} puntos")

res_df = pd.DataFrame(all_results)

# Filtrar outliers numéricos antes del merge

print(f"Puntos tras filtro outliers (|delta| < 5): {len(res_df):,}")

# Merge con df original
df = df.merge(
    res_df[["Date","m_round","tau_round",
            "delta","gamma","vega","price","n_eff","cond"]],
    on=["Date","m_round","tau_round"],
    how="left"
)

# Delta via Euler: delta_financiera = price - m * delta_lpr
df["delta_euler"] = df["price"] - df["m_round"] * df["delta"]

# GEX/DEX por contrato
df["gex_contribution"] = df["gamma"]       * df["OpenInterest"]
df["dex_contribution"] = df["delta_euler"] * df["OpenInterest"]

# Agregación diaria LPR
gex_daily = df.groupby("Date").agg(
    GEX_lpr     = ("gex_contribution", "sum"),
    DEX_lpr     = ("dex_contribution", "sum"),
    n_contracts = ("OptionID", "nunique")
).reset_index()

print("\nGEX/DEX diario (LPR) — primeras filas:")
print(gex_daily.head(10).to_string(index=False))

# =============================================================================
# 5. DELTA EMPÍRICA AGRUPADA POR BUCKET (OPCIÓN 3)
# =============================================================================

print("\nCalculando delta empírica por bucket...")

df_emp = df.sort_values(["OptionID","Date"]).copy()
df_emp["dC"] = df_emp.groupby("OptionID")["MidPrice"].diff()

# Cambio diario del subyacente
spot_daily = (df_emp.groupby("Date")["SpotPrice"]
                    .first()
                    .reset_index()
                    .rename(columns={"SpotPrice": "S_today"}))
spot_daily["S_prev"] = spot_daily["S_today"].shift(1)
spot_daily["dS"]     = spot_daily["S_today"] - spot_daily["S_prev"]
spot_daily["ret_S"]  = spot_daily["dS"] / spot_daily["S_prev"]

df_emp = df_emp.merge(spot_daily[["Date","dS","ret_S"]], on="Date", how="left")

# Filtro: solo días con movimiento suficiente
df_emp = df_emp[df_emp["ret_S"].abs() >= MIN_RET_S].copy()
print(f"Días con |ret_S| >= {MIN_RET_S*100:.1f}%: {df_emp['Date'].nunique():,}")

# Asignar buckets
df_emp["m_bucket"]   = pd.cut(df_emp["m"],   bins=M_BINS,   labels=M_LABELS)
df_emp["tau_bucket"] = pd.cut(df_emp["tau"], bins=TAU_BINS,  labels=TAU_LABELS)

df_emp = df_emp.dropna(subset=["dC","m_bucket","tau_bucket"])

def delta_emp_cell(group):
    oi       = group["OpenInterest"].values
    dC       = group["dC"].values
    dS       = group["dS"].values[0]
    n        = len(group)
    oi_total = oi.sum()

    if oi_total < MIN_OI_CELL or dS == 0:
        return pd.Series({
            "delta_emp":   np.nan,
            "n_contracts": n,
            "oi_total":    oi_total
        })

    delta = np.sum(oi * dC) / (dS * oi_total)

    return pd.Series({
        "delta_emp":   delta,
        "n_contracts": n,
        "oi_total":    oi_total
    })

delta_by_cell = (
    df_emp
    .groupby(["Date","m_bucket","tau_bucket"], observed=True)
    .apply(delta_emp_cell)
    .reset_index()
)

valid_emp = delta_by_cell["delta_emp"].notna().sum()
print(f"Celdas con delta empírica válida: {valid_emp:,} de {len(delta_by_cell):,}")

# DEX empírico diario
delta_by_cell["dex_emp"] = delta_by_cell["delta_emp"] * delta_by_cell["oi_total"]

dex_emp_daily = (
    delta_by_cell
    .groupby("Date")["dex_emp"]
    .sum()
    .reset_index()
    .rename(columns={"dex_emp": "DEX_emp"})
)

# =============================================================================
# 6. COMPARACIÓN LPR vs EMPÍRICA
# =============================================================================

df["m_bucket"]   = pd.cut(df["m"],   bins=M_BINS,   labels=M_LABELS)
df["tau_bucket"] = pd.cut(df["tau"], bins=TAU_BINS,  labels=TAU_LABELS)

delta_lpr_by_cell = (
    df.dropna(subset=["delta_euler","m_bucket","tau_bucket"])
    .groupby(["Date","m_bucket","tau_bucket"], observed=True)
    .apply(lambda g: pd.Series({
        "delta_lpr": np.average(
            g["delta_euler"],
            weights=g["OpenInterest"]
        ) if len(g) > 0 else np.nan
    }))
    .reset_index()
)

comparison = delta_by_cell.merge(
    delta_lpr_by_cell,
    on=["Date","m_bucket","tau_bucket"],
    how="inner"
).dropna(subset=["delta_emp","delta_lpr"])

comparison["diff"] = comparison["delta_emp"] - comparison["delta_lpr"]

print("\n=== COMPARACIÓN DELTA EMPÍRICA vs LPR (Euler) ===")
print(comparison.groupby("m_bucket", observed=True)[["delta_emp","delta_lpr","diff"]]
                .mean()
                .round(4)
                .to_string())

corr_by_bucket = (
    comparison
    .groupby("m_bucket", observed=True)
    .apply(lambda g: g["delta_emp"].corr(g["delta_lpr"]))
    .rename("corr_emp_lpr")
)
print("\nCorrelación delta empírica vs LPR por bucket moneyness:")
print(corr_by_bucket.round(3).to_string())

# =============================================================================
# 7. GRÁFICOS
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f"LPR Diario Modo B — {OPTION_TYPE} — 2023\n"
             f"h_m={H_M}, h_τ={H_TAU}, {KERNEL}",
             fontsize=12)

# GEX diario LPR
ax = axes[0]
ax.bar(range(len(gex_daily)), gex_daily["GEX_lpr"],
       color="steelblue", alpha=0.8, width=0.8)
ax.set_title("GEX diario (LPR)")
ax.set_xlabel("Día")
ax.set_ylabel("GEX = gamma × OI")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(0, len(gex_daily), max(1, len(gex_daily)//6)))
ax.set_xticklabels(
    gex_daily["Date"].iloc[::max(1, len(gex_daily)//6)].dt.strftime("%Y-%m-%d"),
    rotation=45, ha="right", fontsize=7
)

# Scatter delta empírica vs LPR — ATM
ax = axes[1]
atm = comparison[comparison["m_bucket"] == "0.95-1.00"]
if len(atm) > 0:
    ax.scatter(atm["delta_lpr"], atm["delta_emp"],
               s=5, alpha=0.5, color="darkorange")
    lim = max(atm[["delta_lpr","delta_emp"]].abs().max().max(), 0.1)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, alpha=0.5)
ax.set_xlabel("Delta LPR (Euler)")
ax.set_ylabel("Delta empírica")
ax.set_title("Delta LPR vs Empírica — ATM [0.95-1.00]")

# Correlación por bucket
ax = axes[2]
corr_vals = corr_by_bucket.values
labels    = corr_by_bucket.index.tolist()
colors    = ["seagreen" if c > 0 else "red" for c in corr_vals]
ax.barh(range(len(labels)), corr_vals, color=colors, alpha=0.8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Correlación")
ax.set_title("Corr(delta_emp, delta_LPR) por bucket")
ax.axvline(0, color="black", lw=0.8)

plt.tight_layout()
fname_plot = f"{PATH_OUTPUT}\\lpr_vs_emp_{OPTION_TYPE}_2023_nf.png"
plt.savefig(fname_plot, dpi=150, bbox_inches="tight")
plt.show()
print(f"\nGráfico guardado: {fname_plot}")

# =============================================================================
# 8. GUARDAR
# =============================================================================

fname_real = f"{PATH_OUTPUT}\\lpr_real_2023_{OPTION_TYPE}_nf.parquet"
df.to_parquet(fname_real, index=False)

fname_emp = f"{PATH_OUTPUT}\\delta_emp_2023_{OPTION_TYPE}_nf.csv"
delta_by_cell.to_csv(fname_emp, index=False)

fname_gex = f"{PATH_OUTPUT}\\gex_daily_2023_{OPTION_TYPE}_nf.csv"
gex_daily.merge(dex_emp_daily, on="Date", how="left").to_csv(fname_gex, index=False)

print(f"\nArchivos guardados:")
print(f"  {fname_real}")
print(f"  {fname_emp}")
print(f"  {fname_gex}")
print("\n=== FIN ===")