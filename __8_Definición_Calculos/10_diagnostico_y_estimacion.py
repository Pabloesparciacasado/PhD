# In[]: Diagnóstico de ruido + estimación con contenido económico
"""
10_diagnostico_y_estimacion.py

Sustituye al placebo (08_). Razón: con la forma cerrada γ̂ = Γ·(1−g) verificada
en simulación (R²=0.998), `g` se calcula directamente del panel y la
descomposición es exacta. Un Monte Carlo de 500 réplicas estimaría por
simulación algo que ya se conoce analíticamente.

PARTE A — DIAGNÓSTICO DE RUIDO (directo, sin Monte Carlo)
    A1  calcula g y comprueba γ̂ = Γ^BS·(1−g) sobre datos reales
    A2  descompone el factor agregado: ¿cuánto es Γ y cuánto es g?
    A3  ¿el factor mensual es un proxy de volatilidad?

PARTE B — ESTIMACIÓN CON CONTENIDO ECONÓMICO
    B1  γ̃_total   : corrige solo theta  -> derivada TOTAL (incluye dinámica de vol)
    B2  γ̃_parcial : corrige theta y vega -> derivada PARCIAL (σ fija)
    B3  canal vanna = γ̃_total − γ̃_parcial  <- ESTE es el objeto económico
    B4  σ'(K) por regresión ΔIV ~ ΔS       <- la dinámica del smile, directa
    B5  Γ_K por sección cruzada de strikes (Fornberg) como referencia limpia

No modifica el pipeline. Solo lee.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from tabulate import tabulate

# ------------------------------- Configuración ------------------------------
NOMBRE = "opt_df_empirical_greeks.parquet"
CANDIDATOS = [
    r"/Volumes/data/OUTPUTS",
    r"/Volumes/data/OptionMetrics/OUTPUTS",
    r"/Users/pablo/codigos_github",
    r"Y:\OUTPUTS",
]
V_MIN, V_MAX = 15, 45
UMBRAL_GAMMA = 1e-8          # bajo esto el error relativo no significa nada
OUT_DIR = None               # se fija junto al parquet

COLS = ["Date", "OptionID", "CallPut", "Days", "Strike", "Moneyness",
        "MidPrice", "SpotPrice", "OpenInterest", "Volume", "Bid",
        "ImpliedVolatility", "Delta", "Gamma", "Vega", "Theta",
        "delta_emp", "gamma_emp"]

PATH = next((os.path.join(c, NOMBRE) for c in CANDIDATOS
             if os.path.exists(os.path.join(c, NOMBRE))), None)
if PATH is None:
    raise FileNotFoundError("No encuentro el parquet en:\n  " + "\n  ".join(CANDIDATOS))
OUT_DIR = os.path.dirname(PATH)
print(f"Parquet: {PATH}")

df = pd.read_parquet(PATH, columns=COLS)
df = df[(df["Days"] > V_MIN) & (df["Days"] <= V_MAX) & (df["Bid"] > 0)].copy()
df = df[df["OpenInterest"] >= 11]
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["OptionID", "Date"]).reset_index(drop=True)
print(f"Panel (15,45], Bid>0, OI>=11: {len(df):,} filas, "
      f"{df.OptionID.nunique():,} contratos, "
      f"{df.Date.dt.year.min()}–{df.Date.dt.year.max()}")


# In[]: Construcción de lags — Δt REAL en días naturales (no 1/252)

g = df.groupby("OptionID", sort=False)
for c, n in [("MidPrice", "C"), ("SpotPrice", "S"), ("ImpliedVolatility", "IV")]:
    df[f"{n}_l"]  = g[c].shift(1)
    df[f"{n}_l2"] = g[c].shift(2)
df["Date_l"]  = g["Date"].shift(1)
df["Date_l2"] = g["Date"].shift(2)
df["Theta_l"] = g["Theta"].shift(1)
df["Vega_l"]  = g["Vega"].shift(1)

df["gap_t"]   = (df["Date"]   - df["Date_l"]).dt.days.astype(float)
df["gap_lag"] = (df["Date_l"] - df["Date_l2"]).dt.days.astype(float)
df["dt_t"]    = df["gap_t"]   / 365.0
df["dt_lag"]  = df["gap_lag"] / 365.0

df["dS"]     = df["SpotPrice"] - df["S_l"]
df["dS_lag"] = df["S_l"]       - df["S_l2"]
df["dS2"]    = df["SpotPrice"] - df["S_l2"]
df["dC"]     = df["MidPrice"]  - df["C_l"]
df["dC_lag"] = df["C_l"]       - df["C_l2"]
df["dIV"]     = df["ImpliedVolatility"] - df["IV_l"]
df["dIV_lag"] = df["IV_l"] - df["IV_l2"]

d = df[(df[["C_l2", "S_l2"]].notna().all(axis=1)) &
       (df["dS"].abs() > 0) & (df["dS_lag"].abs() > 0) & (df["dS2"].abs() > 0)].copy()
print(f"Tríos válidos: {len(d):,}")
print(f"Distribución de gap_t (días naturales): "
      f"{d.gap_t.value_counts().head(5).to_dict()}")

# d1 implícito desde la Delta de IvyDB (evita suponer r y q)
dl = d["Delta"].clip(-0.999999, 0.999999)
d["d1"] = np.where(d["CallPut"] == "C", norm.ppf(dl.clip(1e-6, 1 - 1e-6)),
                                        norm.ppf((1 + dl).clip(1e-6, 1 - 1e-6)))


# In[]: A1 — ¿se cumple γ̂ = Γ^BS·(1 − g) sobre datos reales?

s, S = d["IV_l"], d["S_l"]                       # evaluadas en t-1
u_t, u_lag = d["dt_t"] / d["dS"], d["dt_lag"] / d["dS_lag"]
d["g"] = (s ** 2 * S ** 2) * (u_t - u_lag) / d["dS2"]

dt_bar = 0.5 * (d["dt_t"] + d["dt_lag"])
esc = s * S * np.sqrt(dt_bar)
d["z_t"], d["z_lag"] = d["dS"] / esc, d["dS_lag"] / esc
d["z_min"] = np.minimum(d["z_t"].abs(), d["z_lag"].abs())

d["gamma_pred"] = d["Gamma"] * (1 - d["g"])

def ajuste(sub, y="gamma_emp", p="gamma_pred"):
    m = np.isfinite(sub[y]) & np.isfinite(sub[p])
    if m.sum() < 30:
        return None
    Y, P = sub.loc[m, y].values, sub.loc[m, p].values
    A = np.c_[np.ones(len(P)), P]
    b, *_ = np.linalg.lstsq(A, Y, rcond=None)
    res = Y - A @ b
    return dict(N=int(m.sum()), corr=np.corrcoef(Y, P)[0, 1],
                pend=b[1], inter=b[0], R2=1 - res.var() / Y.var(),
                med_abs_g=np.median(np.abs(sub.loc[m, "g"])))

print("\n" + "=" * 74)
print("A1 — forma cerrada sobre datos reales")
print("=" * 74)
filas = []
cortes = [(-np.inf, -2), (-2, -1), (-1, -0.5), (-0.5, 0.5),
          (0.5, 1), (1, 2), (2, np.inf)]
for lo, hi in cortes:
    r = ajuste(d[(d.d1 > lo) & (d.d1 <= hi)])
    if r:
        filas.append(dict(bucket_d1=f"({lo:g}, {hi:g}]", **r))
r_all = ajuste(d);              filas.append(dict(bucket_d1="TODO", **r_all))
r_c   = ajuste(d[d.d1.abs() < 1]); filas.append(dict(bucket_d1="|d1|<1", **r_c))
print(tabulate(filas, headers="keys", floatfmt=".4f", tablefmt="github"))
print(f"\nCRITERIO (corr > 0.7 en |d1|<1): {r_c['corr']:.4f} -> "
      f"{'H_A ESTABLECIDA' if r_c['corr'] > 0.7 else 'NO se establece'}")

# Robustez: Gamma de IvyDB vs recalculada
tau = d["Days"] / 365.0
gam_re = norm.pdf(d["d1"]) / (d["SpotPrice"] * d["ImpliedVolatility"] * np.sqrt(tau))
rel = (gam_re - d["Gamma"]) / d["Gamma"].replace(0, np.nan)
print(f"Gamma IvyDB vs recalculada: mediana |dif rel| = {np.nanmedian(np.abs(rel)):.4f}"
      f"  ({'OK' if np.nanmedian(np.abs(rel)) < 0.01 else 'REVISAR >1%'})")


# In[]: A2 — descomposición del agregado diario

def wa(x, w):
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return np.average(x[m], weights=w[m]) if m.any() else np.nan

diario = (d.groupby("Date")
          .apply(lambda G: pd.Series({
              "factor_emp": wa(G.gamma_emp.values, G.OpenInterest.values),
              "Gamma_BS":   wa(G.Gamma.values,     G.OpenInterest.values),
              "g_med":      np.median(G.g),
              "abs_dS":     np.abs(G.dS.iloc[0]),
              "n":          len(G)}), include_groups=False)
          .dropna(subset=["factor_emp"]))
diario["pred"] = diario["Gamma_BS"] * (1 - diario["g_med"])

print("\n" + "=" * 74)
print("A2 — el factor agregado diario")
print("=" * 74)
print(f"corr(factor_emp, Gamma_BS)            = "
      f"{diario[['factor_emp','Gamma_BS']].corr().iloc[0,1]:+.4f}   <- señal")
print(f"corr(factor_emp, Gamma_BS·(1−g))      = "
      f"{diario[['factor_emp','pred']].corr().iloc[0,1]:+.4f}   <- señal+contaminación")
print(f"corr(factor_emp, 1/|ΔS|)              = "
      f"{diario[['factor_emp']].assign(x=1/diario.abs_dS).corr().iloc[0,1]:+.4f}")
print(f"% días con factor negativo            = {np.mean(diario.factor_emp < 0)*100:.1f}%")


# In[]: A3 — ¿es el factor mensual un proxy de volatilidad?

mens = diario.copy()
mens["mes"] = mens.index.to_period("M")
spot = d[["Date", "SpotPrice"]].drop_duplicates("Date").sort_values("Date")
spot["ret"] = np.log(spot.SpotPrice).diff()
rv = spot.groupby(spot.Date.dt.to_period("M")).ret.apply(lambda x: np.sqrt((x**2).sum()))

M = mens.groupby("mes").agg(factor_ult=("factor_emp", "last"),
                            factor_media=("factor_emp", "mean"),
                            g_med=("g_med", "mean"),
                            GBS=("Gamma_BS", "mean")).join(rv.rename("RV"))
M["invRV"] = 1 / M["RV"]

print("\n" + "=" * 74)
print("A3 — correlaciones mensuales del factor")
print("=" * 74)
sub = M[["factor_ult", "factor_media", "GBS", "g_med", "RV", "invRV"]].dropna()
print(tabulate(sub.corr().round(3), headers="keys", tablefmt="github"))
print("\nLECTURA: si |corr(factor, invRV)| >> |corr(factor, GBS)|, el factor es")
print("         principalmente una función de la volatilidad realizada.")
print("         Compara también factor_ult vs factor_media: el muestreo de un")
print("         solo día no diversifica el ruido diario.")


# In[]: B — ESTIMACIÓN CON CONTENIDO ECONÓMICO

print("\n" + "=" * 74)
print("B — estimadores corregidos")
print("=" * 74)

# Convención de Theta de IvyDB: se resuelve empíricamente
print("Resolviendo la convención de Theta (anual vs diaria)...")
mejor, best_med = None, np.inf
for nom, k in [("anual (Θ·Δt años)", 1.0), ("diaria (Θ·días)", 365.0)]:
    delta_c = (d["dC"] - d["Theta_l"] * d["dt_t"] * k) / d["dS"]
    med = np.nanmedian(np.abs(delta_c - d["Delta"]))
    print(f"   {nom:24s} -> mediana |δ̃ − Delta_IvyDB| = {med:.4f}")
    if med < best_med:
        mejor, best_med = k, med
K_TH = mejor
print(f"   -> se usa k = {K_TH:g}")

# B1 — TOTAL: corrige solo theta (retiene la co-movimiento de la vol)
d["delta_tot"]     = (d["dC"]     - d["Theta_l"] * d["dt_t"]   * K_TH) / d["dS"]
d["delta_tot_lag"] = (d["dC_lag"] - d["Theta_l"] * d["dt_lag"] * K_TH) / d["dS_lag"]
d["gamma_tot"] = 2 * (d["delta_tot"] - d["delta_tot_lag"]) / d["dS2"]

# B2 — PARCIAL: corrige theta y vega (σ fija)
d["delta_par"]     = (d["dC"]     - d["Theta_l"] * d["dt_t"]   * K_TH
                      - d["Vega_l"] * d["dIV"]) / d["dS"]
d["delta_par_lag"] = (d["dC_lag"] - d["Theta_l"] * d["dt_lag"] * K_TH
                      - d["Vega_l"] * d["dIV_lag"]) / d["dS_lag"]
d["gamma_par"] = 2 * (d["delta_par"] - d["delta_par_lag"]) / d["dS2"]

# B3 — canal vanna: la diferencia ES la dinámica del smile
d["canal_vanna"] = d["gamma_tot"] - d["gamma_par"]

val = d[d["Gamma"] > UMBRAL_GAMMA].copy()
filas = []
for nom, col in [("γ̂ cruda (pipeline)", "gamma_emp"),
                 ("γ̃ TOTAL (theta)", "gamma_tot"),
                 ("γ̃ PARCIAL (theta+vega)", "gamma_par")]:
    for etq, sub in [("todo", val), ("|d1|<2", val[val.d1.abs() < 2]),
                     ("|d1|<1", val[val.d1.abs() < 1])]:
        v = sub[col]; m = np.isfinite(v)
        if m.sum() < 30:
            continue
        filas.append(dict(estimador=nom, muestra=etq, N=int(m.sum()),
                          MedARE=np.median(np.abs(v[m] - sub.Gamma[m]) / sub.Gamma[m]),
                          pct_neg=np.mean(v[m] < 0)))
print("\nB1–B2 — precisión frente a la Gamma BSM de IvyDB")
print(tabulate(filas, headers="keys", floatfmt=".4f", tablefmt="github"))

c = val[val.d1.abs() < 1]["canal_vanna"]
gp = val[val.d1.abs() < 1]["gamma_par"]
print(f"\nB3 — canal vanna (|d1|<1): mediana = {np.nanmedian(c):+.3e}, "
      f"peso relativo mediano = {np.nanmedian(np.abs(c) / np.abs(gp)):.3f}")
print("     Este es el objeto económico: la parte de la convexidad realizada")
print("     que NO está en la gamma BSM, sino en cómo se mueve la superficie.")


# In[]: B4 — σ'(K): dinámica del smile por regresión directa

print("\nB4 — σ'(K) = dIV/dS por zona de moneyness (pendiente y R²)")
val["zona"] = pd.cut(val["Moneyness"], [0, .5, .7, .9, 1.1, np.inf],
                     labels=["very_deep", "deep", "near", "ATM", "OTM_call"])
filas = []
for (z, cp), G in val.groupby(["zona", "CallPut"], observed=True):
    G = G[np.isfinite(G.dIV) & np.isfinite(G.dS)]
    if len(G) < 200:
        continue
    x, y = G.dS.values, G.dIV.values
    b = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    filas.append(dict(zona=z, cp=cp, N=len(G), sigma_prima=b[0], R2=r ** 2))
print(tabulate(filas, headers="keys", floatfmt=".3e", tablefmt="github"))
print("LECTURA: σ' = 0 sería sticky-strike; σ' < 0 y creciente en |moneyness|")
print("         indica sticky-moneyness. Es la dinámica del smile, medible.")


# In[]: Guardado

out = os.path.join(OUT_DIR, "diagnostico_gamma.parquet")
cols_out = ["Date", "OptionID", "CallPut", "Moneyness", "d1", "Days",
            "OpenInterest", "Gamma", "gamma_emp", "gamma_pred", "g",
            "z_t", "z_lag", "gamma_tot", "gamma_par", "canal_vanna"]
d[cols_out].to_parquet(out, compression="snappy", index=False)
print(f"\nGuardado: {out}")
M.to_csv(os.path.join(OUT_DIR, "diagnostico_mensual.csv"))
print(f"Guardado: {os.path.join(OUT_DIR, 'diagnostico_mensual.csv')}")

# %%
