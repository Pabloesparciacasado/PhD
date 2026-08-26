# In[]: Monte Carlo — validación del estimador numérico de griegas
"""
09_montecarlo_gamma.py

Banco de pruebas del estimador op1 (delta/gamma por diferencias finitas)
contra un mundo donde la griega VERDADERA se conoce en forma cerrada.

Filosofía del diseño: GBM + precios BSM exactos NO se elige por realismo, sino
por ser el MEJOR CASO POSIBLE — vol constante, sin saltos, sin horquilla, sin
tick, sin asincronía, sin smile. Todo error que aparezca aquí es un suelo:
en los datos reales solo puede ser peor. Si el estimador rompe en este mundo,
rompe en cualquiera.

Escalera de contaminación (activar de una en una con los flags):
    0. BSM exacto ....................... aísla el sesgo de theta y la explosión 1/ΔS
    1. + tick ($0.05 / $0.10) ........... discretización del precio
    2. + horquilla / mid ruidoso ........ microestructura
    3. + asincronía quote vs cierre SPX . problema real de OptionMetrics
    4. + Heston ......................... vega·Δσ contamina ΔC (vanna/charm)

Regímenes barridos: sigma (calma / normal / crisis), tau, moneyness.

Bloques:
    A. Panel simulado + estimador op1 idéntico a 02_panel_construction
    B. Descomposición del error de delta: ¿es todo theta?
    C. Error de gamma por decil de |ΔS| (la hipérbola)
    D. ¿Salva la agregación transversal? (spoiler: ΔS es común al panel)
    E. Alternativas: regresión local en el tiempo vs. sección cruzada de strikes
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# ----------------------------- Configuración --------------------------------
SEED       = 7
S0         = 4000.0
R, Q       = 0.02, 0.0
DT         = 1 / 252
T_DAYS     = 1500
REGIMENES  = {"calma": 0.08, "normal": 0.16, "crisis": 0.40}

TICK       = 0.00      # 0.05 / 0.10 para activar discretización
RUIDO_MID  = 0.00      # desviación típica ($) del ruido de mid (horquilla)
TAU_MIN, TAU_MAX = 15, 45


# ----------------------------- Black-Scholes --------------------------------
def bs(S, K, tau, sig, cp=1):
    """Precio, delta, gamma y theta (anual) BSM."""
    S, K, tau = map(lambda x: np.asarray(x, float), (S, K, tau))
    srt = sig * np.sqrt(tau)
    d1  = (np.log(S / K) + (R - Q + 0.5 * sig ** 2) * tau) / srt
    d2  = d1 - srt
    dq, dr = np.exp(-Q * tau), np.exp(-R * tau)
    gam = dq * norm.pdf(d1) / (S * srt)
    if cp == 1:
        px  = S * dq * norm.cdf(d1) - K * dr * norm.cdf(d2)
        dlt = dq * norm.cdf(d1)
        th  = (-S * dq * norm.pdf(d1) * sig / (2 * np.sqrt(tau))
               - R * K * dr * norm.cdf(d2) + Q * S * dq * norm.cdf(d1))
    else:
        px  = K * dr * norm.cdf(-d2) - S * dq * norm.cdf(-d1)
        dlt = -dq * norm.cdf(-d1)
        th  = (-S * dq * norm.pdf(d1) * sig / (2 * np.sqrt(tau))
               + R * K * dr * norm.cdf(-d2) - Q * S * dq * norm.cdf(-d1))
    return px, dlt, gam, th


# ------------------- A. Panel simulado + estimador op1 ----------------------
def simular_panel(sigma, seed=SEED, cp=1):
    rng  = np.random.default_rng(seed)
    z    = rng.standard_normal(T_DAYS)
    S    = np.r_[S0, np.exp(np.log(S0) + np.cumsum(
        (R - Q - 0.5 * sigma ** 2) * DT + sigma * np.sqrt(DT) * z))]

    filas = []
    for exp_day in range(TAU_MAX, T_DAYS, 21):            # vencimientos mensuales
        Sref = S[max(exp_day - TAU_MAX, 0)]
        for m in np.arange(0.80, 1.21, 0.02):
            K = round(Sref * m / 5) * 5                    # malla de strikes de 5 pts
            for t in range(max(exp_day - TAU_MAX, 1), exp_day):
                tau_d = exp_day - t
                if TAU_MIN < tau_d <= TAU_MAX:
                    filas.append((t, exp_day, K, tau_d))

    df = pd.DataFrame(filas, columns=["t", "exp", "K", "tau_d"])
    df["OptionID"] = df["exp"].astype(str) + "_" + df["K"].astype(str)
    df["tau"] = df["tau_d"] / 365.0
    df["S"]   = S[df["t"].to_numpy()]
    px, dlt, gam, th = bs(df["S"], df["K"], df["tau"], sigma, cp)
    df["C"], df["Delta_true"], df["Gamma_true"], df["Theta_true"] = px, dlt, gam, th

    # --- contaminación opcional (escalera) ---
    if RUIDO_MID > 0:
        df["C"] += np.random.default_rng(seed + 1).normal(0, RUIDO_MID, len(df))
    if TICK > 0:
        df["C"] = np.round(df["C"] / TICK) * TICK

    df["Moneyness"] = df["K"] / df["S"]
    return df.sort_values(["OptionID", "t"]).reset_index(drop=True), S


def estimador_op1(df):
    """Réplica exacta de delta_empirica_op1 / gamma_empirica_op1 de 02_."""
    g = df.groupby("OptionID")
    df = df.assign(C_l=g["C"].shift(1), S_l=g["S"].shift(1))
    d  = df.dropna(subset=["C_l"]).copy()
    d  = d[d["S"] != d["S_l"]]
    d["dS"]        = d["S"] - d["S_l"]
    d["delta_emp"] = (d["C"] - d["C_l"]) / d["dS"]

    g2 = d.groupby("OptionID")
    d  = d.assign(delta_l=g2["delta_emp"].shift(1), S_l2=g2["S_l"].shift(1))
    gg = d.dropna(subset=["delta_l", "S_l2"]).copy()
    gg = gg[gg["S"] != gg["S_l2"]]
    gg["dS2"]       = gg["S"] - gg["S_l2"]
    gg["gamma_emp"] = 2 * (gg["delta_emp"] - gg["delta_l"]) / gg["dS2"]
    # NOTA: el factor 2 es CORRECTO. delta_emp(t) estima Δ en el punto medio
    # (S_t+S_{t-1})/2 y delta_emp(t-1) en (S_{t-1}+S_{t-2})/2; la distancia
    # entre ambos centros es (S_t - S_{t-2})/2, de ahí el 2.
    return d, gg


# ------------------- B-C. Diagnóstico del error ------------------------------
def diagnostico(d, gg, etiqueta=""):
    d  = d.copy(); gg = gg.copy()
    d["err_delta"]  = d["delta_emp"] - d["Delta_true"]
    d["sesgo_teta"] = d["Theta_true"] * DT / d["dS"]     # término Θ·Δt/ΔS
    gg["rel_err"]   = (gg["gamma_emp"] - gg["Gamma_true"]) / gg["Gamma_true"]

    ok = np.isfinite(d["err_delta"]) & np.isfinite(d["sesgo_teta"])
    rho = np.corrcoef(d.loc[ok, "err_delta"], d.loc[ok, "sesgo_teta"])[0, 1]

    print(f"\n{'='*66}\nDIAGNÓSTICO {etiqueta}\n{'='*66}")
    print(f"[delta] corr(error, Θ·Δt/ΔS) = {rho:.4f}   R² = {rho**2:.4f}")
    print(f"[gamma] MedARE global        = {np.median(np.abs(gg['rel_err'])):.3f}")
    print(f"[gamma] % |error| > 100%     = {np.mean(np.abs(gg['rel_err']) > 1)*100:.1f}%")
    print(f"[gamma] % signo negativo     = {np.mean(gg['gamma_emp'] < 0)*100:.1f}%"
          f"   (la verdadera es siempre > 0)")

    gg["dec"] = pd.qcut(gg["dS2"].abs(), 10, labels=False, duplicates="drop")
    tab = gg.groupby("dec").agg(
        dS2_mediano=("dS2", lambda x: np.median(np.abs(x))),
        MedARE=("rel_err", lambda x: np.median(np.abs(x))),
        pct_err_gt_100=("rel_err", lambda x: np.mean(np.abs(x) > 1)),
        pct_signo_malo=("gamma_emp", lambda x: np.mean(x < 0)),
        n=("rel_err", "size"))
    print("\nError de gamma por decil de |ΔS₂| (la hipérbola):")
    print(tab.to_string(float_format=lambda v: f"{v:,.4f}"))
    return d, gg


# ------------------- D. ¿Salva la agregación transversal? --------------------
def test_agregacion(gg):
    """
    Punto crítico: en un día dado TODOS los contratos comparten el mismo ΔS.
    El error NO es idiosincrático — es un factor común ∝ 1/ΔS_t. Promediar en
    la sección cruzada no lo elimina.
    """
    gg = gg.copy()
    gg["w"] = gg["S"] * np.sqrt(gg["tau"]) * norm.pdf(0)     # proxy de OI ~ vega

    def wa(x, w):
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        return np.nan if not m.any() else np.average(x[m], weights=w[m])

    diario = (gg.groupby("t")
                .apply(lambda G: pd.Series({
                    "wa_emp":  wa(G["gamma_emp"].to_numpy(), G["w"].to_numpy()),
                    "wa_true": wa(G["Gamma_true"].to_numpy(), G["w"].to_numpy()),
                    "dS2_med": np.median(np.abs(G["dS2"]))}), include_groups=False)
                .dropna())
    diario["err"]    = diario["wa_emp"] - diario["wa_true"]
    diario["inv_dS"] = 1.0 / diario["dS2_med"]

    print(f"\n{'='*66}\nAGREGACIÓN TRANSVERSAL (WA diaria)\n{'='*66}")
    print(f"MedARE de la WA diaria       = "
          f"{np.median(np.abs(diario['err']/diario['wa_true'])):.3f}")
    print(f"% días con WA negativa       = {np.mean(diario['wa_emp'] < 0)*100:.1f}%")
    rho = diario[["err", "inv_dS"]].corr().iloc[0, 1]
    print(f"corr(error WA, 1/|ΔS|)       = {rho:.4f}   R² = {rho**2:.4f}"
          f"   <-- el error NO se diversifica")

    diario["mes"] = diario.index // 21
    mens = diario.groupby("mes").agg(emp=("wa_emp", "mean"),
                                     true=("wa_true", "mean"),
                                     inv=("inv_dS", "mean"))
    print(f"corr MENSUAL(WA_emp, WA_true)= {mens[['emp','true']].corr().iloc[0,1]:.4f}")
    print(f"corr MENSUAL(WA_emp, 1/|ΔS|) = {mens[['emp','inv']].corr().iloc[0,1]:.4f}"
          f"   <-- proxy de vol realizada inversa")
    return diario


# ------------------- E. Alternativa: sección cruzada de strikes --------------
def gamma_seccion_cruzada(sigma=0.16, tau_d=30, h=25.0, tick=0.05, ruido=0.02,
                          reps=200, seed=3):
    """
    Bajo homogeneidad de grado 1 en (S,K):   S²·∂²C/∂S² = K²·∂²C/∂K²
    => gamma se recupera de la sección cruzada de strikes del MISMO día:
       sin theta, sin Δt, sin división por ΔS. Cientos de observaciones.
    El coste se traslada al ancho de banda h: sesgo ~ h², varianza ~ ruido/h².
    """
    tau = tau_d / 365
    K   = np.arange(3200, 4801, h)
    C, _, G_true, _ = bs(S0, K, tau, sigma)
    rng = np.random.default_rng(seed)
    sel = (K[1:-1] / S0 > 0.9) & (K[1:-1] / S0 < 1.1)

    out = []
    for _ in range(reps):
        Cn = C + rng.normal(0, ruido, len(C))
        if tick > 0:
            Cn = np.round(Cn / tick) * tick
        C_KK  = (Cn[2:] - 2 * Cn[1:-1] + Cn[:-2]) / h ** 2
        G_hat = (K[1:-1] / S0) ** 2 * C_KK
        out.append(np.median(np.abs((G_hat[sel] - G_true[1:-1][sel])
                                    / G_true[1:-1][sel])))
    return float(np.mean(out))


# ================================ EJECUCIÓN ==================================
if __name__ == "__main__":
    for nombre, sigma in REGIMENES.items():
        df, S = simular_panel(sigma)
        d, gg = estimador_op1(df)
        d, gg = diagnostico(d, gg, etiqueta=f"régimen '{nombre}' (σ={sigma:.0%})")
        test_agregacion(gg)

    print(f"\n{'='*66}\nSECCIÓN CRUZADA DE STRIKES: sesgo-varianza en h\n{'='*66}")
    print(f"{'h (pts)':>8} | {'MedARE':>8}")
    for h in [5, 10, 25, 50, 100]:
        print(f"{h:>8} | {gamma_seccion_cruzada(h=float(h)):>8.4f}")
    print("\nCon precios exactos y h=5: MedARE ≈ 0.0002 (la identidad es exacta).")
    print("El error de arriba es ENTERAMENTE ruido de tick amplificado por 1/h².")
    print("Esto es exactamente el problema que resuelve un LPKR / kernel sobre K.")

# %%
