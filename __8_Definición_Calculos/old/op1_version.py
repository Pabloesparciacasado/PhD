# ============================================================
# 01_Ablation_Numerical_Examination.py   —   VERSIÓN OP1
# ============================================================
#
# Misma estructura que la versión con bump. Único cambio de fondo:
# las griegas se obtienen con OP1 sobre el camino realizado, no con
# un bump controlado de paso h.
#
#   delta_emp(t) = [C_t - C_{t-1}] / [S_t - S_{t-1}]
#   gamma_emp(t) = 2*[delta_emp(t) - delta_emp(t-1)] / [S_t - S_{t-2}]
#
# Cambios respecto a la versión con h:
#   - fuera  h_s, S_plus, S_min, C_plus, C_min, numeric_greek
#   - dentro estimador_op1 y el oracle en el CENTRO EFECTIVO de op1
#   - OptionID = (K, Sim), para que groupby no mezcle strikes
#   - vanna con el signo corregido:  -e^{-qτ}·φ(d1)·d2/σ
# ============================================================

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from scipy.stats import norm
from tabulate import tabulate


# ============================================================
# 1. PARÁMETROS DEL EXPERIMENTO
# ============================================================

@dataclass(frozen=True)
class Params:

    # Monte Carlo
    n_paths: int = 1_000
    seed: int = 123

    # Spot
    S0: float = 100.0
    mu: float = 0.0
    spot_vol: float = 0.20

    # Volatilidad implícita inicial
    sigma0: float = 0.20

    # sigma(S) = sigma0 * (S/S0)^beta ;  beta < 0 -> leverage effect
    beta: float = -0.50

    # Shock ortogonal de volatilidad
    sigma_shock_sd: float = 0.01

    # Paso diario
    dt: float = 1 / 252

    T: int = 5 * 252

    # tau0 cubre todo el horizonte: en config B nunca llega a tau <= 0
    tau0: float = (5 * 252 + 1) / 252

    r: float = 0.02
    q: float = 0.00

    # convención del script: moneyness = S0 / K
    moneyness: list = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 1.75])

    option: str = "call"


# ============================================================
# 2. BLACK-SCHOLES: PRECIO + GREEKS
# ============================================================

def bs_price_greeks(S, K, tau, r, q, sigma, option="call"):
    """
    Devuelve price, delta, gamma, vega, volga, vanna, c_tau
    con c_tau = ∂C/∂tau   (Theta_calendar = -c_tau).
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if np.any(tau <= 0):
        raise ValueError("tau must be strictly positive.")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive.")

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    discount_q = np.exp(-q * tau)
    discount_r = np.exp(-r * tau)

    if option == "call":
        price = S * discount_q * norm.cdf(d1) - K * discount_r * norm.cdf(d2)
        delta = discount_q * norm.cdf(d1)
        c_tau = (S * discount_q * norm.pdf(d1) * sigma / (2 * sqrt_tau)
                 + r * K * discount_r * norm.cdf(d2)
                 - q * S * discount_q * norm.cdf(d1))

    elif option == "put":
        price = K * discount_r * norm.cdf(-d2) - S * discount_q * norm.cdf(-d1)
        delta = discount_q * (norm.cdf(d1) - 1)
        c_tau = (S * discount_q * norm.pdf(d1) * sigma / (2 * sqrt_tau)
                 - r * K * discount_r * norm.cdf(-d2)
                 + q * S * discount_q * norm.cdf(-d1))
    else:
        raise ValueError("option must be 'call' or 'put'.")

    gamma = discount_q * norm.pdf(d1) / (S * sigma * sqrt_tau)
    vega = S * discount_q * norm.pdf(d1) * sqrt_tau
    volga = vega * (d1 * d2) / sigma

    # CORREGIDO: vanna = ∂²C/∂S∂σ = -e^{-qτ}·φ(d1)·d2/σ   (el signo iba al revés)
    vanna = -discount_q * norm.pdf(d1) * (d2 / sigma)

    return price, delta, gamma, vega, volga, vanna, c_tau


# ============================================================
# 3. RELACIÓN SISTEMÁTICA SPOT-VOL
# ============================================================

def sigma_systematic(S, S0, sigma0, beta):
    """sigma(S) = sigma0 * (S/S0)^beta ;  beta<0: spot baja -> IV sube."""
    S = np.asarray(S, dtype=float)
    return sigma0 * (S / S0) ** beta


# ============================================================
# 4. SIMULACIÓN DE LOS CAMINOS DE SPOT
# ============================================================

def simulate_base_paths(p):
    """
    T+1 fechas por camino. Todos los escenarios A-D comparten los MISMOS
    caminos de spot y los MISMOS shocks: comparación contrafactual path-by-path.
    """
    rng = np.random.default_rng(p.seed)

    z = rng.standard_normal(size=(p.n_paths, p.T))
    S = np.empty((p.n_paths, p.T + 1))
    S[:, 0] = p.S0
    for j in range(1, p.T + 1):
        S[:, j] = S[:, j - 1] * np.exp((p.mu - 0.5 * p.spot_vol**2) * p.dt
                                       + p.spot_vol * np.sqrt(p.dt) * z[:, j - 1])

    eps_sigma = rng.normal(loc=0.0, scale=p.sigma_shock_sd,
                           size=(p.n_paths, p.T + 1))

    n_money = len(p.moneyness)
    n_base = p.n_paths * (p.T + 1)
    K_levels = np.array([p.S0 / m for m in p.moneyness])

    base = pd.DataFrame({
        "K": np.repeat(K_levels, n_base),
        "Sim": np.tile(np.repeat(np.arange(p.n_paths), p.T + 1), n_money) + 1,
        "t": np.tile(np.tile(np.arange(p.T + 1), p.n_paths), n_money),
        "S": np.tile(S.ravel(), n_money),
        "eps_sigma": np.tile(eps_sigma.ravel(), n_money),
    })

    # identificador único de contrato: sin esto, groupby("Sim") mezcla strikes.
    # ENTERO, no string: sobre millones de filas una columna de str se come GB.
    idx_k = np.repeat(np.arange(n_money), n_base)
    base["OptionID"] = (idx_k * (p.n_paths + 1) + base["Sim"]).astype(np.int64)
    return base


# ============================================================
# 5. CONSTRUCCIÓN DE CADA CONFIGURACIÓN
# ============================================================

def build_config_panel(base, config, p):
    """
    A: S cambia; tau fija; sigma fija
    B: S cambia; tau cambia; sigma fija
    C: S cambia; tau fija; sigma = f(S)
    D: S cambia; tau fija; sigma = f(S) + epsilon_sigma
    """
    df = base.copy()

    sigma_sys = sigma_systematic(S=df["S"], S0=p.S0, sigma0=p.sigma0, beta=p.beta)

    if config == "A":
        df["tau"] = p.tau0
        df["sigma"] = p.sigma0
    elif config == "B":
        df["tau"] = p.tau0 - df["t"] * p.dt
        df["sigma"] = p.sigma0
    elif config == "C":
        df["tau"] = p.tau0
        df["sigma"] = sigma_sys
    elif config == "D":
        df["tau"] = p.tau0
        df["sigma"] = sigma_sys + df["eps_sigma"]
    else:
        raise ValueError("config must be A, B, C or D.")

    if (df["sigma"] <= 0).any():
        raise ValueError("Non-positive volatility generated.")

    price, delta, gamma, vega, volga, vanna, c_tau = bs_price_greeks(
        S=df["S"], K=df["K"], tau=df["tau"], r=p.r, q=p.q,
        sigma=df["sigma"], option=p.option)

    df["C"] = price
    df["delta_bs"] = delta
    df["gamma_bs_row"] = gamma
    df["vega"] = vega
    df["volga"] = volga
    df["vanna"] = vanna
    df["c_tau"] = c_tau
    return df


# ============================================================
# 6. ESTIMADOR EMPÍRICO OP1        <-- sustituye a numeric_greek
# ============================================================

def estimador_op1(df):
    """
    delta_emp(t) = [C_t - C_{t-1}] / [S_t - S_{t-1}]
    gamma_emp(t) = 2*[delta_emp(t) - delta_emp(t-1)] / [S_t - S_{t-2}]

    El dataframe debe venir ordenado por OptionID y t.
    Se descartan denominadores exactamente nulos (nunca en GBM continuo,
    pero el filtro replica el pipeline empírico).
    """
    df = df.sort_values(["OptionID", "t"])
    g = df.groupby("OptionID", sort=False)

    df = df.assign(C_l=g["C"].shift(1), S_l=g["S"].shift(1),
                   tau_l=g["tau"].shift(1), sigma_l=g["sigma"].shift(1))

    d = df.dropna(subset=["C_l"]).copy()
    d = d[d["S"] != d["S_l"]]
    d["dS"] = d["S"] - d["S_l"]
    d["delta_emp"] = (d["C"] - d["C_l"]) / d["dS"]

    g2 = d.groupby("OptionID", sort=False)
    d = d.assign(delta_l=g2["delta_emp"].shift(1),
                 S_l2=g2["S_l"].shift(1),
                 tau_l2=g2["tau_l"].shift(1),
                 sigma_l2=g2["sigma_l"].shift(1))

    gg = d.dropna(subset=["delta_l", "S_l2"]).copy()
    gg = gg[gg["S"] != gg["S_l2"]]
    gg["dS2"] = gg["S"] - gg["S_l2"]
    gg["gamma_emp"] = 2 * (gg["delta_emp"] - gg["delta_l"]) / gg["dS2"]

    # El factor 2 es correcto: delta_emp(t) está centrada en (S_t+S_{t-1})/2 y
    # delta_emp(t-1) en (S_{t-1}+S_{t-2})/2; la distancia entre centros es
    # (S_t - S_{t-2})/2.
    return d, gg


# ============================================================
# 7. ORACLE EN EL CENTRO EFECTIVO DE OP1
# ============================================================

def attach_oracle(gg, p):
    """
    gamma_emp está centrada en el estado [x0 + 2*x1 + x2]/4 para S, tau y sigma.
    Ahí evaluamos:
        gamma_bs      -> Gamma BSM PARCIAL
        gamma_oracle  -> derivada TOTAL de la configuración
                         (en C y D:  Gamma + 2·Vanna·σ' + Volga·σ'² + Vega·σ'')
    """
    out = gg.copy()

    out["S_gamma"] = (out["S_l2"] + 2 * out["S_l"] + out["S"]) / 4
    out["tau_gamma"] = (out["tau_l2"] + 2 * out["tau_l"] + out["tau"]) / 4
    out["sigma_gamma"] = (out["sigma_l2"] + 2 * out["sigma_l"] + out["sigma"]) / 4

    _, _, gamma_bs, vega, volga, vanna, _ = bs_price_greeks(
        S=out["S_gamma"], K=out["K"], tau=out["tau_gamma"], r=p.r, q=p.q,
        sigma=out["sigma_gamma"], option=p.option)

    out["gamma_bs"] = gamma_bs

    # sigma' y sigma'' de la ley de potencia (0 en A y B, donde sigma es fija)
    var_sigma = out["sigma_gamma"].to_numpy()
    S_g = out["S_gamma"].to_numpy()
    sp = p.beta * var_sigma / S_g
    spp = p.beta * (p.beta - 1.0) * var_sigma / S_g**2

    out["gamma_oracle"] = np.where(
        out["config"].isin(["C", "D"]),
        gamma_bs + 2 * vanna * sp + volga * sp**2 + vega * spp,
        gamma_bs)

    out["abs_r01"] = np.abs((out["S_l"] - out["S_l2"]) / out["S_l2"])
    out["abs_r12"] = np.abs((out["S"] - out["S_l"]) / out["S_l"])
    return out


# ============================================================
# 8. MÉTRICAS
# ============================================================

def resumen(out, umbral_gamma=1e-10):
    rows = []
    for (cfg, K), d in out.groupby(["config", "K"], sort=True):
        d = d[(d["gamma_bs"] > umbral_gamma) & np.isfinite(d["gamma_emp"])]
        if len(d) < 50:
            continue
        e_or = np.abs(d["gamma_emp"] / d["gamma_oracle"] - 1)
        e_bs = np.abs(d["gamma_emp"] / d["gamma_bs"] - 1)
        rows.append({
            "config": cfg, "K": round(K, 1),
            "K_sobre_S0": round(K / 100.0, 2),
            "N": len(d),
            "Gamma_BS_med": d["gamma_bs"].median(),
            "canal_eco_pct": 100 * np.median(d["gamma_oracle"] / d["gamma_bs"] - 1),
            "MedARE_vs_ORACLE": np.median(e_or),
            "MedARE_vs_BS": np.median(e_bs),
            "pct_gamma_neg": np.mean(d["gamma_emp"] < 0)})
    return pd.DataFrame(rows)


# ============================================================
# 9. EJECUCIÓN
# ============================================================

def run(p):
    """
    Resume DENTRO del bucle y descarta el panel: con T grande, guardar los
    cuatro paneles completos se come varios GB sin necesidad.
    """
    from dataclasses import replace
    tablas = []
    # un strike cada vez: el panel completo (todos los strikes x todas las
    # fechas x 4 configs) no cabe en memoria con T grande.
    for m in p.moneyness:
        pm = replace(p, moneyness=[m])
        base = simulate_base_paths(pm)
        for config in ("A", "B", "C", "D"):
            panel = build_config_panel(base, config, pm)
            _, gg = estimador_op1(panel)
            del panel
            gg["config"] = config
            tablas.append(resumen(attach_oracle(gg, pm)))
            del gg
        del base
    return pd.concat(tablas, ignore_index=True).sort_values(
        ["K_sobre_S0", "config"]).reset_index(drop=True)


if __name__ == "__main__":

    p = Params(n_paths=1_000, seed=123, S0=100.0, mu=0.0, spot_vol=0.20,
               sigma0=0.20, beta=-0.50, sigma_shock_sd=0.01, dt=1/252,
               T=5*252, tau0=(5*252 + 1)/252, r=0.02, q=0.00,
               moneyness=[0.5, 0.75, 1.0, 1.5, 1.75], option="call")

    tab = run(p)

    print(f"\nOP1 sobre el camino realizado — {p.n_paths} caminos, "
          f"T={p.T} pasos, beta={p.beta}\n")
    print(tabulate(tab, headers="keys", tablefmt="rounded_outline",
                   floatfmt=".5g", showindex=False))
