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
#   - dentro estimador_op1 y un oracle ANALÍTICO independiente de OP1
#   - OptionID = (K, Sim), para que groupby no mezcle strikes
#   - vanna con el signo corregido:  -e^{-qτ}·φ(d1)·d2/σ
# ============================================================

from dataclasses import dataclass, field,replace

import numpy as np
import pandas as pd

from scipy.stats import norm
from tabulate import tabulate

import time
import os
import tracemalloc



import time, threading
import psutil



# ============================================================
# 0. RENDIMIENTO DE COMOPUTACION
# ============================================================

class Bench:
    """Mide tiempo, pico de RSS real y núcleos efectivos usados."""
    def __init__(self, nombre="bloque", intervalo=0.05, registro=None):
        self.nombre, self.intervalo = nombre, intervalo
        self.registro = registro
        self.proc = psutil.Process()
        self.pico_rss = 0
        self._stop = threading.Event()

    def _rss_total(self):
        rss = self.proc.memory_info().rss
        for h in self.proc.children(recursive=True):
            try:
                rss += h.memory_info().rss
            except psutil.Error:
                pass
        return rss

    def _cpu_total(self):
        t = self.proc.cpu_times()
        seg = t.user + t.system
        for h in self.proc.children(recursive=True):
            try:
                ht = h.cpu_times()
                seg += ht.user + ht.system
            except psutil.Error:
                pass
        return seg

    def _muestrear(self):
        while not self._stop.is_set():
            try:
                self.pico_rss = max(self.pico_rss, self._rss_total())
            except psutil.Error:
                pass
            self._stop.wait(self.intervalo)

    def __enter__(self):
        self.cpu0 = self._cpu_total()
        self.t0 = time.perf_counter()
        self.hilo = threading.Thread(target=self._muestrear, daemon=True)
        self.hilo.start()
        return self

    def __exit__(self, *args):
        pared = time.perf_counter() - self.t0
        cpu = self._cpu_total() - self.cpu0
        self._stop.set(); self.hilo.join()
        fila = {"bloque": self.nombre,
                "pared_s": round(pared, 2),
                "cpu_s": round(cpu, 2),
                "nucleos": round(cpu / pared, 2) if pared > 0 else 0,
                "pico_MB": round(self.pico_rss / 1024**2, 1)}
        if self.registro is None:
            print(fila)
        else:
            self.registro.append(fila)

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


def bs_delta_tau(S, K, tau, r, q, sigma, option="call"):
    """
    Derivada analítica de Delta respecto al tiempo a vencimiento tau:

        delta_tau = ∂Delta / ∂tau = C_{S tau}.

    Si se usa Charm con tiempo calendario t:

        Charm = ∂Delta/∂t = -delta_tau,

    porque tau = T - t.
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
    log_m = np.log(S / K)
    a = r - q + 0.5 * sigma**2

    d1 = (log_m + a * tau) / (sigma * sqrt_tau)
    discount_q = np.exp(-q * tau)

    if option == "call":
        delta = discount_q * norm.cdf(d1)
    elif option == "put":
        delta = discount_q * (norm.cdf(d1) - 1.0)
    else:
        raise ValueError("option must be 'call' or 'put'.")

    d1_tau = (a * tau - log_m) / (2.0 * sigma * tau**1.5)

    delta_tau = ( -q * delta + discount_q * norm.pdf(d1) * d1_tau ) 

    return delta_tau


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
# 6. ESTIMADOR EMPÍRICO OP1    
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
                   tau_l=g["tau"].shift(1), sigma_l=g["sigma"].shift(1),
                   eps_sigma_l=g["eps_sigma"].shift(1))

    d = df.dropna(subset=["C_l"]).copy()
    d = d[d["S"] != d["S_l"]]
    d["dS"] = d["S"] - d["S_l"]
    d["delta_emp"] = (d["C"] - d["C_l"]) / d["dS"]

    g2 = d.groupby("OptionID", sort=False)
    d = d.assign(delta_l=g2["delta_emp"].shift(1),
                 S_l2=g2["S_l"].shift(1),
                 tau_l2=g2["tau_l"].shift(1),
                 sigma_l2=g2["sigma_l"].shift(1),
                 eps_sigma_l2=g2["eps_sigma_l"].shift(1))

    gg = d.dropna(subset=["delta_l", "S_l2"]).copy()
    gg = gg[gg["S"] != gg["S_l2"]]
    gg["dS2"] = gg["S"] - gg["S_l2"]
    gg["gamma_emp"] = 2 * (gg["delta_emp"] - gg["delta_l"]) / gg["dS2"]

    # El factor 2 es correcto: delta_emp(t) está centrada en (S_t+S_{t-1})/2 y
    # delta_emp(t-1) en (S_{t-1}+S_{t-2})/2; la distancia entre centros es
    # (S_t - S_{t-2})/2.
    return d, gg

# ============================================================
# 7. ORACLE ANALÍTICO EN EL PUNTO MEDIO PONDERADO
# ============================================================

def attach_oracle(gg, p):
    """
    Oracle completamente independiente de OP1.

    Punto de evaluación:

        x_mid = (x_{t-2} + 2*x_{t-1} + x_t) / 4

    para S, tau y sigma cuando corresponda.

    A — Spot only
        Gamma_oracle = Gamma_BS.

    B — Spot + Time
        Consideramos Delta = Delta(S, tau). A primer orden:

            dDelta = Gamma*dS + Delta_tau*dTau.

        Por tanto, a lo largo del movimiento local observado:

            Gamma_oracle = Gamma_BS + Delta_tau*(dTau/dS).

        Los términos de segundo orden de la expansión de Delta,

            0.5*Delta_SS*dS^2 + Delta_Stau*dS*dTau + 0.5*Delta_tautau*dTau^2,

        se cancelan al formar el cociente centrado entre los dos centros
        efectivos (t-2,t-1) y (t-1,t), porque ambos son simétricos alrededor
        del punto medio ponderado. Por eso no aparece una corrección cuadrática
        adicional; el siguiente error es de orden cúbico.

    C — Spot + systematic IV
        sigma = sigma_sys(S), por lo que:

        Gamma_oracle = d^2 C(S, sigma(S)) / dS^2
                     = Gamma+ 2*Vanna*sigma'+ Volga*(sigma')^2 + Vega*sigma''.

    D — systematic IV + orthogonal IV shock
        El shock epsilon modifica el NIVEL de IV observado y, por tanto,
        el nivel local de las Greeks, pero por construcción no constituye
        una relación estructural con S:

            d epsilon / dS = 0.

        El oracle es la sensibilidad analítica condicional al nivel local
        observado de sigma, manteniendo epsilon fijo respecto a S:

            Gamma_oracle = Gamma + 2*Vanna*sigma_sys' + Volga*(sigma_sys')^2 + Vega*sigma_sys''.

        Así, las variaciones realizadas de epsilon entran en los precios que
        observa OP1, pero NO se convierten artificialmente en una derivada
        spot-vol dentro del oracle.
    """
    out = gg.copy()

    if out.empty:
        return out

    assert out["config"].nunique() == 1, \
        "attach_oracle expects a single configuration."

    cfg = out["config"].iloc[0]

    # --------------------------------------------------------
    # Punto medio de evaluación
    # --------------------------------------------------------

    out["S_gamma"] = ( out["S_l2"] + 2.0 * out["S_l"] + out["S"]) / 4.0

    out["tau_gamma"] = ( out["tau_l2"] + 2.0 * out["tau_l"] + out["tau"] ) / 4.0
        
    out["sigma_gamma"] = ( out["sigma_l2"] + 2.0 * out["sigma_l"] + out["sigma"] ) / 4.0
        

    # Relación estructural sigma_sys(S) evaluada en S_mid.
    # En D NO incluye epsilon.
    sigma_sys_gamma = sigma_systematic( S=out["S_gamma"],  S0=p.S0,  sigma0=p.sigma0, beta=p.beta)
        

    # ========================================================
    # A — SPOT ONLY
    # ========================================================

    if cfg == "A":

        _, _, gamma_bs, _, _, _, _ = bs_price_greeks(
            S=out["S_gamma"],
            K=out["K"],
            tau=out["tau_gamma"],
            r=p.r,
            q=p.q,
            sigma=out["sigma_gamma"],
            option=p.option
        )

        out["gamma_bs"] = gamma_bs
        out["gamma_oracle"] = gamma_bs

    # ========================================================
    # B — SPOT + TIME
    # ========================================================

    elif cfg == "B":

        _, _, gamma_bs, _, _, _, _ = bs_price_greeks(
            S=out["S_gamma"],
            K=out["K"],
            tau=out["tau_gamma"],
            r=p.r,
            q=p.q,
            sigma=out["sigma_gamma"],
            option=p.option
        )

        delta_tau = bs_delta_tau(
            S=out["S_gamma"],
            K=out["K"],
            tau=out["tau_gamma"],
            r=p.r,
            q=p.q,
            sigma=out["sigma_gamma"],
            option=p.option
        )

        # Pendiente local del camino tau frente a S entre los dos
        # centros efectivos. El factor 1/2 se cancela en numerador
        # y denominador:
        #
        #   dTau_center/dS_center
        #   = [(tau_t-tau_{t-2})/2] / [(S_t-S_{t-2})/2].
        tau_slope = (  (out["tau"] - out["tau_l2"])  / (out["S"] - out["S_l2"]) )
           
        out["gamma_bs"] = gamma_bs
        out["gamma_oracle"] = (  gamma_bs + delta_tau * tau_slope )
        
    # ========================================================
    # C — SPOT + SYSTEMATIC IV
    # ========================================================

    elif cfg == "C":

        # Greeks evaluadas en el punto medio observado.
        _, _, gamma_bs, vega, volga, vanna, _ = bs_price_greeks(
            S=out["S_gamma"],
            K=out["K"],
            tau=out["tau_gamma"],
            r=p.r,
            q=p.q,
            sigma=out["sigma_gamma"],
            option=p.option
        )

        sigma_prime = (p.beta * sigma_sys_gamma  / out["S_gamma"])
            
        sigma_second = (p.beta * (p.beta - 1.0) * sigma_sys_gamma / out["S_gamma"]**2)
            
        out["gamma_bs"] = gamma_bs
        out["gamma_oracle"] = ( gamma_bs + 2.0 * vanna * sigma_prime + volga * sigma_prime**2 + vega * sigma_second)
            
    # ========================================================
    # D — SYSTEMATIC IV + ORTHOGONAL IV SHOCK
    # ========================================================

    elif cfg == "D":

        # El nivel observado de sigma incluye epsilon y afecta a las Greeks.
        # Sin embargo, las derivadas d sigma/dS y d² sigma/dS² pertenecen
        # únicamente al componente estructural sigma_sys(S).
        _, _, gamma_bs, vega, volga, vanna, _ = bs_price_greeks(
            S=out["S_gamma"],
            K=out["K"],
            tau=out["tau_gamma"],
            r=p.r,
            q=p.q,
            sigma=out["sigma_gamma"],
            option=p.option
        )

        sigma_prime = (  p.beta * sigma_sys_gamma / out["S_gamma"])

        sigma_second = (  p.beta * (p.beta - 1.0) * sigma_sys_gamma / out["S_gamma"]**2)
                    
        out["gamma_bs"] = gamma_bs
        out["gamma_oracle"] = ( gamma_bs + 2.0 * vanna * sigma_prime + volga * sigma_prime**2  + vega * sigma_second)
            
    else:
        raise ValueError("config must be A, B, C or D.")

    # --------------------------------------------------------
    # Diagnósticos:
    # --------------------------------------------------------

    # out["abs_r01"] = np.abs((out["S_l"] - out["S_l2"]) / out["S_l2"])
    # out["abs_r12"] = np.abs((out["S"] - out["S_l"])  / out["S_l"])

    out["abs_r01"] = np.abs((out["S_l"] - out["S_l2"]) / out["S_l2"])
    out["abs_r02"] = np.abs((out["S"] - out["S_l2"]) / out["S_l2"])
    out["abs_r12"] = np.abs((out["S"] - out["S_l"])  / out["S_l"])


    

    return out


# ============================================================
# 8. MÉTRICAS
# ============================================================

def resumen(out, umbral_gamma=1e-10, umbral_spot = 1e-3):

# CALCULO MANUAL DE LA CORRELACIÓN DE PEARSON:
    # def stable_corr(x, y):
    #     if len(x) < 2:
    #         return np.nan
    #     # Varianza casi nula: std <= 1e-10 en escala absoluta,
    #     # o <= 1e-10 del maximo absoluto cuando este supera 1.
    #     sx, sy = np.std(x), np.std(y)
    #     tol_x = 1e-10 * max(1.0, np.max(np.abs(x)))
    #     tol_y = 1e-10 * max(1.0, np.max(np.abs(y)))
    #     if (not np.isfinite(sx) or not np.isfinite(sy)
    #             or sx <= tol_x or sy <= tol_y):
    #         return np.nan
    #     return float(np.clip(np.mean(((x - x.mean()) / sx)
    #                                  * ((y - y.mean()) / sy)), -1.0, 1.0))

    def stable_corr(x, y):
        if len(x) < 2:
            return np.nan

        # Umbral absoluto para considerar la varianza casi nula.
        if np.var(x) <= 1e-20 or np.var(y) <= 1e-20:
            return np.nan

        return np.corrcoef(x, y)[0, 1]


    rows = []
    for (cfg, K), d in out.groupby(["config", "K"], sort=True):
        # d = d[(d["gamma_bs"] > umbral_gamma) & np.isfinite(d["gamma_emp"])]
        d = d[(d["gamma_bs"] > umbral_gamma) & np.isfinite(d[["gamma_emp", "gamma_bs", "gamma_oracle"]]).all(axis=1)].copy()

        if len(d) < 50:
            continue

        d["Dummy_S"] = ((d["abs_r01"] < umbral_spot)  | (d["abs_r12"] < umbral_spot) | (d["abs_r02"] < umbral_spot))
        
        x = d[d["Dummy_S"] == 0]


        # e_or = np.abs(d["gamma_emp"] / d["gamma_oracle"] - 1)
        # e_bs = np.abs(d["gamma_emp"] / d["gamma_bs"] - 1)
        rows.append({
            "config": cfg, "K": round(K, 1),
            "moneyness_0": round(K / p.S0, 2),
            "N": len(d),

            "Gamma_BS_med": d["gamma_bs"].median(),
            "pct_gamma_neg": 100* np.mean(d["gamma_emp"] < 0),
            "Pct_small_Spot": 100* np.mean(d["Dummy_S"]),

            "MARe_oracle_vs_BS":  np.mean( np.abs(d["gamma_oracle"] - d["gamma_bs"] )) / np.median(np.abs(d["gamma_bs"])), # RE structural_effect
            "MAE_oracle_vs_BS":  np.mean( np.abs(d["gamma_oracle"] - d["gamma_bs"] )), # structural_effect

            "MARe_emp_vs_BS":  np.mean( np.abs(d["gamma_emp"] - d["gamma_bs"] )) / np.median(np.abs(d["gamma_bs"])),
            "MAE_emp_vs_BS":  np.mean( np.abs(d["gamma_emp"] - d["gamma_bs"] )),

            "MARe_emp_vs_oracle":  np.mean( np.abs(d["gamma_emp"] - d["gamma_oracle"] )) / np.median(np.abs(d["gamma_oracle"])), # residual_effect
            "MAE_emp_vs_oracle":  np.mean( np.abs(d["gamma_emp"] - d["gamma_oracle"] )), # residual_effect

            "corr_structural_residual": stable_corr(d["gamma_oracle"] - d["gamma_bs"], d["gamma_emp"] - d["gamma_oracle"] ),
            "Cond_corr_str_res": stable_corr(x["gamma_oracle"] - x["gamma_bs"], x["gamma_emp"] - x["gamma_oracle"] ),



            # "MedARE_oracle_vs_bs_pct": 100 * np.median( d["gamma_oracle"] / d["gamma_bs"] - 1),
            # "med_oracle_vs_bs": np.median( d["gamma_oracle"] - d["gamma_bs"] ),
            # "MedARE_vs_ORACLE": np.median(e_or),
            # "MedARE_vs_BS": np.median(e_bs),

            "time_effect_med": (np.median(d["gamma_oracle"] - d["gamma_bs"] ) # time effect
                                            if cfg == "B"  else np.nan),
            "pct_time_effect_neg": (100 * np.mean(d["gamma_oracle"] - d["gamma_bs"] < 0) 
                                                if cfg == "B"  else np.nan),


            })
    return pd.DataFrame(rows)


# ============================================================
# 9. EJECUCIÓN
# ============================================================

def run(p): 
    
    start = time.time()
    tracemalloc.start()


    tablas = []
    # un strike cada vez: el panel completo (todos los strikes x todas las 
    # fechas x 4 configs) no cabe en memoria con T grande. 
    for m in p.moneyness:
        pm = replace(p, moneyness=[m])
        base = simulate_base_paths(pm)
        for config in ("A", "B", "C", "D"):
            panel = build_config_panel(base, config, pm)
            d, gg = estimador_op1(panel)
            del d
            del panel
            gg["config"] = config
            tablas.append(resumen(attach_oracle(gg, pm)))
            del gg
        del base

    end = time.time()
    print(f"Time elapsed:{end-start:.5f} seconds")
    actual, pico = tracemalloc.get_traced_memory()

    tracemalloc.stop()
    print(f"Memoria actual: {actual / (1024**2):.2f} MB")
    print(f"Pico máximo de RAM usado: {pico / (1024**2):.2f} MB")

    return pd.concat(tablas,axis=0, ignore_index=True).sort_values(
        ["moneyness_0", "config"]).reset_index(drop=True)


############################################################################################
############################################################################################
# def run_info(p):
#     reg = []
#     with Bench("TOTAL", registro=reg):
#         tablas = []
#         for m in p.moneyness:
#             pm = replace(p, moneyness=[m])
#             with Bench(f"simulate_base m={m}", registro=reg):
#                 base = simulate_base_paths(pm)
#             for config in ("A", "B", "C", "D"):
#                 with Bench(f"panel m={m} cfg={config}", registro=reg):
#                     panel = build_config_panel(base, config, pm)
#                 with Bench(f"op1   m={m} cfg={config}", registro=reg):
#                     d, gg = estimador_op1(panel)
#                 del d, panel
#                 gg["config"] = config
#                 tablas.append(resumen(attach_oracle(gg, pm)))
#                 del gg
#             del base

#     print(tabulate(pd.DataFrame(reg), headers="keys",
#                    tablefmt="rounded_outline", showindex=False))
#     return pd.concat(tablas, axis=0, ignore_index=True).sort_values(
#         ["moneyness_0", "config"]).reset_index(drop=True)


# # SI QUEREMOS EJECUTAR EL CÓDIGO EN DIFERENTES NUCLEOS:
# from joblib import Parallel, delayed

# def _una_combinacion(p, m, config):
#     pm = replace(p, moneyness=[m])
#     base = simulate_base_paths(pm)          # 0,26 s, recalcularlo sale gratis
#     panel = build_config_panel(base, config, pm)
#     d, gg = estimador_op1(panel)
#     del d, panel, base
#     gg["config"] = config
#     return resumen(attach_oracle(gg, pm))

# def run_kernel(p, n_jobs=6):
    t0 = time.perf_counter()
    combos = [(m, c) for m in p.moneyness for c in ("A", "B", "C", "D")]
    tablas = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_una_combinacion)(p, m, c) for m, c in combos)
    print(f"Tiempo: {time.perf_counter() - t0:.1f} s")
    return pd.concat(tablas, axis=0, ignore_index=True).sort_values(
        ["moneyness_0", "config"]).reset_index(drop=True)
############################################################################################
############################################################################################


if __name__ == "__main__":

    p = Params(n_paths=1_000, seed=123, S0=100.0, mu=0.0, spot_vol=0.20,
               sigma0=0.20, beta=-0.50, sigma_shock_sd=0.01, dt=1/252,
               T=1*252, tau0=(1*252 + 1)/252, r=0.02, q=0.00,
               moneyness=[0.5, 0.75, 1.0, 1.5, 1.75], option="call")

    tab = run(p)

    print(f"\nOP1 sobre el camino realizado — {p.n_paths} caminos, "
          f"T={p.T} pasos, beta={p.beta}\n")
    print(tabulate(tab, headers="keys", tablefmt="rounded_outline",
                   floatfmt=".5g", showindex=False) )


