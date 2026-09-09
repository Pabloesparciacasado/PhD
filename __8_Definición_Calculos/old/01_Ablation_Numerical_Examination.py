# In[]: Monte Carlo — validación del estimador numérico de griegas
"""
01_Ablation_Numerical_Examination.py 


Banco de pruebas del estimador op1 (delta/gamma por diferencias finitas)
contra un mundo donde la griega VERDADERA se conoce en forma cerrada.

Filosofía del diseño: GBM + precios BSM exactos NO se elige por realismo, sino
por ser el MEJOR CASO POSIBLE — vol constante, sin saltos, sin horquilla, sin
tick, sin asincronía, sin smile. Todo error que aparezca aquí es un suelo:
en los datos reales solo puede ser peor. Si el estimador rompe en este mundo,
rompe en cualquiera.


| Config. | \(S\)  | \(\tau\) | \(\sigma\)                  | Oracle                                                   |
| ------- | ------ | -------- | --------------------------- | -------------------------------------------------------- |
| A       | cambia | fija     | fija                        | \(\Gamma^{BS}\)                                          |
| B       | cambia | cambia   | fija                        | \(\Gamma^{BS}+\Delta_\tau\frac{\Delta\tau}{\Delta S}\)   |
| C       | cambia | fija     | \(\sigma=f(S)\)             | \(\Gamma^{BS}+Vanna\frac{\Delta\sigma^{sys}}{\Delta S}\) |
| D       | cambia | fija     | \(f(S)+\varepsilon^\sigma\) | anterior + contaminación ortogonal                       |


"""


# ============================================================
# 0. IMPORTS
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
    n_paths: int = 100_000
    seed: int = 123

    # Spot
    S0: float = 100.0
    mu: float = 0.0
    spot_vol: float = 0.20
    h_s: float = 1.0

    # Volatilidad implícita inicial
    sigma0: float = 0.20

    # Relación sistemática spot-vol:
    #
    # sigma(S) = sigma0 * (S/S0)^beta
    #
    # beta < 0 implica leverage effect
    beta: float = -0.50

    # Shock ortogonal de volatilidad
    sigma_shock_sd: float = 0.01

    # Paso diario
    dt: float = 1 / 252

    # 5 años
    T: int = 5*252

    # Maturity inicial: debe cubrir todo el horizonte T para que la config B
    # (tau decreciente con t) nunca llegue a tau <= 0. El +1 deja tau = dt
    # (estrictamente positivo) en el último día simulado, en vez de 0.
    tau0: float = (5*252 + 1) / 252

    # Tipos
    r: float = 0.02
    q: float = 0.00

    moneyness: list = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 1.75])

    # Tipo de opción
    option: str = "call"


# ============================================================
# 2. BLACK-SCHOLES:
#    PRECIO + GREEKS NECESARIAS
# ============================================================

def bs_price_greeks(S, K, tau, r, q, sigma, option="call"):
    """
    Black-Scholes price and Greeks.

    Returns
    -------
    price
    delta
    gamma
    vega
    c_tau

    donde:

        c_tau = ∂C / ∂tau

    Atención:
        tau = tiempo restante hasta vencimiento.

    Por tanto:

        Theta_calendar = -c_tau
    """

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if np.any(tau <= 0):
        raise ValueError( "tau must be strictly positive." )

    if np.any(sigma <= 0):
        raise ValueError( "sigma must be strictly positive." )

    sqrt_tau = np.sqrt(tau)

    d1 = ( np.log(S / K) + ( r - q + 0.5 * sigma**2) * tau) / (sigma * sqrt_tau )

    d2 = ( d1 - sigma * sqrt_tau )

    discount_q = np.exp(-q * tau)
    discount_r = np.exp(-r * tau) 

    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    if option == "call":

        price = ( S * discount_q * norm.cdf(d1) - K * discount_r * norm.cdf(d2) )

        delta = ( discount_q * norm.cdf(d1) ) 

        # ∂C / ∂tau
        c_tau = ( S  * discount_q * norm.pdf(d1) * sigma / (2 * sqrt_tau)
           
            + r * K * discount_r * norm.cdf(d2) - q * S * discount_q * norm.cdf(d1) )

    # --------------------------------------------------------
    # PUT
    # --------------------------------------------------------

    elif option == "put":

        price = ( K * discount_r * norm.cdf(-d2) - S * discount_q * norm.cdf(-d1) )

        delta = ( discount_q * ( norm.cdf(d1)- 1) )

        # ∂P / ∂tau
        c_tau = (
            S * discount_q * norm.pdf(d1) * sigma / (2 * sqrt_tau)
            - r * K * discount_r * norm.cdf(-d2) + q * S * discount_q * norm.cdf(-d1) 
            )

    else:

        raise ValueError(
            "option must be 'call' or 'put'."
        )

    # --------------------------------------------------------
    # GAMMA
    # --------------------------------------------------------

    gamma = (
        discount_q * norm.pdf(d1) / ( S * sigma * sqrt_tau ) 
        )

    # --------------------------------------------------------
    # VEGA
    # --------------------------------------------------------

    vega = (
        S * discount_q * norm.pdf(d1) * sqrt_tau
        )

    
    # --------------------------------------------------------
    # Volga
    # --------------------------------------------------------

    volga = (
        vega * (d1*d2)/sigma
        )

    # --------------------------------------------------------
    # Vanna
    # --------------------------------------------------------

    vanna = (
        discount_q*norm.pdf(d1)*(d2/sigma)
        )
    

    return ( price, delta, gamma, vega, volga, vanna, c_tau  )


# ============================================================
# 3. RELACIÓN SISTEMÁTICA SPOT-VOL
# ============================================================

def sigma_systematic(S, S0, sigma0, beta):
    
    """
    Systematic IV response:

        sigma(S) = sigma0 * (S/S0)^beta

    beta < 0:
        spot down -> IV up
        spot up   -> IV down
    """

    S = np.asarray( S, dtype=float) 

    return ( sigma0 * (S / S0) ** beta )


# ============================================================
# 4. SIMULACIÓN DE LOS CAMINOS DE SPOT
# ============================================================

def simulate_base_paths(p):
    """
    Genera T fechas por OptionID

    Esto es exactamente lo necesario para calcular:

        delta_emp(1)
        gamma_emp(1)

    Todos los escenarios A-D utilizan LOS MISMOS caminos de
    spot y LOS MISMOS shocks aleatorios.

    Esto permite hacer comparaciones contrafactuales path-by-path.
    
    """

    rng = np.random.default_rng( p.seed )


    z = rng.standard_normal( size = (p.n_paths, p.T)  )

    S = np.empty( (p.n_paths, p.T +1))
       

    S[:, 0] = p.S0

    for j in range(1, p.T+1):

        S[:, j] = (S[:, j - 1] *np.exp((p.mu  - 0.5* p.spot_vol**2)* p.dt 
                                    + p.spot_vol* np.sqrt(p.dt)* z[:, j - 1]))

    # --------------------------------------------------------
    # Shocks ortogonales de IV
    # --------------------------------------------------------

    eps_sigma = rng.normal( loc=0.0, scale=p.sigma_shock_sd,size=(p.n_paths, p.T +1))


    # --------------------------------------------------------
    # Panel largo
    # --------------------------------------------------------
 
    n_money = len(p.moneyness)
    n_base = p.n_paths * (p.T + 1)
    K_levels = np.array([p.S0 / m for m in p.moneyness])   # 1 K por moneyness

    base = pd.DataFrame({
        "K": np.repeat(K_levels, n_base),
        "Sim": np.tile(np.repeat(np.arange(p.n_paths), p.T + 1), n_money)+1,
        "t": np.tile(np.tile(np.arange(p.T + 1), p.n_paths), n_money),
        "S": np.tile(S.ravel(), n_money),
        "S_plus": np.tile(S.ravel() + p.h_s, n_money),
        "S_min": np.tile(S.ravel() - p.h_s, n_money),
        "eps_sigma": np.tile(eps_sigma.ravel(), n_money),
    })

    return base


# ============================================================
# 5. CONSTRUCCIÓN DE CADA CONFIGURACIÓN
# ============================================================

def build_config_panel( base,config,p):
    """
    A:
        S cambia; tau fija; sigma fija

    B:
        S cambia; tau cambia; sigma fija

    C:
        S cambia; tau fija; sigma = f(S)

    D:
        S cambia; tau fija; sigma = f(S) + epsilon_sigma

    """
    df = base.copy()

    # --------------------------------------------------------
    # Componente sistemática de IV
    # --------------------------------------------------------

    sigma_sys = sigma_systematic(S=df["S"], S0=p.S0, sigma0=p.sigma0, beta=p.beta)
        

    # ========================================================
    # A — SPOT ONLY
    # ========================================================

    if config == "A":

        df["tau"] = p.tau0
        df["sigma"] = p.sigma0

    # ========================================================
    # B — SPOT + TIME
    # ========================================================

    elif config == "B":

        df["tau"] = (p.tau0  - df["t"] * p.dt )
        df["sigma"] = p.sigma0 

    # ========================================================
    # C — SPOT + SYSTEMATIC IV
    # ========================================================

    elif config == "C":

        df["tau"] = p.tau0
        df["sigma"] = sigma_sys

    # ========================================================
    # D — SYSTEMATIC IV + ORTHOGONAL IV SHOCK
    # ========================================================

    elif config == "D":

        df["tau"] = p.tau0
        df["sigma"] = sigma_sys + df["eps_sigma"]

    else:
        raise ValueError("config must be A, B, C or D.")

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if (df["sigma"] <= 0).any():
        raise ValueError("Non-positive volatility generated.")

    # --------------------------------------------------------
    # Exact BSM prices and Greeks
    # --------------------------------------------------------
        #Centrado en t (el simulado)
    price, delta, gamma, vega, volga, vanna, c_tau = bs_price_greeks(

        S=df["S"], K=df["K"], tau=df["tau"], r=p.r, q=p.q, sigma=df["sigma"], option=p.option
    )


    df["C"] = price
    df["delta_bs"] = delta
    df["gamma_bs_row"] = gamma
    df["vega"] = vega
    df["volga"] = volga
    df["vanna"] = vanna
    df["c_tau"] = c_tau

        #Centrado en S_t + h 
    price_plus, *_ = bs_price_greeks(S=df["S_plus"], K=df["K"], tau=df["tau"],  r=p.r, q=p.q, sigma=df["sigma"], option=p.option)

    df["C_plus"] = price_plus

        #Centrado en S_t - h
    price_min, *_ = bs_price_greeks(S=df["S_min"], K=df["K"], tau=df["tau"],  r=p.r, q=p.q, sigma=df["sigma"], option=p.option)

    df["C_min"] = price_min

    return df


# ============================================================
# 6. ESTIMADOR EMPÍRICO OP1
# ============================================================

def estimador_op1(df):
    """
    Réplica exacta del estimador utilizado en 02_.

    El dataframe debe estar ordenado por:
        K and t
    """

    # ========================================================
    # DELTA EMPÍRICA
    # ========================================================

    
    data = df.groupby("Sim")

   

    # ========================================================
    # GAMMA EMPÍRICA
    # ========================================================

    

    # --------------------------------------------------------
    # El factor 2 es correcto:
    #
    # delta_emp(t) está aproximadamente centrada en:
    #
    #       (S_t + S_{t-1}) / 2
    #
    # delta_emp(t-1) está aproximadamente centrada en:
    #
    #       (S_{t-1} + S_{t-2}) / 2
    #
    # La distancia entre ambos centros es:
    #
    #       (S_t - S_{t-2}) / 2
    #
    # --------------------------------------------------------

    return d, gg


# ============================================================
# 7. ORACLE ANALÍTICO PARA GAMMA
# ============================================================

def attach_gamma_oracle(
    gg,
    panel,
    p
):
    """
    Añade la Gamma Black-Scholes parcial correspondiente al
    punto central natural del estimador OP1.

    Tenemos:

    delta_emp(1):
        centro ~ (state0 + state1)/2

    delta_emp(2):
        centro ~ (state1 + state2)/2

    Por tanto, gamma_emp está aproximadamente centrada en:

        [state0 + 2*state1 + state2] / 4

    para S, tau y sigma.
    """

    # --------------------------------------------------------
    # Pasamos las tres observaciones a formato ancho
    # --------------------------------------------------------

    states = panel.pivot(

        index="OptionID",

        columns="t",

        values=[
            "S",
            "tau",
            "sigma"
        ]
    )

    # --------------------------------------------------------
    # Flatten MultiIndex
    #
    # ('S',0) -> S0
    # ('S',1) -> S1
    # ...
    # --------------------------------------------------------

    states.columns = [

        f"{variable}{int(t)}"

        for variable, t
        in states.columns
    ]

    states = (
        states
        .reset_index()
    )

    out = gg.merge(

        states,

        on="OptionID",

        how="left"
    )

    # --------------------------------------------------------
    # Centro efectivo del estimador Gamma
    # --------------------------------------------------------

    out["S_gamma"] = (

        out["S0"]
        + 2 * out["S1"]
        + out["S2"]

    ) / 4

    out["tau_gamma"] = (

        out["tau0"]
        + 2 * out["tau1"]
        + out["tau2"]

    ) / 4

    out["sigma_gamma"] = (

        out["sigma0"]
        + 2 * out["sigma1"]
        + out["sigma2"]

    ) / 4

    # --------------------------------------------------------
    # Gamma BSM parcial
    # --------------------------------------------------------

    K = (
        p.S0
        / p.moneyness
    )

    (
        _,
        _,
        gamma_bs,
        _,
        _
    ) = bs_price_greeks(

        S=out["S_gamma"],
        K=K,
        tau=out["tau_gamma"],
        r=p.r,
        q=p.q,
        sigma=out["sigma_gamma"],
        option=p.option
    )

    out["gamma_bs"] = (
        gamma_bs
    )

    # ========================================================
    # DIAGNÓSTICO DE DENOMINADORES
    # ========================================================

    # S0 -> S1
    out["abs_r01"] = np.abs(

        (
            out["S1"]
            - out["S0"]
        )

        / out["S0"]
    )

    # S1 -> S2
    out["abs_r12"] = np.abs(

        (
            out["S2"]
            - out["S1"]
        )

        / out["S1"]
    )

    # S0 -> S2
    out["abs_r02"] = np.abs(

        (
            out["S2"]
            - out["S0"]
        )

        / out["S0"]
    )

    return out


# ============================================================
# 8. MÉTRICAS DE EVALUACIÓN
# ============================================================

def evaluation_metrics(
    df,
    target="gamma_bs"
):
    """
    Métricas de gamma_emp frente al benchmark indicado.
    """

    y_hat = (
        df["gamma_emp"]
        .to_numpy()
    )

    y = (
        df[target]
        .to_numpy()
    )

    error = (
        y_hat
        - y
    )

    rmse = np.sqrt(
        np.mean(
            error**2
        )
    )

    scale = np.mean(
        np.abs(y)
    )

    return {

        "N":
            len(df),

        "Bias":
            np.mean(
                error
            ),

        "MAE":
            np.mean(
                np.abs(error)
            ),

        "RMSE":
            rmse,

        # RMSE relativo al tamaño medio
        # de la Gamma analítica
        "NRMSE":
            (
                rmse / scale
                if scale > 0
                else np.nan
            ),

        # Mediana del error absoluto:
        # mucho más robusta a explosiones
        "MedAE":
            np.median(
                np.abs(error)
            ),

        # Percentil 95 del error absoluto
        "Q95_AE":
            np.quantile(
                np.abs(error),
                0.95
            ),

        "Correlation":
            np.corrcoef(
                y_hat,
                y
            )[0, 1]
    }


# ============================================================
# 9. TABLA PRINCIPAL DEL ABLATION
# ============================================================

def ablation_summary(
    results
):
    """
    Compara SIEMPRE gamma_emp contra gamma_bs.

    Ésta es la tabla importante para responder:

        ¿está OP1 recuperando la Gamma BSM parcial?
    """

    rows = []

    for (
        config,
        df
    ) in results.items():

        metrics = (
            evaluation_metrics(
                df,
                target="gamma_bs"
            )
        )

        rows.append({

            "Config":
                config,

            **metrics
        })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 10. FILTRO EXTERNO DE DENOMINADORES
# ============================================================

def denominator_filter(
    df,
    min_1d=0.0,
    min_2d=0.0
):
    """
    IMPORTANTE:

    Este filtro NO forma parte de estimador_op1.

    Se utiliza únicamente para estudiar sensibilidad a
    denominadores pequeños.

    min_1d:
        mínimo movimiento absoluto S0->S1
        y S1->S2.

    min_2d:
        mínimo movimiento absoluto S0->S2.

    Por ejemplo:

        min_1d = 0.001

    significa exigir movimientos de al menos 0.10%.
    """

    filtered = df[

        (df["abs_r01"] >= min_1d)

        &

        (df["abs_r12"] >= min_1d)

        &

        (df["abs_r02"] >= min_2d)

    ].copy()

    return filtered


# ============================================================
# 11. OPERADOR GAMMA OP1
# ============================================================

def gamma_operator(
    panel,
    signal
):
    """
    Aplica el mismo operador lineal que gamma_emp,
    pero a una señal genérica x_t.

    Para x0, x1, x2:

        slope_01 =
            (x1-x0)/(S1-S0)

        slope_12 =
            (x2-x1)/(S2-S1)

        G[x] =
            2*(slope_12-slope_01)/(S2-S0)


    Esto es útil porque, para un mismo camino de S:

        G[C_B] - G[C_A]
        =
        G[C_B - C_A]

    exactamente.

    Así podemos aislar canales de forma contrafactual.
    """

    tmp = panel[
        [
            "OptionID",
            "t",
            "S"
        ]
    ].copy()

    tmp["x"] = np.asarray(
        signal,
        dtype=float
    )

    wide = tmp.pivot(

        index="OptionID",

        columns="t",

        values=[
            "S",
            "x"
        ]
    )

    S0 = wide[
        ("S", 0)
    ]

    S1 = wide[
        ("S", 1)
    ]

    S2 = wide[
        ("S", 2)
    ]

    x0 = wide[
        ("x", 0)
    ]

    x1 = wide[
        ("x", 1)
    ]

    x2 = wide[
        ("x", 2)
    ]

    dS01 = (
        S1
        - S0
    )

    dS12 = (
        S2
        - S1
    )

    dS02 = (
        S2
        - S0
    )

    valid = (

        (dS01 != 0)

        &

        (dS12 != 0)

        &

        (dS02 != 0)
    )

    value = (

        2
        *
        (
            (x2 - x1) / dS12

            -

            (x1 - x0) / dS01
        )

        / dS02
    )

    value = (
        value[valid]
        .rename("value")
    )

    return value


# ============================================================
# 12. DESCOMPOSICIÓN CONTRAFACTUAL
# ============================================================

def channel_decomposition(
    results,
    panels
):
    """
    Aísla exactamente los tres canales:

        Time:
            B - A

        Systematic vol:
            C - A

        Orthogonal vol:
            D - C


    Como todos utilizan exactamente los mismos caminos de spot,
    estas diferencias son comparaciones path-by-path.
    """

    # --------------------------------------------------------
    # Alinear gammas empíricas por OptionID
    # --------------------------------------------------------

    aligned = pd.DataFrame(

        index=
            results["A"][
                "OptionID"
            ]
    )

    aligned.index.name = (
        "OptionID"
    )

    for config in (
        "A",
        "B",
        "C",
        "D"
    ):

        aligned[config] = (

            results[config]

            .set_index(
                "OptionID"
            )[
                "gamma_emp"
            ]
        )

    # ========================================================
    # EFECTOS EXACTOS OBSERVADOS
    # ========================================================

    aligned[
        "Time_actual"
    ] = (

        aligned["B"]
        - aligned["A"]
    )

    aligned[
        "SysVol_actual"
    ] = (

        aligned["C"]
        - aligned["A"]
    )

    aligned[
        "OrthVol_actual"
    ] = (

        aligned["D"]
        - aligned["C"]
    )

    # --------------------------------------------------------
    # Paneles
    # --------------------------------------------------------

    pA = panels["A"]
    pB = panels["B"]
    pC = panels["C"]
    pD = panels["D"]

    # ========================================================
    # 12.1. IDENTIDAD EXACTA
    # ========================================================
    #
    # Como G es lineal en precios:
    #
    #     Gamma_B - Gamma_A
    #       =
    #     G[C_B - C_A]
    #
    # etc.
    #
    # Estos RMSE deberían estar prácticamente en precisión
    # de máquina.
    # ========================================================

    exact = {

        "Time":

            gamma_operator(

                panel=pA,

                signal=(
                    pB["C"].to_numpy()
                    - pA["C"].to_numpy()
                )
            ),

        "SysVol":

            gamma_operator(

                panel=pA,

                signal=(
                    pC["C"].to_numpy()
                    - pA["C"].to_numpy()
                )
            ),

        "OrthVol":

            gamma_operator(

                panel=pC,

                signal=(
                    pD["C"].to_numpy()
                    - pC["C"].to_numpy()
                )
            )
    }

    # ========================================================
    # 12.2. PREDICCIÓN DE PRIMER ORDEN USANDO GREEKS
    # ========================================================

    # --------------------------------------------------------
    # TIME
    #
    # C_B - C_A
    #
    # aproximadamente:
    #
    # C_tau * (tau_B - tau_A)
    # --------------------------------------------------------

    time_price_prediction = (

        pA["c_tau"].to_numpy()

        *

        (
            pB["tau"].to_numpy()
            - pA["tau"].to_numpy()
        )
    )

    # --------------------------------------------------------
    # SYSTEMATIC VOL
    #
    # C_C - C_A
    #
    # aproximadamente:
    #
    # Vega_A * (sigma_C - sigma_A)
    # --------------------------------------------------------

    sysvol_price_prediction = (

        pA["vega"].to_numpy()

        *

        (
            pC["sigma"].to_numpy()
            - pA["sigma"].to_numpy()
        )
    )

    # --------------------------------------------------------
    # ORTHOGONAL VOL
    #
    # C_D - C_C
    #
    # aproximadamente:
    #
    # Vega_C * epsilon_sigma
    # --------------------------------------------------------

    orthvol_price_prediction = (

        pC["vega"].to_numpy()

        *

        (
            pD["sigma"].to_numpy()
            - pC["sigma"].to_numpy()
        )
    )

    predicted = {

        "Time":

            gamma_operator(
                pA,
                time_price_prediction
            ),

        "SysVol":

            gamma_operator(
                pA,
                sysvol_price_prediction
            ),

        "OrthVol":

            gamma_operator(
                pC,
                orthvol_price_prediction
            )
    }

    # ========================================================
    # MÉTRICAS
    # ========================================================

    mapping = {

        "Time":
            "Time_actual",

        "SysVol":
            "SysVol_actual",

        "OrthVol":
            "OrthVol_actual"
    }

    rows = []

    for (
        channel,
        actual_column
    ) in mapping.items():

        common_index = (

            aligned.index

            .intersection(
                predicted[
                    channel
                ].index
            )
        )

        actual = (

            aligned
            .loc[
                common_index,
                actual_column
            ]
            .to_numpy()
        )

        exact_channel = (

            exact[
                channel
            ]
            .loc[
                common_index
            ]
            .to_numpy()
        )

        prediction = (

            predicted[
                channel
            ]
            .loc[
                common_index
            ]
            .to_numpy()
        )

        # ----------------------------------------------------
        # Exact operator check
        # ----------------------------------------------------

        exact_error = (

            actual
            - exact_channel
        )

        exact_rmse = np.sqrt(

            np.mean(
                exact_error**2
            )
        )

        # ----------------------------------------------------
        # Greek approximation check
        # ----------------------------------------------------

        prediction_error = (

            actual
            - prediction
        )

        prediction_rmse = np.sqrt(

            np.mean(
                prediction_error**2
            )
        )

        effect_rms = np.sqrt(

            np.mean(
                actual**2
            )
        )

        rows.append({

            "Channel":
                channel,

            "Effect_mean":
                np.mean(actual),

            "Effect_RMS":
                effect_rms,

            "Effect_MedAbs":
                np.median(
                    np.abs(actual)
                ),

            # Debe ser ~ 0
            "Exact_operator_RMSE":
                exact_rmse,

            # ¿Explica el Greek la contaminación?
            "Greek_prediction_corr":
                np.corrcoef(
                    actual,
                    prediction
                )[0, 1],

            "Greek_prediction_RMSE":
                prediction_rmse,

            "Greek_prediction_rel_RMSE":
                (
                    prediction_rmse
                    / effect_rms

                    if effect_rms > 0

                    else np.nan
                )
        })

    channel_summary = pd.DataFrame(
        rows
    )

    return (
        aligned,
        channel_summary
    )


# ============================================================
# 13. SENSIBILIDAD A DENOMINADORES PEQUEÑOS
# ============================================================

def denominator_sensitivity(
    results,
    thresholds=(
        0.0,
        0.0005,
        0.001,
        0.002,
        0.005,
        0.010
    )
):
    """
    Estudia cómo cambia el comportamiento del estimador
    al exigir movimientos mínimos de spot.

    0.0005 = 5 bps
    0.0010 = 10 bps
    0.0020 = 20 bps
    0.0050 = 50 bps
    0.0100 = 100 bps

    IMPORTANTE:
    esto es diagnóstico de robustez.
    No forma parte de OP1.
    """

    rows = []

    for threshold in thresholds:

        for (
            config,
            df
        ) in results.items():

            filtered = denominator_filter(

                df,

                min_1d=
                    threshold,

                min_2d=
                    threshold
            )

            metrics = evaluation_metrics(
                filtered,
                target="gamma_bs"
            )

            rows.append({

                "Threshold":
                    threshold,

                "Config":
                    config,

                "N":
                    len(filtered),

                "Kept_pct":
                    (
                        100
                        * len(filtered)
                        / len(df)
                    ),

                "RMSE":
                    metrics[
                        "RMSE"
                    ],

                "NRMSE":
                    metrics[
                        "NRMSE"
                    ],

                "MedAE":
                    metrics[
                        "MedAE"
                    ],

                "Q95_AE":
                    metrics[
                        "Q95_AE"
                    ],

                "Correlation":
                    metrics[
                        "Correlation"
                    ]
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# 14. EJECUCIÓN COMPLETA DEL TEST
# ============================================================

def run_ablation(p):
    """
    Ejecuta Test 1 completo.
    """

    # ========================================================
    # 1. CAMINOS BASE
    # ========================================================

    base = simulate_base_paths(
        p
    )

    panels = {}
    results = {}

    # ========================================================
    # 2. CONFIGURACIONES A-D
    # ========================================================

    for config in (
        "A",
        "B",
        "C",
        "D"
    ):

        panel = build_config_panel(

            base=base,

            config=config,

            p=p
        )

        # ----------------------------------------------------
        # Aplicamos EXACTAMENTE OP1
        # ----------------------------------------------------

        (
            delta_results,
            gamma_results
        ) = estimador_op1(
            panel
        )

        # ----------------------------------------------------
        # Añadimos Gamma BSM oracle
        # ----------------------------------------------------

        gamma_results = (
            attach_gamma_oracle(

                gg=gamma_results,

                panel=panel,

                p=p
            )
        )

        panels[
            config
        ] = panel

        results[
            config
        ] = gamma_results

    # ========================================================
    # 3. TABLA PRINCIPAL
    # ========================================================

    summary = (
        ablation_summary(
            results
        )
    )

    # ========================================================
    # 4. DESCOMPOSICIÓN DE CANALES
    # ========================================================

    (
        aligned,
        channels
    ) = channel_decomposition(

        results=results,

        panels=panels
    )

    # ========================================================
    # 5. SENSIBILIDAD A DENOMINADORES
    # ========================================================

    sensitivity = (
        denominator_sensitivity(
            results
        )
    )

    return (

        base,
        panels,
        results,
        summary,
        aligned,
        channels,
        sensitivity
    )


# ============================================================
# 15. PRINT DE RESULTADOS
# ============================================================

def print_results(
    summary,
    channels,
    sensitivity
):

    # ========================================================
    # TABLE 1
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "TABLE 1 — EMPIRICAL GAMMA VS PARTIAL BLACK-SCHOLES GAMMA"
    )

    print(
        "============================================================"
    )

    print(

        tabulate(

            summary,

            headers="keys",

            tablefmt=
                "rounded_outline",

            floatfmt=
                ".6f",

            showindex=False
        )
    )

    # ========================================================
    # TABLE 2
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "TABLE 2 — COUNTERFACTUAL CHANNEL DECOMPOSITION"
    )

    print(
        "============================================================"
    )

    print(

        tabulate(

            channels,

            headers="keys",

            tablefmt=
                "rounded_outline",

            floatfmt=
                ".6f",

            showindex=False
        )
    )

    # ========================================================
    # TABLE 3
    # ========================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "TABLE 3 — DENOMINATOR SENSITIVITY"
    )

    print(
        "============================================================"
    )

    print(

        tabulate(

            sensitivity,

            headers="keys",

            tablefmt=
                "rounded_outline",

            floatfmt=
                ".6f",

            showindex=False
        )
    )


# ============================================================
# 16. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Parámetros
    # --------------------------------------------------------

    p = Params(

        n_paths=100_000,

        seed=123,

        S0=100.0,

        h_s = 1,

        mu=0.0,

        spot_vol=0.20,

        sigma0=0.20,

        beta=-0.50,

        sigma_shock_sd=0.005,

        tau0=90 / 252,

        r=0.02,

        q=0.01,

        moneyness=1.0,

        option="call"
    )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    (
        base,
        panels,
        results,
        summary,
        aligned,
        channels,
        sensitivity
    ) = run_ablation(
        p
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(

        summary=summary,

        channels=channels,

        sensitivity=sensitivity
    )
# %%
