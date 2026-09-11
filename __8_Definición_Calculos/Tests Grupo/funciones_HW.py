"""
Funciones de la replicación Hull-White (2017) Minimum Variance Delta.

Módulo sin código de ejecución: solo definiciones, para poder importarse
desde otros scripts (ej. 02_Replication_delta_HW.py, 04_Gamma_MV.py)
sin disparar carga de datos, estimación ni gráficos.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

#%% Función: dataset_preparation
"""
Los datos importados tan solo tiene el filtro de quela implied volatility,
    se ha puesto formato fecha a tau (Days) ,
    el strike divido /1000,
    enemos el precio del subyacente,
    Y tenemos los datos filtrados para el SP500.

    VER "01_limpieza_preliminar"
"""

def dataset_preparation(df, next_trad_date):

    # Hacemos filtro de pairs dates. Necesitamos al menos dos fechas consecutivas de trading.
    df = df.sort_values(["OptionID", "Date"]).copy()

    g = df.groupby("OptionID", sort=False)

    df["price_next"] = g["MidPrice"].shift(-1)
    df["spot_next"] = g["SpotPrice"].shift(-1)
    df["next_date"] = g["Date"].shift(-1)

    df["next_market_date"] = df["Date"].map(next_trad_date)

    df = df[df["next_date"] == df["next_market_date"]].copy()

    df = df[df["Days"] >= 14].copy()

    df["T"] = df["Days"] / 365

    # BID / ASK /VI and greeks
    df = df[(df["Bid"] > 0) & (df["Ask"] > 0) & (df["ImpliedVolatility"] != -99.99)].copy()

    # Quitamos deltas extremas:
    df = df[(np.abs(df["Delta"].round(2)) >= 0.05) & (np.abs(df["Delta"].round(2)) <= 0.95)].copy()

    #Normalizamos el spot para el primer día del par sea de 1:

    df["ds"] = (df["spot_next"] - df["SpotPrice"]) / df["SpotPrice"]
    df["df_option"] = (df["price_next"] - df["MidPrice"]) / df["SpotPrice"]
    df["vega_norm"] = df["Vega"] / df["SpotPrice"]

    # Metemos la ecuación de regresión:

        # Y = Δf - delta_BS ΔS
          # Y = vega*ds/T *(a + bDelta +cDelta^2)
          # Y = a* vegads/T  + b*vega*ds/T*Delta +c*vega*ds/TDelta^2)

    df["Y"] = df["df_option"] - df["Delta"] * df["ds"]
    df["Z"] = df["vega_norm"] * df["ds"] / np.sqrt(df["T"])
    df["X_a"] = df["Z"]
    df["X_b"] = df["Z"] * df["Delta"]
    df["X_c"] = df["Z"] * df["Delta"]**2

    return df.reset_index(drop=True)


# ============================================================
#  ESTIMACIÓN DE a, b, c
# ============================================================

def fit_hw_coefficients(df_train):
    """
    Estima Eq. (6) SIN intercepto:

        Y = a*X_a + b*X_b + c*X_c + epsilon

    equivalente a:

        Δf - delta_BS ΔS = [vega / sqrt(T)] [ΔS/S](a + b delta_BS + c delta_BS²)  + epsilon

    """

    cols = ["Y", "X_a", "X_b", "X_c"]
    d = (df_train[cols].replace([np.inf, -np.inf], np.nan).dropna())

# LSTSQ (más abajo) no permite meter DataFrames, es preferible trabajar con las matrices de Numpy
  # y usamos np.linalg.lstsq se basa en la norma 2 y devuelve información clave del álgebra su diagnóstico como rango... y es más rápdido.

    y = d["Y"].to_numpy()

    X = d[["X_a", "X_b", "X_c"]].to_numpy()

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    return {"a": beta[0], "b": beta[1], "c": beta[2], "N": len(d)}


# ============================================================
# una vez estimados los parametros. calculamos los errores:
# ============================================================

def apply_greeks(df_test, params, ej_gamma: bool=True):

    out = df_test.copy()

    a = params["a"]
    b = params["b"]
    c = params["c"]

    delta = out["Delta"]

    # --------------------------------------------------------
    # Función Cuadrática de HW
    # --------------------------------------------------------

    out["hw_poly"] = (a + b * delta + c * delta**2)

    # ========================================================
    # Derivada empírica de IV respecto al spot:
    # ∂E[sigma_imp] / ∂S
    # ========================================================

    out["sigma_S_hat"] = (out["hw_poly"] / (out["SpotPrice"] * np.sqrt(out["T"])))

    # ========================================================
    # Minimum Variance Delta:
    # delta_MV= delta_BS + Vega * sigma_S_hat
    # ========================================================

    out["delta_mv"] = (out["Delta"] + out["Vega"] * out["sigma_S_hat"])

    # --------------------------------------------------------
    # Hedging errors usando variables normalizadas
    # --------------------------------------------------------

    out["eps_bs_delta"] = (out["df_option"] - out["Delta"] * out["ds"])

    out["eps_mv_delta"] = (out["df_option"] - out["delta_mv"] * out["ds"])

    # --------------------------------------------------------
    # GAMMA empírica:
    # --------------------------------------------------------
    if ej_gamma:
        out["sigma_SS_bracket"] =  ( b + 2*c * delta)*out["Gamma"] -  out["hw_poly"]/out["SpotPrice"]
    #     out["sigma_SS_hat"] = (out["sigma_SS_bracket"] / (out["SpotPrice"] * np.sqrt(out["T"])))

    #     out["gamma_emp"] = (
    #     out["Gamma"]+ 2 * out["Vanna"] * out["sigma_S_hat"] + out["Volga"] * out["sigma_S_hat"]**2 + out["Vega"] * out["sigma_SS_hat"]   
    # )

        out["gamma_emp"] = out["Gamma"] + 2*out["Vanna"]*out["sigma_S_hat"] + out["Volga"]*(out["sigma_S_hat"])**2 + (out["Vega"] /(out["SpotPrice"] * np.sqrt(out["T"])))* out["sigma_SS_bracket"]

        out["eps_delta_bs_gamma_bs"] = (out["df_option"] - out["Delta"] * out["ds"] - 0.5*(out["Gamma"] * out["SpotPrice"]* out["ds"]**2))

        out["eps_delta_mv_gamma_emp"] = (out["df_option"] - out["delta_mv"] * out["ds"] - 0.5*(out["gamma_emp"] * out["SpotPrice"]* out["ds"]**2))

        out["eps_delta_bs_gamma_emp"] = (out["df_option"] - out["Delta"] * out["ds"] - 0.5*(out["gamma_emp"] * out["SpotPrice"]* out["ds"]**2))

        out["eps_delta_mv_gamma_bs"] = (out["df_option"] - out["delta_mv"] * out["ds"] - 0.5*(out["Gamma"] * out["SpotPrice"]* out["ds"]**2))

    else:
        return out
    return out


# ============================================================
# ROLLING WINDOW. Dates from HW(2017)
# ============================================================

def rolling_window(
    pairs,
    ej_gamma: bool = True,
    f1="2004-01-01",  # primera fecha de estimación, tras el tiempo de espera en la primera estimación-
    f2="2015-08-31",
    window_months_est=36,
    month_pred=1,
    cp_col="CallPut"
):
    """
    HW:
        36 meses de estimación -> parámetros aplicados al siguiente mes.

    Calls y puts se estiman separadamente.
    """

    df = pairs.copy()

    #filtro harcodeado de fechas usadas en HW:

    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= pd.Timestamp(f1)) & (df["Date"] <= pd.Timestamp(f2))]  #pandas intenta interpretar el string como fecha, pero con Timestamp mejor

    first_month = pd.Timestamp(df["Date"].min()).to_period("M")
    first_test_month = (first_month + window_months_est)
    last_month = pd.Timestamp(df["Date"].max()).to_period("M")

    months = pd.period_range(start=first_test_month, end=last_month, freq="M")

    results = []
    coefficients = []

    for m in months:
        train_start = (m - window_months_est).start_time
        train_end = (m).start_time

        testing_start = (m).start_time
        testing_end = (m + month_pred).start_time

        train = df[(df["Date"] >= train_start) & (df["Date"] < train_end)]
        test = df[(df["Date"] >= testing_start) & (df["Date"] < testing_end)]

        # HW estima puts y calls separadamente
        for cp in ["C", "P"]:

            train_cp = train[train[cp_col] == cp]
            test_cp = test[test[cp_col] == cp]

            if len(train_cp) == 0 or len(test_cp) == 0:
                continue

            params = fit_hw_coefficients(train_cp)

            coefficients.append({
                "test_month": m,
                "cp_flag": cp,
                "train_start": train_start,
                "train_end": train_end,
                **params})

            pred = apply_greeks(test_cp, params,ej_gamma)

            pred["test_month"] = m

            results.append(pred)

    oos = pd.concat(results, ignore_index=True)

    coef = pd.DataFrame(coefficients)

    return oos, coef


# ============================================================
# VALIDACIÓN HULL-WHITE: GAIN
# ============================================================

def gain_hw(df):
    """
    Hull & White (2017), Eq. (3):

        Gain = 1 - SSE(eps_MV) / SSE(eps_BS)
    """

    sse_mv = np.sum(df["eps_mv_delta"] ** 2)
    sse_bs = np.sum(df["eps_bs_delta"] ** 2)
  
    return 1 - sse_mv / sse_bs


# =======================================================================================================
### =============================================================
#   ADAPTACIÓN DE FUNCIONES A LA GAMMA
### =============================================================
# =======================================================================================================

def bs_price_greeks(S, K, tau, r, sigma, option="C",q = 0):
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

    is_call = np.asarray(option) == "C"

    price = np.where(is_call,
            S * discount_q * norm.cdf(d1) - K * discount_r * norm.cdf(d2),
            K * discount_r * norm.cdf(-d2) - S * discount_q * norm.cdf(-d1)
    )

    delta = np.where(is_call,
            discount_q * norm.cdf(d1),
            discount_q * (norm.cdf(d1) - 1)
                     )
    c_tau = np.where(is_call,
            (S * discount_q * norm.pdf(d1) * sigma / (2 * sqrt_tau)
                 + r * K * discount_r * norm.cdf(d2)
                 - q * S * discount_q * norm.cdf(d1)),
            (S * discount_q * norm.pdf(d1) * sigma / (2 * sqrt_tau)
                 - r * K * discount_r * norm.cdf(-d2)
                 + q * S * discount_q * norm.cdf(-d1))
                     )


    # gamma = discount_q * norm.pdf(d1) / (S * sigma * sqrt_tau)
    vega = S * discount_q * norm.pdf(d1) * sqrt_tau
    volga = vega * (d1 * d2) / sigma
    volga_norm = norm.pdf(d1) * (d1 * d2) / sigma
    # CORREGIDO: vanna = ∂²C/∂S∂σ = -e^{-qτ}·φ(d1)·d2/σ   (el signo iba al revés)
    vanna = -discount_q * norm.pdf(d1) * (d2 / sigma)

    return volga, vanna, volga_norm


def rolling_window_estimated(
    pairs,
    coef,
    f1="2004-01-01",
    f2="2015-08-31",
    ej_gamma:bool =True,
    cp_col="CallPut" ):

    df = pairs.copy()
    coef = coef.copy()

    # --------------------------------------------------------
    # Formatos de fecha
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(df["Date"])

    coef["train_start"] = pd.to_datetime(coef["train_start"])
    coef["train_end"] = pd.to_datetime(coef["train_end"])

    # --------------------------------------------------------
    # Filtrado de muestra
    # --------------------------------------------------------

    df = df[(df["Date"] >= pd.Timestamp(f1)) & (df["Date"] <= pd.Timestamp(f2))].copy()
            
    results = []

    # --------------------------------------------------------
    # Aplicamos cada conjunto de coeficientes al mes siguiente
    # --------------------------------------------------------

    for (_, train_end, cp), g in coef.groupby(["train_start", "train_end", "cp_flag"],sort=True):
        
        # Los parámetros estimados hasta train_end
        # se utilizan en el mes inmediatamente posterior

        testing_start = train_end
        testing_end = (pd.Timestamp(train_end)  + pd.DateOffset(months=1) )
            
        test = df[(df["Date"] >= testing_start) & (df["Date"] < testing_end) & (df[cp_col] == cp) ].copy()
            
        if test.empty:
            continue

        # Una fila = un conjunto a,b,c
        params = g.iloc[0]

        pred = apply_greeks( test, params,ej_gamma )

        pred["test_month"] = (testing_start.to_period("M"))

        results.append(pred)

    oos = pd.concat(results,ignore_index=True)

    return oos


def gain_gamma(df):
    """
    Hull & White (2017), Eq. (3):

        Gain = 1 - SSE(eps_MV) / SSE(eps_BS)
    """
    sse_mv_emp = np.sum(df["eps_delta_mv_gamma_emp"] ** 2)
    sse_bs_bs = np.sum(df["eps_delta_bs_gamma_bs"] ** 2)
    sse_bs_emp = np.sum(df["eps_delta_bs_gamma_emp"] ** 2)
    sse_mv_bs = np.sum(df["eps_delta_mv_gamma_bs"] ** 2)

    gain_gamma1 = (1 - sse_mv_emp / sse_bs_bs)*100
    gain_gamma2 = (1 - sse_mv_emp / sse_mv_bs)*100
    gain_gamma3 = (1 - sse_mv_emp / sse_bs_emp)*100
    gain_gamma4 = (1 - sse_bs_emp / sse_bs_bs)*100



    return gain_gamma1, gain_gamma2, gain_gamma3, gain_gamma4



def sse_normalizado(df):
    """
    Hull & White (2017), Eq. (3):

        Gain = 1 - SSE(eps_MV) / SSE(eps_BS)
    """
    sse_mv_emp = np.sum(df["eps_delta_mv_gamma_emp"] ** 2)
    sse_bs_bs = np.sum(df["eps_delta_bs_gamma_bs"] ** 2)
    sse_bs_emp = np.sum(df["eps_delta_bs_gamma_emp"] ** 2)
    sse_mv_bs = np.sum(df["eps_delta_mv_gamma_bs"] ** 2)

    SSE_bs_bs   = (1- sse_bs_bs / sse_bs_bs)*100
    SSE_bs_emp   = (1- sse_bs_emp / sse_bs_bs)*100
    SSE_mv_bs   = (1- sse_mv_bs / sse_bs_bs)*100
    SSE_mv_emp   = (1- sse_mv_emp / sse_bs_bs)*100



    return SSE_bs_bs, SSE_bs_emp, SSE_mv_bs, SSE_mv_emp
