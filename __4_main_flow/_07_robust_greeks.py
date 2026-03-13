"""
Cálculo robusto de Delta, Vega y Gamma siguiendo Bates (2005) / Black-76.

Mejoras respecto a _07_greeks.py:
  1. Se agrupa por Date únicamente (no por Date+CallPut) → las derivadas
     del smile usan la sonrisa completa (puts+calls juntos), evitando
     diferencias unilaterales en el borde ATM.
  2. Se usan las derivadas analíticas dsigma_dm / d2sigma_dm2 del ajuste
     Shimko si están disponibles en el parquet, convirtiendo a espacio K
     mediante:  dsigma_dK = dsigma_dm / F,  d2sigma_dK2 = d2sigma_dm2 / F²
  3. Se añade el factor de descuento e^{-rT} en los griegos Black-76
     (delta_bs, gamma_bs, vega), que faltaba en la versión anterior.
  4. Función check_pcp_greeks(): verifica paridad put-call en las griegas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
import duckdb


# =============================================================================
# Derivadas analíticas m → K  (si vienen del ajuste Shimko)
# =============================================================================

def _smile_derivatives_analytical(
    dsigma_dm: np.ndarray,
    d2sigma_dm2: np.ndarray,
    F: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte derivadas analíticas respecto a m = K/F  a derivadas
    respecto a K, usando la regla de la cadena:
        dm/dK = 1/F
        dsigma/dK   = dsigma/dm  * (1/F)
        d²sigma/dK² = d²sigma/dm² * (1/F²)
    """
    dsigma_dK   = dsigma_dm   / F
    d2sigma_dK2 = d2sigma_dm2 / (F * F)
    return dsigma_dK, d2sigma_dK2


# =============================================================================
# Derivadas de smile vía diferencias finitas centradas — respaldo
# =============================================================================

def _smile_derivatives_fd(
    K_arr: np.ndarray,
    iv_arr: np.ndarray,
    min_spacing: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    d_sigma/dK y d²_sigma/dK² vía diferencias finitas centradas
    con espaciado no uniforme en K.  Usado como alternativa cuando
    no están disponibles las derivadas analíticas de Shimko.
    """
    n = len(K_arr)
    dsigma_dK   = np.zeros(n)
    d2sigma_dK2 = np.zeros(n)

    if n < 2:
        return dsigma_dK, d2sigma_dK2

    for i in range(n):
        if i == 0:
            h = K_arr[1] - K_arr[0]
            if abs(h) < min_spacing:
                continue
            dsigma_dK[i] = (iv_arr[1] - iv_arr[0]) / h

        elif i == n - 1:
            h = K_arr[-1] - K_arr[-2]
            if abs(h) < min_spacing:
                continue
            dsigma_dK[i] = (iv_arr[-1] - iv_arr[-2]) / h

        else:
            h1 = K_arr[i]     - K_arr[i - 1]
            h2 = K_arr[i + 1] - K_arr[i]

            if abs(h1) < min_spacing or abs(h2) < min_spacing:
                continue

            dsigma_dK[i] = (
                iv_arr[i + 1] * h1**2
                + iv_arr[i]   * (h2**2 - h1**2)
                - iv_arr[i - 1] * h2**2
            ) / (h1 * h2 * (h1 + h2))

            d2sigma_dK2[i] = 2.0 * (
                iv_arr[i + 1] / (h2 * (h1 + h2))
                - iv_arr[i]   / (h1 * h2)
                + iv_arr[i - 1] / (h1 * (h1 + h2))
            )

    if n > 2:
        d2sigma_dK2[0]  = d2sigma_dK2[1]
        d2sigma_dK2[-1] = d2sigma_dK2[-2]

    return dsigma_dK, d2sigma_dK2


# =============================================================================
# BS/Black-76 Greeks
# =============================================================================

def _bs_greeks(
    F: float,
    K: np.ndarray,
    T: float,
    r: float,
    sigma: np.ndarray,
    cp: np.ndarray,
) -> dict:
    """
    Sensibilidades Black-76 con factor de descuento e^{-rT} correcto.

    Black-76 (opciones sobre forward/futuro):
        C = e^{-rT} [F·N(d1) - K·N(d2)]
        P = e^{-rT} [K·N(-d2) - F·N(-d1)]
        d1 = [ln(F/K) + 0.5·σ²·T] / (σ·√T)
        d2 = d1 - σ·√T

    Griegas respecto a F (delta de forward):
        delta_C = e^{-rT}·N(d1)
        delta_P = e^{-rT}·(N(d1) - 1) = -e^{-rT}·N(-d1)
        gamma   = e^{-rT}·φ(d1) / (F·σ·√T)
        vega    = e^{-rT}·F·√T·φ(d1)
        vanna_K = -e^{-rT}·φ(d1)·d2 / (σ·√T·K)
        volga   = vega·d1·d2/σ
    """
    iv_s   = np.maximum(sigma, 1e-8)
    sqrt_T = np.sqrt(T)
    df_r   = np.exp(-r * T)          # factor de descuento

    d1 = (np.log(F / K) + 0.5 * iv_s**2 * T) / (iv_s * sqrt_T)
    d2 = d1 - iv_s * sqrt_T

    npd1 = norm.pdf(d1)

    vega     = df_r * F * sqrt_T * npd1
    gamma_bs = df_r * npd1 / (F * iv_s * sqrt_T)
    delta_bs = np.where(
        cp == "C",
        df_r * norm.cdf(d1),
        df_r * (norm.cdf(d1) - 1.0),
    )
    vanna_K  = -df_r * npd1 * d2 / (iv_s * sqrt_T * K)
    volga    = vega * d1 * d2 / iv_s

    return {
        "d1":        d1,
        "d2":        d2,
        "npd1":      npd1,
        "delta_bs":  delta_bs,
        "vega":      vega,
        "gamma_bs":  gamma_bs,
        "vanna_K":   vanna_K,
        "volga":     volga,
        "df_r":      df_r,
    }


# =============================================================================
# Pipeline principal — Bates (2005) ec. 6 y 7  (versión robusta)
# =============================================================================

def compute_greeks(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Delta y Gamma smile-adjusted siguiendo Bates (2005) ecuaciones (6) y (7).

    Cambios respecto a _07_greeks.py:
      · Agrupa por Date únicamente → derivadas del smile usan la sonrisa
        completa (puts+calls), sin truncar el smile en el borde ATM.
      · Usa dsigma_dm / d2sigma_dm2 analíticos de Shimko si están en df_in.
      · Incluye factor de descuento e^{-rT} en delta_bs, gamma_bs y vega.
    """
    df = df_in.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    has_analytical = (
        "dsigma_dm" in df.columns
        and "d2sigma_dm2" in df.columns
        and df["dsigma_dm"].notna().any()
    )

    results = []

    for date, grp_date in df.groupby("Date"):
        grp_date = grp_date.sort_values("Strike").reset_index(drop=True)

        if len(grp_date) < 3:
            continue

        F = float(grp_date["forward"].iloc[0])
        r = float(grp_date["rate"].iloc[0])
        T = float(grp_date["T"].iloc[0])

        if T <= 0:
            continue

        K  = grp_date["Strike"].to_numpy(dtype=float)
        iv = grp_date["implied_vol"].to_numpy(dtype=float)
        cp = grp_date["CallPut"].to_numpy()

        # --- derivadas del smile ---
        if has_analytical:
            dsigma_dm_arr   = grp_date["dsigma_dm"].to_numpy(dtype=float)
            d2sigma_dm2_arr = grp_date["d2sigma_dm2"].to_numpy(dtype=float)
            dsigma_dK, d2sigma_dK2 = _smile_derivatives_analytical(
                dsigma_dm_arr, d2sigma_dm2_arr, F
            )
        else:
            dsigma_dK, d2sigma_dK2 = _smile_derivatives_fd(K, iv)

        # --- griegas BS/Black-76 ---
        bs = _bs_greeks(F, K, T, r, iv, cp)

        # --- Bates (2005) correcciones del smile ---
        # Delta corregida (ec. 6)
        delta = bs["delta_bs"] - bs["vega"] * (K / F) * dsigma_dK

        # Gamma corregida (ec. 7)
        correction = (K / F)**2 * (
            2.0 * bs["vanna_K"] * dsigma_dK
            +     bs["volga"]   * dsigma_dK**2
            +     bs["vega"]    * d2sigma_dK2
        )
        gamma = bs["gamma_bs"] + correction

        grp_out = grp_date.copy()
        grp_out["d1"]          = bs["d1"]
        grp_out["d2"]          = bs["d2"]
        grp_out["df_r"]        = bs["df_r"]
        grp_out["delta_bs"]    = bs["delta_bs"]
        grp_out["vega"]        = bs["vega"]
        grp_out["gamma_bs"]    = bs["gamma_bs"]
        grp_out["vanna_K"]     = bs["vanna_K"]
        grp_out["volga"]       = bs["volga"]
        grp_out["dsigma_dK"]   = dsigma_dK
        grp_out["d2sigma_dK2"] = d2sigma_dK2
        grp_out["delta"]       = delta
        grp_out["gamma"]       = gamma

        results.append(grp_out)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# =============================================================================
# Sanity checks — griegas
# =============================================================================

def check_greeks(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Verifica propiedades básicas de las griegas calculadas.
    """
    rows = []

    for (date, callput), grp in df.groupby(["Date", "CallPut"]):
        delta    = grp["delta"].to_numpy()
        vega     = grp["vega"].to_numpy()
        gamma_bs = grp["gamma_bs"].to_numpy()
        gamma    = grp["gamma"].to_numpy()
        df_r     = grp["df_r"].to_numpy()

        # Black-76 delta ∈ [-e^{-rT}, 0]  para puts, [0, e^{-rT}]  para calls
        if callput == "C":
            n_delta_bad = int(((delta < -0.01) | (delta > df_r.max() + 0.01)).sum())
        elif callput == "P":
            n_delta_bad = int(((delta < -df_r.max() - 0.01) | (delta > 0.01)).sum())
        else:
            n_delta_bad = 0

        n_vega_bad     = int((vega     < -1e-6).sum())
        n_gamma_bs_bad = int((gamma_bs < -1e-6).sum())
        n_gamma_bad    = int((gamma    < -0.01).sum())

        rows.append({
            "Date":          date,
            "CallPut":       callput,
            "n_points":      len(grp),
            "n_delta_bad":   n_delta_bad,
            "n_vega_bad":    n_vega_bad,
            "n_gamma_bs_bad":n_gamma_bs_bad,
            "n_gamma_bad":   n_gamma_bad,
            "flag_ok": (
                n_delta_bad + n_vega_bad + n_gamma_bs_bad + n_gamma_bad
            ) == 0,
        })

    check_df = pd.DataFrame(rows)

    if verbose and not check_df.empty:
        n    = len(check_df)
        n_ok = int(check_df["flag_ok"].sum())

        print("=" * 55)
        print("SANITY CHECKS — GRIEGAS (Bates 2005 robusto)")
        print("=" * 55)
        print(f"  Slices totales : {n:,}")
        print(f"  Slices OK      : {n_ok:,} ({n_ok / n * 100:.1f}%)")
        print()

        for col, label in [
            ("n_delta_bad",    "Delta fuera de bounds  "),
            ("n_vega_bad",     "Vega negativa          "),
            ("n_gamma_bs_bad", "Gamma BS negativa      "),
            ("n_gamma_bad",    "Gamma smile < -0.01    "),
        ]:
            n_bad = int((check_df[col] > 0).sum())
            print(f"  {label}: {n_bad:,} slices ({n_bad / n * 100:.1f}%)")

        print("=" * 55)

    return check_df


# =============================================================================
# Verificación de paridad put-call en griegas
# =============================================================================

def check_pcp_greeks(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Verifica paridad put-call en las griegas:
        gamma_C  = gamma_P          (idénticas)
        delta_C  - delta_P = e^{-rT}  (Black-76)
        vega_C   = vega_P            (idénticas)

    Requiere que el DataFrame tenga puts Y calls en el mismo strike y fecha.
    Devuelve estadísticos de la violación de paridad.
    """
    # Separar puts y calls
    calls = df[df["CallPut"] == "C"].set_index(["Date", "Strike"])
    puts  = df[df["CallPut"] == "P"].set_index(["Date", "Strike"])

    common = calls.index.intersection(puts.index)
    if len(common) == 0:
        if verbose:
            print("check_pcp_greeks: no hay strikes comunes C/P.")
        return pd.DataFrame()

    C = calls.loc[common]
    P = puts.loc[common]

    rows = []
    for (date, strike) in common:
        c_row = C.loc[(date, strike)]
        p_row = P.loc[(date, strike)]

        df_r = float(c_row["df_r"])

        rows.append({
            "Date":            date,
            "Strike":          strike,
            # delta_C - delta_P = e^{-rT}
            "pcp_delta_err":   float(c_row["delta"] - p_row["delta"]) - df_r,
            # gamma_C = gamma_P
            "pcp_gamma_err":   float(c_row["gamma"] - p_row["gamma"]),
            # vega_C = vega_P
            "pcp_vega_err":    float(c_row["vega"]  - p_row["vega"]),
            # gamma_bs_C = gamma_bs_P
            "pcp_gamma_bs_err":float(c_row["gamma_bs"] - p_row["gamma_bs"]),
        })

    pcp_df = pd.DataFrame(rows)

    if verbose:
        print("=" * 60)
        print("VERIFICACIÓN PARIDAD PUT-CALL — GRIEGAS")
        print("=" * 60)
        print(f"  Pares comunes (Date, Strike): {len(pcp_df):,}")
        print()
        for col, label in [
            ("pcp_delta_err",    "delta_C - delta_P - e^{-rT}"),
            ("pcp_gamma_err",    "gamma_C - gamma_P           "),
            ("pcp_gamma_bs_err", "gamma_bs_C - gamma_bs_P     "),
            ("pcp_vega_err",     "vega_C - vega_P             "),
        ]:
            arr = pcp_df[col].to_numpy()
            print(f"  {label}")
            print(f"    media={arr.mean():.2e}  std={arr.std():.2e}  "
                  f"max|err|={np.abs(arr).max():.2e}")
        print("=" * 60)

    return pcp_df


# =============================================================================
# EJECUCIÓN
# =============================================================================

con = duckdb.connect()

suf_clean = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\superficie_con_precios_limpio_shimko_2.parquet')
""").df()

df_greeks  = compute_greeks(suf_clean)
check_df   = check_greeks(df_greeks, verbose=True)
pcp_df     = check_pcp_greeks(df_greeks, verbose=True)

PARQUET_OUTPUT = (
    r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output"
    r"\superficie_con_greeks_shimko_robust.parquet"
)
duckdb.from_df(df_greeks).write_parquet(PARQUET_OUTPUT, compression="snappy")

print("=" * 55)
print("Generada la superficie con griegas robusta correctamente")
print(f"  → {PARQUET_OUTPUT}")
print("=" * 55)
