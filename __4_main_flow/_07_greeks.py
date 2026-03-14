
"""
Cálculo de Delta, Vega y Gamma siguiendo Bates (2005)
"Hedging the Smirk", ecuaciones (6) y (7).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm
import duckdb

# =============================================================================
# Derivadas de smile via diferencias finitas centradas — respecto a K
# =============================================================================

def _smile_derivatives_K(
    K_arr: np.ndarray,
    iv_arr: np.ndarray,
    min_spacing: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    d_sigma/dK y d²_sigma/dK² via diferencias finitas centradas
    con spacing no uniforme en K.
    """
    n = len(K_arr)
    dsigma_dK = np.zeros(n)
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
        d2sigma_dK2[0] = d2sigma_dK2[1]
        d2sigma_dK2[-1] = d2sigma_dK2[-2]

    return dsigma_dK, d2sigma_dK2


# =============================================================================
# BS Greeks primitivas
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
    Calcula las sensibilidades BS necesarias al estilo Bates (2005).
    """
    iv_s = np.maximum(sigma, 1e-8)
    sqrt_T = np.sqrt(T)

    d1 = (np.log(F / K) + 0.5 * iv_s**2 * T) / (iv_s * sqrt_T)
    d2 = d1 - iv_s * sqrt_T

    npd1 = norm.pdf(d1)

    # Vega
    vega = F * sqrt_T * npd1

    # Gamma BS
    gamma_bs = npd1 / (F * iv_s * sqrt_T)

    # Delta BS
    delta_bs = np.where(cp == "C", norm.cdf(d1), norm.cdf(d1) - 1.0)

    # Vanna respecto a K
    vanna_K = -npd1 * d2 / (iv_s * sqrt_T * K)

    # Volga
    volga = vega * d1 * d2 / iv_s

    return {
        "d1": d1,
        "d2": d2,
        "npd1": npd1,
        "delta_bs": delta_bs,
        "vega": vega,
        "gamma_bs": gamma_bs,
        "vanna_K": vanna_K,
        "volga": volga,
    }


# =============================================================================
# Pipeline principal — Bates (2005) ec. 6 y 7
# =============================================================================

def compute_greeks(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Delta y Gamma siguiendo Bates (2005) ecuaciones (6) y (7).
    """
    df = df_in.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    results = []

    # IMPORTANTE: agrupar por Date Y CallPut
    for (date, callput), grp_date in df.groupby(["Date", "CallPut"]):
        grp_date = grp_date.sort_values("Strike").reset_index(drop=True)

        if len(grp_date) < 3:
            continue

        F = float(grp_date["forward"].iloc[0])
        r = float(grp_date["rate"].iloc[0])
        T = float(grp_date["T"].iloc[0])
        S = F

        if T <= 0:
            continue

        K = grp_date["Strike"].to_numpy(dtype=float)
        iv = grp_date["implied_vol"].to_numpy(dtype=float)
        cp = grp_date["CallPut"].to_numpy()

        # BS greeks
        bs = _bs_greeks(F, K, T, r, iv, cp)

        # Derivadas de smile
        dsigma_dK, d2sigma_dK2 = _smile_derivatives_K(K, iv)

        # Delta corregida
        delta = bs["delta_bs"] - bs["vega"] * (K / S) * dsigma_dK

        # Gamma corregida
        correction = (K / S)**2 * (
            2.0 * bs["vanna_K"] * dsigma_dK
            +     bs["volga"]   * dsigma_dK**2
            +     bs["vega"]    * d2sigma_dK2
        )
        gamma = bs["gamma_bs"] + correction

        grp_out = grp_date.copy()
        grp_out["d1"] = bs["d1"]
        grp_out["d2"] = bs["d2"]
        grp_out["delta_bs"] = bs["delta_bs"]
        grp_out["vega"] = bs["vega"]
        grp_out["gamma_bs"] = bs["gamma_bs"]
        grp_out["vanna_K"] = bs["vanna_K"]
        grp_out["volga"] = bs["volga"]
        grp_out["dsigma_dK"] = dsigma_dK
        grp_out["d2sigma_dK2"] = d2sigma_dK2
        grp_out["delta"] = delta
        grp_out["gamma"] = gamma

        results.append(grp_out)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


# =============================================================================
# Sanity checks
# =============================================================================

def check_greeks(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Verifica propiedades básicas de las griegas calculadas.
    """
    rows = []

    for (date, callput), grp in df.groupby(["Date", "CallPut"]):
        delta = grp["delta"].to_numpy()
        vega = grp["vega"].to_numpy()
        gamma_bs = grp["gamma_bs"].to_numpy()
        gamma = grp["gamma"].to_numpy()

        if callput == "C":
            n_delta_bad = int(((delta < -0.01) | (delta > 1.01)).sum())
        else:
            n_delta_bad = int(((delta < -1.01) | (delta > 0.01)).sum())

        n_vega_bad = int((vega < -1e-6).sum())
        n_gamma_bs_bad = int((gamma_bs < -1e-6).sum())
        n_gamma_bad = int((gamma < -0.01).sum())

        rows.append({
            "Date": date,
            "CallPut": callput,
            "n_points": len(grp),
            "n_delta_bad": n_delta_bad,
            "n_vega_bad": n_vega_bad,
            "n_gamma_bs_bad": n_gamma_bs_bad,
            "n_gamma_bad": n_gamma_bad,
            "flag_ok": (
                n_delta_bad + n_vega_bad + n_gamma_bs_bad + n_gamma_bad
            ) == 0,
        })

    check_df = pd.DataFrame(rows)

    if verbose and not check_df.empty:
        n = len(check_df)
        n_ok = int(check_df["flag_ok"].sum())

        print("=" * 55)
        print("SANITY CHECKS — GRIEGAS (Bates 2005)")
        print("=" * 55)
        print(f"  Slices totales : {n:,}")
        print(f"  Slices OK      : {n_ok:,} ({n_ok/n*100:.1f}%)")
        print()

        for col, label in [
            ("n_delta_bad",    "Delta fuera de bounds"),
            ("n_vega_bad",     "Vega negativa"),
            ("n_gamma_bs_bad", "Gamma BS negativa"),
            ("n_gamma_bad",    "Gamma smile < -0.01"),
        ]:
            n_bad = int((check_df[col] > 0).sum())
            print(f"  {label:30s}: {n_bad:,} slices ({n_bad/n*100:.1f}%)")

        print("=" * 55)

    return check_df


# =============================================================================
# EJECUCIÓN
# =============================================================================

con = duckdb.connect()

# Usa el fichero limpio, no el bruto
suf_clean = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\superficie_con_precios_limpio_shimko_2_prueba.parquet')
""").df()

df_greeks = compute_greeks(suf_clean)
check_df = check_greeks(df_greeks, verbose=True)

PARQUET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\superficie_con_greeks_shimko_2_prueba.parquet"
duckdb.from_df(df_greeks).write_parquet(PARQUET_OUTPUT, compression='snappy')

print("======================================================")
print("Generada la superficie con griegas correctamente")
print("======================================================")




