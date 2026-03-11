"""
06_surface_valid_quant_checks.py
================================

Validación cuantitativa de la superficie 30d y de las griegas calculadas.

Este script sustituye tests poco informativos o no válidos para una
superficie construida por smiles independientes en moneyness.

Checks incluidos
----------------
1. Bounds estáticos de no-arbitraje
2. Monotonicidad en strike
3. Convexidad en strike
4. Densidad RN discreta >= 0
5. Integración aproximada de la densidad RN
6. Distribución y outliers de implied volatility
7. Calidad de griegas:
   - delta dentro de rango
   - vega no negativa
   - gamma no excesivamente negativa
8. Suavidad local de smile:
   - dsigma/dK
   - d2sigma/dK2
   con desglose entre zona observada y extrapolada

Inputs
------
- superficie_con_precios_limpio.parquet
- superficie_con_greeks.parquet

Outputs
-------
- resúmenes por pantalla
- CSVs con slices y puntos problemáticos
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import duckdb
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

PATH_PRICES = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\superficie_con_precios_limpio.parquet"
PATH_GREEKS = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\superficie_con_greeks.parquet"

OUT_DIR = Path(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Tolerancias de arbitraje
TOL_BOUNDS = 1e-6
TOL_MONO = 1e-4
TOL_CONV = 1e-3
TOL_RND_NEG = 1e-10

# Griegas
DELTA_TOL = 0.01
GAMMA_NEG_TOL = -0.01
VEGA_NEG_TOL = -1e-6

# Smile smoothness
MIN_POINTS_SMILE = 7
Z_THRESHOLD_D1 = 6.0
Z_THRESHOLD_D2 = 6.0
MIN_SPACING = 1e-10

# IV reasonableness
IV_UPPER_ALERT = 3.0   # 300%
IV_LOWER_ALERT = 0.01  # 1%


# ============================================================
# UTILIDADES
# ============================================================

def robust_zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def smile_derivatives_K(
    K_arr: np.ndarray,
    iv_arr: np.ndarray,
    min_spacing: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(K_arr)
    dsigma_dK = np.zeros(n)
    d2sigma_dK2 = np.zeros(n)

    if n < 2:
        return dsigma_dK, d2sigma_dK2

    for i in range(n):
        if i == 0:
            h = K_arr[1] - K_arr[0]
            if abs(h) >= min_spacing:
                dsigma_dK[i] = (iv_arr[1] - iv_arr[0]) / h

        elif i == n - 1:
            h = K_arr[-1] - K_arr[-2]
            if abs(h) >= min_spacing:
                dsigma_dK[i] = (iv_arr[-1] - iv_arr[-2]) / h

        else:
            h1 = K_arr[i] - K_arr[i - 1]
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


# ============================================================
# CHECK 1 — BOUNDS
# ============================================================

def check_bounds(df: pd.DataFrame, tol: float = 1e-6) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy()

    is_call = x["CallPut"] == "C"

    x["lower_bound"] = np.where(
        is_call,
        x["discount_factor"] * np.maximum(x["forward"] - x["Strike"], 0.0),
        x["discount_factor"] * np.maximum(x["Strike"] - x["forward"], 0.0),
    )
    x["upper_bound"] = np.where(
        is_call,
        x["discount_factor"] * x["forward"],
        x["discount_factor"] * x["Strike"],
    )

    x["flag_lower_ok"] = x["Precio_Modelo"] >= x["lower_bound"] - tol
    x["flag_upper_ok"] = x["Precio_Modelo"] <= x["upper_bound"] + tol
    x["flag_bounds_ok"] = x["flag_lower_ok"] & x["flag_upper_ok"]

    bad = x[~x["flag_bounds_ok"]].copy()
    return x, bad


# ============================================================
# CHECK 2-4 — MONO / CONV / RND
# ============================================================

def check_shape_and_rnd(
    df: pd.DataFrame,
    tol_mono: float = 1e-4,
    tol_conv: float = 1e-3,
    tol_rnd_neg: float = 1e-10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    points = []

    for (date, callput), grp in df.groupby(["Date", "CallPut"]):
        grp = grp.sort_values("Strike").reset_index(drop=True)

        K = grp["Strike"].to_numpy(dtype=float)
        P = grp["Precio_Modelo"].to_numpy(dtype=float)
        r = float(grp["rate"].iloc[0])
        T = float(grp["T"].iloc[0])

        zone = grp["flag_inside_observed_range"].to_numpy(dtype=bool) if "flag_inside_observed_range" in grp.columns else np.ones(len(grp), dtype=bool)

        if len(grp) < 3:
            rows.append({
                "Date": date,
                "CallPut": callput,
                "n_points": len(grp),
                "n_mono_viol": 0,
                "n_conv_viol": 0,
                "n_rnd_neg": 0,
                "rnd_mass": np.nan,
                "flag_mono_ok": True,
                "flag_conv_ok": True,
                "flag_rnd_ok": True,
            })
            continue

        # Monotonicidad
        dP = np.diff(P)
        mono_bad_idx = np.where(dP > tol_mono)[0] if callput == "C" else np.where(dP < -tol_mono)[0]

        # Convexidad + densidad RN discreta
        conv_bad = 0
        rnd_bad = 0
        rnd_mass = 0.0

        for i in range(1, len(K) - 1):
            h1 = K[i] - K[i - 1]
            h2 = K[i + 1] - K[i]

            if h1 <= 0 or h2 <= 0:
                continue

            slope_left = (P[i] - P[i - 1]) / h1
            slope_right = (P[i + 1] - P[i]) / h2

            curvature = slope_right - slope_left

            # Aproximación discreta de segunda derivada en malla no uniforme
            second_deriv = 2.0 * curvature / (h1 + h2)
            q_i = np.exp(r * T) * second_deriv

            # masa aproximada de densidad: q(K_i) * dK local
            dk_local = 0.5 * (h1 + h2)
            rnd_mass += max(q_i, 0.0) * dk_local

            if curvature < -tol_conv:
                conv_bad += 1

            if q_i < -tol_rnd_neg:
                rnd_bad += 1
                points.append({
                    "Date": date,
                    "CallPut": callput,
                    "Strike": K[i],
                    "Precio_Modelo": P[i],
                    "q_discrete": q_i,
                    "curvature": curvature,
                    "inside_observed": bool(zone[i]),
                })

        rows.append({
            "Date": date,
            "CallPut": callput,
            "n_points": len(grp),
            "n_mono_viol": int(len(mono_bad_idx)),
            "n_conv_viol": int(conv_bad),
            "n_rnd_neg": int(rnd_bad),
            "rnd_mass": float(rnd_mass),
            "flag_mono_ok": len(mono_bad_idx) == 0,
            "flag_conv_ok": conv_bad == 0,
            "flag_rnd_ok": rnd_bad == 0,
        })

    return pd.DataFrame(rows), pd.DataFrame(points)


# ============================================================
# CHECK 5 — GREEKS
# ============================================================

def check_greeks_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bad_points = []

    for (date, callput), grp in df.groupby(["Date", "CallPut"]):
        grp = grp.sort_values("Strike").reset_index(drop=True)

        delta = grp["delta"].to_numpy(dtype=float)
        vega = grp["vega"].to_numpy(dtype=float)
        gamma = grp["gamma"].to_numpy(dtype=float)

        if callput == "C":
            delta_bad_mask = (delta < -DELTA_TOL) | (delta > 1.0 + DELTA_TOL)
        else:
            delta_bad_mask = (delta < -1.0 - DELTA_TOL) | (delta > DELTA_TOL)

        vega_bad_mask = vega < VEGA_NEG_TOL
        gamma_bad_mask = gamma < GAMMA_NEG_TOL

        n_delta_bad = int(delta_bad_mask.sum())
        n_vega_bad = int(vega_bad_mask.sum())
        n_gamma_bad = int(gamma_bad_mask.sum())

        rows.append({
            "Date": date,
            "CallPut": callput,
            "n_points": len(grp),
            "n_delta_bad": n_delta_bad,
            "n_vega_bad": n_vega_bad,
            "n_gamma_bad": n_gamma_bad,
            "pct_gamma_neg": 100.0 * n_gamma_bad / len(grp) if len(grp) else np.nan,
            "flag_delta_ok": n_delta_bad == 0,
            "flag_vega_ok": n_vega_bad == 0,
            "flag_gamma_ok": n_gamma_bad == 0,
        })

        bad_mask = delta_bad_mask | vega_bad_mask | gamma_bad_mask
        if bad_mask.any():
            tmp = grp.loc[bad_mask, [
                "Date", "CallPut", "moneyness", "Strike",
                "implied_vol", "delta", "vega", "gamma"
            ]].copy()
            tmp["flag_delta_bad"] = delta_bad_mask[bad_mask]
            tmp["flag_vega_bad"] = vega_bad_mask[bad_mask]
            tmp["flag_gamma_bad"] = gamma_bad_mask[bad_mask]
            bad_points.append(tmp)

    if bad_points:
        bad_points_df = pd.concat(bad_points, ignore_index=True)
    else:
        bad_points_df = pd.DataFrame(columns=[
            "Date", "CallPut", "moneyness", "Strike",
            "implied_vol", "delta", "vega", "gamma",
            "flag_delta_bad", "flag_vega_bad", "flag_gamma_bad"
        ])

    return pd.DataFrame(rows), bad_points_df


# ============================================================
# CHECK 6 — IMPLIED VOL OUTLIERS
# ============================================================

def check_iv_distribution(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = df.copy()

    x["flag_iv_low"] = x["implied_vol"] < IV_LOWER_ALERT
    x["flag_iv_high"] = x["implied_vol"] > IV_UPPER_ALERT
    x["flag_iv_outlier"] = x["flag_iv_low"] | x["flag_iv_high"]

    summary = pd.DataFrame([{
        "n_rows": len(x),
        "iv_min": float(x["implied_vol"].min()),
        "iv_p01": float(x["implied_vol"].quantile(0.01)),
        "iv_p50": float(x["implied_vol"].quantile(0.50)),
        "iv_p99": float(x["implied_vol"].quantile(0.99)),
        "iv_max": float(x["implied_vol"].max()),
        "n_iv_low": int(x["flag_iv_low"].sum()),
        "n_iv_high": int(x["flag_iv_high"].sum()),
        "n_iv_outlier": int(x["flag_iv_outlier"].sum()),
        "pct_iv_outlier": 100.0 * float(x["flag_iv_outlier"].mean()),
    }])

    bad = x[x["flag_iv_outlier"]].copy()
    return summary, bad


# ============================================================
# CHECK 7 — SMILE SMOOTHNESS CON DESGLOSE
# ============================================================

def check_smile_smoothness(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    points = []

    for (date, callput), grp in df.groupby(["Date", "CallPut"]):
        grp = grp.sort_values("Strike").reset_index(drop=True)

        if len(grp) < MIN_POINTS_SMILE:
            continue

        K = grp["Strike"].to_numpy(dtype=float)
        iv = grp["implied_vol"].to_numpy(dtype=float)

        if "flag_inside_observed_range" in grp.columns:
            inside = grp["flag_inside_observed_range"].to_numpy(dtype=bool)
        else:
            inside = np.ones(len(grp), dtype=bool)

        dsigma_dK, d2sigma_dK2 = smile_derivatives_K(K, iv, min_spacing=MIN_SPACING)

        z1 = robust_zscore(dsigma_dK)
        z2 = robust_zscore(d2sigma_dK2)

        spike_mask = (np.abs(z1) > Z_THRESHOLD_D1) | (np.abs(z2) > Z_THRESHOLD_D2)

        n_spikes = int(spike_mask.sum())
        n_spikes_inside = int((spike_mask & inside).sum())
        n_spikes_outside = int((spike_mask & ~inside).sum())

        rows.append({
            "Date": date,
            "CallPut": callput,
            "n_points": len(grp),
            "n_spikes": n_spikes,
            "n_spikes_inside": n_spikes_inside,
            "n_spikes_outside": n_spikes_outside,
            "max_abs_z_dsigma_dK": float(np.nanmax(np.abs(z1))),
            "max_abs_z_d2sigma_dK2": float(np.nanmax(np.abs(z2))),
            "flag_smooth_ok": n_spikes == 0,
        })

        if n_spikes > 0:
            tmp = grp.loc[spike_mask, [
                "Date", "CallPut", "moneyness", "Strike",
                "implied_vol", "flag_inside_observed_range"
            ]].copy() if "flag_inside_observed_range" in grp.columns else grp.loc[spike_mask, [
                "Date", "CallPut", "moneyness", "Strike", "implied_vol"
            ]].copy()

            tmp["dsigma_dK"] = dsigma_dK[spike_mask]
            tmp["d2sigma_dK2"] = d2sigma_dK2[spike_mask]
            tmp["z_dsigma_dK"] = z1[spike_mask]
            tmp["z_d2sigma_dK2"] = z2[spike_mask]
            points.append(tmp)

    summary_df = pd.DataFrame(rows)

    if points:
        points_df = pd.concat(points, ignore_index=True)
    else:
        cols = [
            "Date", "CallPut", "moneyness", "Strike", "implied_vol",
            "dsigma_dK", "d2sigma_dK2", "z_dsigma_dK", "z_d2sigma_dK2"
        ]
        if "flag_inside_observed_range" in df.columns:
            cols.insert(5, "flag_inside_observed_range")
        points_df = pd.DataFrame(columns=cols)

    return summary_df, points_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    con = duckdb.connect()

    print("=" * 72)
    print("CARGANDO PARQUETS")
    print("=" * 72)

    prices = con.execute(f"""
        SELECT *
        FROM read_parquet('{PATH_PRICES}')
    """).df()

    greeks = con.execute(f"""
        SELECT *
        FROM read_parquet('{PATH_GREEKS}')
    """).df()

    prices["Date"] = pd.to_datetime(prices["Date"])
    greeks["Date"] = pd.to_datetime(greeks["Date"])

    print(f"Superficie con precios : {prices.shape}")
    print(f"Superficie con griegas : {greeks.shape}")

    # --------------------------------------------------------
    # CHECK 1 — BOUNDS
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 1 — BOUNDS ESTÁTICOS")
    print("=" * 72)

    prices_bounds, bad_bounds = check_bounds(prices, tol=TOL_BOUNDS)
    print(f"Violaciones bounds: {len(bad_bounds):,} filas ({100*len(bad_bounds)/len(prices_bounds):.4f}%)")

    # --------------------------------------------------------
    # CHECK 2-4 — MONO / CONV / RND
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 2-4 — MONOTONICIDAD / CONVEXIDAD / DENSIDAD RN")
    print("=" * 72)

    shape_summary, rnd_bad_points = check_shape_and_rnd(
        prices_bounds,
        tol_mono=TOL_MONO,
        tol_conv=TOL_CONV,
        tol_rnd_neg=TOL_RND_NEG,
    )

    if not shape_summary.empty:
        n_slices = len(shape_summary)
        mono_bad = int((~shape_summary["flag_mono_ok"]).sum())
        conv_bad = int((~shape_summary["flag_conv_ok"]).sum())
        rnd_bad = int((~shape_summary["flag_rnd_ok"]).sum())

        print(f"Slices analizados       : {n_slices:,}")
        print(f"Slices mono con fallos  : {mono_bad:,} ({100*mono_bad/n_slices:.2f}%)")
        print(f"Slices conv con fallos  : {conv_bad:,} ({100*conv_bad/n_slices:.2f}%)")
        print(f"Slices RND con q<0      : {rnd_bad:,} ({100*rnd_bad/n_slices:.2f}%)")

        mass = shape_summary["rnd_mass"].dropna()
        if not mass.empty:
            print("\nIntegración aproximada de q(K):")
            print(mass.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())

    # --------------------------------------------------------
    # CHECK 5 — GREEKS
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 5 — CALIDAD DE GRIEGAS")
    print("=" * 72)

    greeks_summary, greeks_bad_points = check_greeks_quality(greeks)

    if not greeks_summary.empty:
        n_slices = len(greeks_summary)
        delta_bad = int((~greeks_summary["flag_delta_ok"]).sum())
        vega_bad = int((~greeks_summary["flag_vega_ok"]).sum())
        gamma_bad = int((~greeks_summary["flag_gamma_ok"]).sum())

        print(f"Slices analizados       : {n_slices:,}")
        print(f"Slices delta con fallos : {delta_bad:,} ({100*delta_bad/n_slices:.2f}%)")
        print(f"Slices vega con fallos  : {vega_bad:,} ({100*vega_bad/n_slices:.2f}%)")
        print(f"Slices gamma con fallos : {gamma_bad:,} ({100*gamma_bad/n_slices:.2f}%)")

        gamma_neg_rate = (greeks["gamma"] < GAMMA_NEG_TOL).mean() * 100.0
        print(f"% global de puntos con gamma < {GAMMA_NEG_TOL}: {gamma_neg_rate:.4f}%")

    # --------------------------------------------------------
    # CHECK 6 — IV DISTRIBUTION
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 6 — DISTRIBUCIÓN DE IMPLIED VOL")
    print("=" * 72)

    iv_summary, iv_bad = check_iv_distribution(prices_bounds)
    print(iv_summary.to_string(index=False))

    # --------------------------------------------------------
    # CHECK 7 — SMILE SMOOTHNESS
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("CHECK 7 — SUAVIDAD LOCAL DE LA SMILE")
    print("=" * 72)

    smooth_summary, smooth_bad_points = check_smile_smoothness(prices_bounds)

    if not smooth_summary.empty:
        n_slices = len(smooth_summary)
        n_bad = int((~smooth_summary["flag_smooth_ok"]).sum())
        print(f"Slices analizados       : {n_slices:,}")
        print(f"Slices con spikes       : {n_bad:,} ({100*n_bad/n_slices:.2f}%)")

        total_inside = smooth_summary["n_spikes_inside"].sum()
        total_outside = smooth_summary["n_spikes_outside"].sum()
        total_spikes = smooth_summary["n_spikes"].sum()

        print(f"Spikes totales          : {total_spikes:,}")
        print(f"Spikes dentro observado : {total_inside:,}")
        print(f"Spikes fuera observado  : {total_outside:,}")

    # --------------------------------------------------------
    # EXPORTS
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("GUARDANDO RESULTADOS")
    print("=" * 72)

    shape_summary.to_csv(OUT_DIR / "valid_check_shape_summary.csv", index=False)
    rnd_bad_points.to_csv(OUT_DIR / "valid_check_rnd_bad_points.csv", index=False)
    greeks_summary.to_csv(OUT_DIR / "valid_check_greeks_summary.csv", index=False)
    greeks_bad_points.to_csv(OUT_DIR / "valid_check_greeks_bad_points.csv", index=False)
    iv_summary.to_csv(OUT_DIR / "valid_check_iv_summary.csv", index=False)
    iv_bad.to_csv(OUT_DIR / "valid_check_iv_bad_points.csv", index=False)
    smooth_summary.to_csv(OUT_DIR / "valid_check_smooth_summary.csv", index=False)
    smooth_bad_points.to_csv(OUT_DIR / "valid_check_smooth_bad_points.csv", index=False)
    bad_bounds.to_csv(OUT_DIR / "valid_check_bounds_bad_points.csv", index=False)

    print("CSVs guardados correctamente en:")
    print(OUT_DIR)

    # --------------------------------------------------------
    # RESUMEN FINAL
    # --------------------------------------------------------
    print("\n" + "=" * 72)
    print("RESUMEN FINAL")
    print("=" * 72)

    print(f"Bounds bad points         : {len(bad_bounds):,}")
    print(f"RND bad points            : {len(rnd_bad_points):,}")
    print(f"Greeks bad points         : {len(greeks_bad_points):,}")
    print(f"IV outlier points         : {len(iv_bad):,}")
    print(f"Smile smooth bad points   : {len(smooth_bad_points):,}")
    print("=" * 72)


if __name__ == "__main__":
    main()