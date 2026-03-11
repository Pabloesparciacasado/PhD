"""
07_validation_plots.py
======================

Gráficos de validación de superficie 30d para presentación.

Produce:
1. Dashboard resumen con barras
2. Histograma de rnd_mass
3. Scatter de bad points de densidad RN
4. Scatter de bad points de smoothness
5. Subplots de peores slices por smoothness
6. Subplots de slices con puntos RND negativos
7. Histograma de gamma
8. Histograma de implied vol

Outputs:
- PNGs guardados en carpeta output/plots_validation
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output")
PLOTS_DIR = BASE_DIR / "plots_validation"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

PATH_PRICES = BASE_DIR / "superficie_con_precios_limpio.parquet"
PATH_GREEKS = BASE_DIR / "superficie_con_greeks.parquet"

PATH_SHAPE_SUMMARY = BASE_DIR / "valid_check_shape_summary.csv"
PATH_RND_BAD = BASE_DIR / "valid_check_rnd_bad_points.csv"
PATH_SMOOTH_SUMMARY = BASE_DIR / "valid_check_smooth_summary.csv"
PATH_SMOOTH_BAD = BASE_DIR / "valid_check_smooth_bad_points.csv"
PATH_GREEKS_SUMMARY = BASE_DIR / "valid_check_greeks_summary.csv"
PATH_IV_SUMMARY = BASE_DIR / "valid_check_iv_summary.csv"

TOP_N_SMOOTH = 12
TOP_N_RND = 8


# ============================================================
# CARGA
# ============================================================

def load_data():
    con = duckdb.connect()

    prices = con.execute(f"""
        SELECT *
        FROM read_parquet('{str(PATH_PRICES)}')
    """).df()

    greeks = con.execute(f"""
        SELECT *
        FROM read_parquet('{str(PATH_GREEKS)}')
    """).df()

    shape_summary = pd.read_csv(PATH_SHAPE_SUMMARY)
    rnd_bad = pd.read_csv(PATH_RND_BAD)
    smooth_summary = pd.read_csv(PATH_SMOOTH_SUMMARY)
    smooth_bad = pd.read_csv(PATH_SMOOTH_BAD)
    greeks_summary = pd.read_csv(PATH_GREEKS_SUMMARY)
    iv_summary = pd.read_csv(PATH_IV_SUMMARY)

    for df in [prices, greeks, shape_summary, rnd_bad, smooth_summary, smooth_bad, greeks_summary]:
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

    return prices, greeks, shape_summary, rnd_bad, smooth_summary, smooth_bad, greeks_summary, iv_summary


# ============================================================
# UTILIDADES
# ============================================================

def savefig(name: str):
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name, dpi=180, bbox_inches="tight")
    plt.close()


def _nrows_for(n_items: int, ncols: int) -> int:
    return math.ceil(n_items / ncols)


# ============================================================
# 1. DASHBOARD RESUMEN
# ============================================================

def plot_summary_dashboard(shape_summary, greeks_summary, iv_summary, smooth_summary):
    n_slices_shape = len(shape_summary)
    n_slices_greeks = len(greeks_summary)
    n_slices_smooth = len(smooth_summary)

    mono_bad = int((~shape_summary["flag_mono_ok"]).sum())
    conv_bad = int((~shape_summary["flag_conv_ok"]).sum())
    rnd_bad = int((~shape_summary["flag_rnd_ok"]).sum())

    delta_bad = int((~greeks_summary["flag_delta_ok"]).sum())
    vega_bad = int((~greeks_summary["flag_vega_ok"]).sum())
    gamma_bad = int((~greeks_summary["flag_gamma_ok"]).sum())

    smooth_bad = int((~smooth_summary["flag_smooth_ok"]).sum())

    iv_outlier = int(iv_summary["n_iv_outlier"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Panel 1: arbitraje
    ax = axes[0, 0]
    labels = ["Mono", "Convexidad", "RND < 0"]
    values = [mono_bad, conv_bad, rnd_bad]
    ax.bar(labels, values)
    ax.set_title("Violaciones de forma")
    ax.set_ylabel("Número de slices")

    # Panel 2: griegas
    ax = axes[0, 1]
    labels = ["Delta", "Vega", "Gamma"]
    values = [delta_bad, vega_bad, gamma_bad]
    ax.bar(labels, values)
    ax.set_title("Slices con fallos en griegas")
    ax.set_ylabel("Número de slices")

    # Panel 3: smoothness
    ax = axes[1, 0]
    labels = ["Smoothness bad", "IV outliers"]
    values = [smooth_bad, iv_outlier]
    ax.bar(labels, values)
    ax.set_title("Calidad local de la superficie")
    ax.set_ylabel("Conteo")

    # Panel 4: texto resumen
    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"Slices shape      : {n_slices_shape:,}\n"
        f"Slices greeks     : {n_slices_greeks:,}\n"
        f"Slices smoothness : {n_slices_smooth:,}\n\n"
        f"Monotonicidad bad : {mono_bad:,}\n"
        f"Convexidad bad    : {conv_bad:,}\n"
        f"RND bad           : {rnd_bad:,}\n\n"
        f"Delta bad         : {delta_bad:,}\n"
        f"Vega bad          : {vega_bad:,}\n"
        f"Gamma bad         : {gamma_bad:,}\n\n"
        f"Smoothness bad    : {smooth_bad:,}\n"
        f"IV outliers       : {iv_outlier:,}"
    )
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=12)
    ax.set_title("Resumen cuantitativo")

    fig.suptitle("Dashboard de validación de la superficie 30d", fontsize=16, fontweight="bold")
    savefig("01_dashboard_validacion.png")


# ============================================================
# 2. HISTOGRAMA DE MASA RND
# ============================================================

def plot_rnd_mass_hist(shape_summary):
    x = shape_summary["rnd_mass"].dropna()

    plt.figure(figsize=(10, 5))
    plt.hist(x, bins=50)
    plt.title("Distribución de la masa aproximada de densidad RN")
    plt.xlabel("rnd_mass")
    plt.ylabel("Frecuencia")
    savefig("02_hist_rnd_mass.png")


# ============================================================
# 3. BAD POINTS RND
# ============================================================

def plot_rnd_bad_points(rnd_bad):
    if rnd_bad.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(rnd_bad["Strike"], rnd_bad["q_discrete"], s=18, alpha=0.7)
    plt.axhline(0.0, linewidth=1)
    plt.title("Puntos con densidad RN discreta negativa")
    plt.xlabel("Strike")
    plt.ylabel("q_discrete")
    savefig("03_rnd_bad_points_scatter.png")

    if "inside_observed" in rnd_bad.columns:
        plt.figure(figsize=(10, 6))
        for flag, label in [(True, "Observed"), (False, "Extrapolated")]:
            tmp = rnd_bad[rnd_bad["inside_observed"] == flag]
            if len(tmp) > 0:
                plt.scatter(tmp["Strike"], tmp["q_discrete"], s=18, alpha=0.7, label=label)
        plt.axhline(0.0, linewidth=1)
        plt.title("RND bad points: observado vs extrapolado")
        plt.xlabel("Strike")
        plt.ylabel("q_discrete")
        plt.legend()
        savefig("04_rnd_bad_points_observed_vs_extrapolated.png")


# ============================================================
# 4. BAD POINTS SMOOTHNESS
# ============================================================

def plot_smooth_bad_points(smooth_bad):
    if smooth_bad.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(smooth_bad["Strike"], smooth_bad["z_d2sigma_dK2"], s=10, alpha=0.5)
    plt.axhline(0.0, linewidth=1)
    plt.title("Bad points de smoothness: z-score de d²σ/dK²")
    plt.xlabel("Strike")
    plt.ylabel("z_d2sigma_dK2")
    savefig("05_smooth_bad_points_curvature.png")

    plt.figure(figsize=(10, 6))
    plt.scatter(smooth_bad["moneyness"], smooth_bad["z_dsigma_dK"], s=10, alpha=0.5)
    plt.axhline(0.0, linewidth=1)
    plt.title("Bad points de smoothness: z-score de dσ/dK")
    plt.xlabel("Moneyness")
    plt.ylabel("z_dsigma_dK")
    savefig("06_smooth_bad_points_slope.png")

    if "flag_inside_observed_range" in smooth_bad.columns:
        plt.figure(figsize=(10, 6))
        for flag, label in [(True, "Observed"), (False, "Extrapolated")]:
            tmp = smooth_bad[smooth_bad["flag_inside_observed_range"] == flag]
            if len(tmp) > 0:
                plt.scatter(tmp["Strike"], tmp["z_d2sigma_dK2"], s=10, alpha=0.5, label=label)
        plt.axhline(0.0, linewidth=1)
        plt.title("Smoothness bad points: observado vs extrapolado")
        plt.xlabel("Strike")
        plt.ylabel("z_d2sigma_dK2")
        plt.legend()
        savefig("07_smooth_bad_points_observed_vs_extrapolated.png")


# ============================================================
# 5. GALERÍA DE PEORES SLICES POR SMOOTHNESS
# ============================================================

def plot_top_smooth_slices(prices, smooth_summary, smooth_bad, top_n=12):
    if smooth_summary.empty:
        return

    worst = (
        smooth_summary
        .sort_values(["n_spikes", "max_abs_z_d2sigma_dK2"], ascending=[False, False])
        .head(top_n)
        .copy()
    )

    ncols = 3
    nrows = _nrows_for(len(worst), ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, worst.iterrows()):
        date = row["Date"]
        cp = row["CallPut"]

        sl = prices[
            (prices["Date"] == date) &
            (prices["CallPut"] == cp)
        ].sort_values("Strike").copy()

        bad = smooth_bad[
            (smooth_bad["Date"] == date) &
            (smooth_bad["CallPut"] == cp)
        ].copy()

        ax.plot(sl["Strike"], sl["implied_vol"], linewidth=1.5)
        ax.scatter(sl["Strike"], sl["implied_vol"], s=12)

        if len(bad) > 0:
            ax.scatter(bad["Strike"], bad["implied_vol"], s=32, marker="x", zorder=5)

        ax.set_title(
            f"{pd.Timestamp(date).date()} | {cp}\n"
            f"spikes={int(row['n_spikes'])}, max|z2|={row['max_abs_z_d2sigma_dK2']:.1f}",
            fontsize=9
        )
        ax.set_xlabel("Strike")
        ax.set_ylabel("IV")

    for ax in axes[len(worst):]:
        ax.set_visible(False)

    fig.suptitle("Peores slices por smoothness", fontsize=16, fontweight="bold")
    savefig("08_top_smooth_slices.png")


# ============================================================
# 6. GALERÍA DE SLICES CON RND NEGATIVA
# ============================================================

def plot_rnd_problem_slices(prices, rnd_bad, top_n=8):
    if rnd_bad.empty:
        return

    worst = (
        rnd_bad.groupby(["Date", "CallPut"])
        .size()
        .reset_index(name="n_bad_points")
        .sort_values("n_bad_points", ascending=False)
        .head(top_n)
    )

    ncols = 2
    nrows = _nrows_for(len(worst), ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, (_, row) in zip(axes, worst.iterrows()):
        date = row["Date"]
        cp = row["CallPut"]

        sl = prices[
            (prices["Date"] == date) &
            (prices["CallPut"] == cp)
        ].sort_values("Strike").copy()

        bad = rnd_bad[
            (rnd_bad["Date"] == date) &
            (rnd_bad["CallPut"] == cp)
        ].copy()

        ax.plot(sl["Strike"], sl["Precio_Modelo"], linewidth=1.5)
        ax.scatter(sl["Strike"], sl["Precio_Modelo"], s=12)

        if len(bad) > 0:
            # mapear el precio del punto malo usando Strike como índice
            price_map = sl.set_index("Strike")["Precio_Modelo"]
            bad_plot = bad.copy()
            bad_plot["Precio_Modelo_plot"] = bad_plot["Strike"].map(price_map)

            bad_plot = bad_plot.dropna(subset=["Precio_Modelo_plot"])

            ax.scatter(
                bad_plot["Strike"],
                bad_plot["Precio_Modelo_plot"],
                s=38,
                marker="x",
                zorder=5
            )

        ax.set_title(
            f"{pd.Timestamp(date).date()} | {cp}\n"
            f"RND bad points={int(row['n_bad_points'])}",
            fontsize=10
        )
        ax.set_xlabel("Strike")
        ax.set_ylabel("Precio_Modelo")

    for ax in axes[len(worst):]:
        ax.set_visible(False)

    fig.suptitle("Slices con densidad RN negativa discreta", fontsize=16, fontweight="bold")
    savefig("09_rnd_problem_slices.png")


# ============================================================
# 7. DISTRIBUCIÓN DE GAMMA
# ============================================================

def plot_gamma_distribution(greeks):
    plt.figure(figsize=(10, 5))
    plt.hist(greeks["gamma"].dropna(), bins=80)
    plt.title("Distribución de gamma")
    plt.xlabel("Gamma")
    plt.ylabel("Frecuencia")
    savefig("10_hist_gamma.png")

    plt.figure(figsize=(10, 5))
    x = greeks["gamma"].dropna()
    x = x[(x >= x.quantile(0.01)) & (x <= x.quantile(0.99))]
    plt.hist(x, bins=80)
    plt.title("Distribución de gamma (entre percentiles 1% y 99%)")
    plt.xlabel("Gamma")
    plt.ylabel("Frecuencia")
    savefig("11_hist_gamma_trimmed.png")


# ============================================================
# 8. DISTRIBUCIÓN DE IMPLIED VOL
# ============================================================

def plot_iv_distribution(prices):
    plt.figure(figsize=(10, 5))
    plt.hist(prices["implied_vol"].dropna(), bins=80)
    plt.title("Distribución de implied volatility")
    plt.xlabel("Implied vol")
    plt.ylabel("Frecuencia")
    savefig("12_hist_implied_vol.png")

    plt.figure(figsize=(10, 5))
    plt.scatter(prices["moneyness"], prices["implied_vol"], s=4, alpha=0.15)
    plt.title("Implied vol vs moneyness")
    plt.xlabel("Moneyness")
    plt.ylabel("Implied vol")
    savefig("13_iv_vs_moneyness.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 72)
    print("CARGANDO DATOS")
    print("=" * 72)

    prices, greeks, shape_summary, rnd_bad, smooth_summary, smooth_bad, greeks_summary, iv_summary = load_data()

    print(f"Prices shape        : {prices.shape}")
    print(f"Greeks shape        : {greeks.shape}")
    print(f"Shape summary       : {shape_summary.shape}")
    print(f"RND bad points      : {rnd_bad.shape}")
    print(f"Smooth summary      : {smooth_summary.shape}")
    print(f"Smooth bad points   : {smooth_bad.shape}")

    print("\nGenerando gráficos...")

    plot_summary_dashboard(shape_summary, greeks_summary, iv_summary, smooth_summary)
    plot_rnd_mass_hist(shape_summary)
    plot_rnd_bad_points(rnd_bad)
    plot_smooth_bad_points(smooth_bad)
    plot_top_smooth_slices(prices, smooth_summary, smooth_bad, top_n=TOP_N_SMOOTH)
    plot_rnd_problem_slices(prices, rnd_bad, top_n=TOP_N_RND)
    plot_gamma_distribution(greeks)
    plot_iv_distribution(prices)

    print("\nGráficos guardados en:")
    print(PLOTS_DIR)


if __name__ == "__main__":
    main()