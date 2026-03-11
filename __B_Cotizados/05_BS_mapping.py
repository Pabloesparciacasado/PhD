# In[]
import pandas as pd
import numpy as np
from scipy.stats import norm
import sys
import duckdb
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from __3_Functions.valuation import model_valuation


con = duckdb.connect()

vol_surface = con.execute("""
SELECT *
FROM read_parquet("C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\volatility_surface_30_B.parquet")
""").df()

#### Aplicamos BSM para mapear las volatilidades a precios de opciones. #####

model = model_valuation(curve_df = vol_surface)
resultado_europeo = model.price_BS_general(vol_surface)
resultado_europeo = resultado_europeo.rename(columns={"BS_Price": "Precio_Modelo"})


PARQET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\superficie_con_precios.parquet"
duckdb.from_df(resultado_europeo).write_parquet(PARQET_OUTPUT, compression='snappy')



print(vol_surface.columns)

















# In[]

#############################
######### ANALISYS de arbitraje (informal) ##########

df = resultado_europeo.copy()
df["Date"] = pd.to_datetime(df["Date"])

print(df.columns)
print(df.head())

# ============================================================
# 1. SANITY CHECKS BÁSICOS
# ============================================================

print("\nResumen de datos:")
print(df[["Date", "Days", "moneyness", "Strike", "implied_vol", "Precio_Modelo"]].describe(include="all"))

print("\nValores nulos:")
print(df[["Date", "Days", "moneyness", "Strike", "implied_vol", "Precio_Modelo"]].isna().sum())

# ============================================================
# 2. ANÁLISIS DE SALTOS TEMPORALES
# ============================================================

atm_band = 0.02

df["atm_like"] = (df["moneyness"] >= 1 - atm_band) & (df["moneyness"] <= 1 + atm_band)

atm_ts = (
    df[df["atm_like"]]
    .groupby(["Date", "CallPut"])
    .agg(
        iv_mean=("implied_vol", "mean"),
        price_mean=("Precio_Modelo", "mean"),
        forward_mean=("forward", "mean")
    )
    .reset_index()
    .sort_values(["CallPut", "Date"])
)

atm_ts["iv_diff"]         = atm_ts.groupby("CallPut")["iv_mean"].diff()
atm_ts["price_diff"]      = atm_ts.groupby("CallPut")["price_mean"].diff()
atm_ts["iv_pct_change"]   = atm_ts.groupby("CallPut")["iv_mean"].pct_change()
atm_ts["price_pct_change"]= atm_ts.groupby("CallPut")["price_mean"].pct_change()

def rolling_zscore(x, window=20):
    mean = x.rolling(window).mean()
    std  = x.rolling(window).std()
    return (x - mean) / std

atm_ts["iv_jump_z"]    = atm_ts.groupby("CallPut")["iv_diff"].transform(lambda s: rolling_zscore(s, 20))
atm_ts["price_jump_z"] = atm_ts.groupby("CallPut")["price_diff"].transform(lambda s: rolling_zscore(s, 20))

jump_threshold = 4

jumps_iv    = atm_ts[np.abs(atm_ts["iv_jump_z"])    > jump_threshold].copy()
jumps_price = atm_ts[np.abs(atm_ts["price_jump_z"]) > jump_threshold].copy()

print("\nSaltos grandes en IV ATM-like:")
print(jumps_iv[["Date", "CallPut", "iv_mean", "iv_diff", "iv_jump_z"]].head(20))

print("\nSaltos grandes en precio ATM-like:")
print(jumps_price[["Date", "CallPut", "price_mean", "price_diff", "price_jump_z"]].head(20))

for cp in sorted(atm_ts["CallPut"].dropna().unique()):
    tmp = atm_ts[atm_ts["CallPut"] == cp].copy()
    plt.figure(figsize=(12, 4))
    plt.plot(tmp["Date"], tmp["iv_mean"])
    plt.title(f"ATM-like implied vol - {cp}")
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(tmp["Date"], tmp["price_mean"])
    plt.title(f"ATM-like model price - {cp}")
    plt.show()

# ============================================================
# 3. NO-ARBITRAJE ESTÁTICO: BOUNDS
# ============================================================

is_call = df["CallPut"] == "C"
is_put  = df["CallPut"] == "P"

df["lower_bound"] = np.where(
    is_call,
    df["discount_factor"] * np.maximum(df["forward"] - df["Strike"], 0),
    df["discount_factor"] * np.maximum(df["Strike"]  - df["forward"], 0)
)
df["upper_bound"] = np.where(
    is_call,
    df["discount_factor"] * df["forward"],
    df["discount_factor"] * df["Strike"]
)

tol = 1e-8

df["flag_lower_ok"] = df["Precio_Modelo"] >= df["lower_bound"] - tol
df["flag_upper_ok"] = df["Precio_Modelo"] <= df["upper_bound"] + tol
df["flag_bounds_ok"] = df["flag_lower_ok"] & df["flag_upper_ok"]

print("\nViolaciones de bounds:")
print((~df["flag_bounds_ok"]).value_counts())

viol_bounds = df[~df["flag_bounds_ok"]].copy()
print(viol_bounds[["Date", "CallPut", "Strike", "forward", "Precio_Modelo",
                    "lower_bound", "upper_bound"]].head(20))

# ============================================================
# 4. MONOTONICIDAD Y CONVEXIDAD EN STRIKE
#    con separación interpolado vs extrapolado
# ============================================================

rows = []

for (date, days, callput), grp in df.groupby(["Date", "Days", "CallPut"]):
    grp = grp.sort_values("Strike").copy()

    # Zona interpolada vs extrapolada
    if "flag_inside_observed_range" in grp.columns:
        grp["zone"] = np.where(
            grp["flag_inside_observed_range"], "interpolated", "extrapolated"
        )
    else:
        grp["zone"] = "unknown"

    K = grp["Strike"].to_numpy(dtype=float)
    P = grp["Precio_Modelo"].to_numpy(dtype=float)
    Z = grp["zone"].to_numpy()

    if len(grp) < 3:
        continue

    # ---- Monotonicidad ----
    first_diff = np.diff(P)
    if callput == "C":
        mono_viol = np.where(first_diff >  tol)[0]
    else:
        mono_viol = np.where(first_diff < -tol)[0]

    # ---- Convexidad discreta — separada por zona ----
    conv_viol_interp = []
    conv_viol_extrap = []
    conv_viol_unknown = []

    for i in range(1, len(K) - 1):
        h1 = K[i]     - K[i - 1]
        h2 = K[i + 1] - K[i]

        if h1 <= 0 or h2 <= 0:
            continue

        slope_left  = (P[i]     - P[i - 1]) / h1
        slope_right = (P[i + 1] - P[i])     / h2

        if slope_right - slope_left < -1e-8:
            zone_i = Z[i]
            if zone_i == "interpolated":
                conv_viol_interp.append(i)
            elif zone_i == "extrapolated":
                conv_viol_extrap.append(i)
            else:
                conv_viol_unknown.append(i)

    n_conv_interp  = len(conv_viol_interp)
    n_conv_extrap  = len(conv_viol_extrap)
    n_conv_unknown = len(conv_viol_unknown)
    n_conv_total   = n_conv_interp + n_conv_extrap + n_conv_unknown

    rows.append({
        "Date":           date,
        "Days":           days,
        "CallPut":        callput,
        "n_points":       len(grp),
        "n_mono_viol":    len(mono_viol),
        "n_conv_viol":    n_conv_total,
        "n_conv_interp":  n_conv_interp,
        "n_conv_extrap":  n_conv_extrap,
        "flag_mono_ok":   len(mono_viol) == 0,
        "flag_conv_ok":   n_conv_total == 0,
        "flag_conv_interp_ok": n_conv_interp == 0,
        "flag_conv_extrap_ok": n_conv_extrap == 0,
    })

arb_summary = pd.DataFrame(rows)

# ---- Resumen global ----
print("\nResumen de no-arbitraje por slice (primeras filas):")
print(arb_summary.head())

print("\nSlices con violación de monotonicidad:")
print((~arb_summary["flag_mono_ok"]).sum())

print("\nSlices con violación de convexidad (total):")
print((~arb_summary["flag_conv_ok"]).sum())

print("\nSlices con violación de convexidad DENTRO del rango observado:")
print((~arb_summary["flag_conv_interp_ok"]).sum())

print("\nSlices con violación de convexidad EN ZONA EXTRAPOLADA:")
print((~arb_summary["flag_conv_extrap_ok"]).sum())

# ---- Desglose de violaciones por punto ----
total_conv   = arb_summary["n_conv_viol"].sum()
interp_conv  = arb_summary["n_conv_interp"].sum()
extrap_conv  = arb_summary["n_conv_extrap"].sum()

print(f"\nPuntos con violación de convexidad:")
print(f"  Total          : {total_conv}")
print(f"  Interpolados   : {interp_conv}  ({interp_conv/total_conv*100:.1f}%)" if total_conv > 0 else "  Interpolados   : 0")
print(f"  Extrapolados   : {extrap_conv}  ({extrap_conv/total_conv*100:.1f}%)" if total_conv > 0 else "  Extrapolados   : 0")

# ---- Slices problemáticos ----
problem_slices = arb_summary[
    (~arb_summary["flag_mono_ok"]) | (~arb_summary["flag_conv_ok"])
].copy()

print("\nPrimeros slices problemáticos:")
print(problem_slices.head(20))

# ---- Por año ----
arb_summary["year"] = pd.to_datetime(arb_summary["Date"]).dt.year

viol_by_year = (
    arb_summary.groupby("year")
    .agg(
        total_slices   = ("Date",               "size"),
        mono_bad       = ("flag_mono_ok",        lambda s: (~s).sum()),
        conv_bad       = ("flag_conv_ok",        lambda s: (~s).sum()),
        conv_interp_bad= ("flag_conv_interp_ok", lambda s: (~s).sum()),
        conv_extrap_bad= ("flag_conv_extrap_ok", lambda s: (~s).sum()),
    )
    .reset_index()
)

viol_by_year["mono_bad_pct"]        = 100 * viol_by_year["mono_bad"]        / viol_by_year["total_slices"]
viol_by_year["conv_bad_pct"]        = 100 * viol_by_year["conv_bad"]        / viol_by_year["total_slices"]
viol_by_year["conv_interp_bad_pct"] = 100 * viol_by_year["conv_interp_bad"] / viol_by_year["total_slices"]
viol_by_year["conv_extrap_bad_pct"] = 100 * viol_by_year["conv_extrap_bad"] / viol_by_year["total_slices"]

print("\nViolaciones por año (con desglose interp/extrap):")
print(viol_by_year.to_string(index=False))

# ============================================================
# 5. EXPORTS
# ============================================================

problem_slices.to_csv("problem_slices_no_arbitrage.csv", index=False)

if not problem_slices.empty:
    problem_points = df.merge(
        problem_slices[["Date", "Days", "CallPut"]],
        on=["Date", "Days", "CallPut"],
        how="inner"
    )
    problem_points.to_csv("problem_points_no_arbitrage.csv", index=False)

viol_by_year.to_csv("no_arbitrage_violations_by_year.csv", index=False)

# ============================================================
# 6. VISUALIZACIÓN DE UN SLICE PROBLEMÁTICO
# ============================================================

if not problem_slices.empty:
    ex = problem_slices.iloc[0]

    ex_points = df[
        (df["Date"]    == ex["Date"])    &
        (df["Days"]    == ex["Days"])    &
        (df["CallPut"] == ex["CallPut"])
    ].sort_values("Strike")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Precios
    ax = axes[0]
    ax.plot(ex_points["Strike"], ex_points["Precio_Modelo"], marker="o", label="Precio")
    if "flag_inside_observed_range" in ex_points.columns:
        obs  = ex_points[ex_points["flag_inside_observed_range"]]
        extr = ex_points[~ex_points["flag_inside_observed_range"]]
        ax.scatter(obs["Strike"],  obs["Precio_Modelo"],  color="blue",  zorder=5, label="Interpolado", s=40)
        ax.scatter(extr["Strike"], extr["Precio_Modelo"], color="orange", zorder=5, label="Extrapolado", s=40)
    ax.set_title(f"Precios: {ex['Date']} | {ex['Days']}d | {ex['CallPut']}")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Precio_Modelo")
    ax.legend()

    # IV
    ax2 = axes[1]
    ax2.plot(ex_points["Strike"], ex_points["implied_vol"], marker="o", color="green", label="IV")
    if "flag_inside_observed_range" in ex_points.columns:
        obs  = ex_points[ex_points["flag_inside_observed_range"]]
        extr = ex_points[~ex_points["flag_inside_observed_range"]]
        ax2.scatter(obs["Strike"],  obs["implied_vol"],  color="blue",   zorder=5, s=40)
        ax2.scatter(extr["Strike"], extr["implied_vol"], color="orange",  zorder=5, s=40)
    ax2.set_title(f"IV: {ex['Date']} | {ex['Days']}d | {ex['CallPut']}")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Implied Vol")
    ax2.legend()

    plt.tight_layout()
    plt.show()






print(df[["discount_factor", "forward", "Strike", 
          "Precio_Modelo", "lower_bound", "upper_bound",
          "flag_lower_ok", "flag_upper_ok"]].head(10))







# In[]  





# In[]






















# vol_surface_2 = vol_surface[(vol_surface["flag_wing_clipped"]) == False]
# #print(vol_surface_2.head(50).sort_values("Days"))

# clipped_extrap = vol_surface[
#     (vol_surface["flag_wing_clipped"]) & 
#     (vol_surface["flag_inside_observed_range"])
# ]
# print(clipped_extrap)


# # In[]

# # Coge una smile cualquiera
# grupo = list(vol_surface.groupby(["Date", "Expiration"]))
# smile = grupo[0][1]

# print("n_grid points:", len(smile))
# print("m_obs_min:", smile["m_obs_min"].iloc[0])
# print("m_obs_max:", smile["m_obs_max"].iloc[0])
# print("moneyness min:", smile["moneyness"].min())
# print("moneyness max:", smile["moneyness"].max())
# print(smile["flag_inside_observed_range"].value_counts())

# # In[]


# stats_by_smile = (
#     vol_surface.groupby(["Date", "Expiration"])["flag_wing_clipped"]
#     .value_counts()
#     .unstack(fill_value=0)
# )

# print(stats_by_smile)
# # %%

# summary = (
#     vol_surface.groupby(["Date","Expiration"])
#     .agg(
#         n_points=("flag_wing_clipped","size"),
#         n_clipped=("flag_wing_clipped","sum"),
#         has_clipping=("flag_wing_clipped","any")
#     )
# )

# print(summary.head())


# # %%
# has_false = (
#     vol_surface.groupby(["Date", "Expiration"])["flag_wing_clipped"]
#     .apply(lambda s: (~s).any())
# )

# all_false = vol_surface.groupby(["Date", "Expiration"])["flag_wing_clipped"].any()
# all_false.mean()
# # %%

# %%
