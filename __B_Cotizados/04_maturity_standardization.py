import pandas as pd
import numpy as np
import duckdb

# ============================================================
# 0. CARGA DEL FICHERO DE SUPERFICIE DE VOLATILIDAD
# ============================================================

con = duckdb.connect()

vol_surface = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\volatility_surface_shimko.parquet')
""").df()

print(vol_surface.columns)

# ============================================================
# 1. PARÁMETROS
# ============================================================

target_days = 30
target_T = target_days / 365.0

use_only_observed = True
apply_maturity_filter = True

max_gap = 45
max_below_dist = 20
max_above_dist = 20

# ============================================================
# 2. BASE DE TRABAJO
# ============================================================

vs = vol_surface.copy()

if use_only_observed:
    vs = vs[vs["flag_inside_observed_range"]].copy()

vs["Date"] = pd.to_datetime(vs["Date"])
vs["Expiration"] = pd.to_datetime(vs["Expiration"])

# ============================================================
# 2.1. COLAPSAR DUPLICADOS POR DATE × DAYS × MONEYNESS × CALLPUT
# ============================================================
# Motivo:
# puede haber varias Expiration distintas con el mismo número de Days
# dentro de una misma fecha. Para interpolar a 30d necesitamos una sola
# smile por Date × Days. Aquí agregamos esas duplicidades.

group_cols = ["Date", "Days", "moneyness", "CallPut"]

agg_dict = {
    "log_moneyness": "first",
    "implied_vol": "median",
    "total_variance": "median",
    "forward": "median",
    "rate": "median",
    "discount_factor": "median",
    "Strike": "median",
    "T": "first",
    "m_obs_min": "median",
    "m_obs_max": "median",
    "k_obs_min": "median",
    "k_obs_max": "median",
    "flag_inside_observed_range": "all",
    "flag_wing_clipped": "any",
}

# Añadir columnas Shimko solo si existen
optional_cols = ["shimko_rmse", "dsigma_dm", "d2sigma_dm2"]
for col in optional_cols:
    if col in vs.columns:
        agg_dict[col] = "median"

# Expiration representativa:
# - tomamos la mínima si hay varias con mismo Days
#   (en la práctica debería coincidir casi siempre)
agg_dict["Expiration"] = "min"

vs = (
    vs.groupby(group_cols, as_index=False)
      .agg(agg_dict)
      .sort_values(["Date", "Days", "CallPut", "moneyness"])
      .reset_index(drop=True)
)

# Reordenar columnas de forma cómoda
base_cols = [
    "Date", "Expiration", "Days", "T",
    "moneyness", "log_moneyness", "CallPut",
    "implied_vol", "total_variance",
    "forward", "rate", "discount_factor", "Strike",
    "m_obs_min", "m_obs_max", "k_obs_min", "k_obs_max",
    "flag_inside_observed_range", "flag_wing_clipped"
]
extra_cols = [c for c in vs.columns if c not in base_cols]
vs = vs[base_cols + extra_cols]

# Check de unicidad tras colapsar
dup_check = vs.duplicated(subset=["Date", "Days", "moneyness", "CallPut"]).sum()
print(f"Duplicados tras colapsar Date×Days×moneyness×CallPut: {dup_check}")

# ============================================================
# 3. IDENTIFICAR VENCIMIENTOS QUE RODEAN 30 DÍAS
# ============================================================

days_by_date = (
    vs[["Date", "Days"]]
    .drop_duplicates()
    .sort_values(["Date", "Days"])
)

rows = []

for date, df_date in days_by_date.groupby("Date"):
    days = np.sort(df_date["Days"].unique())

    lower = days[days < target_days]
    upper = days[days > target_days]
    exact = days[days == target_days]

    if len(exact) > 0:
        d1 = exact[0]
        d2 = exact[0]
    else:
        d1 = lower.max() if len(lower) > 0 else np.nan
        d2 = upper.min() if len(upper) > 0 else np.nan

    if pd.notna(d1) and pd.notna(d2):
        gap         = d2 - d1
        below_dist  = target_days - d1
        above_dist  = d2 - target_days
        basic_valid = True
    else:
        gap = below_dist = above_dist = np.nan
        basic_valid = False

    if apply_maturity_filter and basic_valid:
        valid = (
            (gap <= max_gap) and
            (below_dist <= max_below_dist) and
            (above_dist <= max_above_dist)
        )
    else:
        valid = basic_valid

    rows.append({
        "Date":             date,
        "Days_1":           d1,
        "Days_2":           d2,
        "gap":              gap,
        "below_dist":       below_dist,
        "above_dist":       above_dist,
        "basic_valid":      basic_valid,
        "valid_30d_interp": valid,
        "is_exact_30":      bool(pd.notna(d1) and pd.notna(d2) and d1 == d2),
    })

pairs_30 = pd.DataFrame(rows)
pairs_30 = pairs_30[pairs_30["valid_30d_interp"]].copy()

print("Fechas utilizables para 30 días:", pairs_30["Date"].nunique())
print("\nResumen de distancias:")
print(pairs_30[["gap", "below_dist", "above_dist"]].describe())

# ============================================================
# 4. EXTRAER VENCIMIENTO INFERIOR Y SUPERIOR
# ============================================================

pairs_cols = ["Date", "Days_1", "Days_2", "gap", "below_dist", "above_dist", "is_exact_30"]

vs1 = vs.merge(
    pairs_30[pairs_cols],
    left_on=["Date", "Days"],
    right_on=["Date", "Days_1"],
    how="inner",
    validate="many_to_one"
).copy()

pairs_30_non_exact = pairs_30[~pairs_30["is_exact_30"]].copy()

vs2 = vs.merge(
    pairs_30_non_exact[["Date", "Days_1", "Days_2", "gap", "below_dist", "above_dist"]],
    left_on=["Date", "Days"],
    right_on=["Date", "Days_2"],
    how="inner",
    validate="many_to_one"
).copy()

# ============================================================
# 5. CASO EXACTO 30 DÍAS (SIN SELF-MERGE)
# ============================================================

exact_30 = vs1[vs1["is_exact_30"]].copy()

if not exact_30.empty:
    exact_30["w1"] = 1.0
    exact_30["w2"] = 0.0

    exact_30["Days"] = target_days
    exact_30["T"]    = target_T

    exact_30["Expiration_1"] = exact_30["Expiration"]
    exact_30["Expiration_2"] = exact_30["Expiration"]

    exact_30_final = exact_30[[
        "Date",
        "Days", "T",
        "Days_1", "Days_2",
        "Expiration_1", "Expiration_2",
        "gap", "below_dist", "above_dist",
        "w1", "w2",
        "moneyness", "log_moneyness", "CallPut",
        "forward", "rate", "discount_factor",
        "Strike",
        "implied_vol", "total_variance",
        "flag_inside_observed_range", "flag_wing_clipped"
    ]].copy()
else:
    exact_30_final = pd.DataFrame()

# ============================================================
# 6. CASO INTERPOLADO ENTRE DOS MADURECES
# ============================================================

vs1_non_exact = vs1[~vs1["is_exact_30"]].copy()

cols_keep = [
    "Date", "Expiration", "Days", "T",
    "moneyness", "log_moneyness", "CallPut",
    "implied_vol", "total_variance",
    "forward", "rate", "discount_factor", "Strike",
    "flag_inside_observed_range", "flag_wing_clipped",
    "gap", "below_dist", "above_dist"
]

# conservar también columnas Shimko si existen
extra_keep = [c for c in ["shimko_rmse", "dsigma_dm", "d2sigma_dm2"] if c in vs1_non_exact.columns]
cols_keep = cols_keep + extra_keep

vs1_non_exact = vs1_non_exact[cols_keep].rename(columns={
    "Expiration":       "Expiration_1",
    "Days":             "Days_1",
    "T":                "T_1",
    "implied_vol":      "implied_vol_1",
    "total_variance":   "total_variance_1",
    "forward":          "forward_1",
    "rate":             "rate_1",
    "discount_factor":  "discount_factor_1",
    "Strike":           "Strike_1",
    "flag_inside_observed_range": "flag_inside_observed_range_1",
    "flag_wing_clipped": "flag_wing_clipped_1",
    "shimko_rmse":      "shimko_rmse_1" if "shimko_rmse" in cols_keep else "shimko_rmse",
    "dsigma_dm":        "dsigma_dm_1" if "dsigma_dm" in cols_keep else "dsigma_dm",
    "d2sigma_dm2":      "d2sigma_dm2_1" if "d2sigma_dm2" in cols_keep else "d2sigma_dm2",
})

vs2 = vs2[cols_keep].rename(columns={
    "Expiration":       "Expiration_2",
    "Days":             "Days_2",
    "T":                "T_2",
    "implied_vol":      "implied_vol_2",
    "total_variance":   "total_variance_2",
    "forward":          "forward_2",
    "rate":             "rate_2",
    "discount_factor":  "discount_factor_2",
    "Strike":           "Strike_2",
    "flag_inside_observed_range": "flag_inside_observed_range_2",
    "flag_wing_clipped": "flag_wing_clipped_2",
    "shimko_rmse":      "shimko_rmse_2" if "shimko_rmse" in cols_keep else "shimko_rmse",
    "dsigma_dm":        "dsigma_dm_2" if "dsigma_dm" in cols_keep else "dsigma_dm",
    "d2sigma_dm2":      "d2sigma_dm2_2" if "d2sigma_dm2" in cols_keep else "d2sigma_dm2",
})

surface_30_interp = vs1_non_exact.merge(
    vs2,
    on=["Date", "moneyness", "CallPut", "gap", "below_dist", "above_dist"],
    how="inner",
    validate="one_to_one"
).copy()

print("Filas tras merge temporal (interpoladas):", len(surface_30_interp))

# ============================================================
# 7. PESOS DE INTERPOLACIÓN TEMPORAL
# ============================================================

surface_30_interp["w1"] = (
    (surface_30_interp["Days_2"] - target_days) /
    (surface_30_interp["Days_2"] - surface_30_interp["Days_1"])
)

surface_30_interp["w2"] = (
    (target_days - surface_30_interp["Days_1"]) /
    (surface_30_interp["Days_2"] - surface_30_interp["Days_1"])
)

# ============================================================
# 8. INTERPOLACIÓN EN VARIANZA TOTAL, FORWARD Y RATE
# ============================================================

surface_30_interp["total_variance"] = (
    surface_30_interp["w1"] * surface_30_interp["total_variance_1"] +
    surface_30_interp["w2"] * surface_30_interp["total_variance_2"]
)

surface_30_interp["implied_vol"] = np.sqrt(
    np.maximum(surface_30_interp["total_variance"] / target_T, 1e-12)
)

surface_30_interp["forward"] = np.exp(
    surface_30_interp["w1"] * np.log(surface_30_interp["forward_1"]) +
    surface_30_interp["w2"] * np.log(surface_30_interp["forward_2"])
)

log_df1 = -surface_30_interp["rate_1"] * surface_30_interp["T_1"]
log_df2 = -surface_30_interp["rate_2"] * surface_30_interp["T_2"]

log_df_target = (
    surface_30_interp["w1"] * log_df1 +
    surface_30_interp["w2"] * log_df2
)

surface_30_interp["rate"] = -log_df_target / target_T
surface_30_interp["discount_factor"] = np.exp(log_df_target)

surface_30_interp["Strike"] = surface_30_interp["moneyness"] * surface_30_interp["forward"]

surface_30_interp["flag_inside_observed_range"] = (
    surface_30_interp["flag_inside_observed_range_1"] &
    surface_30_interp["flag_inside_observed_range_2"]
)

surface_30_interp["flag_wing_clipped"] = (
    surface_30_interp["flag_wing_clipped_1"] |
    surface_30_interp["flag_wing_clipped_2"]
)

# ============================================================
# 9. COLUMNAS FINALES
# ============================================================

surface_30_interp["Days"] = target_days
surface_30_interp["T"] = target_T

surface_30_interp_final = surface_30_interp[[
    "Date",
    "Days", "T",
    "Days_1", "Days_2",
    "Expiration_1", "Expiration_2",
    "gap", "below_dist", "above_dist",
    "w1", "w2",
    "moneyness", "CallPut",
    "forward", "rate", "discount_factor",
    "Strike",
    "implied_vol", "total_variance",
    "flag_inside_observed_range", "flag_wing_clipped"
]].copy()

surface_30_interp_final["log_moneyness"] = np.log(surface_30_interp_final["moneyness"])

surface_30_interp_final = surface_30_interp_final[[
    "Date",
    "Days", "T",
    "Days_1", "Days_2",
    "Expiration_1", "Expiration_2",
    "gap", "below_dist", "above_dist",
    "w1", "w2",
    "moneyness", "log_moneyness", "CallPut",
    "forward", "rate", "discount_factor",
    "Strike",
    "implied_vol", "total_variance",
    "flag_inside_observed_range", "flag_wing_clipped"
]]

surface_30_final = pd.concat(
    [exact_30_final, surface_30_interp_final],
    ignore_index=True
)

surface_30_final = surface_30_final.sort_values(
    ["Date", "CallPut", "moneyness"]
).reset_index(drop=True)

print(surface_30_final.head())
print(surface_30_final.shape)


print(surface_30_interp.columns.tolist())


# ============================================================
# 10. CHECK FINAL DE UNICIDAD
# ============================================================

dup_final = surface_30_final.duplicated(subset=["Date", "moneyness", "CallPut"]).sum()
print(f"Duplicados finales Date×moneyness×CallPut: {dup_final}")

# ============================================================
# 11. GUARDAR
# ============================================================

PARQUET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\volatility_surface_30_B.parquet"
duckdb.from_df(surface_30_final).write_parquet(PARQUET_OUTPUT, compression="snappy")

print("Generada la superficie de volatilidad estandarizada a 30 días con éxito")