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
    })

pairs_30 = pd.DataFrame(rows)
pairs_30 = pairs_30[pairs_30["valid_30d_interp"]].copy()

print("Fechas utilizables para 30 días:", pairs_30["Date"].nunique())
print("\nResumen de distancias:")
print(pairs_30[["gap", "below_dist", "above_dist"]].describe())

# ============================================================
# 4. EXTRAER VENCIMIENTO INFERIOR Y SUPERIOR
# ============================================================

vs1 = vs.merge(
    pairs_30[["Date", "Days_1", "Days_2", "gap", "below_dist", "above_dist"]],
    left_on=["Date", "Days"],
    right_on=["Date", "Days_1"],
    how="inner"
).copy()

vs2 = vs.merge(
    pairs_30[["Date", "Days_1", "Days_2", "gap", "below_dist", "above_dist"]],
    left_on=["Date", "Days"],
    right_on=["Date", "Days_2"],
    how="inner"
).copy()

# ============================================================
# 5. RENOMBRAR COLUMNAS
# ============================================================

cols_keep = [
    "Date", "Expiration", "Days", "T",
    "moneyness", "log_moneyness", "CallPut",
    "implied_vol", "total_variance",
    "forward", "rate", "discount_factor", "Strike",
    "gap", "below_dist", "above_dist"
]

vs1 = vs1[cols_keep].rename(columns={
    "Expiration":       "Expiration_1",
    "Days":             "Days_1",
    "T":                "T_1",
    "implied_vol":      "implied_vol_1",
    "total_variance":   "total_variance_1",
    "forward":          "forward_1",
    "rate":             "rate_1",
    "discount_factor":  "discount_factor_1",
    "Strike":           "Strike_1",
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
})

# ============================================================
# 6. MERGE POR FECHA + MONEYNESS + LADO CALL/PUT
# ============================================================

surface_30 = vs1.merge(
    vs2,
    on=["Date", "moneyness", "log_moneyness", "CallPut",
        "gap", "below_dist", "above_dist"],
    how="inner"
).copy()

print("Filas tras merge temporal:", len(surface_30))

# ============================================================
# 7. PESOS DE INTERPOLACIÓN TEMPORAL
# ============================================================

exact_mask = surface_30["Days_1"] == surface_30["Days_2"]

surface_30["w1"] = np.where(
    exact_mask,
    1.0,
    (surface_30["Days_2"] - target_days) / (surface_30["Days_2"] - surface_30["Days_1"])
)
surface_30["w2"] = np.where(
    exact_mask,
    0.0,
    (target_days - surface_30["Days_1"]) / (surface_30["Days_2"] - surface_30["Days_1"])
)

# ============================================================
# 8. INTERPOLACIÓN EN VARIANZA TOTAL, FORWARD Y RATE
# ============================================================

# ----------------------------------------------------------
# Varianza total — interpolación lineal en w = sigma^2 * T
# Estándar en la literatura (Conrad et al. 2013, CBOE VIX)
# ----------------------------------------------------------
surface_30["total_variance"] = (
    surface_30["w1"] * surface_30["total_variance_1"] +
    surface_30["w2"] * surface_30["total_variance_2"]
)

surface_30["implied_vol"] = np.sqrt(
    np.maximum(surface_30["total_variance"] / target_T, 1e-12)
)

# ----------------------------------------------------------
# Forward — interpolación log-lineal
# F(T*) = exp(w1 * log(F1) + w2 * log(F2))
# Más correcto que interpolación lineal porque el forward
# crece exponencialmente con T bajo cost-of-carry constante.
# ----------------------------------------------------------
surface_30["forward"] = np.exp(
    surface_30["w1"] * np.log(surface_30["forward_1"]) +
    surface_30["w2"] * np.log(surface_30["forward_2"])
)

# ----------------------------------------------------------
# Rate — interpolación log-lineal en factores de descuento
# DF(T*) = exp(w1 * log(DF1) + w2 * log(DF2))
# r(T*)  = -log(DF(T*)) / T*
#
# Estándar de industria: los factores de descuento se
# interpolan log-linealmente (equivalente a interpolar
# linealmente las tasas continuas ponderadas por plazo,
# que es la convención de bootstrapping de curvas).
# ----------------------------------------------------------
log_df1 = -surface_30["rate_1"] * surface_30["T_1"]  # = log(DF1)
log_df2 = -surface_30["rate_2"] * surface_30["T_2"]  # = log(DF2)

log_df_target = (
    surface_30["w1"] * log_df1 +
    surface_30["w2"] * log_df2
)

surface_30["rate"]            = -log_df_target / target_T
surface_30["discount_factor"] = np.exp(log_df_target)

# ----------------------------------------------------------
# Strike consistente con el forward interpolado
# K = moneyness * F(T*)
# ----------------------------------------------------------
surface_30["Strike"] = surface_30["moneyness"] * surface_30["forward"]

# ============================================================
# 9. COLUMNAS FINALES
# ============================================================

surface_30["Days"] = target_days
surface_30["T"]    = target_T

surface_30_final = surface_30[[
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
]].copy()

print(surface_30_final.head())
print(surface_30_final.shape)

# ============================================================
# 10. GUARDAR
# ============================================================

PARQUET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\volatility_surface_30.parquet"
duckdb.from_df(surface_30_final).write_parquet(PARQUET_OUTPUT, compression="snappy")

print("Generada la superficie de volatilidad estandarizada a 30 días con éxito")