import pandas as pd
import numpy as np
import duckdb

# ============================================================
# 0. CARGA DEL FICHERO DE SUPERFICIE DE VOLATILIDAD
# ============================================================

con = duckdb.connect()

vol_surface = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\volatility_surface.parquet')
""").df()

print(vol_surface.columns)

# ============================================================
# 1. PARÁMETROS
# ============================================================

target_days = 30
target_T = target_days / 365.0

# Usar solo la parte observada de cada smile
use_only_observed = True

# Activar o no el filtro de calidad temporal
apply_maturity_filter = True

# Umbrales del filtro temporal
max_gap = 45          # máximo Days_2 - Days_1
max_below_dist = 20   # máximo 30 - Days_1
max_above_dist = 20   # máximo Days_2 - 30

# ============================================================
# 2. BASE DE TRABAJO
# ============================================================

vs = vol_surface.copy()

if use_only_observed:
    vs = vs[vs["flag_inside_observed_range"]].copy()

# Nos aseguramos de que Date esté en datetime
vs["Date"] = pd.to_datetime(vs["Date"])

# ============================================================
# 3. IDENTIFICAR, PARA CADA FECHA, LOS VENCIMIENTOS QUE RODEAN 30 DÍAS
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

    # Si existe exactamente 30 días, tomamos ese mismo vencimiento
    if len(exact) > 0:
        d1 = exact[0]
        d2 = exact[0]
    else:
        d1 = lower.max() if len(lower) > 0 else np.nan
        d2 = upper.min() if len(upper) > 0 else np.nan

    # Distancias al target
    if pd.notna(d1) and pd.notna(d2):
        gap = d2 - d1
        below_dist = target_days - d1
        above_dist = d2 - target_days
        basic_valid = True
    else:
        gap = np.nan
        below_dist = np.nan
        above_dist = np.nan
        basic_valid = False

    # Filtro temporal opcional
    if apply_maturity_filter and basic_valid:
        valid = (
            (gap <= max_gap) and
            (below_dist <= max_below_dist) and
            (above_dist <= max_above_dist)
        )
    else:
        valid = basic_valid

    rows.append({
        "Date": date,
        "Days_1": d1,
        "Days_2": d2,
        "gap": gap,
        "below_dist": below_dist,
        "above_dist": above_dist,
        "basic_valid": basic_valid,
        "valid_30d_interp": valid
    })

pairs_30 = pd.DataFrame(rows)

# Nos quedamos solo con fechas utilizables
pairs_30 = pairs_30[pairs_30["valid_30d_interp"]].copy()

print("Fechas utilizables para 30 días:", pairs_30["Date"].nunique())
print("\nResumen de distancias:")
print(pairs_30[["gap", "below_dist", "above_dist"]].describe())

# ============================================================
# 4. EXTRAER EL VENCIMIENTO INFERIOR Y SUPERIOR
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
# 5. RENOMBRAR COLUMNAS PARA COMBINAR AMBOS VENCIMIENTOS
# ============================================================

cols_keep = [
    "Date", "Expiration", "Days", "T",
    "moneyness", "log_moneyness", "CallPut",
    "implied_vol", "total_variance",
    "forward", "rate", "discount_factor", "Strike",
    "gap", "below_dist", "above_dist"
]

vs1 = vs1[cols_keep].rename(columns={
    "Expiration": "Expiration_1",
    "Days": "Days_1",
    "T": "T_1",
    "implied_vol": "implied_vol_1",
    "total_variance": "total_variance_1",
    "forward": "forward_1",
    "rate": "rate_1",
    "discount_factor": "discount_factor_1",
    "Strike": "Strike_1"
})

vs2 = vs2[cols_keep].rename(columns={
    "Expiration": "Expiration_2",
    "Days": "Days_2",
    "T": "T_2",
    "implied_vol": "implied_vol_2",
    "total_variance": "total_variance_2",
    "forward": "forward_2",
    "rate": "rate_2",
    "discount_factor": "discount_factor_2",
    "Strike": "Strike_2"
})

# ============================================================
# 6. MERGE POR FECHA + MONEyness + LADO CALL/PUT
# ============================================================

surface_30 = vs1.merge(
    vs2,
    on=[
        "Date",
        "moneyness",
        "log_moneyness",
        "CallPut",
        "gap",
        "below_dist",
        "above_dist"
    ],
    how="inner"
).copy()

print("Filas tras merge temporal:", len(surface_30))

# ============================================================
# 7. INTERPOLACIÓN LINEAL EN VARIANZA TOTAL
# ============================================================

# Caso exacto: ya hay vencimiento a 30 días
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

# Varianza total a 30 días
surface_30["total_variance"] = (
    surface_30["w1"] * surface_30["total_variance_1"] +
    surface_30["w2"] * surface_30["total_variance_2"]
)

# Recuperar IV a 30 días
surface_30["implied_vol"] = np.sqrt(
    np.maximum(surface_30["total_variance"] / target_T, 1e-12)
)

# ============================================================
# 8. INTERPOLACIÓN SIMPLE DE FORWARD Y RATE
# ============================================================

surface_30["forward"] = (
    surface_30["w1"] * surface_30["forward_1"] +
    surface_30["w2"] * surface_30["forward_2"]
)

surface_30["rate"] = (
    surface_30["w1"] * surface_30["rate_1"] +
    surface_30["w2"] * surface_30["rate_2"]
)

surface_30["discount_factor"] = np.exp(-surface_30["rate"] * target_T)

# Strike consistente con el nuevo forward
surface_30["Strike"] = surface_30["moneyness"] * surface_30["forward"]

# ============================================================
# 9. COLUMNAS FINALES
# ============================================================

surface_30["Days"] = target_days
surface_30["T"] = target_T

surface_30_final = surface_30[
    [
        "Date",
        "Days", "T",
        "Days_1", "Days_2",
        "Expiration_1", "Expiration_2",
        "gap", "below_dist", "above_dist",
        "w1", "w2",
        "moneyness", "log_moneyness", "CallPut",
        "forward", "rate", "discount_factor",
        "Strike",
        "implied_vol", "total_variance"
    ]
].copy()

print(surface_30_final.head())
print(surface_30_final.shape)

# In[]:



PARQET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\volatility_surface_30.parquet"
duckdb.from_df(surface_30_final).write_parquet(PARQET_OUTPUT, compression='snappy')


print("Generada la superficie de volatilidad estadarizada a 30 días con éxito")


# In[]: