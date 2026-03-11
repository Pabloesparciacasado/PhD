# In[]:
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline
import duckdb
from typing import Literal

con = duckdb.connect()

quoted_option = con.execute("""
SELECT *
FROM read_parquet('C:\\Users\\pablo.esparcia\\Documents\\OptionMetrics\\output\\opt_df_clean.parquet')
""").df()

quoted_option = quoted_option[quoted_option["flag_otm"]]
quoted_option = quoted_option[(quoted_option["Moneyness"] >= 0.3) & (quoted_option["Moneyness"] <= 1.7)]



quoted_option.groupby(quoted_option["Date"].dt.year).size().rename("n_contracts")
# In[]:
# Función de interpolación cubic spline:


# def interpolate_smile_slice(
#     df_slice: pd.DataFrame,
#     n_grid: int = 100,
#     use_observed_grid: bool = True,
#     global_moneyness_min: float = 0.5,
#     global_moneyness_max: float = 1.5,
#     min_nodes: int = 5,
#     min_iv: float = 0.03,          # suelo económico de IV, no numérico
#     max_iv: float = 5.00,          # opcional: techo defensivo
#     bc_type: str = "natural"       # más estable en extremos que not-a-knot
# ) -> pd.DataFrame | None:
#     """
#     Interpola una smile de IV en función de moneyness (K/F),
#     pero tratando las alas en log-moneyness k = log(K/F).

#     Dentro del rango observado:
#         - CubicSpline sobre IV(k)

#     Fuera del rango observado (si use_observed_grid=False):
#         - extrapolación lineal en varianza total w(k) = sigma(k)^2 * T
#         - usando la derivada del spline en el extremo observado
#         - con clipping de pendiente para evitar alas decrecientes
#         - con suelo en IV mínima

#     Devuelve:
#         DataFrame con:
#         moneyness, log_moneyness, implied_vol,
#         total_variance, m_obs_min, m_obs_max,
#         k_obs_min, k_obs_max,
#         flag_inside_observed_range, flag_wing_clipped
#     """

#     required_cols = {"Moneyness", "ImpliedVolatility", "Days"}
#     missing = required_cols - set(df_slice.columns)
#     if missing:
#         raise ValueError(f"Faltan columnas requeridas: {missing}")

#     df = df_slice.copy()
#     df = df[np.isfinite(df["Moneyness"])]
#     df = df[np.isfinite(df["ImpliedVolatility"])]
#     df = df[(df["Moneyness"] > 0) & (df["ImpliedVolatility"] > 0)]

#     if df.empty:
#         return None

#     # Agrupar duplicados de moneyness de forma robusta
#     df = (
#         df.groupby("Moneyness", as_index=False)
#           .agg({
#               "ImpliedVolatility": "median",
#               "Days": "first"
#           })
#           .sort_values("Moneyness")
#           .reset_index(drop=True)
#     )

#     m = df["Moneyness"].to_numpy(dtype=float)
#     iv = df["ImpliedVolatility"].to_numpy(dtype=float)

#     # Pasamos a log-moneyness
#     k = np.log(m)

#     # Recuento de nodos a cada lado del ATM, ya sin duplicados
#     n_left = np.sum(k < 0.0)
#     n_right = np.sum(k > 0.0)

#     if len(k) < min_nodes or n_left < 2 or n_right < 2:
#         return None

#     T = float(df["Days"].iloc[0]) / 365.0
#     if not np.isfinite(T) or T <= 0:
#         return None

#     # Suelo de varianza total derivado de min_iv
#     min_total_variance = (min_iv ** 2) * T
#     max_total_variance = (max_iv ** 2) * T

#     k_min_obs = float(k.min())
#     k_max_obs = float(k.max())
#     m_min_obs = float(m.min())
#     m_max_obs = float(m.max())

#     # Spline sobre IV(k) dentro del rango observado
#     interp = CubicSpline(k, iv, bc_type=bc_type, extrapolate=False)
#     #interp = PchipInterpolator(k, iv)

#     # Grid
#     if use_observed_grid:
#         k_grid = np.linspace(k_min_obs, k_max_obs, n_grid)

#     else:
#         if global_moneyness_min <= 0 or global_moneyness_max <= 0:
#             raise ValueError("Los extremos globales de moneyness deben ser > 0.")
#         if global_moneyness_min >= global_moneyness_max:
#             raise ValueError("global_moneyness_min debe ser menor que global_moneyness_max.")

#         k_global_min = np.log(global_moneyness_min)
#         k_global_max = np.log(global_moneyness_max)

#         if k_global_min > k_min_obs or k_global_max < k_max_obs:
#             raise ValueError(
#                 "El grid global no contiene completamente el rango observado. "
#                 f"Observado: [{m_min_obs:.4f}, {m_max_obs:.4f}] | "
#                 f"Global: [{global_moneyness_min:.4f}, {global_moneyness_max:.4f}]"
#             )

#         k_grid = np.linspace(k_global_min, k_global_max, n_grid)

#     m_grid = np.exp(k_grid)

#     # Interpolación interior
#     iv_grid = interp(k_grid)

#     # Flags
#     flag_inside = (k_grid >= k_min_obs) & (k_grid <= k_max_obs)
#     flag_wing_clipped = np.zeros_like(k_grid, dtype=bool)

#     # Extrapolación en alas
#     if not use_observed_grid:
#         w_obs = iv**2 * T

#         # Derivadas del spline de IV respecto a k
#         div_dk_left = float(interp(k_min_obs, 1))
#         div_dk_right = float(interp(k_max_obs, 1))

#         # dw/dk = 2 * sigma * T * d(sigma)/dk
#         slope_left_w = 2.0 * iv[0] * T * div_dk_left
#         slope_right_w = 2.0 * iv[-1] * T * div_dk_right

#         # Clipping de pendientes para evitar colapso de alas:
#         # - ala izquierda: al moverte a k más pequeño, no queremos que w baje
#         #   => slope_left_w NO debe ser positiva
#         # - ala derecha: al moverte a k más grande, no queremos que w baje
#         #   => slope_right_w NO debe ser negativa
#         slope_left_w = min(slope_left_w, 0.0)
#         slope_right_w = max(slope_right_w, 0.0)

#         left_mask = k_grid < k_min_obs
#         right_mask = k_grid > k_max_obs

#         if np.any(left_mask):
#             w_left = w_obs[0] + slope_left_w * (k_grid[left_mask] - k_min_obs)
#             clipped_left = (w_left <= min_total_variance)
#             w_left = np.clip(w_left, min_total_variance, max_total_variance)
#             iv_grid[left_mask] = np.sqrt(w_left / T)
#             flag_wing_clipped[left_mask] = clipped_left

#         if np.any(right_mask):
#             w_right = w_obs[-1] + slope_right_w * (k_grid[right_mask] - k_max_obs)
#             clipped_right = (w_right <= min_total_variance)
#             w_right = np.clip(w_right, min_total_variance, max_total_variance)
#             iv_grid[right_mask] = np.sqrt(w_right / T)
#             flag_wing_clipped[right_mask] = clipped_right

#     # Protección final por si hubiera ruido numérico dentro del rango observado
#     iv_grid = np.clip(iv_grid, min_iv, max_iv)
#     total_variance_grid = iv_grid**2 * T

#     smile = pd.DataFrame({
#         "moneyness": m_grid,
#         "log_moneyness": k_grid,
#         "implied_vol": iv_grid,
#         "total_variance": total_variance_grid,
#         "m_obs_min": m_min_obs,
#         "m_obs_max": m_max_obs,
#         "k_obs_min": k_min_obs,
#         "k_obs_max": k_max_obs,
#         "flag_inside_observed_range": flag_inside,
#         "flag_wing_clipped": flag_wing_clipped,
#     })

#     return smile



def interpolate_smile_slice(
    df_slice: pd.DataFrame,
    n_grid: int = 100,
    extrapolation_method: Literal["observed", "linear", "flat"] = "flat",
    global_moneyness_min: float = 0.3,
    global_moneyness_max: float = 1.7,
    min_nodes: int = 5,
    min_iv: float = 0.03,
    max_iv: float = 5.00,
    bc_type: str = "natural",
    use_log_moneyness: bool = True,
) -> pd.DataFrame | None:
    """
    Interpola una smile de IV en función de moneyness.

    Parámetro principal: use_log_moneyness
    ---------------------------------------
    True  (default): spline sobre IV(k) con k = log(K/F), grilla uniforme en k.
                     Asintóticamente más estable. Spacing variable en K.

    False           : spline sobre IV(m) con m = K/F, grilla uniforme en m.
                      Spacing uniforme en K → segunda diferencia más estable
                      para cálculo de gamma numérico.

    Parámetro: extrapolation_method
    --------------------------------
    "observed" : grilla limitada al rango observado. Sin extrapolación.
    "linear"   : extrapolación lineal en varianza total w = sigma^2 * T
                 con pendiente analítica del spline. Pendientes clippeadas
                 para garantizar monotonicidad de w (Lee 2004).
    "flat"     : extrapolación flat en varianza total. Garantiza butterfly
                 no-arbitraje trivialmente: g(k)=1>0 (Gatheral-Jacquier 2014).

    Parameters
    ----------
    df_slice             : columnas requeridas: Moneyness, ImpliedVolatility, Days
    n_grid               : puntos del grid
    extrapolation_method : "observed" | "linear" | "flat"
    global_moneyness_min : extremo inferior del grid global
    global_moneyness_max : extremo superior del grid global
    min_nodes            : nodos mínimos para interpolar
    min_iv               : suelo económico de IV (default 3%)
    max_iv               : techo defensivo de IV (default 500%)
    bc_type              : condición de frontera CubicSpline
                           "natural"    → w''=0 en extremos (conservador)
                           "not-a-knot" → tercera derivada continua en nodos interiores
    use_log_moneyness    : True → spline en IV(k), grilla en k
                           False → spline en IV(m), grilla en m

    Returns
    -------
    pd.DataFrame con columnas:
        moneyness, log_moneyness, implied_vol, total_variance,
        m_obs_min, m_obs_max, k_obs_min, k_obs_max,
        flag_inside_observed_range, flag_wing_clipped
    None si no hay suficientes datos.
    """
    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------
    required_cols = {"Moneyness", "ImpliedVolatility", "Days"}
    missing = required_cols - set(df_slice.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    if extrapolation_method not in ("observed", "linear", "flat"):
        raise ValueError(
            f"extrapolation_method debe ser 'observed', 'linear' o 'flat'. "
            f"Recibido: '{extrapolation_method}'"
        )

    df = df_slice.copy()
    df = df[np.isfinite(df["Moneyness"]) & np.isfinite(df["ImpliedVolatility"])]
    df = df[(df["Moneyness"] > 0) & (df["ImpliedVolatility"] > 0)]

    if df.empty:
        return None

    # Agrupar duplicados de moneyness
    df = (
        df.groupby("Moneyness", as_index=False)
          .agg({"ImpliedVolatility": "median", "Days": "first"})
          .sort_values("Moneyness")
          .reset_index(drop=True)
    )

    m  = df["Moneyness"].to_numpy(dtype=float)
    iv = df["ImpliedVolatility"].to_numpy(dtype=float)
    k  = np.log(m)

    n_left  = np.sum(k < 0.0)
    n_right = np.sum(k > 0.0)

    if len(k) < min_nodes or n_left < 2 or n_right < 2:
        return None

    T = float(df["Days"].iloc[0]) / 365.0
    if not np.isfinite(T) or T <= 0:
        return None

    min_total_variance = (min_iv ** 2) * T
    max_total_variance = (max_iv ** 2) * T

    k_min_obs = float(k.min())
    k_max_obs = float(k.max())
    m_min_obs = float(m.min())
    m_max_obs = float(m.max())

    # ------------------------------------------------------------------
    # Spline: en k (log-moneyness) o en m (moneyness lineal)
    # ------------------------------------------------------------------
    if use_log_moneyness:
        x_nodes = k          # nodos en log-moneyness
        x_min_obs = k_min_obs
        x_max_obs = k_max_obs
    else:
        x_nodes = m          # nodos en moneyness lineal
        x_min_obs = m_min_obs
        x_max_obs = m_max_obs

    interp = CubicSpline(x_nodes, iv, bc_type=bc_type, extrapolate=False)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    if extrapolation_method == "observed":
        x_grid = np.linspace(x_min_obs, x_max_obs, n_grid)
    else:
        if global_moneyness_min <= 0 or global_moneyness_max <= 0:
            raise ValueError("Los extremos globales de moneyness deben ser > 0.")
        if global_moneyness_min >= global_moneyness_max:
            raise ValueError("global_moneyness_min debe ser menor que global_moneyness_max.")

        if use_log_moneyness:
            x_global_min = np.log(global_moneyness_min)
            x_global_max = np.log(global_moneyness_max)
        else:
            x_global_min = global_moneyness_min
            x_global_max = global_moneyness_max

        if x_global_min > x_min_obs or x_global_max < x_max_obs:
            raise ValueError(
                "El grid global no contiene completamente el rango observado. "
                f"Observado: [{m_min_obs:.4f}, {m_max_obs:.4f}] | "
                f"Global: [{global_moneyness_min:.4f}, {global_moneyness_max:.4f}]"
            )

        x_grid = np.linspace(x_global_min, x_global_max, n_grid)

    # Grilla siempre en ambos espacios
    if use_log_moneyness:
        k_grid = x_grid
        m_grid = np.exp(k_grid)
    else:
        m_grid = x_grid
        k_grid = np.log(m_grid)

    # Interpolación interior
    iv_grid = interp(x_grid)

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------
    if use_log_moneyness:
        flag_inside = (k_grid >= k_min_obs) & (k_grid <= k_max_obs)
    else:
        flag_inside = (m_grid >= m_min_obs) & (m_grid <= m_max_obs)

    flag_wing_clipped = np.zeros(len(x_grid), dtype=bool)

    # ------------------------------------------------------------------
    # Extrapolación en alas
    # ------------------------------------------------------------------
    if extrapolation_method in ("linear", "flat"):
        w_obs = iv ** 2 * T

        if use_log_moneyness:
            left_mask  = k_grid < k_min_obs
            right_mask = k_grid > k_max_obs
        else:
            left_mask  = m_grid < m_min_obs
            right_mask = m_grid > m_max_obs

        if extrapolation_method == "flat":
            # ----------------------------------------------------------
            # FLAT en varianza total — g(k)=1>0, butterfly free trivial
            # ----------------------------------------------------------
            if np.any(left_mask):
                w_left = np.full(left_mask.sum(), w_obs[0])
                w_left = np.clip(w_left, min_total_variance, max_total_variance)
                iv_grid[left_mask] = np.sqrt(w_left / T)

            if np.any(right_mask):
                w_right = np.full(right_mask.sum(), w_obs[-1])
                w_right = np.clip(w_right, min_total_variance, max_total_variance)
                iv_grid[right_mask] = np.sqrt(w_right / T)

        else:
            # ----------------------------------------------------------
            # LINEAL en varianza total con pendiente analítica del spline
            # Conversión de dIV/dx a dw/dx:
            #   use_log_moneyness: dw/dk = 2*sigma*T * dIV/dk
            #   use_log_moneyness=False: dw/dm = 2*sigma*T * dIV/dm
            # Clipping de pendientes para monotonicidad de w en alas
            # ----------------------------------------------------------
            div_dx_left  = float(interp(x_min_obs, 1))
            div_dx_right = float(interp(x_max_obs, 1))

            slope_left_w  = min(2.0 * iv[0]  * T * div_dx_left,  0.0)
            slope_right_w = max(2.0 * iv[-1] * T * div_dx_right, 0.0)

            if np.any(left_mask):
                w_left = w_obs[0] + slope_left_w * (x_grid[left_mask] - x_min_obs)
                clipped_left = w_left <= min_total_variance
                w_left = np.clip(w_left, min_total_variance, max_total_variance)
                iv_grid[left_mask] = np.sqrt(w_left / T)
                flag_wing_clipped[left_mask] = clipped_left

            if np.any(right_mask):
                w_right = w_obs[-1] + slope_right_w * (x_grid[right_mask] - x_max_obs)
                clipped_right = w_right <= min_total_variance
                w_right = np.clip(w_right, min_total_variance, max_total_variance)
                iv_grid[right_mask] = np.sqrt(w_right / T)
                flag_wing_clipped[right_mask] = clipped_right

    # Suelo/techo final
    iv_grid = np.clip(iv_grid, min_iv, max_iv)

    return pd.DataFrame({
        "moneyness":                  m_grid,
        "log_moneyness":              k_grid,
        "implied_vol":                iv_grid,
        "total_variance":             iv_grid ** 2 * T,
        "m_obs_min":                  m_min_obs,
        "m_obs_max":                  m_max_obs,
        "k_obs_min":                  k_min_obs,
        "k_obs_max":                  k_max_obs,
        "flag_inside_observed_range": flag_inside,
        "flag_wing_clipped":          flag_wing_clipped,
    })



####################################
# Nueva versión con Shimko
####################################

def interpolate_smile_slice(
    df_slice: pd.DataFrame,
    n_grid: int = 100,
    extrapolation_method: Literal["observed", "linear", "flat"] = "flat",
    global_moneyness_min: float = 0.3,
    global_moneyness_max: float = 1.7,
    min_nodes: int = 5,
    min_iv: float = 0.03,
    max_iv: float = 5.00,
    bc_type: str = "natural",
    use_log_moneyness: bool = True,
    smoothing_method: Literal["spline", "shimko"] = "spline",
    shimko_degree: int = 4,
) -> pd.DataFrame | None:
    """
    Interpola una smile de IV en función de moneyness.

    Parámetro principal: smoothing_method
    ---------------------------------------
    "spline" (default):
        CubicSpline sobre IV(k) o IV(m) según use_log_moneyness.
        Interpolación exacta en los nodos observados.

    "shimko":
        Ajuste polinomial OLS de grado shimko_degree sobre IV(m)
        en la zona observada únicamente (Shimko 1993).
        Fuera del rango observado: extrapolación flat en varianza
        total w = sigma^2 * T (Gatheral-Jacquier 2014).

        Ventajas sobre spline:
        - Suavizado (no interpola exactamente en cada nodo) → menos
          sensible a outliers de IV en strikes poco líquidos
        - Derivadas analíticas exactas: dsigma/dm = poly'(m)/1
          disponibles desde los coeficientes → gamma analítica
          en greeks.py sin diferencias finitas

        Nota: use_log_moneyness se ignora cuando smoothing_method="shimko"
        porque Shimko (1993) trabaja en moneyness lineal m = K/F.

    Parámetro: extrapolation_method
    --------------------------------
    Solo aplica cuando smoothing_method="spline".
    Con smoothing_method="shimko" la extrapolación es siempre flat.

    Parameters
    ----------
    df_slice             : columnas requeridas: Moneyness, ImpliedVolatility, Days
    n_grid               : puntos del grid
    extrapolation_method : "observed" | "linear" | "flat" (solo para spline)
    global_moneyness_min : extremo inferior del grid global
    global_moneyness_max : extremo superior del grid global
    min_nodes            : nodos mínimos para interpolar
    min_iv               : suelo económico de IV (default 3%)
    max_iv               : techo defensivo de IV (default 500%)
    bc_type              : condición de frontera CubicSpline (solo para spline)
    use_log_moneyness    : True → spline en IV(k); False → spline en IV(m)
                           ignorado cuando smoothing_method="shimko"
    smoothing_method     : "spline" | "shimko"
    shimko_degree        : grado del polinomio Shimko (default 4)

    Returns
    -------
    pd.DataFrame con columnas:
        moneyness, log_moneyness, implied_vol, total_variance,
        m_obs_min, m_obs_max, k_obs_min, k_obs_max,
        flag_inside_observed_range, flag_wing_clipped,
        -- solo con shimko --
        shimko_coef       : coeficientes del polinomio [a0, a1, ..., an]
        dsigma_dm         : d_sigma/dm analítica desde el polinomio
        d2sigma_dm2       : d²_sigma/dm² analítica desde el polinomio
    None si no hay suficientes datos.
    """
    # ------------------------------------------------------------------
    # Validación
    # ------------------------------------------------------------------
    required_cols = {"Moneyness", "ImpliedVolatility", "Days"}
    missing = required_cols - set(df_slice.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    if smoothing_method not in ("spline", "shimko"):
        raise ValueError(
            f"smoothing_method debe ser 'spline' o 'shimko'. "
            f"Recibido: '{smoothing_method}'"
        )

    if extrapolation_method not in ("observed", "linear", "flat"):
        raise ValueError(
            f"extrapolation_method debe ser 'observed', 'linear' o 'flat'. "
            f"Recibido: '{extrapolation_method}'"
        )

    df = df_slice.copy()
    df = df[np.isfinite(df["Moneyness"]) & np.isfinite(df["ImpliedVolatility"])]
    df = df[(df["Moneyness"] > 0) & (df["ImpliedVolatility"] > 0)]

    if df.empty:
        return None

    # Agrupar duplicados de moneyness
    df = (
        df.groupby("Moneyness", as_index=False)
          .agg({"ImpliedVolatility": "median", "Days": "first"})
          .sort_values("Moneyness")
          .reset_index(drop=True)
    )

    m  = df["Moneyness"].to_numpy(dtype=float)
    iv = df["ImpliedVolatility"].to_numpy(dtype=float)
    k  = np.log(m)

    n_left  = np.sum(k < 0.0)
    n_right = np.sum(k > 0.0)

    if len(k) < min_nodes or n_left < 2 or n_right < 2:
        return None

    T = float(df["Days"].iloc[0]) / 365.0
    if not np.isfinite(T) or T <= 0:
        return None

    min_total_variance = (min_iv ** 2) * T
    max_total_variance = (max_iv ** 2) * T

    k_min_obs = float(k.min())
    k_max_obs = float(k.max())
    m_min_obs = float(m.min())
    m_max_obs = float(m.max())

    # ------------------------------------------------------------------
    # Grid global
    # ------------------------------------------------------------------
    if global_moneyness_min <= 0 or global_moneyness_max <= 0:
        raise ValueError("Los extremos globales de moneyness deben ser > 0.")
    if global_moneyness_min >= global_moneyness_max:
        raise ValueError("global_moneyness_min debe ser menor que global_moneyness_max.")

    # ------------------------------------------------------------------
    # RAMA SHIMKO
    # ------------------------------------------------------------------
    if smoothing_method == "shimko":
        return _shimko_smile(
            m=m, iv=iv, k=k, T=T,
            m_min_obs=m_min_obs, m_max_obs=m_max_obs,
            k_min_obs=k_min_obs, k_max_obs=k_max_obs,
            global_moneyness_min=global_moneyness_min,
            global_moneyness_max=global_moneyness_max,
            n_grid=n_grid,
            shimko_degree=shimko_degree,
            min_iv=min_iv, max_iv=max_iv,
            min_total_variance=min_total_variance,
            max_total_variance=max_total_variance,
        )

    # ------------------------------------------------------------------
    # RAMA SPLINE (comportamiento original)
    # ------------------------------------------------------------------
    if use_log_moneyness:
        x_nodes   = k
        x_min_obs = k_min_obs
        x_max_obs = k_max_obs
    else:
        x_nodes   = m
        x_min_obs = m_min_obs
        x_max_obs = m_max_obs

    interp = CubicSpline(x_nodes, iv, bc_type=bc_type, extrapolate=False)

    if extrapolation_method == "observed":
        x_grid = np.linspace(x_min_obs, x_max_obs, n_grid)
    else:
        if use_log_moneyness:
            x_global_min = np.log(global_moneyness_min)
            x_global_max = np.log(global_moneyness_max)
        else:
            x_global_min = global_moneyness_min
            x_global_max = global_moneyness_max

        if x_global_min > x_min_obs or x_global_max < x_max_obs:
            raise ValueError(
                "El grid global no contiene completamente el rango observado. "
                f"Observado: [{m_min_obs:.4f}, {m_max_obs:.4f}] | "
                f"Global: [{global_moneyness_min:.4f}, {global_moneyness_max:.4f}]"
            )
        x_grid = np.linspace(x_global_min, x_global_max, n_grid)

    if use_log_moneyness:
        k_grid = x_grid
        m_grid = np.exp(k_grid)
    else:
        m_grid = x_grid
        k_grid = np.log(m_grid)

    iv_grid = interp(x_grid)

    if use_log_moneyness:
        flag_inside = (k_grid >= k_min_obs) & (k_grid <= k_max_obs)
    else:
        flag_inside = (m_grid >= m_min_obs) & (m_grid <= m_max_obs)

    flag_wing_clipped = np.zeros(len(x_grid), dtype=bool)

    if extrapolation_method in ("linear", "flat"):
        w_obs      = iv ** 2 * T
        left_mask  = ~flag_inside & (m_grid < m_min_obs)
        right_mask = ~flag_inside & (m_grid > m_max_obs)

        if extrapolation_method == "flat":
            if np.any(left_mask):
                w_left = np.clip(
                    np.full(left_mask.sum(), w_obs[0]),
                    min_total_variance, max_total_variance
                )
                iv_grid[left_mask] = np.sqrt(w_left / T)
            if np.any(right_mask):
                w_right = np.clip(
                    np.full(right_mask.sum(), w_obs[-1]),
                    min_total_variance, max_total_variance
                )
                iv_grid[right_mask] = np.sqrt(w_right / T)
        else:
            div_dx_left  = float(interp(x_min_obs, 1))
            div_dx_right = float(interp(x_max_obs, 1))
            slope_left_w  = min(2.0 * iv[0]  * T * div_dx_left,  0.0)
            slope_right_w = max(2.0 * iv[-1] * T * div_dx_right, 0.0)

            if np.any(left_mask):
                w_left = w_obs[0] + slope_left_w * (x_grid[left_mask] - x_min_obs)
                clipped = w_left <= min_total_variance
                w_left  = np.clip(w_left, min_total_variance, max_total_variance)
                iv_grid[left_mask]         = np.sqrt(w_left / T)
                flag_wing_clipped[left_mask] = clipped

            if np.any(right_mask):
                w_right = w_obs[-1] + slope_right_w * (x_grid[right_mask] - x_max_obs)
                clipped  = w_right <= min_total_variance
                w_right  = np.clip(w_right, min_total_variance, max_total_variance)
                iv_grid[right_mask]         = np.sqrt(w_right / T)
                flag_wing_clipped[right_mask] = clipped

    iv_grid = np.clip(iv_grid, min_iv, max_iv)

    return pd.DataFrame({
        "moneyness":                  m_grid,
        "log_moneyness":              k_grid,
        "implied_vol":                iv_grid,
        "total_variance":             iv_grid ** 2 * T,
        "m_obs_min":                  m_min_obs,
        "m_obs_max":                  m_max_obs,
        "k_obs_min":                  k_min_obs,
        "k_obs_max":                  k_max_obs,
        "flag_inside_observed_range": flag_inside,
        "flag_wing_clipped":          flag_wing_clipped,
    })


# =============================================================================
# Rama Shimko
# =============================================================================

def _shimko_smile(
    m: np.ndarray,
    iv: np.ndarray,
    k: np.ndarray,
    T: float,
    m_min_obs: float,
    m_max_obs: float,
    k_min_obs: float,
    k_max_obs: float,
    global_moneyness_min: float,
    global_moneyness_max: float,
    n_grid: int,
    shimko_degree: int,
    min_iv: float,
    max_iv: float,
    min_total_variance: float,
    max_total_variance: float,
) -> pd.DataFrame | None:
    """
    Ajuste polinomial Shimko (1993) sobre IV(m) en zona observada.
    Extrapolación flat en varianza total fuera del rango observado.

    sigma(m) = sum_{j=0}^{d} a_j * m^j

    Derivadas analíticas:
        d_sigma/dm   = sum_{j=1}^{d} j * a_j * m^{j-1}
        d²_sigma/dm² = sum_{j=2}^{d} j*(j-1) * a_j * m^{j-2}

    Estas derivadas permiten calcular gamma analíticamente en greeks.py
    sin diferencias finitas.
    """
    # ------------------------------------------------------------------
    # Ajuste OLS del polinomio sobre zona observada
    # ------------------------------------------------------------------
    # Centrar moneyness en ATM (m=1) para mejorar el condicionamiento
    # numérico del sistema de ecuaciones normales
    m_center = 1.0
    m_c = m - m_center

    try:
        coef = np.polyfit(m_c, iv, shimko_degree)  # coef[0] es el de mayor grado
    except (np.linalg.LinAlgError, ValueError):
        return None

    # Verificar que el ajuste es razonable
    iv_fitted = np.polyval(coef, m_c)
    rmse_fit  = np.sqrt(np.mean((iv_fitted - iv)**2))

    # Si el RMSE del ajuste es muy alto (>5 vol points), el polinomio
    # no está capturando bien la smile — devolver None
    if rmse_fit > 0.05:
        return None

    # ------------------------------------------------------------------
    # Grid global en moneyness lineal
    # ------------------------------------------------------------------
    m_grid = np.linspace(global_moneyness_min, global_moneyness_max, n_grid)
    k_grid = np.log(m_grid)
    m_c_grid = m_grid - m_center

    # ------------------------------------------------------------------
    # Evaluar polinomio en toda la grilla
    # ------------------------------------------------------------------
    iv_grid = np.polyval(coef, m_c_grid)

    # ------------------------------------------------------------------
    # Flag zona observada
    # ------------------------------------------------------------------
    flag_inside      = (m_grid >= m_min_obs) & (m_grid <= m_max_obs)
    flag_wing_clipped = np.zeros(n_grid, dtype=bool)

    # ------------------------------------------------------------------
    # Extrapolación flat en varianza total fuera del rango observado
    # El polinomio puede divergir fuera de la zona de ajuste —
    # sustituimos por flat en w para garantizar no-arbitraje en las alas
    # ------------------------------------------------------------------
    w_obs = iv ** 2 * T
    w_at_left  = float(np.polyval(coef, m_min_obs - m_center))**2 * T
    w_at_right = float(np.polyval(coef, m_max_obs - m_center))**2 * T

    # Usar el mínimo entre el valor del polinomio en el borde y w observada
    # para evitar discontinuidades en la unión
    w_left_flat  = np.clip(w_at_left,  min_total_variance, max_total_variance)
    w_right_flat = np.clip(w_at_right, min_total_variance, max_total_variance)

    left_mask  = ~flag_inside & (m_grid < m_min_obs)
    right_mask = ~flag_inside & (m_grid > m_max_obs)

    if np.any(left_mask):
        iv_grid[left_mask] = np.sqrt(w_left_flat / T)

    if np.any(right_mask):
        iv_grid[right_mask] = np.sqrt(w_right_flat / T)

    # ------------------------------------------------------------------
    # Suelo/techo de IV
    # ------------------------------------------------------------------
    iv_grid = np.clip(iv_grid, min_iv, max_iv)

    # ------------------------------------------------------------------
    # Derivadas analíticas del polinomio en toda la grilla
    # Solo válidas dentro del rango observado — fuera son las derivadas
    # del flat (cero) pero las calculamos igualmente para consistencia
    # ------------------------------------------------------------------
    # Derivada del polinomio: si p(x) = sum a_j x^j
    # p'(x) = sum j * a_j * x^{j-1}  → np.polyder
    coef_d1 = np.polyder(coef, 1)   # primera derivada
    coef_d2 = np.polyder(coef, 2)   # segunda derivada

    dsigma_dm   = np.polyval(coef_d1, m_c_grid)
    d2sigma_dm2 = np.polyval(coef_d2, m_c_grid)

    # Fuera del rango observado, derivadas = 0 (flat)
    dsigma_dm[left_mask | right_mask]   = 0.0
    d2sigma_dm2[left_mask | right_mask] = 0.0

    return pd.DataFrame({
        "moneyness":                  m_grid,
        "log_moneyness":              k_grid,
        "implied_vol":                iv_grid,
        "total_variance":             iv_grid ** 2 * T,
        "m_obs_min":                  m_min_obs,
        "m_obs_max":                  m_max_obs,
        "k_obs_min":                  k_min_obs,
        "k_obs_max":                  k_max_obs,
        "flag_inside_observed_range": flag_inside,
        "flag_wing_clipped":          flag_wing_clipped,
        "shimko_rmse":                rmse_fit,
        "dsigma_dm":                  dsigma_dm,
        "d2sigma_dm2":                d2sigma_dm2,
    })
# In[]:

surface_slices = []

for (date, expiry), df_slice in quoted_option.groupby(["Date", "Expiration"]):

    smile = interpolate_smile_slice(
    df_slice,
    n_grid=200,
    smoothing_method="shimko",   # ← activa Shimko
    shimko_degree=4,
    global_moneyness_min=0.3,
    global_moneyness_max=1.7,
)

#     smile = interpolate_smile_slice(
#     df_slice,
#     n_grid=300,
#     extrapolation_method="linear",   #  "flat ", "linear" o "observed"
#     use_log_moneyness=True
# )

    if smile is None:
        continue

    F = df_slice["forward_index"].iloc[0]
    r = df_slice["Rate"].iloc[0]
    T_days = df_slice["Days"].iloc[0]
    T = T_days / 365.0

    smile["Date"] = date
    smile["Expiration"] = expiry
    smile["Days"] = T_days
    smile["T"] = T
    smile["forward"] = F
    smile["rate"] = r
    smile["discount_factor"] = np.exp(-r * T)

    smile["Strike"] = smile["moneyness"] * smile["forward"]
    smile["CallPut"] = np.where(smile["moneyness"] < 1, "P", "C")
    smile.loc[np.isclose(smile["moneyness"], 1.0), "CallPut"] = "ATM"

    surface_slices.append(smile)


# In[]:


x = pd.concat(surface_slices, ignore_index=True)

PARQET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\volatility_surface_shimko.parquet"
duckdb.from_df(x).write_parquet(PARQET_OUTPUT, compression='snappy')


print("Generada la superficie de volatilidad con éxito")






























# def interpolate_smile_slice_1(df_slice, n_grid=50):

#     df_slice = df_slice.sort_values("Moneyness")

#     k = df_slice["Moneyness"].values
#     iv = df_slice["ImpliedVolatility"].values

#     # eliminar duplicados si los hubiera
#     uniq = ~pd.Series(k).duplicated().values
#     k = k[uniq]
#     iv = iv[uniq]

#     if len(k) < 4:
#         return None

#     # interpolador PCHIP
#     interp = PchipInterpolator(k, iv)

#     k_min = k.min()
#     k_max = k.max()

#     k_grid = np.linspace(k_min, k_max, n_grid)

#     iv_grid = interp(k_grid)

#     return pd.DataFrame({
#         "moneyness": k_grid,
#         "implied_vol": iv_grid
#     })



# def interpolate_smile_slice_2(
#     df_slice: pd.DataFrame,
#     n_grid: int = 50,
#     use_observed_grid: bool = True,
#     global_moneyness_min: float = 0.1,
#     global_moneyness_max: float = 3.0,
#     min_nodes: int = 4,
#     min_total_variance: float = 1e-12
# ) -> pd.DataFrame | None:
#     """
#     Interpola una smile de IV en función de moneyness.

#     - Dentro del rango observado: PCHIP
#     - Fuera del rango observado (solo si use_observed_grid=False):
#       extrapolación lineal en varianza total w = sigma^2 * T

#     Parameters
#     ----------
#     df_slice : pd.DataFrame
#         Slice de opciones para una fecha/vencimiento.
#         Debe contener al menos:
#         - Moneyness
#         - ImpliedVolatility
#         - Days

#     n_grid : int, default 50
#         Número de puntos del grid de interpolación.

#     use_observed_grid : bool, default True
#         Si True, usa [min(Moneyness observado), max(Moneyness observado)].
#         Si False, usa [global_moneyness_min, global_moneyness_max].

#     global_moneyness_min : float, default 0.1
#         Extremo inferior del grid homogéneo global.

#     global_moneyness_max : float, default 3.0
#         Extremo superior del grid homogéneo global.

#     min_nodes : int, default 4
#         Número mínimo de nodos distintos de moneyness para interpolar.

#     min_total_variance : float, default 1e-12
#         Suelo numérico para evitar varianza total negativa o nula al extrapolar.

#     Returns
#     -------
#     pd.DataFrame | None
#     """
#     df_slice = df_slice.sort_values("Moneyness").copy()

#     m = df_slice["Moneyness"].to_numpy(dtype=float)

#     n_left = (m < 1).sum()
#     n_right = (m > 1).sum()


#     iv = df_slice["ImpliedVolatility"].to_numpy(dtype=float)

#     # Eliminar duplicados exactos de moneyness conservando el primero
#     uniq = ~pd.Series(m).duplicated().to_numpy()
#     m = m[uniq]
#     iv = iv[uniq]

#     if len(m) < min_nodes or n_left < 2 or n_right < 2:
#         return None

#     m_min_obs = float(np.min(m))
#     m_max_obs = float(np.max(m))

#     # Tiempo en años para la extrapolación en varianza total
#     T = float(df_slice["Days"].iloc[0]) / 365.0
#     if T <= 0:
#         return None

#     # PCHIP SOLO dentro del rango observado
#     interp = PchipInterpolator(m, iv, extrapolate=False)

#     # Grid
#     if use_observed_grid:
#         m_grid = np.linspace(m_min_obs, m_max_obs, n_grid)
#     else:
#         if global_moneyness_min <= 0 or global_moneyness_max <= 0:
#             raise ValueError("Los extremos del grid global de moneyness deben ser positivos.")
#         if global_moneyness_min >= global_moneyness_max:
#             raise ValueError("global_moneyness_min debe ser menor que global_moneyness_max.")
#         if not (global_moneyness_min <= m_min_obs and m_max_obs <= global_moneyness_max):
#             raise ValueError(
#                 f"El rango observado [{m_min_obs:.4f}, {m_max_obs:.4f}] "
#                 f"no está contenido en el grid global "
#                 f"[{global_moneyness_min:.4f}, {global_moneyness_max:.4f}]."
#             )

#         m_grid = np.linspace(global_moneyness_min, global_moneyness_max, n_grid)

#     # Interpolación interior
#     iv_grid = interp(m_grid)

# # FUTURA EXTRAPOLACIÓN EN WINGS (DE MOMENTO DESACTIVADA)
#     # Extrapolación lineal en varianza total si el grid es global
#     if not use_observed_grid:
#         w = (iv ** 2) * T

#         # Pendiente izquierda en varianza total
#         slope_left = (w[1] - w[0]) / (m[1] - m[0])

#         # Pendiente derecha en varianza total
#         slope_right = (w[-1] - w[-2]) / (m[-1] - m[-2])

#         left_mask = m_grid < m_min_obs
#         right_mask = m_grid > m_max_obs

#         # Ala izquierda
#         if np.any(left_mask):
#             w_left_ext = w[0] + slope_left * (m_grid[left_mask] - m[0])
#             w_left_ext = np.maximum(w_left_ext, min_total_variance)
#             iv_grid[left_mask] = np.sqrt(w_left_ext / T)

#         # Ala derecha
#         if np.any(right_mask):
#             w_right_ext = w[-1] + slope_right * (m_grid[right_mask] - m[-1])
#             w_right_ext = np.maximum(w_right_ext, min_total_variance)
#             iv_grid[right_mask] = np.sqrt(w_right_ext / T)

#     smile = pd.DataFrame({
#         "moneyness": m_grid,
#         "implied_vol": iv_grid
#     })

#     # Metadatos
#     smile["m_obs_min"] = m_min_obs
#     smile["m_obs_max"] = m_max_obs
#     smile["flag_inside_observed_range"] = (
#         (smile["moneyness"] >= m_min_obs) &
#         (smile["moneyness"] <= m_max_obs)
#     )

#     return smile


# def interpolate_smile_slice_A(
#     df_slice: pd.DataFrame,
#     n_grid: int = 100,
#     use_observed_grid: bool = True,
#     global_moneyness_min: float = 0.1,
#     global_moneyness_max: float = 3.0,
#     min_nodes: int = 4,
#     min_total_variance: float = 1e-12,
# ) -> pd.DataFrame | None:
#     """
#     Interpola una smile de IV en función de moneyness (K/F).

#     - Dentro del rango observado: CubicSpline (C², segunda derivada continua)
#     - Fuera del rango observado (solo si use_observed_grid=False):
#       extrapolación lineal en varianza total w = sigma^2 * T,
#       usando la pendiente del spline en el extremo observado
#       (convertida de dIV/dm a dw/dm), con suelo en min_total_variance.

#     Parameters
#     ----------
#     df_slice : pd.DataFrame
#         Slice de opciones para una fecha/vencimiento.
#         Debe contener al menos:
#         - Moneyness      : K/F
#         - ImpliedVolatility
#         - Days

#     n_grid : int, default 50
#         Número de puntos del grid de interpolación.

#     use_observed_grid : bool, default True
#         Si True,  usa [min(Moneyness observado), max(Moneyness observado)].
#         Si False, usa [global_moneyness_min, global_moneyness_max].

#     global_moneyness_min : float, default 0.1
#     global_moneyness_max : float, default 3.0
#         Extremos del grid homogéneo global (solo si use_observed_grid=False).

#     min_nodes : int, default 4
#         Número mínimo de nodos distintos de moneyness para interpolar.

#     min_total_variance : float, default 1e-12
#         Suelo numérico para evitar varianza total negativa o nula.

#     Returns
#     -------
#     pd.DataFrame con columnas:
#         moneyness, implied_vol, m_obs_min, m_obs_max,
#         flag_inside_observed_range
#     None si no hay suficientes datos.

#     Notes
#     -----
#     Cambios respecto a la versión anterior (PCHIP):
#     - CubicSpline (not-a-knot) sustituye a PCHIP → segunda derivada
#       continua en todos los nodos, necesaria para gamma via
#       Breeden-Litzenberger.
#     - La pendiente de extrapolación se obtiene de la derivada analítica
#       del spline en el extremo observado, no de la diferencia finita entre
#       los dos últimos puntos observados (que era sensible al ruido en los
#       strikes menos líquidos).
#     - La pendiente se convierte correctamente de dIV/dm a dw/dm:
#           dw/dm = 2 * sigma * T * (dIV/dm)
#     - Se mantiene moneyness K/F (no log-moneyness).
#     """
#     df_slice = df_slice.sort_values("Moneyness").copy()

#     m = df_slice["Moneyness"].to_numpy(dtype=float)
#     iv = df_slice["ImpliedVolatility"].to_numpy(dtype=float)

#     n_left  = (m < 1).sum()
#     n_right = (m > 1).sum()

#     # Eliminar duplicados exactos de moneyness conservando el primero
#     uniq_mask = ~pd.Series(m).duplicated().to_numpy()
#     m  = m[uniq_mask]
#     iv = iv[uniq_mask]

#     if len(m) < min_nodes or n_left < 2 or n_right < 2:
#         return None

#     m_min_obs = float(m.min())
#     m_max_obs = float(m.max())

#     T = float(df_slice["Days"].iloc[0]) / 365.0
#     if T <= 0:
#         return None

#     # ------------------------------------------------------------------
#     # CubicSpline (not-a-knot) sobre IV dentro del rango observado
#     # extrapolate=False → NaN fuera del rango, gestionamos nosotros
#     # ------------------------------------------------------------------
#     interp = CubicSpline(m, iv, bc_type="not-a-knot", extrapolate=False)

#     # Grid
#     if use_observed_grid:
#         m_grid = np.linspace(m_min_obs, m_max_obs, n_grid)
#     else:
#         if global_moneyness_min <= 0 or global_moneyness_max <= 0:
#             raise ValueError(
#                 "Los extremos del grid global de moneyness deben ser positivos."
#             )
#         if global_moneyness_min >= global_moneyness_max:
#             raise ValueError(
#                 "global_moneyness_min debe ser menor que global_moneyness_max."
#             )
#         m_grid = np.linspace(global_moneyness_min, global_moneyness_max, n_grid)

#     # Interpolación interior (NaN fuera del rango observado)
#     iv_grid = interp(m_grid)

#     # ------------------------------------------------------------------
#     # Extrapolación en varianza total (solo si grid global)
#     # ------------------------------------------------------------------
#     if not use_observed_grid:
#         w = iv**2 * T  # varianza total en nodos observados

#         # Pendiente del spline en los extremos (dIV/dm), analítica
#         div_dm_left  = float(interp(m_min_obs, 1))  # primera derivada en m_min
#         div_dm_right = float(interp(m_max_obs, 1))  # primera derivada en m_max

#         # Convertir a dw/dm = 2 * sigma * T * (dIV/dm)
        
#         slope_left_w  = 2.0 * iv[0]  * T * div_dm_left
#         slope_right_w = 2.0 * iv[-1] * T * div_dm_right

#         left_mask  = m_grid < m_min_obs
#         right_mask = m_grid > m_max_obs

#         # Ala izquierda
#         if np.any(left_mask):
#             w_left = w[0] + slope_left_w * (m_grid[left_mask] - m_min_obs)
#             w_left = np.maximum(w_left, min_total_variance)
#             iv_grid[left_mask] = np.sqrt(w_left / T)

#         # Ala derecha
#         if np.any(right_mask):
#             w_right = w[-1] + slope_right_w * (m_grid[right_mask] - m_max_obs)
#             w_right = np.maximum(w_right, min_total_variance)
#             iv_grid[right_mask] = np.sqrt(w_right / T)

#     smile = pd.DataFrame({
#         "moneyness":   m_grid,
#         "implied_vol": iv_grid,
#     })
# #      Flag cuando la extrapolación toca el suelo min_total_variance
# #      indica que la pendiente era tan negativa que w colapsó
#     smile["flag_wing_clipped"] = False
    
#     if not use_observed_grid:
#         flag_clipped = np.zeros(len(m_grid), dtype=bool)
#         if np.any(left_mask):
#             flag_clipped[left_mask]  = (w_left  <= min_total_variance * 10)
#         if np.any(right_mask):
#             flag_clipped[right_mask] = (w_right <= min_total_variance * 10)
#         smile["flag_wing_clipped"] = flag_clipped
#     else:
#         smile["flag_wing_clipped"] = False
#     smile["m_obs_min"] = m_min_obs
#     smile["m_obs_max"] = m_max_obs
#     smile["no_extrapolated_zone"] = (
#         (smile["moneyness"] >= m_min_obs) &
#         (smile["moneyness"] <= m_max_obs)
#     )

#     return smile











# In[]:


############ OLD #########


# surface_slices = []

# for (date, expiry), df_slice in quoted_option.groupby(["Date", "Expiration"]):

#     smile = interpolate_smile_slice(df_slice)

#     if smile is None:
#         continue

#     F = df_slice["forward_index"].iloc[0]
#     r = df_slice["Rate"].iloc[0]
#     T_days = df_slice["Days"].iloc[0]
#     T = T_days / 365.0

#     smile["Date"] = date
#     smile["Expiration"] = expiry
#     smile["Days"] = T_days
#     smile["T"] = T
#     smile["forward"] = F
#     smile["rate"] = r
#     smile["discount_factor"] = np.exp(-r * T)

#     smile["Strike"] = smile["moneyness"] * smile["forward"]
#     smile["log_moneyness"] = np.log(smile["moneyness"])
#     smile["CallPut"] = np.where(smile["moneyness"] < 1, "P", "C")

#     surface_slices.append(smile)

# # %%
# x = pd.concat(surface_slices, ignore_index=True)


# # %%
# print(quoted_option.columns)


# # %%
# print(x)
# # %%