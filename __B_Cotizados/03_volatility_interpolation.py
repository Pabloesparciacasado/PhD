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

# In[]:
# Función de interpolación cubic spline:


def interpolate_smile_slice(
    df_slice: pd.DataFrame,
    n_grid: int = 100,
    use_observed_grid: bool = True,
    global_moneyness_min: float = 0.5,
    global_moneyness_max: float = 1.5,
    min_nodes: int = 5,
    min_iv: float = 0.03,          # suelo económico de IV, no numérico
    max_iv: float = 5.00,          # opcional: techo defensivo
    bc_type: str = "natural"       # más estable en extremos que not-a-knot
) -> pd.DataFrame | None:
    """
    Interpola una smile de IV en función de moneyness (K/F),
    pero tratando las alas en log-moneyness k = log(K/F).

    Dentro del rango observado:
        - CubicSpline sobre IV(k)

    Fuera del rango observado (si use_observed_grid=False):
        - extrapolación lineal en varianza total w(k) = sigma(k)^2 * T
        - usando la derivada del spline en el extremo observado
        - con clipping de pendiente para evitar alas decrecientes
        - con suelo en IV mínima

    Devuelve:
        DataFrame con:
        moneyness, log_moneyness, implied_vol,
        total_variance, m_obs_min, m_obs_max,
        k_obs_min, k_obs_max,
        flag_inside_observed_range, flag_wing_clipped
    """

    required_cols = {"Moneyness", "ImpliedVolatility", "Days"}
    missing = required_cols - set(df_slice.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df = df_slice.copy()
    df = df[np.isfinite(df["Moneyness"])]
    df = df[np.isfinite(df["ImpliedVolatility"])]
    df = df[(df["Moneyness"] > 0) & (df["ImpliedVolatility"] > 0)]

    if df.empty:
        return None

    # Agrupar duplicados de moneyness de forma robusta
    df = (
        df.groupby("Moneyness", as_index=False)
          .agg({
              "ImpliedVolatility": "median",
              "Days": "first"
          })
          .sort_values("Moneyness")
          .reset_index(drop=True)
    )

    m = df["Moneyness"].to_numpy(dtype=float)
    iv = df["ImpliedVolatility"].to_numpy(dtype=float)

    # Pasamos a log-moneyness
    k = np.log(m)

    # Recuento de nodos a cada lado del ATM, ya sin duplicados
    n_left = np.sum(k < 0.0)
    n_right = np.sum(k > 0.0)

    if len(k) < min_nodes or n_left < 2 or n_right < 2:
        return None

    T = float(df["Days"].iloc[0]) / 365.0
    if not np.isfinite(T) or T <= 0:
        return None

    # Suelo de varianza total derivado de min_iv
    min_total_variance = (min_iv ** 2) * T
    max_total_variance = (max_iv ** 2) * T

    k_min_obs = float(k.min())
    k_max_obs = float(k.max())
    m_min_obs = float(m.min())
    m_max_obs = float(m.max())

    # Spline sobre IV(k) dentro del rango observado
    interp = CubicSpline(k, iv, bc_type=bc_type, extrapolate=False)

    # Grid
    if use_observed_grid:
        k_grid = np.linspace(k_min_obs, k_max_obs, n_grid)
    else:
        if global_moneyness_min <= 0 or global_moneyness_max <= 0:
            raise ValueError("Los extremos globales de moneyness deben ser > 0.")
        if global_moneyness_min >= global_moneyness_max:
            raise ValueError("global_moneyness_min debe ser menor que global_moneyness_max.")

        k_global_min = np.log(global_moneyness_min)
        k_global_max = np.log(global_moneyness_max)

        if k_global_min > k_min_obs or k_global_max < k_max_obs:
            raise ValueError(
                "El grid global no contiene completamente el rango observado. "
                f"Observado: [{m_min_obs:.4f}, {m_max_obs:.4f}] | "
                f"Global: [{global_moneyness_min:.4f}, {global_moneyness_max:.4f}]"
            )

        k_grid = np.linspace(k_global_min, k_global_max, n_grid)

    m_grid = np.exp(k_grid)

    # Interpolación interior
    iv_grid = interp(k_grid)

    # Flags
    flag_inside = (k_grid >= k_min_obs) & (k_grid <= k_max_obs)
    flag_wing_clipped = np.zeros_like(k_grid, dtype=bool)

    # Extrapolación en alas
    if not use_observed_grid:
        w_obs = iv**2 * T

        # Derivadas del spline de IV respecto a k
        div_dk_left = float(interp(k_min_obs, 1))
        div_dk_right = float(interp(k_max_obs, 1))

        # dw/dk = 2 * sigma * T * d(sigma)/dk
        slope_left_w = 2.0 * iv[0] * T * div_dk_left
        slope_right_w = 2.0 * iv[-1] * T * div_dk_right

        # Clipping de pendientes para evitar colapso de alas:
        # - ala izquierda: al moverte a k más pequeño, no queremos que w baje
        #   => slope_left_w NO debe ser positiva
        # - ala derecha: al moverte a k más grande, no queremos que w baje
        #   => slope_right_w NO debe ser negativa
        slope_left_w = min(slope_left_w, 0.0)
        slope_right_w = max(slope_right_w, 0.0)

        left_mask = k_grid < k_min_obs
        right_mask = k_grid > k_max_obs

        if np.any(left_mask):
            w_left = w_obs[0] + slope_left_w * (k_grid[left_mask] - k_min_obs)
            clipped_left = (w_left <= min_total_variance)
            w_left = np.clip(w_left, min_total_variance, max_total_variance)
            iv_grid[left_mask] = np.sqrt(w_left / T)
            flag_wing_clipped[left_mask] = clipped_left

        if np.any(right_mask):
            w_right = w_obs[-1] + slope_right_w * (k_grid[right_mask] - k_max_obs)
            clipped_right = (w_right <= min_total_variance)
            w_right = np.clip(w_right, min_total_variance, max_total_variance)
            iv_grid[right_mask] = np.sqrt(w_right / T)
            flag_wing_clipped[right_mask] = clipped_right

    # Protección final por si hubiera ruido numérico dentro del rango observado
    iv_grid = np.clip(iv_grid, min_iv, max_iv)
    total_variance_grid = iv_grid**2 * T

    smile = pd.DataFrame({
        "moneyness": m_grid,
        "log_moneyness": k_grid,
        "implied_vol": iv_grid,
        "total_variance": total_variance_grid,
        "m_obs_min": m_min_obs,
        "m_obs_max": m_max_obs,
        "k_obs_min": k_min_obs,
        "k_obs_max": k_max_obs,
        "flag_inside_observed_range": flag_inside,
        "flag_wing_clipped": flag_wing_clipped,
    })

    return smile



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
) -> pd.DataFrame | None:
    """
    Interpola una smile de IV en función de log-moneyness k = log(K/F).

    Parámetro principal: extrapolation_method
    -----------------------------------------
    "observed" : grid limitado al rango observado [k_min_obs, k_max_obs].
                 Solo interpolación, sin extrapolación.
                 Útil como sanity check o cuando no se necesita
                 comparabilidad cross-sectional.

    "linear"   : grid global [log(global_moneyness_min), log(global_moneyness_max)].
                 Extrapolación lineal en varianza total w(k) = sigma^2 * T
                 más allá del rango observado, usando la derivada analítica
                 del spline en el extremo. Pendientes clippeadas para
                 garantizar monotonicidad de w en las alas
                 (condición necesaria para no-arbitraje local, Lee 2004).

    "flat"     : grid global [log(global_moneyness_min), log(global_moneyness_max)].
                 Extrapolación flat en varianza total: w(k) = w(k_obs_min)
                 para k < k_obs_min, w(k) = w(k_obs_max) para k > k_obs_max.
                 Garantiza butterfly no-arbitraje en las alas de forma trivial:
                 con w' = 0 y w'' = 0, la condición de Gatheral (2004)
                 g(k) = 1 > 0 se satisface en todo el dominio extrapolado.
                 Recomendado para cálculo de griegas y momentos BKM.

    Parameters
    ----------
    df_slice             : slice de opciones para una fecha/vencimiento.
                           Columnas requeridas: Moneyness, ImpliedVolatility, Days.
    n_grid               : número de puntos del grid.
    extrapolation_method : "observed" | "linear" | "flat"
    global_moneyness_min : extremo inferior del grid global (solo linear/flat).
    global_moneyness_max : extremo superior del grid global (solo linear/flat).
    min_nodes            : número mínimo de nodos distintos para interpolar.
    min_iv               : suelo económico de IV (default 3%).
    max_iv               : techo defensivo de IV (default 500%).
    bc_type              : condición de frontera del CubicSpline.
                           "natural"    → segunda derivada cero en extremos
                                          (conservador, estable en borders).
                           "not-a-knot" → tercera derivada continua en
                                          primeros y últimos nodos interiores.

    Returns
    -------
    pd.DataFrame con columnas:
        moneyness, log_moneyness, implied_vol, total_variance,
        m_obs_min, m_obs_max, k_obs_min, k_obs_max,
        flag_inside_observed_range, flag_wing_clipped
    None si no hay suficientes datos o el slice no pasa los filtros.
    """
    # ------------------------------------------------------------------
    # Validación y limpieza
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

    # Agrupar duplicados de moneyness de forma robusta
    df = (
        df.groupby("Moneyness", as_index=False)
          .agg({"ImpliedVolatility": "median", "Days": "first"})
          .sort_values("Moneyness")
          .reset_index(drop=True)
    )

    m  = df["Moneyness"].to_numpy(dtype=float)
    iv = df["ImpliedVolatility"].to_numpy(dtype=float)
    k  = np.log(m)  # log-moneyness

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
    # Spline sobre IV(k) dentro del rango observado
    # ------------------------------------------------------------------
    interp = CubicSpline(k, iv, bc_type=bc_type, extrapolate=False)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    if extrapolation_method == "observed":
        k_grid = np.linspace(k_min_obs, k_max_obs, n_grid)
    else:
        # "linear" o "flat" — grid global
        if global_moneyness_min <= 0 or global_moneyness_max <= 0:
            raise ValueError("Los extremos globales de moneyness deben ser > 0.")
        if global_moneyness_min >= global_moneyness_max:
            raise ValueError("global_moneyness_min debe ser menor que global_moneyness_max.")

        k_global_min = np.log(global_moneyness_min)
        k_global_max = np.log(global_moneyness_max)

        if k_global_min > k_min_obs or k_global_max < k_max_obs:
            raise ValueError(
                "El grid global no contiene completamente el rango observado. "
                f"Observado: [{m_min_obs:.4f}, {m_max_obs:.4f}] | "
                f"Global: [{global_moneyness_min:.4f}, {global_moneyness_max:.4f}]"
            )

        k_grid = np.linspace(k_global_min, k_global_max, n_grid)

    m_grid = np.exp(k_grid)

    # Interpolación interior
    iv_grid = interp(k_grid)

    # ------------------------------------------------------------------
    # Flags (se rellenan en el bloque de extrapolación si aplica)
    # ------------------------------------------------------------------
    flag_inside      = (k_grid >= k_min_obs) & (k_grid <= k_max_obs)
    flag_wing_clipped = np.zeros(len(k_grid), dtype=bool)

    # ------------------------------------------------------------------
    # Extrapolación en alas
    # ------------------------------------------------------------------
    if extrapolation_method in ("linear", "flat"):
        w_obs     = iv ** 2 * T
        left_mask  = k_grid < k_min_obs
        right_mask = k_grid > k_max_obs

        if extrapolation_method == "flat":
            # ----------------------------------------------------------
            # FLAT en varianza total
            # w(k) = w(k_obs_min) para k < k_obs_min
            # w(k) = w(k_obs_max) para k > k_obs_max
            #
            # Con w' = 0 y w'' = 0, la condición de Gatheral (2004):
            #   g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2
            # se reduce a g(k) = 1 > 0 en todo el dominio extrapolado.
            # Butterfly no-arbitraje garantizado de forma trivial.
            # ----------------------------------------------------------
            if np.any(left_mask):
                w_left = np.full(left_mask.sum(), w_obs[0])
                w_left = np.clip(w_left, min_total_variance, max_total_variance)
                iv_grid[left_mask] = np.sqrt(w_left / T)
                # flat nunca clippa por definición, pero flag = False

            if np.any(right_mask):
                w_right = np.full(right_mask.sum(), w_obs[-1])
                w_right = np.clip(w_right, min_total_variance, max_total_variance)
                iv_grid[right_mask] = np.sqrt(w_right / T)

        else:
            # ----------------------------------------------------------
            # LINEAL en varianza total
            # Pendiente analítica del spline en el extremo, convertida
            # de dIV/dk a dw/dk = 2 * sigma * T * dIV/dk.
            # Pendientes clippeadas para garantizar monotonicidad de w:
            #   - ala izquierda: slope <= 0  (w no crece hacia -inf)
            #   - ala derecha:   slope >= 0  (w no decrece hacia +inf)
            # ----------------------------------------------------------
            div_dk_left  = float(interp(k_min_obs, 1))
            div_dk_right = float(interp(k_max_obs, 1))

            slope_left_w  = min(2.0 * iv[0]  * T * div_dk_left,  0.0)
            slope_right_w = max(2.0 * iv[-1] * T * div_dk_right, 0.0)

            if np.any(left_mask):
                w_left = w_obs[0] + slope_left_w * (k_grid[left_mask] - k_min_obs)
                clipped_left = w_left <= min_total_variance
                w_left = np.clip(w_left, min_total_variance, max_total_variance)
                iv_grid[left_mask] = np.sqrt(w_left / T)
                flag_wing_clipped[left_mask] = clipped_left

            if np.any(right_mask):
                w_right = w_obs[-1] + slope_right_w * (k_grid[right_mask] - k_max_obs)
                clipped_right = w_right <= min_total_variance
                w_right = np.clip(w_right, min_total_variance, max_total_variance)
                iv_grid[right_mask] = np.sqrt(w_right / T)
                flag_wing_clipped[right_mask] = clipped_right

    # Suelo/techo numérico final sobre toda la grilla
    iv_grid = np.clip(iv_grid, min_iv, max_iv)

    return pd.DataFrame({
        "moneyness":                m_grid,
        "log_moneyness":            k_grid,
        "implied_vol":              iv_grid,
        "total_variance":           iv_grid ** 2 * T,
        "m_obs_min":                m_min_obs,
        "m_obs_max":                m_max_obs,
        "k_obs_min":                k_min_obs,
        "k_obs_max":                k_max_obs,
        "flag_inside_observed_range": flag_inside,
        "flag_wing_clipped":        flag_wing_clipped,
    })

# In[]:

surface_slices = []

for (date, expiry), df_slice in quoted_option.groupby(["Date", "Expiration"]):

    smile = interpolate_smile_slice(
    df_slice,
    n_grid=100,
    extrapolation_method="flat",   # o "linear" o "observed"
    global_moneyness_min=0.3,
    global_moneyness_max=1.7,
    bc_type="natural",
)

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

PARQET_OUTPUT = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\volatility_surface.parquet"
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