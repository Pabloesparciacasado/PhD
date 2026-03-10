import pandas as pd
import numpy as np
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


### Funciones de interpolacion para rates y dividends ########################


def interpolate_rates(
    curve_df: pd.DataFrame,
    expiry_days: pd.Series,
    currency: int,
    base: float = 365) -> pd.Series:
    """
    Interpolación de tasas de interés continuously compounded.

    Flujo:
        Se interpolan linealmente los log(DF) = -r·t/base, lo que equivale a
        interpolación log-lineal de factores de descuento. Se recuperan luego
        las tasas interpoladas a partir del log(DF) interpolado.

    Parameters
    ----------
    curve_df     : DataFrame con columnas [Currency, Date, Days, Rate]
    expiry_days  : Series con los días a interpolar
    currency     : Código de moneda (e.g. 272, 864, 1, ...)
    base         : Base de días para la convención day-count (default 360)

    Returns
    -------
    Series con los rates interpolados, mismo índice que expiry_days
    """
  
    curve = (
        curve_df[curve_df["Currency"] == currency]
        .sort_values("Days")
        .reset_index(drop=True)
    )
    days  = curve["Days"].values.astype(float)
    rates = curve["Rate"].values

 
    log_df = -rates * days / base

 
    t = expiry_days.values.astype(float)
    log_df_interp = np.interp(t, days, log_df)

   
    rates_interp = np.where(
        t > 0,
        -log_df_interp * base / t,
        rates[0] 
    )

    return pd.DataFrame(
        {"Days": expiry_days.values, "Rate": rates_interp},
        index=expiry_days.index,
    )




def interpolate_dividends(
    div_df: pd.DataFrame,
    security_id: int,
    date: pd.Timestamp,
    expiry_days: pd.Series,
    base: float = 360.0,
    MISSING: float = -99.99
) -> np.ndarray:
    """
    Interpola el dividend yield q (continuously compounded) para los días dados.

    Parameters
    ----------
    div_df      : DataFrame [SecurityID, Date, Expiration(YYYYMMDD int), Rate]
    security_id : ID del subyacente
    date        : Fecha de observación
    expiry_days : Series con los días a interpolar (mismo índice que vol_fecha)
    base        : Convención day-count (default 360)

    Returns
    -------
    Array numpy con q interpolado. Devuelve ceros si no hay datos.
    """
    sub = div_df[
        (div_df["SecurityID"] == security_id) &
        (div_df["Date"]       == date)        &
        (div_df["Rate"]       >  MISSING)
    ].copy()

    if sub.empty:
        return np.zeros(len(expiry_days))

    # Expiration ya es datetime (convertido en cargar_parquet) → días desde la fecha de observación
    sub["Days"] = (sub["Expiration"] - date).dt.days.astype(float)
    sub_pos      = sub[sub["Days"] > 0].sort_values("Days")

    # Caso flat yield: Expiration = 19000101 (sentinel de IvyDB sin estructura temporal)
    if sub_pos.empty:
        flat_rate = sub["Rate"].iloc[-1]   # rate de esa fecha
        return np.full(len(expiry_days), flat_rate)

    days  = sub_pos["Days"].values
    rates = sub_pos["Rate"].values
    log_df = -rates * days / base

    t             = expiry_days.values.astype(float)
    log_df_interp = np.interp(t, days, log_df)
    q_interp      = np.where(t > 0, -log_df_interp * base / t, rates[0])
    # Extrapolación izquierda: t < days[0] → usar rate del primer punto directamente
    q_interp      = np.where(t < days[0], rates[0], q_interp)
    return q_interp


### Funciones de nivel superior para superficies de volatilidad ###############


def interpolate_rates_surface(
    curve_df: pd.DataFrame,
    vol_fecha: pd.DataFrame,
    fecha: pd.Timestamp,
    currency: int,
    base: float = 360.0,
) -> pd.Series:
    """
    Interpola r para todos los vencimientos de una fecha de la superficie.

    Parameters
    ----------
    curve_df  : DataFrame completo de zero curve [Currency, Date, Days, Rate]
    vol_fecha : DataFrame de opciones para una única fecha
    fecha     : Fecha de observación
    currency  : Código de divisa
    base      : Convención day-count (default 360)

    Returns
    -------
    pd.Series con r interpolado, mismo índice que vol_fecha
    """
    curva_fecha = curve_df[curve_df["Date"] == fecha]
    return interpolate_rates(curva_fecha, vol_fecha["Days"], currency, base)["Rate"]


def interpolate_dividends_surface(
    div_df: pd.DataFrame | None,
    vol_fecha: pd.DataFrame,
    fecha: pd.Timestamp,
    base: float = 360.0,
) -> pd.Series:
    """
    Interpola q para todos los SecurityIDs de una fecha de la superficie.

    Itera sobre los SecurityIDs presentes en vol_fecha y llama a
    interpolate_dividends para cada uno.

    Parameters
    ----------
    div_df    : DataFrame de dividendos [SecurityID, Date, Expiration, Rate].
                None → devuelve ceros.
    vol_fecha : DataFrame de opciones para una única fecha
    fecha     : Fecha de observación
    base      : Convención day-count (default 360)

    Returns
    -------
    pd.Series con q interpolado, mismo índice que vol_fecha
    """
    q = pd.Series(0.0, index=vol_fecha.index)
    if div_df is None:
        return q
    div_fecha = div_df[div_df["Date"] == fecha]
    for sid, grp in vol_fecha.groupby("SecurityID"):
        q.loc[grp.index] = interpolate_dividends(div_fecha, sid, fecha, grp["Days"], base)
    return q
