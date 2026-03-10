
import numpy as np
import pandas as pd

MISSING = -99.99
BASE    = 360.0


# ─── Helpers internos ─────────────────────────────────────────────────────────

def _interpolar_rate(curve_fecha: pd.DataFrame, days_target: float, base: float) -> float:
    """Recomputa r para un único days_target dado los puntos de curva de esa fecha."""
    c     = curve_fecha.sort_values("Days").reset_index(drop=True)
    days  = c["Days"].values.astype(float)
    rates = c["Rate"].values
    log_df        = -rates * days / base
    log_df_interp = np.interp(days_target, days, log_df)
    return float(-log_df_interp * base / days_target) if days_target > 0 else float(rates[0])


def _interpolar_q(sub_div: pd.DataFrame, days_target: float, base: float) -> tuple[float, str]:
    """
    Recomputa q para un único days_target dado los datos de dividendo ya filtrados.
    Devuelve (q_value, tipo) donde tipo es 'flat' o 'term_structure'.
    """
    sub = sub_div[sub_div["Rate"] > MISSING].copy()
    if sub.empty:
        return 0.0, "sin_datos"

    if "Days" not in sub.columns:
        raise ValueError("sub_div debe tener columna 'Days' ya calculada")

    sub_pos = sub[sub["Days"] > 0].sort_values("Days")

    if sub_pos.empty:
        flat_rate = float(sub["Rate"].iloc[-1])
        return flat_rate, "flat"

    days  = sub_pos["Days"].values.astype(float)
    rates = sub_pos["Rate"].values
    log_df        = -rates * days / base
    log_df_interp = np.interp(days_target, days, log_df)
    if days_target <= 0:
        q = float(rates[0])
    elif days_target < days[0]:
        q = float(rates[0])   # extrapolación izquierda: rate del primer punto
    else:
        q = float(-log_df_interp * base / days_target)
    return q, "term_structure"


# ─── Funciones públicas ───────────────────────────────────────────────────────

def validar_rate(row: pd.Series, curve_df: pd.DataFrame, base: float = BASE) -> None:
    """
    Muestra los puntos de curva usados para interpolar r de una fila de price_BS.

    Parameters
    ----------
    row       : Fila del DataFrame devuelto por price_BS
    curve_df  : DataFrame de zero_curve (columnas: Currency, Date, Days, Rate)
    base      : Convención day-count (default 360)
    """
    date     = row["Date"]
    currency = int(row["Currency"])
    days_t   = float(row["Days"])

    curve_fecha = curve_df[
        (curve_df["Date"]     == date) &
        (curve_df["Currency"] == currency)
    ].sort_values("Days").reset_index(drop=True)

    print(f"\n--- Tipo de interés (r) ---")
    print(f"Curva Zero  Currency={currency} | Date={date.date() if hasattr(date,'date') else date}")

    if curve_fecha.empty:
        print("  ⚠ Sin datos de curva para esta fecha/moneda.")
        return

    # Identificar los dos puntos bracketing
    days_arr = curve_fecha["Days"].values.astype(float)
    idx_right = np.searchsorted(days_arr, days_t)
    idx_left  = idx_right - 1

    rows_to_show = list(range(max(0, idx_left - 1), min(len(curve_fecha), idx_right + 2)))

    for i, r_row in curve_fecha.iterrows():
        if i not in rows_to_show:
            continue
        marker = ""
        if i == idx_left:
            marker = "  ← izquierda"
        elif i == idx_right:
            marker = "  ← derecha"
        print(f"  Days={int(r_row['Days']):5d}  Rate={r_row['Rate']:.6f}{marker}")

    r_calc = _interpolar_rate(curve_fecha, days_t, base)
    print(f"  [Days={int(days_t)}]  r interpolado = {r_calc:.6f}")

    if "r" in row.index:
        match = "✓" if abs(row["r"] - r_calc) < 1e-8 else "✗"
        print(f"  r en row      = {row['r']:.6f}  {match}")


def validar_dividendo(
    row: pd.Series,
    div_df: pd.DataFrame,
    base: float = BASE,
) -> None:
    """
    Muestra los datos de dividendo usados para interpolar q de una fila de price_BS.

    Parameters
    ----------
    row    : Fila del DataFrame devuelto por price_BS
    div_df : DataFrame de index_dividend (columnas: SecurityID, Date, Expiration, Rate)
    base   : Convención day-count (default 360)
    """
    date        = row["Date"]
    security_id = int(row["SecurityID"])
    days_t      = float(row["Days"])

    sub = div_df[
        (div_df["SecurityID"] == security_id) &
        (div_df["Date"]       == date)        &
        (div_df["Rate"]       >  MISSING)
    ].copy()

    print(f"\n--- Dividend yield (q) ---")
    print(f"IndexDividend  SecurityID={security_id} | Date={date.date() if hasattr(date,'date') else date}")

    if sub.empty:
        print("  ⚠ Sin datos de dividendo → q = 0.0")
        return

    # Calcular días desde la fecha de observación
    sub["Days"] = (sub["Expiration"] - date).dt.days.astype(float)
    sub_pos     = sub[sub["Days"] > 0].sort_values("Days").reset_index(drop=True)

    if sub_pos.empty:
        # Flat yield
        flat_rate = float(sub["Rate"].iloc[-1])
        exp_str   = str(sub["Expiration"].iloc[-1].date()) if hasattr(sub["Expiration"].iloc[-1], "date") else str(sub["Expiration"].iloc[-1])
        print(f"  Tipo: FLAT YIELD (Expiration={exp_str}, sentinel US index)")
        print(f"  Rate = {flat_rate:.6f}")
        q_calc = flat_rate
    else:
        # Term structure
        print(f"  Tipo: TERM STRUCTURE ({len(sub_pos)} puntos)")
        days_arr  = sub_pos["Days"].values.astype(float)
        idx_right = np.searchsorted(days_arr, days_t)
        idx_left  = idx_right - 1

        rows_to_show = list(range(max(0, idx_left - 1), min(len(sub_pos), idx_right + 2)))
        for i, d_row in sub_pos.iterrows():
            if i not in rows_to_show:
                continue
            exp_str = str(d_row["Expiration"].date()) if hasattr(d_row["Expiration"], "date") else str(d_row["Expiration"])
            marker = ""
            if i == idx_left:
                marker = "  ← izquierda"
            elif i == idx_right:
                marker = "  ← derecha"
            print(f"  Exp={exp_str}  Days={int(d_row['Days']):5d}  Rate={d_row['Rate']:.6f}{marker}")

        q_calc, _ = _interpolar_q(sub_pos.assign(Rate=sub_pos["Rate"]), days_t, base)

    print(f"  [Days={int(days_t)}]  q interpolado = {q_calc:.6f}")

    if "q" in row.index:
        match = "✓" if abs(float(row["q"]) - q_calc) < 1e-8 else "✗"
        print(f"  q en row      = {float(row['q']):.6f}  {match}")


def validar_fila(
    row: pd.Series,
    curve_df: pd.DataFrame,
    div_df: pd.DataFrame | None = None,
    base: float = BASE,
) -> None:
    """
    Valida r y q para una fila del output de price_BS.

    Parameters
    ----------
    row      : Fila del DataFrame devuelto por price_BS
    curve_df : DataFrame de zero_curve
    div_df   : DataFrame de index_dividend (opcional)
    base     : Convención day-count (default 360)
    """
    # Cabecera
    sid    = int(row["SecurityID"])
    date   = row["Date"]
    days   = int(row["Days"])
    delta  = row.get("Delta", "?")
    strike = row.get("Strike", "?")
    cp     = row.get("CallPut", "?")

    date_str = date.date() if hasattr(date, "date") else date
    print(f"\n{'='*70}")
    print(f"SecurityID={sid} | Date={date_str} | Days={days} | "
          f"CallPut={cp} | Delta={delta} | Strike={strike}")

    for col in ("ImpliedVol", "q", "BS_Price", "Premium"):
        if col in row.index:
            print(f"  {col:12s} = {row[col]:.6f}")
    print(f"{'='*70}")

    validar_rate(row, curve_df, base)

    if div_df is not None:
        validar_dividendo(row, div_df, base)
    else:
        print("\n--- Dividend yield (q) ---")
        print("  div_df no proporcionado → q = 0.0 por defecto")
