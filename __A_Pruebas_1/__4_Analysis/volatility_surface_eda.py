
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import norm
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from __0_Validaciones.validar_bs_inputs import validar_fila
from __2_Files.volatility_surface import VolatilitySurface
from __2_Files.zero_curve import ZeroCurve
from __2_Files.index_dividend import IndexDividend
from __3_Functions.valuation import model_valuation



 # In[]

# --- Configuración ------------------------------------------------------------

CSV_RUTA     = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\volatility_surface.csv"
PARQUET_RUTA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\volatility_surface.parquet"
MISSING      = -99.99   



# ─── 1. Cargar datos ──────────────────────────────────────────────────────────

vs = VolatilitySurface()


vs.cargar_parquet(ruta=PARQUET_RUTA)




#--------- Cargamos zero rates ----------------
zc = ZeroCurve(sep='\t')
zc.cargar_parquet(
    ruta = str(Path(PARQUET_RUTA).parent / "zero_curve.parquet")
    )

#--------- Cargamos dividendos ----------------
id_ = IndexDividend()
id_.cargar_parquet(
    ruta = str(Path(PARQUET_RUTA).parent / "index_dividend.parquet")
    )
 # In[]


def interpolate_rates(
    curve_df: pd.DataFrame,
    expiry_days: pd.Series,
    currency: int,
    base: float = 360.0) -> pd.Series:
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





 # In[]

MISSING = -99.99

def interpolate_dividends(
    div_df: pd.DataFrame,
    security_id: int,
    date: pd.Timestamp,
    expiry_days: pd.Series,
    base: float = 360.0,
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


def price_BS(
    volatility_data: pd.DataFrame,
    curve_df: pd.DataFrame,
    currency: int,
    div_df: pd.DataFrame | None = None,
    base: float = 360.0,
) -> pd.DataFrame:
    """
    Recupera S_t y calcula el precio BSM con ajuste de dividendo continuo.

    Fórmula con dividendo q:
        Δ*  = Δ/100·e^(q·τ)          para calls (Δ≥0)
        Δ*  = Δ/100·e^(q·τ) + 1      para puts  (Δ<0)
        d1  = Φ⁻¹(Δ*)
        d2  = d1 − σ·√τ
        S   = K · exp[ d1·σ·√τ − (r − q + σ²/2)·τ ]
        Call = S·e^(−q·τ)·N(d1)  − K·e^(−r·τ)·N(d2)
        Put  = K·e^(−r·τ)·N(−d2) − S·e^(−q·τ)·N(−d1)

    Parameters
    ----------
    volatility_data : DataFrame [Date, SecurityID, Days, Delta, CallPut, ImpliedVol, Strike]
    curve_df        : DataFrame de la zero curve [Currency, Date, Days, Rate]
    currency        : Código de moneda para filtrar curve_df
    div_df          : DataFrame de dividendos [SecurityID, Date, Expiration, Rate]. None → q=0
    base            : Convención day-count (default 360)
    """
    bloques = []

    for fecha, vol_fecha in volatility_data.groupby("Date"):
        curva_fecha = curve_df[curve_df["Date"] == fecha]
        if curva_fecha.empty:
            continue

        # Filtrar filas con datos missing antes de calcular
        vol_fecha = vol_fecha[
            (vol_fecha["ImpliedVol"] > MISSING) &
            (vol_fecha["Strike"]     > 0)
        ]
        if vol_fecha.empty:
            continue

        result = vol_fecha.copy()
        r = interpolate_rates(curva_fecha, vol_fecha["Days"], currency, base)["Rate"]
        result["r"] = r.values

        # ── Dividend yield q por SecurityID ───────────────────────────────────
        result["q"] = 0.0
        if div_df is not None:
            div_fecha = div_df[div_df["Date"] == fecha]
            for sid, grp in vol_fecha.groupby("SecurityID"):
                result.loc[grp.index, "q"] = interpolate_dividends(
                    div_fecha, sid, fecha, grp["Days"], base
                )

        # ── Subyacente implícito ───────────────────────────────────────────────
        # Con dividendo: Δ* = Δ/100·e^(q·τ) para calls, +1 para puts
        delta = result["Delta"]
        sigma = result["ImpliedVol"]
        K     = result["Strike"]
        t     = (result["Days"] / base)
        q     = result["q"]

        adj   = delta / 100 * np.exp(q * t)
        d1    = norm.ppf(np.where(delta >= 0, adj, adj + 1))
        d2    = d1 - sigma * np.sqrt(t)
        S     = K * np.exp(d1 * sigma * np.sqrt(t) - (r - q + sigma**2 / 2) * t)
        result["Underlying"] = S

        # ── Precio BSM ────────────────────────────────────────────────────────
        # Call: S·e^(−q·τ)·N(d1)  − K·e^(−r·τ)·N(d2)
        # Put:  K·e^(−r·τ)·N(−d2) − S·e^(−q·τ)·N(−d1)
        disc_q = np.exp(-q * t)
        disc_r = np.exp(-r * t)
        result["BS_Price"] = np.nan
        mask_call = result["CallPut"] == "C"
        result.loc[ mask_call, "BS_Price"] = (
            S[mask_call] * disc_q[mask_call] * norm.cdf( d1[mask_call])
            - K[mask_call] * disc_r[mask_call] * norm.cdf( d2[mask_call])
        )
        result.loc[~mask_call, "BS_Price"] = (
            K[~mask_call] * disc_r[~mask_call] * norm.cdf(-d2[~mask_call])
            - S[~mask_call] * disc_q[~mask_call] * norm.cdf(-d1[~mask_call])
        )
        bloques.append(result)

    return pd.concat(bloques)
 # In[]

# ── Prueba ────────────────────────────────────────────────────────────
ANIOS_PRUEBA = [2019, 2020, 2021, 2022, 2023, 2024]

datos_3y = vs.df[
    vs.df["Date"].dt.year.isin(ANIOS_PRUEBA) & (vs.df["Currency"] == 333)
]
curva_3y = zc.df[
    zc.df["Date"].dt.year.isin(ANIOS_PRUEBA) & (zc.df["Currency"] == 333)
]
divs_3y = id_.df[
    id_.df["Date"].dt.year.isin(ANIOS_PRUEBA)
]

model   = model_valuation(curve_df=curva_3y, currency=333, div_df=divs_3y)
ejemplo = model.price_BS(datos_3y)
ejemploo = ejemplo.iloc[0:20000]
ejemploo.to_csv(r"C:\Users\pablo.esparcia\Downloads\ejemplo_underlying.csv", index=False)
 # In[]
# ── Validación de inputs (r y q) ──────────────────────────────────────────────

# Cambia el índice para inspeccionar cualquier fila
validar_fila(ejemplo.iloc[0], curva_3y, divs_3y)

 # In[]

print(ejemplo.describe())
print(f"Filas resultado: {len(ejemplo):,}  |  Fechas únicas: {ejemplo['Date'].nunique()}")


 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 # In[]
##################################################################
############## EJEMPLO DE USO: ANÁLISIS (CLAUDE) #################
##################################################################

# ─── 2. Elegir subyacente y fecha de referencia ───────────────────────────────

# Subyacente más frecuente en el dataset
sid   = int(vs.df['SecurityID'].value_counts().index[0])
fecha = vs.fechas_disponibles(sid)[-1]          # última fecha disponible
days_list = vs.maturities_disponibles(sid, fecha)

print(f"\nSecurityID : {sid}")
print(f"Fecha      : {fecha}")
print(f"Maturities  : {days_list}")


# ─── 3. Smile de volatilidad (varias Maturities) ───────────────────────────────

fig, ax = plt.subplots(figsize=(9, 5))
palette = cm.viridis(np.linspace(0.1, 0.9, min(len(days_list), 6)))

for color, days in zip(palette, days_list[:6]):
    smile = vs.smile(sid, fecha, days)
    smile = smile[smile['ImpliedVol'] > MISSING]
    if smile.empty:
        continue
    ax.plot(smile['Delta'], smile['ImpliedVol'],
            marker='o', ms=4, linewidth=1.4,
            color=color, label=f'{days}d')

ax.set_title(f'Volatility Smile  –  SecurityID {sid}  ({fecha})')
ax.set_xlabel('Delta')
ax.set_ylabel('Implied Volatility')
ax.legend(title='Days to expiry')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('smile.png', dpi=150)
plt.show()

                                                                                                                                                                                                                                                                                                                                                                                                         
# ─── 4. Superficie de volatilidad (heatmap) ───────────────────────────────────

surf = vs.superficie(sid, fecha).replace(MISSING, np.nan)

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(surf.values, aspect='auto', cmap='RdYlGn_r',
               origin='lower', vmin=0)
plt.colorbar(im, ax=ax, label='Implied Volatility')

ax.set_xticks(range(len(surf.columns)))
ax.set_xticklabels(surf.columns.astype(int))
ax.set_yticks(range(len(surf.index)))
ax.set_yticklabels(surf.index.astype(int))
ax.set_xlabel('Delta')
ax.set_ylabel('Days to Expiry')
ax.set_title(f'Volatility Surface  –  SecurityID {sid}  ({fecha})')
plt.tight_layout()
plt.savefig('superficie.png', dpi=150)
plt.show()


# ─── 5. Serie temporal de vol ATM ─────────────────────────────────────────────

# Tenor de referencia: la segunda más corta (la más corta puede ser ruidosa)
days_ref  = days_list[1] if len(days_list) > 1 else days_list[0]

# Delta ATM: el call cuyo delta está más cerca de 50
deltas_call = (
    vs.df[(vs.df['SecurityID'] == sid) & (vs.df['CallPut'] == 'C')]
    ['Delta'].unique()
)
delta_atm = int(deltas_call[np.abs(deltas_call - 50).argmin()])

# Query vectorizada al DataFrame (mucho más rápido que iterar con smile())
mask = (
    (vs.df['SecurityID'] == sid)      &
    (vs.df['Days']       == days_ref) &
    (vs.df['Delta']      == delta_atm) &
    (vs.df['CallPut']    == 'C')      &
    (vs.df['ImpliedVol'] >  MISSING)
)
serie = (vs.df[mask][['Date', 'ImpliedVol']]
           .set_index('Date')
           .sort_index())

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(serie.index, serie['ImpliedVol'], linewidth=1.2, color='steelblue')
ax.set_title(f'ATM Implied Vol  ({days_ref}d, Δ={delta_atm})  –  SecurityID {sid}')
ax.set_xlabel('Date')
ax.set_ylabel('Implied Volatility')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('vol_atm_serie.png', dpi=150)
plt.show()
