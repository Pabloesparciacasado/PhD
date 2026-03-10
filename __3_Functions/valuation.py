import pandas as pd
import numpy as np
from scipy.stats import norm
import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __3_Functions.interpolation import interpolate_rates_surface, interpolate_dividends_surface


class model_valuation:
    """
    Clase de valoración de opciones sobre índices.

    Parameters
    ----------
    curve_df : pd.DataFrame
        Zero curve [Currency, Date, Days, Rate]
    currency : int
        Código de divisa IvyDB (ej. 333=USD, 978=EUR)
    div_df : pd.DataFrame | None
        Dividendos [SecurityID, Date, Expiration, Rate]. None → q=0
    base : float
        Convención day-count (default 360)

    Examples
    --------
    >>> model = model_valuation(curve_df=curva_3y, currency=333, div_df=divs_3y)
    >>> europeas  = model.price_BS(datos_europeos)
    >>> americanas = model.price_american(datos_americanos)
    >>> todas     = model.price(datos_con_exercise_style)
    >>> model.pricing_error(europeas)
    """

    MISSING: float = -99.99

    def __init__(
        self,
        curve_df: pd.DataFrame,
        currency: int | None = None,
        div_df: pd.DataFrame | None = None,
        base: float = 365.0
    ) -> None:
        self.curve_df = curve_df
        self.currency = currency
        self.div_df   = div_df
        self.base     = base

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos pricing desde fichero de volatilidad (v1)
    # ──────────────────────────────────────────────────────────────────────────

    def price_BS(self, volatility_data: pd.DataFrame) -> pd.DataFrame:
        """
        Valoración Black-Scholes-Merton para opciones europeas.

        Recupera S implícito invirtiendo la Delta y aplica la fórmula de Merton
        con dividendo continuo q.

        Fórmulas:
            Δ*   = Δ/100·e^(q·τ)          (calls, Δ≥0)
            Δ*   = Δ/100·e^(q·τ) + 1      (puts,  Δ<0)
            d1   = Φ⁻¹(Δ*)
            d2   = d1 − σ·√τ
            S    = K · exp[d1·σ·√τ − (r−q+σ²/2)·τ]
            Call = S·e^(−q·τ)·N(d1)  − K·e^(−r·τ)·N(d2)
            Put  = K·e^(−r·τ)·N(−d2) − S·e^(−q·τ)·N(−d1)

        Parameters
        ----------
        volatility_data : pd.DataFrame
            Superficie de volatilidad [Date, SecurityID, Days, Delta,
            CallPut, ImpliedVol, Strike, Premium]

        Returns
        -------
        pd.DataFrame
            Input con columnas añadidas: r, q, Underlying, BS_Price
        """
        bloques = []

        for fecha, vol_fecha in volatility_data.groupby("Date"):
            if not (self.curve_df["Date"] == fecha).any():
                continue

            vol_fecha = self._filter_missing(vol_fecha)
            if vol_fecha.empty:
                continue

            result        = vol_fecha.copy()
            r             = interpolate_rates_surface(self.curve_df, vol_fecha, fecha, self.currency, self.base)
            result["r"]   = r.values
            result["q"]   = interpolate_dividends_surface(self.div_df, vol_fecha, fecha, self.base)
            d1, d2, S     = self._recover_S(result, r)
            result["Underlying"] = S

            disc_q = np.exp(-result["q"] * (result["Days"] / self.base))
            disc_r = np.exp(-r           * (result["Days"] / self.base))
            result["BS_Price"] = np.nan
            mask_call = result["CallPut"] == "C"

            result.loc[ mask_call, "BS_Price"] = (
                S[mask_call] * disc_q[mask_call] * norm.cdf( d1[mask_call])
                - result.loc[mask_call, "Strike"] * disc_r[mask_call] * norm.cdf( d2[mask_call])
            )
            result.loc[~mask_call, "BS_Price"] = (
                result.loc[~mask_call, "Strike"] * disc_r[~mask_call] * norm.cdf(-d2[~mask_call])
                - S[~mask_call] * disc_q[~mask_call] * norm.cdf(-d1[~mask_call])
            )
            bloques.append(result)

        return pd.concat(bloques)
    # ──────────────────────────────────────────────────────────────────────────
    # Métodos pricing general (v2) "vale para fichero vol_surface si se incluye el precio forward"
    # ──────────────────────────────────────────────────────────────────────────

    def price_BS_general(self, copmlete_volatility_data: pd.DataFrame) -> pd.DataFrame:
        """
        Valoración Black-Scholes-Merton para opciones europeas. 
       
        Recupera S implícito desde el precio forward (implicitamente si hubiera dividendos los estaríamos descontando)
        
        Fórmulas:
            S    = K · exp[d1·σ·√τ − (r−q+σ²/2)·τ]
            Call = S·e^(−q·τ)·N(d1)  − K·e^(−r·τ)·N(d2)
            Put  = K·e^(−r·τ)·N(−d2) − S·e^(−q·τ)·N(−d1)

        Returns
        -------
        pd.DataFrame
            Input con columnas añadidas: r, forward, BS_Price
        """
        bloques = []

        for fecha, vol_fecha in copmlete_volatility_data.groupby("Date"):
            if not (self.curve_df["Date"] == fecha).any():
                continue

            vol_fecha = self._filter_missing(vol_fecha)
            if vol_fecha.empty:
                continue

            result        = vol_fecha.copy()
            # r             = interpolate_rates_surface(self.curve_df, vol_fecha, fecha, self.currency, self.base)
            # result["r"]   = r.values
            F = result["forward"]
            K = result["Strike"]
            
            d1 = (np.log(F/K) + ( 0.5*result["implied_vol"]**2)*result["T"])/(result["implied_vol"]*np.sqrt(result["T"]))
            d2 = d1 - result["implied_vol"]*np.sqrt(result["T"])
            disc_r = result["discount_factor"]
            
            mask_call = result["CallPut"] == "C"
      
            result["BS_Price"] = np.nan
            result.loc[ mask_call, "BS_Price"] = disc_r*( F[mask_call]*norm.cdf(d1[mask_call]) - K[mask_call]*norm.cdf(d2[mask_call]) )
            result.loc[~mask_call, "BS_Price"] = disc_r*( - F[~mask_call]*norm.cdf(-d1[~mask_call]) + K[~mask_call]*norm.cdf(-d2[~mask_call]) )
            bloques.append(result)
    
        return pd.concat(bloques)

    def price_american(
        self,
        volatility_data: pd.DataFrame,
        n_steps: int = 200,
    ) -> pd.DataFrame:
        """
        Valoración por árbol binomial CRR para opciones americanas.

        Construye un árbol recombinante de Cox-Ross-Rubinstein con n_steps pasos
        y aplica la condición de ejercicio anticipado en cada nodo.

        Parámetros del árbol:
            u = exp(σ·√(τ/n))
            d = 1/u
            p = (exp((r−q)·τ/n) − d) / (u − d)   [prob. riesgo-neutral]

        Parameters
        ----------
        volatility_data : pd.DataFrame
            Mismas columnas que price_BS
        n_steps : int
            Número de pasos del árbol (default 200). Mayor → más preciso y lento.

        Returns
        -------
        pd.DataFrame
            Input con columnas añadidas: r, q, Underlying, American_Price
        """
        bloques = []

        for fecha, vol_fecha in volatility_data.groupby("Date"):
            if not (self.curve_df["Date"] == fecha).any():
                continue

            vol_fecha = self._filter_missing(vol_fecha)
            if vol_fecha.empty:
                continue

            result      = vol_fecha.copy()
            r           = interpolate_rates_surface(self.curve_df, vol_fecha, fecha, self.currency, self.base)
            result["r"] = r.values
            result["q"] = interpolate_dividends_surface(self.div_df, vol_fecha, fecha, self.base)
            _, _, S     = self._recover_S(result, r)
            result["Underlying"]    = S
            result["American_Price"] = np.nan

            for idx, row in result.iterrows():
                result.at[idx, "American_Price"] = self._crr_price(
                    S      = float(S[idx]),
                    K      = float(row["Strike"]),
                    r      = float(r[idx]),
                    q      = float(row["q"]),
                    sigma  = float(row["ImpliedVol"]),
                    tau    = float(row["Days"]) / self.base,
                    is_call= row["CallPut"] == "C",
                    n      = n_steps,
                )
            bloques.append(result)

        return pd.concat(bloques)

    def price(
        self,
        volatility_data: pd.DataFrame,
        exercise_style_col: str = "ExerciseStyle",
    ) -> pd.DataFrame:
        """
        Método unificado: despacha a price_BS o price_american según ExerciseStyle.

        Si la columna exercise_style_col no existe, usa price_BS para todas las filas.

        Parameters
        ----------
        volatility_data : pd.DataFrame
            Superficie de volatilidad. Puede incluir columna ExerciseStyle ('E' / 'A').
        exercise_style_col : str
            Nombre de la columna con el estilo de ejercicio (default 'ExerciseStyle').

        Returns
        -------
        pd.DataFrame
            Resultado combinado con columnas de precio según cada modelo.
        """
        if exercise_style_col not in volatility_data.columns:
            return self.price_BS(volatility_data)

        europeas  = volatility_data[volatility_data[exercise_style_col] == "E"]
        americanas = volatility_data[volatility_data[exercise_style_col] == "A"]

        partes = []
        if not europeas.empty:
            partes.append(self.price_BS(europeas))
        if not americanas.empty:
            partes.append(self.price_american(americanas))

        return pd.concat(partes).sort_index()

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos públicos de utilidad
    # ──────────────────────────────────────────────────────────────────────────

    def pricing_error(
        self,
        priced_df: pd.DataFrame,
        price_col: str = "BS_Price",
    ) -> pd.DataFrame:
        """
        Calcula el error entre el precio calculado y el Premium de IvyDB.

        Parameters
        ----------
        priced_df : pd.DataFrame
            Output de price_BS o price_american (debe contener Premium y price_col)
        price_col : str
            Columna con el precio calculado (default 'BS_Price')

        Returns
        -------
        pd.DataFrame
            Tabla Days × Delta con error porcentual medio.
        """
        df = priced_df[priced_df["Premium"] > self.MISSING].copy()
        df["dif"]     = df[price_col] - df["Premium"]
        df["dif_pct"] = df["dif"].abs() / df["Premium"].abs() * 100
        return (
            df.groupby(["Days", "Delta"])["dif_pct"]
            .agg(["mean", "std", "max"])
            .round(2)
        )

    def greeks(self, volatility_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula las griegas analíticas BSM (solo europeas).

        Griegas implementadas:
            Delta  = e^(−q·τ)·N(d1)                          (call)
                   = e^(−q·τ)·(N(d1)−1)                      (put)
            Gamma  = e^(−q·τ)·n(d1) / (S·σ·√τ)
            Vega   = S·e^(−q·τ)·n(d1)·√τ
            Theta  = −S·e^(−q·τ)·n(d1)·σ/(2·√τ)
                     −r·K·e^(−r·τ)·N(d2)  + q·S·e^(−q·τ)·N(d1)   (call)
            Rho    = K·τ·e^(−r·τ)·N(d2)                       (call)
                   = −K·τ·e^(−r·τ)·N(−d2)                    (put)

        Returns
        -------
        pd.DataFrame
            Input con columnas añadidas: r, q, Underlying, Delta_calc,
            Gamma, Vega, Theta, Rho
        """
        bloques = []

        for fecha, vol_fecha in volatility_data.groupby("Date"):
            if not (self.curve_df["Date"] == fecha).any():
                continue

            vol_fecha = self._filter_missing(vol_fecha)
            if vol_fecha.empty:
                continue

            result      = vol_fecha.copy()
            r           = interpolate_rates_surface(self.curve_df, vol_fecha, fecha, self.currency, self.base)
            result["r"] = r.values
            result["q"] = interpolate_dividends_surface(self.div_df, vol_fecha, fecha, self.base)
            d1, d2, S   = self._recover_S(result, r)
            result["Underlying"] = S

            q   = result["q"]
            K   = result["Strike"]
            t   = result["Days"] / self.base
            sig = result["ImpliedVol"]

            disc_q  = np.exp(-q * t)
            disc_r  = np.exp(-r * t)
            n_d1    = norm.pdf(d1)   # densidad normal en d1
            sqrt_t  = np.sqrt(t)

            mask_call = result["CallPut"] == "C"

            # Delta
            result["Delta_calc"] = np.where(
                mask_call,
                disc_q * norm.cdf( d1),
                disc_q * (norm.cdf(d1) - 1),
            )

            # Gamma (igual para calls y puts)
            result["Gamma"] = disc_q * n_d1 / (S * sig * sqrt_t)

            # Vega (igual para calls y puts), en términos de 1% de σ
            result["Vega"] = S * disc_q * n_d1 * sqrt_t / 100

            # Theta (por día calendario)
            theta_common = -S * disc_q * n_d1 * sig / (2 * sqrt_t)
            result["Theta"] = np.where(
                mask_call,
                (theta_common - r * K * disc_r * norm.cdf( d2) + q * S * disc_q * norm.cdf( d1)) / self.base,
                (theta_common + r * K * disc_r * norm.cdf(-d2) - q * S * disc_q * norm.cdf(-d1)) / self.base,
            )

            # Rho (por 1% de r)
            result["Rho"] = np.where(
                mask_call,
                 K * t * disc_r * norm.cdf( d2) / 100,
                -K * t * disc_r * norm.cdf(-d2) / 100,
            )

            bloques.append(result)

        return pd.concat(bloques)

    # ──────────────────────────────────────────────────────────────────────────
    # Métodos internos
    # ──────────────────────────────────────────────────────────────────────────

    def _filter_missing(self, vol_fecha: pd.DataFrame) -> pd.DataFrame:
        """Descarta filas con volatilidad implícita o strike inválidos."""

        vol_col = "ImpliedVol" if "ImpliedVol" in vol_fecha.columns else "implied_vol"

        return vol_fecha[
            (vol_fecha[vol_col] > self.MISSING) &
            (vol_fecha["Strike"] > 0)
        ]
    def _recover_S(
        self,
        result: pd.DataFrame,
        r: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Invierte la Delta para obtener d1 y recupera S implícito.

        Returns
        -------
        (d1, d2, S) como pandas Series con el índice de result.
        """
        delta = result["Delta"]
        sigma = result["ImpliedVol"]
        K     = result["Strike"]
        t     = result["Days"] / self.base
        q     = result["q"]

        adj = delta / 100 * np.exp(q * t)
        d1  = norm.ppf(np.where(delta >= 0, adj, adj + 1))
        d2  = d1 - sigma * np.sqrt(t)
        S   = K * np.exp(d1 * sigma * np.sqrt(t) - (r - q + sigma**2 / 2) * t)

        # Convertir d1, d2 a Series preservando el índice
        d1 = pd.Series(d1, index=result.index)
        d2 = pd.Series(d2, index=result.index)
        return d1, d2, S

    @staticmethod
    def _crr_price(
        S: float,
        K: float,
        r: float,
        q: float,
        sigma: float,
        tau: float,
        is_call: bool,
        n: int,
    ) -> float:
        """
        Árbol binomial CRR para una sola opción americana.

        Parameters
        ----------
        S, K    : Precio subyacente y strike
        r, q    : Tipo libre de riesgo y dividend yield (continuously compounded)
        sigma   : Volatilidad implícita
        tau     : Tiempo al vencimiento en años
        is_call : True para call, False para put
        n       : Número de pasos del árbol

        Returns
        -------
        float : Precio de la opción americana
        """
        if tau <= 0 or sigma <= 0:
            payoff = max(S - K, 0) if is_call else max(K - S, 0)
            return float(payoff)

        dt   = tau / n
        u    = np.exp(sigma * np.sqrt(dt))
        d    = 1.0 / u
        disc = np.exp(-r * dt)
        p    = (np.exp((r - q) * dt) - d) / (u - d)
        q_rn = 1.0 - p                             # prob. bajada riesgo-neutral

        # Precios del subyacente en el nodo final (vectorizado)
        j      = np.arange(n + 1)
        S_T    = S * (u ** (n - j)) * (d ** j)

        # Valores intrínsecos en el vencimiento
        if is_call:
            V = np.maximum(S_T - K, 0.0)
        else:
            V = np.maximum(K - S_T, 0.0)

        # Inducción hacia atrás con early-exercise
        for i in range(n - 1, -1, -1):
            S_i       = S * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
            continuation = disc * (p * V[:i + 1] + q_rn * V[1:i + 2])
            if is_call:
                intrinsic = np.maximum(S_i - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_i, 0.0)
            V = np.maximum(continuation, intrinsic)

        return float(V[0])
