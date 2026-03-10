import pandas as pd
import duckdb
import os
import sys
import time
import numpy as np
from scipy.stats import norm
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __1_Input_Data.data_ingestion import DataLoader


# ─── CLASE: Volatility Surface ────────────────────────────────────────────────

class VolatilitySurface:
    """
    Carga y consulta la superficie de volatilidad implícita (IvyDB GI – IVYVSURFD).

    Estructura del dato:
        SecurityID | Date | Days | Delta | CallPut | ImpliedVol | Strike | Premium | Dispersion | Currency

    Dimensiones clave:
        - SecurityID : subyacente
        - Date       : fecha de observación
        - Days       : días hasta vencimiento  (eje de tenor)
        - Delta      : delta de la opción      (eje de moneyness)
        - CallPut    : 'C' o 'P'
    """

    # Dtypes compactos: ~60 % menos RAM que los defaults de pandas.
    _DTYPE_IVYVSURFD: dict[str, str] = {
        'SecurityID': 'int32',
        'Days':       'int16',
        'Delta':      'int16',
        'CallPut':    'category',
        'ImpliedVol': 'float32',
        'Strike':     'float32',
        'Premium':    'float32',
        'Dispersion': 'float32',
        'Currency':   'int16',
    }

    def __init__(self, sep: str = '\t', encoding: str = 'utf-8'):
        self._loader = DataLoader(sep=sep, encoding=encoding)
        self.df: pd.DataFrame | None = None
        

    # --- Carga ─────────────────────────────────────────────────────────────────

    def cargar_datos(
        self,
        carpeta:    str,
        guardar_en: str | None = None,
        limite:     int | None = None,
    ) -> pd.DataFrame:
        """
        Compila todos los archivos .txt de la carpeta IVYVSURFD,
        guarda el resultado como CSV y lo carga en self.df con dtypes compactos.

        :param carpeta:    Ruta a la carpeta raíz (contiene la subcarpeta GI.ALL.IVYVSURFD).
        :param guardar_en: Ruta donde guardar el CSV compilado.
        :param limite:     Si se indica, procesa solo los N archivos más recientes.
                           Útil para pruebas rápidas (ej. limite=3).
        """
        inicio = time.perf_counter()
        self._loader.compilar_carpeta(
            os.path.join(carpeta, "GI.ALL.IVYVSURFD"),
            guardar_en = guardar_en,
            dtype      = self._DTYPE_IVYVSURFD,
            limite     = limite,
        )
        self._cargar_desde_csv(self._loader.ruta_csv)
        fin = time.perf_counter()
        print(f"Tiempo total: {fin - inicio:.0f} s")
        return self.df

    def cargar_csv(self, ruta: str) -> pd.DataFrame:
        """
        Carga desde un CSV ya compilado con dtypes compactos.
        Útil cuando el CSV existe de una ejecución anterior.

        :param ruta: Ruta al CSV compilado (p.ej. volatility_surface.csv).
        """
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró: {ruta}")
        inicio = time.perf_counter()
        self._cargar_desde_csv(ruta)
        fin = time.perf_counter()
        print(f"Cargado ({os.path.getsize(ruta) / 1e9:.1f} GB en disco): {fin - inicio:.0f} s")
        return self.df

    def exportar_parquet(self, ruta: str) -> None:
        """
        Exporta self.df a Parquet (ejecutar una sola vez).
        Requiere haber llamado antes a cargar_datos() o cargar_csv().

        :param ruta: Ruta de destino del Parquet (p.ej. volatility_surface.parquet).
        """
        if self.df is None:
            raise RuntimeError("Llama primero a cargar_datos() o cargar_csv().")
        print("Exportando a Parquet…")
        inicio = time.perf_counter()
        ruta_p = ruta.replace('\\', '/')
        duckdb.from_df(self.df).write_parquet(ruta_p, compression='snappy')
        fin    = time.perf_counter()
        tam_gb = os.path.getsize(ruta) / 1e9
        print(f"Guardado: {ruta} ({tam_gb:.1f} GB) en {fin - inicio:.0f} s")

    def cargar_parquet(
        self,
        ruta:     str,
        desde:    str | None       = None,
        hasta:    str | None       = None,
        columnas: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Carga datos del Parquet en self.df como un pandas DataFrame normal.

        :param ruta:     Ruta al archivo Parquet.
        :param desde:    Fecha inicio en formato 'YYYY-MM-DD' (inclusive).
        :param hasta:    Fecha fin   en formato 'YYYY-MM-DD' (inclusive).
        :param columnas: Lista de columnas a cargar. None carga todas.
        """
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró: {ruta}")

        ruta_p = ruta.replace('\\', '/')
        cols   = ', '.join(columnas) if columnas else '*'

        # Detectar si Date está guardado como entero YYYYMMDD o como tipo fecha
        schema_df = duckdb.sql(f"DESCRIBE SELECT Date FROM '{ruta_p}' LIMIT 0").df()
        date_type = schema_df.iloc[0]['column_type'] if not schema_df.empty else ''
        date_is_int = 'INT' in date_type.upper() or 'BIGINT' in date_type.upper()

        def _fmt_fecha(fecha_str: str) -> str:
            """Devuelve el literal SQL correcto según el tipo de la columna Date."""
            if date_is_int:
                return fecha_str.replace('-', '')          # '2024-02-29' → 20240229
            return f"'{fecha_str}'"                        # → '2024-02-29'

        condiciones = []
        if desde:
            condiciones.append(f"Date >= {_fmt_fecha(desde)}")
        if hasta:
            condiciones.append(f"Date <= {_fmt_fecha(hasta)}")
        where = ('WHERE ' + ' AND '.join(condiciones)) if condiciones else ''

        print("Cargando en RAM…")
        inicio   = time.perf_counter()
        self.df  = duckdb.sql(f"SELECT {cols} FROM '{ruta_p}' {where}").df()
        fin      = time.perf_counter()

        if 'Date' in self.df.columns:
            col = self.df['Date']
            if pd.api.types.is_integer_dtype(col):
                # Parquet guardó la fecha como entero YYYYMMDD
                self.df['Date'] = pd.to_datetime(
                    col.astype(str), format='%Y%m%d', errors='coerce'
                )
            else:
                # DuckDB devuelve string con formatos mixtos: YYYYMMDD y YYYY-MM-DD
                # format='mixed' (pandas ≥2.0) parsea cada valor individualmente
                self.df['Date'] = pd.to_datetime(col, format='mixed', errors='coerce')

        if 'Currency' in self.df.columns:
            self.df['Currency'] = pd.to_numeric(self.df['Currency'], errors='coerce').astype('int16')

        ram_gb = self.df.memory_usage(deep=True).sum() / 1e9
        print(f"Cargadas {len(self.df):,} filas × {len(self.df.columns)} columnas")
        print(f"RAM: {ram_gb:.1f} GB  |  Tiempo: {fin - inicio:.0f} s")
        return self.df

    # ── Ejemplo de Consultas ─────────────────────────────────────────────────────────────

    def superficie(self, security_id: int, fecha: str) -> pd.DataFrame:
        """
        Devuelve la superficie de vol para un subyacente en una fecha concreta,
        pivotada como matriz Days × Delta.

        :param security_id: ID del subyacente (ej. 102434)
        :param fecha:       Fecha en formato 'YYYY-MM-DD'
        :return:            DataFrame pivotado (index=Days, columns=Delta)
        """
        self._verificar_datos()
        fecha_dt = pd.Timestamp(fecha)
        filtro = (
            (self.df['SecurityID'] == security_id) &
            (self.df['Date']       == fecha_dt)
        )
        sub = self.df[filtro]
        if sub.empty:
            raise ValueError(f"Sin datos para SecurityID={security_id} en {fecha}")

        return sub.pivot_table(
            index='Days', columns='Delta', values='ImpliedVol', aggfunc='mean'
        )

    def smile(self, security_id: int, fecha: str, days: int) -> pd.DataFrame:
        """
        Devuelve el smile de volatilidad (Delta → ImpliedVol) para un subyacente,
        fecha y tenor concretos.

        :param security_id: ID del subyacente
        :param fecha:       Fecha en formato 'YYYY-MM-DD'
        :param days:        Días hasta vencimiento
        :return:            DataFrame con columnas [Delta, CallPut, ImpliedVol, Strike]
        """
        self._verificar_datos()
        fecha_dt = pd.Timestamp(fecha)
        filtro = (
            (self.df['SecurityID'] == security_id) &
            (self.df['Date']       == fecha_dt)    &
            (self.df['Days']       == days)
        )
        sub = self.df[filtro][['Delta', 'CallPut', 'ImpliedVol', 'Strike']].copy()
        if sub.empty:
            raise ValueError(
                f"Sin datos para SecurityID={security_id}, fecha={fecha}, days={days}"
            )
        return sub.sort_values('Delta').reset_index(drop=True)

    def maturities_disponibles(self, security_id: int, fecha: str) -> list[int]:
        """Devuelve los valores de Days disponibles para un subyacente y fecha."""
        self._verificar_datos()
        fecha_dt = pd.Timestamp(fecha)
        filtro = (
            (self.df['SecurityID'] == security_id) &
            (self.df['Date']       == fecha_dt)
        )
        return sorted(self.df[filtro]['Days'].unique().tolist())

    def fechas_disponibles(self, security_id: int) -> list[str]:
        """Devuelve las fechas disponibles para un subyacente concreto."""
        self._verificar_datos()
        fechas = self.df[self.df['SecurityID'] == security_id]['Date'].dropna().unique()
        return sorted(pd.Timestamp(f).strftime('%Y-%m-%d') for f in fechas)

    def resumen(self) -> None:
        """Imprime un resumen del dataset cargado."""
        self._verificar_datos()
        print(f"Filas            : {len(self.df):,}")
        print(f"Subyacentes      : {self.df['SecurityID'].nunique():,}")
        print(f"Rango de fechas  : {self.df['Date'].min().date()} → {self.df['Date'].max().date()}")
        print(f"Days únicos      : {sorted(self.df['Days'].unique().tolist())}")
        print(f"Deltas únicos    : {sorted(self.df['Delta'].unique().tolist())}")
        print(f"Columnas         : {self.df.columns.tolist()}")

    # ── Interno ───────────────────────────────────────────────────────────────

    def _verificar_datos(self) -> None:
        if self.df is None:
            raise RuntimeError("Llama primero a cargar_datos() o cargar_csv().")

    def _cargar_desde_csv(self, ruta: str) -> None:
        """Lee el CSV con dtypes compactos y convierte la columna de fecha."""
        self.df = pd.read_csv(
            ruta,
            dtype      = self._DTYPE_IVYVSURFD,
            parse_dates= ['Date'],
            encoding   = 'utf-8',
            low_memory = False,
        )










# ─── EJECUCIÓN ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    CSV     = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\volatility_surface.csv"
    PARQUET = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\volatility_surface.parquet"
    CARPETA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado"

    vs = VolatilitySurface(sep='\t')

    # ── Paso 1 (una sola vez): compilar TXT → CSV ─────────────────────────────
    #vs.cargar_datos(carpeta=CARPETA, guardar_en=CSV)

    # ── Paso 2 (una sola vez): CSV → Parquet ──────────────────────────────────
    # vs.cargar_csv(CSV)
    # vs.exportar_parquet(PARQUET)

    #── Uso habitual: cargar Parquet → self.df (pandas normal) ────────────────
    if Path(PARQUET).exists():
        vs.cargar_parquet(ruta=PARQUET, 
                          desde="2010-01-01", 
                          hasta="2023-12-31")
    else:
        vs.cargar_csv(CSV)

    print(vs.df.columns.tolist())
    vs.resumen()

    #Ejemplo de consulta — ajusta SecurityID y fecha a lo que haya en tus datos
    superficie = vs.superficie(security_id=102434, fecha='2023-03-01')
    print(superficie)

    smile = vs.smile(security_id=102434, fecha='2023-03-01', days=10)
    print(smile)

    smile = vs.smile(security_id=102434, fecha='2023-03-01', days=10)
    print(smile)
