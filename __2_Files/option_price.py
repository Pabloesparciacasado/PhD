import pandas as pd
import duckdb
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __1_Input_Data.data_ingestion import DataLoader


# ─── CLASE: Option Price ────────────────────────────────────────────────────────

class OptionPrice:
    """
    Carga y consulta de Option Price (IvyDB GI.ALL.IVYOPPRCD).

    Estructura del dato:
        SecurityID | Date | OptionID | Exchange | Currency | Expiration | Strike |
        CallPut | Symbol | Bid | Ask | Last | Volume | OpenInterest |
        SpecialSettlement | ImpliedVolatility | Delta | Gamma | Vega | Theta |
        AdjustmentFactor | ExerciseStyle | SymbolFlag | CalculationPrice |
        ReferenceExchange | AMSettlement | ContractSize | ExpiryIndicator

    El CSV compilado (235 M filas, ~36 GB) se consulta directamente con DuckDB
    sin cargarlo en RAM. Solo el subconjunto filtrado se materializa en un DataFrame.

    Requiere: pip install duckdb
    """

    # Nombres de columna en el orden que tienen los TXT (archivos 1996-2023 sin cabecera).
    _NAMES_IVYOPPRCD: list[str] = [
        'SecurityID', 'Date', 'OptionID', 'Exchange', 'Currency', 'Expiration',
        'Strike', 'CallPut', 'Symbol', 'Bid', 'Ask', 'Last', 'Volume',
        'OpenInterest', 'SpecialSettlement', 'ImpliedVolatility', 'Delta',
        'Gamma', 'Vega', 'Theta', 'AdjustmentFactor', 'ExerciseStyle',
        'SymbolFlag', 'CalculationPrice', 'ReferenceExchange', 'AMSettlement',
        'ContractSize', 'ExpiryIndicator',
    ]

    # Dtypes compactos usados durante la compilación TXT → CSV.
    # No se aplican en las consultas DuckDB (DuckDB infiere tipos del CSV).
    _DTYPE_IVYOPPRCD: dict[str, str] = {
        'SecurityID':        'int32',
        'OptionID':          'int32',
        'Exchange':          'int16',
        'Currency':          'int16',
        'Expiration':        'int32',    # YYYYMMDD entero en el CSV
        'Strike':            'float32',
        'CallPut':           'category',
        'Symbol':            'object',   # explícito para evitar DtypeWarning por tipos mixtos
        'Bid':               'float32',
        'Ask':               'float32',
        'Last':              'float32',
        'Volume':            'int32',
        'OpenInterest':      'int32',
        'SpecialSettlement': 'int8',
        'ImpliedVolatility': 'float32',
        'Delta':             'float32',
        'Gamma':             'float32',
        'Vega':              'float32',
        'Theta':             'float32',
        'AdjustmentFactor':  'float32',
        'ExerciseStyle':     'category',
        'SymbolFlag':        'int8',
        'CalculationPrice':  'category',
        'ReferenceExchange': 'int16',
        'AMSettlement':      'int8',
        'ContractSize':      'float32',
        'ExpiryIndicator':   'category',
    }

    def __init__(self, sep: str = '\t', encoding: str = 'utf-8'):
        self._loader   = DataLoader(sep=sep, encoding=encoding)
        self._ruta_csv: str | None = None
        self.df: pd.DataFrame | None = None

    # ── Carga ──────────────────────────────────────────────────────────────────

    def cargar_datos(
        self,
        carpeta:    str,
        guardar_en: str | None = None,
        limite:     int | None = None,
    ) -> None:
        """
        Compila todos los archivos .txt de la carpeta IVYOPPRCD y guarda el
        resultado como CSV. Las consultas se ejecutan directamente sobre el CSV
        con DuckDB (sin cargarlo en RAM).

        :param carpeta:    Ruta raíz (contiene la subcarpeta GI.ALL.IVYOPPRCD).
        :param guardar_en: Ruta donde guardar el CSV compilado.
        :param limite:     Si se indica, procesa solo los N archivos más recientes.
                           Útil para pruebas rápidas (ej. limite=3).
        """
        inicio = time.perf_counter()
        self._loader.compilar_carpeta(
            os.path.join(carpeta, "GI.ALL.IVYOPPRCD"),
            guardar_en = guardar_en,
            dtype      = self._DTYPE_IVYOPPRCD,
            names      = self._NAMES_IVYOPPRCD,
            limite     = limite,
        )
        self._ruta_csv = self._loader.ruta_csv
        fin = time.perf_counter()
        print(f"CSV listo en disco. Consultas directas vía DuckDB (sin carga en RAM).")
        print(f"Tiempo total: {fin - inicio:.0f} s")

    def cargar_csv(self, ruta: str) -> None:
        """
        Conecta con un CSV ya compilado. Las consultas usan DuckDB directamente
        sobre el fichero sin cargarlo en RAM.

        :param ruta: Ruta al CSV compilado (p.ej. option_price.csv).
        """
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró: {ruta}")
        self._ruta_csv = ruta
        tam_gb = os.path.getsize(ruta) / 1e9
        print(f"CSV listo ({tam_gb:.1f} GB en disco). Consultas vía DuckDB.")

    def exportar_parquet(self, ruta: str) -> None:
        """
        Convierte el CSV compilado a Parquet (ejecutar una sola vez).
        El Parquet ocupa ~3–4× menos en disco y las consultas son mucho más rápidas.
        Requiere haber llamado antes a cargar_datos() o cargar_csv().

        :param ruta: Ruta de destino del Parquet (p.ej. option_price.parquet).
        """
        if self._ruta_csv is None:
            raise RuntimeError("Llama primero a cargar_datos() o cargar_csv().")
        ruta_p   = ruta.replace('\\', '/')
        csv_p    = self._csv_path()
        print("Convirtiendo CSV → Parquet (puede tardar varios minutos)…")
        inicio = time.perf_counter()
        duckdb.sql(f"""
            COPY (SELECT * FROM read_csv_auto('{csv_p}', header=true))
            TO '{ruta_p}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        fin    = time.perf_counter()
        tam_gb = os.path.getsize(ruta) / 1e9
        print(f"Guardado: {ruta} ({tam_gb:.1f} GB) en {fin - inicio:.0f} s")

    def cargar_parquet(
        self,
        ruta:        str,
        desde:       str | None        = None,
        hasta:       str | None        = None,
        columnas:    list[str] | None  = None,
        security_id: int | list | None = None,
    ) -> None:
        """
        Carga datos del Parquet en self.df como un pandas DataFrame normal.
        Solo los datos del rango y columnas indicados llegan a RAM.

        :param ruta:     Ruta al archivo Parquet.
        :param desde:    Fecha inicio en formato 'YYYY-MM-DD' (inclusive).
        :param hasta:    Fecha fin   en formato 'YYYY-MM-DD' (inclusive).
        :param columnas: Lista de columnas a cargar. None carga todas (28 columnas).

        Ejemplo — cargar solo gregas de 2015 a 2023:
            op.cargar_parquet(
                ruta     = "option_price.parquet",
                desde    = "2015-01-01",
                hasta    = "2023-12-31",
                columnas = ['SecurityID', 'Date', 'Expiration', 'Strike',
                            'CallPut', 'ImpliedVolatility', 'Delta', 'Volume'],
            )
            op.df.groupby('SecurityID')['Volume'].sum()   # pandas puro
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
        if security_id is not None:
            ids = security_id if isinstance(security_id, list) else [security_id]
            condiciones.append(f"SecurityID IN ({', '.join(str(i) for i in ids)})")
        where = ('WHERE ' + ' AND '.join(condiciones)) if condiciones else ''

        print("Cargando en RAM…")
        inicio    = time.perf_counter()
        self.df   = duckdb.sql(f"SELECT {cols} FROM '{ruta_p}' {where}").df()
        fin       = time.perf_counter()

        for fecha_col in ('Date', 'Expiration'):
            if fecha_col in self.df.columns:
                col = self.df[fecha_col]
                if pd.api.types.is_integer_dtype(col):
                    self.df[fecha_col] = pd.to_datetime(
                        col.astype(str), format='%Y%m%d', errors='coerce'
                    )
                else:
                    # format='mixed' (pandas ≥2.0) parsea cada valor individualmente
                    self.df[fecha_col] = pd.to_datetime(col, format='mixed', errors='coerce')

        if 'Currency' in self.df.columns:
            self.df['Currency'] = pd.to_numeric(self.df['Currency'], errors='coerce').astype('int16')

        ram_gb = self.df.memory_usage(deep=True).sum() / 1e9
        print(f"Cargadas {len(self.df):,} filas × {len(self.df.columns)} columnas")
        print(f"RAM: {ram_gb:.1f} GB  |  Tiempo: {fin - inicio:.0f} s")

        # Apuntar _ruta_csv al Parquet para que resumen() y similares también lo usen
        self._ruta_csv = ruta

    # ── Consultas ──────────────────────────────────────────────────────────────

    def opciones_por_subyacente_fecha(self, security_id: int, fecha: str) -> pd.DataFrame:
        """
        Devuelve todas las opciones de un subyacente en una fecha concreta.

        :param security_id: ID del subyacente.
        :param fecha:       Fecha en formato 'YYYY-MM-DD'.
        """
        self._verificar_datos()
        sql = f"""
            SELECT *
            FROM {self._fuente()}
            WHERE SecurityID = {security_id}
              AND Date = '{fecha}'
            ORDER BY Expiration, Strike, CallPut
        """
        sub = self._ejecutar(sql)
        if sub.empty:
            raise ValueError(f"Sin datos para SecurityID={security_id} en {fecha}")
        return sub

    def opciones_por_vencimiento(
        self,
        security_id: int,
        fecha:       str,
        expiration:  str,
        call_put:    str | None = None,
    ) -> pd.DataFrame:
        """
        Devuelve las opciones para un subyacente, fecha y vencimiento concretos.

        :param security_id: ID del subyacente.
        :param fecha:       Fecha en formato 'YYYY-MM-DD'.
        :param expiration:  Vencimiento en formato 'YYYY-MM-DD'.
        :param call_put:    'C', 'P' o None para ambos.
        """
        self._verificar_datos()
        exp_int = int(expiration.replace('-', ''))   # 'YYYY-MM-DD' → YYYYMMDD int
        and_cp  = f"AND CallPut = '{call_put}'" if call_put else ''
        sql = f"""
            SELECT *
            FROM {self._fuente()}
            WHERE SecurityID = {security_id}
              AND Date = '{fecha}'
              AND Expiration = {exp_int}
              {and_cp}
            ORDER BY Strike
        """
        sub = self._ejecutar(sql)
        if sub.empty:
            raise ValueError(
                f"Sin datos para SecurityID={security_id}, "
                f"fecha={fecha}, expiration={expiration}"
            )
        return sub

    def vencimientos_disponibles(self, security_id: int, fecha: str) -> list[str]:
        """Devuelve los vencimientos disponibles para un subyacente y fecha."""
        self._verificar_datos()
        sql = f"""
            SELECT DISTINCT Expiration
            FROM {self._fuente()}
            WHERE SecurityID = {security_id}
              AND Date = '{fecha}'
              AND Expiration IS NOT NULL
            ORDER BY Expiration
        """
        exps = self._ejecutar(sql)['Expiration'].tolist()
        return [
            f"{str(int(e))[:4]}-{str(int(e))[4:6]}-{str(int(e))[6:8]}"
            for e in exps
        ]

    def fechas_disponibles(self, security_id: int) -> list[str]:
        """Devuelve las fechas disponibles para un subyacente concreto."""
        self._verificar_datos()
        sql = f"""
            SELECT DISTINCT Date
            FROM {self._fuente()}
            WHERE SecurityID = {security_id}
            ORDER BY Date
        """
        return self._ejecutar(sql)['Date'].astype(str).tolist()

    def resumen(self) -> None:
        """Imprime un resumen del dataset (requiere un escaneo completo del CSV)."""
        self._verificar_datos()
        sql = f"""
            SELECT
                COUNT(*)                   AS filas,
                COUNT(DISTINCT SecurityID) AS subyacentes,
                COUNT(DISTINCT OptionID)   AS opciones,
                MIN(Date)                  AS fecha_min,
                MAX(Date)                  AS fecha_max,
                COUNT(DISTINCT Expiration) AS vencimientos,
                COUNT(DISTINCT Strike)     AS strikes
            FROM {self._fuente()}
        """
        r = self._ejecutar(sql).iloc[0]
        print(f"Filas              : {int(r['filas']):,}")
        print(f"Subyacentes        : {int(r['subyacentes']):,}")
        print(f"Opciones únicas    : {int(r['opciones']):,}")
        print(f"Rango de fechas    : {r['fecha_min']} → {r['fecha_max']}")
        print(f"Vencimientos únicos: {int(r['vencimientos']):,}")
        print(f"Strikes únicos     : {int(r['strikes']):,}")

    def query(self, sql: str) -> pd.DataFrame:
        """
        Ejecuta SQL arbitrario sobre el CSV vía DuckDB.
        Usa {csv} como placeholder para la fuente de datos.

        Ejemplo:
            op.query("SELECT SecurityID, SUM(Volume) AS vol FROM {csv} GROUP BY SecurityID")
        """
        self._verificar_datos()
        sql_final = sql.replace(
            '{csv}', f"{self._fuente()}"
        )
        return self._ejecutar(sql_final)

    # ── Interno ────────────────────────────────────────────────────────────────

    def _verificar_datos(self) -> None:
        if self._ruta_csv is None:
            raise RuntimeError("Llama primero a cargar_datos(), cargar_csv() o cargar_parquet().")

    def _csv_path(self) -> str:
        """Normaliza la ruta para DuckDB (barras hacia adelante)."""
        return self._ruta_csv.replace('\\', '/')

    def _fuente(self) -> str:
        """Expresión DuckDB correcta según el formato del fichero (CSV o Parquet)."""
        ruta = self._csv_path()
        if ruta.lower().endswith('.parquet'):
            return f"'{ruta}'"
        return f"read_csv_auto('{ruta}', header=true)"

    def _ejecutar(self, sql: str) -> pd.DataFrame:
        """Ejecuta una consulta DuckDB y devuelve un DataFrame."""
        return duckdb.sql(sql).df()


# ─── EJECUCIÓN ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    CSV     = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\option_price.csv"
    PARQUET = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\option_price.parquet"
    CARPETA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado"

    op = OptionPrice(sep='\t')

    # ── Paso 1 (una sola vez): compilar TXT → CSV ─────────────────────────────
    #op.cargar_datos(carpeta=CARPETA, guardar_en=CSV)

    # ── Paso 2 (una sola vez): CSV → Parquet  (más rápido y pequeño) ──────────
    # op.cargar_csv(CSV)
    # op.exportar_parquet(PARQUET)

    # ── Uso habitual: cargar Parquet → self.df (pandas normal) ────────────────
    if Path(PARQUET).exists():
        op.cargar_parquet(
            ruta     = PARQUET,
            desde    = "2010-01-01",
            hasta    = "2023-12-31",
            columnas = [
                'SecurityID', 'Date', 'Expiration', 'Strike',
                'CallPut', 'ImpliedVolatility', 'Delta', 'Volume', 'OpenInterest',
            ]
        )
    else:
        op.cargar_csv(CSV)
    # A partir de aquí: pandas puro
    print(op.df.head())
    print(op.df.groupby('SecurityID')['Volume'].sum().nlargest(10))
