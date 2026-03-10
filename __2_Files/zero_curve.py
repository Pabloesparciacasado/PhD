import pandas as pd
import duckdb
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __1_Input_Data.data_ingestion import DataLoader


# ─── CLASE: Zero Curve ──────────────────────────────────────────────────────────

class ZeroCurve:
    """
    Carga y consulta de Zero Curve (IvyDB GI.ALL.IVYZEROCD).

    Estructura del dato:
        Currency | Date | Days | Rate

    Contiene la curva de tipos de interés cero cupón por divisa, fecha de
    observación y plazo en días. No tiene SecurityID: la dimensión clave es
    la divisa (Currency code IvyDB).

    El CSV compilado se consulta directamente con DuckDB sin cargarlo en RAM.
    Solo el subconjunto filtrado se materializa en un DataFrame.

    Requiere: pip install duckdb
    """

    # Nombres de columna en el orden que tienen los TXT (archivos sin cabecera).
    _NAMES_IVYZEROCD: list[str] = [
        'Currency', 'Date', 'Days', 'Rate',
    ]

    # Dtypes compactos usados durante la compilación TXT → CSV.
    _DTYPE_IVYZEROCD: dict[str, str] = {
        'Currency': 'int16',
        'Days':     'int16',
        'Rate':     'float32',
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
        Compila todos los archivos .txt de la carpeta IVYZEROCD y guarda el
        resultado como CSV. Las consultas se ejecutan directamente sobre el CSV
        con DuckDB (sin cargarlo en RAM).

        :param carpeta:    Ruta raíz (contiene la subcarpeta GI.ALL.IVYZEROCD).
        :param guardar_en: Ruta donde guardar el CSV compilado.
        :param limite:     Si se indica, procesa solo los N archivos más recientes.
        """
        inicio = time.perf_counter()
        self._loader.compilar_carpeta(
            os.path.join(carpeta, "GI.ALL.IVYZEROCD"),
            guardar_en = guardar_en,
            dtype      = self._DTYPE_IVYZEROCD,
            names      = self._NAMES_IVYZEROCD,
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

        :param ruta: Ruta al CSV compilado (p.ej. zero_curve.csv).
        """
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"No se encontró: {ruta}")
        self._ruta_csv = ruta
        tam_gb = os.path.getsize(ruta) / 1e9
        print(f"CSV listo ({tam_gb:.2f} GB en disco). Consultas vía DuckDB.")

    def exportar_parquet(self, ruta: str) -> None:
        """
        Convierte el CSV compilado a Parquet (ejecutar una sola vez).
        El Parquet ocupa ~3–4× menos en disco y las consultas son mucho más rápidas.
        Requiere haber llamado antes a cargar_datos() o cargar_csv().

        :param ruta: Ruta de destino del Parquet (p.ej. zero_curve.parquet).
        """
        if self._ruta_csv is None:
            raise RuntimeError("Llama primero a cargar_datos() o cargar_csv().")
        ruta_p = ruta.replace('\\', '/')
        csv_p  = self._csv_path()
        print("Convirtiendo CSV → Parquet…")
        inicio = time.perf_counter()
        duckdb.sql(f"""
            COPY (SELECT * FROM read_csv_auto('{csv_p}', header=true))
            TO '{ruta_p}' (FORMAT PARQUET, COMPRESSION SNAPPY)
        """)
        fin    = time.perf_counter()
        tam_gb = os.path.getsize(ruta) / 1e9
        print(f"Guardado: {ruta} ({tam_gb:.2f} GB) en {fin - inicio:.0f} s")

    def cargar_parquet(
        self,
        ruta:     str,
        desde:    str | None       = None,
        hasta:    str | None       = None,
        columnas: list[str] | None = None,
    ) -> None:
        """
        Carga datos del Parquet en self.df como un pandas DataFrame normal.
        Solo los datos del rango y columnas indicados llegan a RAM.

        :param ruta:     Ruta al archivo Parquet.
        :param desde:    Fecha inicio en formato 'YYYY-MM-DD' (inclusive).
        :param hasta:    Fecha fin   en formato 'YYYY-MM-DD' (inclusive).
        :param columnas: Lista de columnas a cargar. None carga todas.

        Ejemplo:
            zc.cargar_parquet(
                ruta  = "zero_curve.parquet",
                desde = "2010-01-01",
                hasta = "2023-12-31",
            )
            zc.df[zc.df['Currency'] == 333]   # pandas puro
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
            if date_is_int:
                return fecha_str.replace('-', '')
            return f"'{fecha_str}'"

        condiciones = []
        if desde:
            condiciones.append(f"Date >= {_fmt_fecha(desde)}")
        if hasta:
            condiciones.append(f"Date <= {_fmt_fecha(hasta)}")
        where = ('WHERE ' + ' AND '.join(condiciones)) if condiciones else ''

        print("Cargando en RAM…")
        inicio  = time.perf_counter()
        self.df = duckdb.sql(f"SELECT {cols} FROM '{ruta_p}' {where}").df()
        fin     = time.perf_counter()

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
        print(f"RAM: {ram_gb:.2f} GB  |  Tiempo: {fin - inicio:.0f} s")

        self._ruta_csv = ruta

    # ── Resumen ────────────────────────────────────────────────────────────────

    def resumen(self) -> None:
        """Imprime un resumen del dataset (requiere un escaneo completo del fichero)."""
        self._verificar_datos()
        sql = f"""
            SELECT
                COUNT(*)                AS filas,
                COUNT(DISTINCT Currency) AS divisas,
                MIN(Date)               AS fecha_min,
                MAX(Date)               AS fecha_max,
                COUNT(DISTINCT Days)    AS plazos
            FROM {self._fuente()}
        """
        r = self._ejecutar(sql).iloc[0]
        print(f"Filas          : {int(r['filas']):,}")
        print(f"Divisas        : {int(r['divisas'])}")
        print(f"Rango de fechas: {r['fecha_min']} → {r['fecha_max']}")
        print(f"Plazos únicos  : {int(r['plazos'])}")

    def query(self, sql: str) -> pd.DataFrame:
        """
        Ejecuta SQL arbitrario sobre el fichero vía DuckDB.
        Usa {csv} como placeholder para la fuente de datos.

        Ejemplo:
            zc.query("SELECT Currency, Days, AVG(Rate) FROM {csv} GROUP BY Currency, Days")
        """
        self._verificar_datos()
        return self._ejecutar(sql.replace('{csv}', self._fuente()))

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

    CSV     = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\zero_curve.csv"
    PARQUET = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\zero_curve.parquet"
    CARPETA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado"

    zc = ZeroCurve(sep='\t')

    # ── Paso 1 (una sola vez): compilar TXT → CSV ─────────────────────────────
    # zc.cargar_datos(carpeta=CARPETA, guardar_en=CSV)

    # ── Paso 2 (una sola vez): CSV → Parquet  (más rápido y pequeño) ──────────
    # zc.cargar_csv(CSV)
    # zc.exportar_parquet(PARQUET)

    # ── Uso habitual: cargar Parquet → self.df (pandas normal) ────────────────
    zc.cargar_parquet(
        ruta  = PARQUET,
        desde = "2010-01-01",   # ajusta el rango según necesites
        hasta = "2023-12-31",
    )

    # A partir de aquí: pandas puro
    print(zc.df.head())
    zc.resumen()
