import zipfile
import shutil
import os
import re
import tempfile
import pandas as pd
import time


# ─── CLASE 1: Gestión de archivos ZIP y organización ──────────────────────────

class ZipExtractor:
    """
    Extrae archivos ZIP y organiza el contenido en subcarpetas por tipo.
    """

    def __init__(self, carpeta_zips: str, carpeta_destino: str):
        self.carpeta_zips    = carpeta_zips
        self.carpeta_destino = carpeta_destino

    # GI.<REGION>.<CÓDIGO>_<YYYYMM>  o  GI.<REGION>.<CÓDIGO>  (sin fecha)
    _PATRON_ZIP = re.compile(r'^(GI\.\w+\.\w+?)(?:_\d+)?\.zip$')

    def _tipo_desde_nombre(self, nombre_zip: str) -> str:
        """Devuelve el prefijo de tipo (ej. 'GI.ALL.IVYVSURFD') a partir del nombre del ZIP."""
        m = self._PATRON_ZIP.match(nombre_zip)
        return m.group(1) if m else "OTROS"

    def _extraer_zip(self, archivo_zip: str) -> None:
        """Extrae un ZIP directamente en la subcarpeta de su tipo."""
        nombre_zip   = os.path.basename(archivo_zip)
        tipo         = self._tipo_desde_nombre(nombre_zip)
        carpeta_tipo = os.path.join(self.carpeta_destino, tipo)
        os.makedirs(carpeta_tipo, exist_ok=True)

        if not zipfile.is_zipfile(archivo_zip):
            print(f"⚠️  '{nombre_zip}' no es un ZIP válido — omitido.")
            return

        with zipfile.ZipFile(archivo_zip, 'r') as zip_ref:
            zip_ref.extractall(carpeta_tipo)

        print(f"✅ '{nombre_zip}' → {tipo}/")

    def procesar_todos(self, limite: int | None = None) -> None:
        """Procesa los .zip dentro de carpeta_zips. Con limite=N procesa solo los N primeros."""
        inicio = time.perf_counter()
        zips = sorted(f for f in os.listdir(self.carpeta_zips) if f.endswith('.zip'))[:limite]

        if not zips:
            print("No se encontraron archivos ZIP.")
            return

        for nombre_zip in zips:
            self._extraer_zip(os.path.join(self.carpeta_zips, nombre_zip))

        fin = time.perf_counter()
        print(f"\n Proceso completado. Contenido acumulado en: {self.carpeta_destino} \n Tiempo transcurrido: {fin-inicio:.1f} s")

    # Categorías oficiales IvyDB GI (código corto → nombre canónico)
    TIPOS_IVYDB: dict[str, str] = {
        'IVYSECNM':    'Security_Name',
        'IVYSECPR':    'Security_Price',
        'IVYFUTPRC':   'Future_Price',
        'IVYFUTPRCD':  'Future_Price',
        'IVYOPPRC':    'Option_Price',
        'IVYIDXDV':    'Index_Dividend',
        'DISTR':       'Distribution',
        'IVYVSURF':    'Volatility_Surface',
        'IVYSTDOP':    'Std_Option_Price',
        'IVYHISTVOL':  'Historical_Volatility',
        'IVYZEROC':    'Zero_Curve',
        'IVYCURRENCY': 'Currency',
        'IVYEXCHNG':   'Exchange',
        'IVYEXCHNGD':  'Exchange',
        'IVYDISTRPROJ':'Distribution_Projection',
        'IVYFWDPR':    'Forward_Price',
    }

    def agrupar_por_tipo(self) -> None:
        """
        Agrupa archivos IvyDB GI en subcarpetas manteniendo el nombre original
        (ej. GI.ALL.IVYVSURFD). Soporta cualquier región y fechas de 6 u 8 dígitos.
        Actúa sobre carpeta_destino.
        """
        patron = re.compile(r'^(GI\.\w+\.\w+)_\d{6,8}$')

        archivos = [f for f in os.listdir(self.carpeta_destino)
                    if os.path.isfile(os.path.join(self.carpeta_destino, f))]

        agrupados: dict[str, list[str]] = {}
        for archivo in archivos:
            nombre_sin_ext = os.path.splitext(archivo)[0]
            match = patron.match(nombre_sin_ext)
            tipo = match.group(1) if match else "OTROS"
            agrupados.setdefault(tipo, []).append(archivo)

        for tipo, lista in agrupados.items():
            carpeta_tipo = os.path.join(self.carpeta_destino, tipo)
            os.makedirs(carpeta_tipo, exist_ok=True)
            for archivo in lista:
                shutil.move(
                    os.path.join(self.carpeta_destino, archivo),
                    os.path.join(carpeta_tipo, archivo)
                )
            print(f"📁 {tipo}: {len(lista)} archivo(s) movido(s)")

        print("\n✅ Agrupación completada.")


# ─── CLASE 2: Carga de datos en DataFrames ────────────────────────────────────

class DataLoader:
    """
    Carga archivos de datos (CSV/TXT tabulados) en DataFrames de pandas.
    """

    def __init__(self, sep: str = '\t', encoding: str = 'utf-8'):
        self.sep      = sep
        self.encoding = encoding
        self.df: pd.DataFrame | None = None
        self.ruta_csv: str | None = None    # ruta al CSV compilado (puede ser >RAM)

    def cargar(self, ruta: str) -> pd.DataFrame:
        """Carga un archivo y lo almacena en self.df."""
        self.df = pd.read_csv(ruta, sep=self.sep, encoding=self.encoding)
        print(f"✅ Cargado: {ruta}  →  {self.df.shape[0]} filas × {self.df.shape[1]} columnas")
        return self.df

    def compilar_carpeta(
        self,
        carpeta:    str,
        guardar_en: str | None  = None,
        dtype:      dict | None = None,
        names:      list | None = None,
        limite:     int  | None = None,
    ) -> pd.DataFrame:
        """
        Lee todos los .txt de una carpeta, los concatena en un único DataFrame
        ordenado por la columna Date y lo almacena en self.df.

        :param carpeta:    Ruta a la carpeta (ej. .../GI.ALL.IVYFUTPRCD)
        :param guardar_en: Si se indica, guarda el resultado como CSV en esa ruta.
        :param dtype:      Diccionario de dtypes para pd.read_csv. Usar tipos
                           compactos (float32, int32, 'category') para ahorrar RAM
                           y evitar OOM en el pd.concat.
        :param names:      Nombres de columna. Si se indica, los archivos sin
                           cabecera se leen con header=None y estos nombres.
                           Los archivos que sí tienen cabecera se leen normalmente.
        :param limite:     Si se indica, procesa solo los N primeros archivos.
                           Útil para pruebas rápidas (ej. limite=3).
        """
        # Descendente: los archivos más recientes (con cabecera) se procesan primero.
        archivos = sorted(
            [f for f in os.listdir(carpeta)
             if f.endswith('.txt') and os.path.isfile(os.path.join(carpeta, f))],
            reverse=True,
        )[:limite]

        if not archivos:
            raise FileNotFoundError(f"No se encontraron archivos .txt en: {carpeta}")

        # Escribe cada trozo directamente al CSV de destino (o a uno temporal)
        # para evitar el pd.concat, que necesita un bloque contiguo gigante en RAM.
        usar_temp   = guardar_en is None
        tmp         = tempfile.NamedTemporaryFile(suffix='.csv', delete=False) if usar_temp else None
        ruta_salida = tmp.name if usar_temp else guardar_en
        if tmp:
            tmp.close()

        try:
            primera = True
            total_filas = 0
            ncols = None
            for nombre in archivos:
                ruta = os.path.join(carpeta, nombre)

                # Detectar si el archivo tiene cabecera comprobando su primera celda.
                if names:
                    primera_celda = pd.read_csv(
                        ruta, sep=self.sep, encoding=self.encoding,
                        nrows=1, header=None,
                    ).iloc[0, 0]
                    tiene_header = str(primera_celda).strip() == names[0]
                    df_trozo = pd.read_csv(
                        ruta, sep=self.sep, encoding=self.encoding, dtype=dtype,
                        header = 0    if tiene_header else None,
                        names  = None if tiene_header else names,
                    )
                else:
                    df_trozo = pd.read_csv(ruta, sep=self.sep, encoding=self.encoding, dtype=dtype)

                if 'Date' in df_trozo.columns:
                    df_trozo['Date'] = pd.to_datetime(df_trozo['Date'], format='%Y%m%d')
                total_filas += len(df_trozo)
                ncols        = df_trozo.shape[1]
                print(f"{nombre}  →  {len(df_trozo):,} filas")
                df_trozo.to_csv(ruta_salida, mode='w' if primera else 'a',
                                header=primera, index=False, encoding='utf-8')
                primera = False

            print(f"\n✅ Compilados {len(archivos)} archivos  →  {total_filas:,} filas × {ncols} columnas")
            if guardar_en:
                print(f"Guardado en: {guardar_en}")

            # El CSV resultante puede superar la RAM disponible (ej. 36 GB para
            # 235 M filas). No se carga completo: solo se almacena la ruta y se
            # lee la cabecera para exponer el esquema.  Las consultas deben usar
            # pd.read_csv(chunksize=...) o, para mejor rendimiento,
            #   pip install duckdb  →  duckdb.query("SELECT … FROM read_csv_auto(ruta)")
            self.ruta_csv = ruta_salida
            self.df = pd.read_csv(ruta_salida, nrows=0, encoding='utf-8')

        finally:
            if usar_temp and os.path.exists(ruta_salida):
                os.remove(ruta_salida)

        return self.df

    def resumen(self) -> None:
        """Imprime un resumen básico del DataFrame cargado."""
        if self.df is None:
            print("No hay datos cargados.")
            return
        print("\n── Primeras filas ──")
        print(self.df.head())
        print("\n── Columnas ──")
        print(self.df.columns.tolist())
        print("\n── Dimensiones ──")
        print(self.df.shape)


# ─── USO DIRECTO ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Extraer y organizar ZIPs
    extractor = ZipExtractor(
        carpeta_zips    = r"C:\Users\pablo.esparcia\OneDrive - UNIVERSIDAD ALICANTE\OptionMetrics\OptionMetrics\v3.3\ALL\History",
        carpeta_destino = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado")
    
    extractor.procesar_todos()  # limite=200

    # 2. Compilar todos los archivos de una carpeta en un único DataFrame
    # loader = DataLoader(sep='\t')
    # df = loader.compilar_carpeta(
    #     carpeta    = r"G:\Mi unidad\OptionMetrics\pruebas\acumulado\GI.ALL.IVYFUTPRCD",
    #     guardar_en = r"G:\Mi unidad\OptionMetrics\pruebas\acumulado\GI.ALL.IVYFUTPRCD.csv",
    # )
    # loader.resumen()
