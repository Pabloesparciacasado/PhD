# Documentación del proyecto — Tratamiento de datos OptionMetrics IvyDB

## Índice

1. [Arquitectura general](#1-arquitectura-general)
2. [Fuente de datos: IvyDB GI](#2-fuente-de-datos-ivydb-gi)
3. [data_ingestion.py — capa de ingesta](#3-data_ingestionpy--capa-de-ingesta)
4. [option_price.py — capa de negocio](#4-option_pricepy--capa-de-negocio)
5. [volatility_surface.py — capa de negocio](#5-volatility_surfacepy--capa-de-negocio)
6. [volatility_surface_eda.py — análisis de superficie de volatilidad](#6-volatility_surface_edapy--análisis-de-superficie-de-volatilidad)
7. [option_price_eda.py — análisis de precios de opciones](#7-option_price_edapy--análisis-de-precios-de-opciones)
8. [Flujo completo de ejecución](#8-flujo-completo-de-ejecución)
9. [Decisiones de diseño y justificaciones](#9-decisiones-de-diseño-y-justificaciones)

---

## 1. Arquitectura general

El proyecto tiene tres capas bien separadas:

```
__1_Input_Data/
    data_ingestion.py         ← Capa de ingesta: ZIP, TXT → CSV
__2_Files/
    option_price.py           ← Capa de negocio: Option Price
    volatility_surface.py     ← Capa de negocio: Volatility Surface
__3_Analysis/
    volatility_surface_eda.py ← Análisis: Volatility Surface
    option_price_eda.py       ← Análisis: Option Price
```

**Capa de ingesta** (`__1_Input_Data`): no sabe nada del dominio financiero.
Solo sabe extraer ZIPs, leer TXT tabulados y escribir CSVs.

**Capa de negocio** (`__2_Files`): no sabe dónde están los archivos ni cómo se leen.
Solo sabe qué columnas tiene cada producto IvyDB y qué consultas financieras tienen sentido.

**Capa de análisis** (`__3_Analysis`): usa las clases de negocio para hacer análisis
y visualizaciones. No lee ficheros directamente.

Esta separación permite reutilizar `DataLoader` para cualquier producto IvyDB
(IVYFUTPRCD, IVYSECPRD, etc.) simplemente creando una nueva clase en `__2_Files`.

### Ciclo de vida de los datos

```
TXT (mensual)  →[compilar_carpeta]→  CSV (36 GB)  →[exportar_parquet]→  Parquet (~9 GB)
                                                                              ↓
                                                                    [cargar_parquet]
                                                                              ↓
                                                                         self.df  (pandas)
```

El CSV es un paso intermedio necesario para la compilación. El Parquet es el
formato final de trabajo: compacto, rápido y compatible con pandas.

---

## 2. Fuente de datos: IvyDB GI

**IvyDB GI** (OptionMetrics) distribuye sus datos en archivos `.txt` tabulados con
separador `\t` (tabulador). Cada archivo cubre un mes: `GI.ALL.IVYOPPRCD_202401.txt`.

### Convención de nombres

```
GI  .  ALL  .  IVYOPPRCD  _  202401  .txt
│      │       │              │
│      │       │              └─ Año + mes (YYYYMM)
│      │       └─ Código del producto
│      └─ Región (ALL = global)
└─ Prefijo IvyDB GI
```

### Productos relevantes en este proyecto

| Código        | Contenido                                 |
|---------------|-------------------------------------------|
| `IVYOPPRCD`   | Option Price — precios, griegas, volumen  |
| `IVYVSURFD`   | Volatility Surface — superficie de vol IV |
| `IVYFUTPRCD`  | Futures Price                             |
| `IVYSECPRD`   | Security Price — precios del subyacente   |

### Importante: cabeceras en los TXT

- **Archivos 1996–2023**: **sin cabecera**. La primera fila ya es un dato.
- **Archivos 2024 en adelante**: **con cabecera**. La primera fila tiene los nombres
  de columna (`SecurityID`, `Date`, etc.)

Este cambio en el formato es clave y explica varias decisiones del código
(véase sección 9).

---

## 3. `data_ingestion.py` — capa de ingesta

### Clase `ZipExtractor`

Gestiona la extracción y organización de los ZIPs descargados de OptionMetrics.

```python
_PATRON_ZIP = re.compile(r'^(GI\.\w+\.\w+?)(?:_\d+)?\.zip$')
```
Expresión regular que extrae el tipo de producto del nombre del ZIP.
Ejemplo: `GI.ALL.IVYOPPRCD_202401.zip` → tipo = `GI.ALL.IVYOPPRCD`.
El `(?:_\d+)?` hace la parte de la fecha opcional (algunos ZIPs no la llevan).

```python
def _extraer_zip(self, archivo_zip):
    tipo         = self._tipo_desde_nombre(nombre_zip)
    carpeta_tipo = os.path.join(self.carpeta_destino, tipo)
    os.makedirs(carpeta_tipo, exist_ok=True)
    zip_ref.extractall(carpeta_tipo)
```
Extrae el ZIP directamente en su subcarpeta de tipo. Así todos los `.txt` de
Option Price van a `Acumulado/GI.ALL.IVYOPPRCD/`, los de Vol Surface a
`Acumulado/GI.ALL.IVYVSURFD/`, etc.

---

### Clase `DataLoader`

Es el núcleo técnico del proyecto. Implementa la compilación de múltiples TXT
en un único CSV sin explotar la RAM.

```python
def __init__(self, sep='\t', encoding='utf-8'):
    self.sep      = sep
    self.encoding = encoding
    self.df:       pd.DataFrame | None = None
    self.ruta_csv: str | None          = None
```
`ruta_csv` se añadió para que las clases de negocio puedan saber la ruta del
CSV resultante sin que `DataLoader` tenga que cargarlo en RAM.

---

#### `compilar_carpeta` — el método más importante

```python
def compilar_carpeta(self, carpeta, guardar_en=None, dtype=None, names=None, limite=None):
```

**Parámetros:**
- `carpeta`: ruta a la subcarpeta con los TXT (ej. `.../GI.ALL.IVYOPPRCD`)
- `guardar_en`: ruta del CSV de salida
- `dtype`: diccionario de tipos compactos (evita OOM en datasets grandes)
- `names`: lista de nombres de columna — necesario para archivos sin cabecera
- `limite`: procesar solo los N archivos más recientes (pruebas)

**Por qué `reverse=True`:**

```python
archivos = sorted([...], reverse=True)[:limite]
```

Los archivos se ordenan de **más reciente a más antiguo**. Esto es crucial porque:
- Los archivos de 2024+ tienen cabecera → pandas detecta los nombres de columna
  correctamente en el primer archivo procesado.
- Los archivos de 1996-2023 no tienen cabecera → se usan los `names` proporcionados.

**Por qué escritura incremental al CSV en lugar de `pd.concat`:**

```python
for nombre in archivos:
    df_trozo.to_csv(ruta_salida, mode='w' if primera else 'a',
                    header=primera, index=False, encoding='utf-8')
```

`pd.concat(trozos)` necesita tener **todos los DataFrames en RAM simultáneamente**.
Con 338 archivos × ~2M filas = 235M filas, esto requeriría ~100 GB de RAM.

La escritura incremental usa solo RAM para un archivo a la vez (~330 MB),
sin importar cuántos archivos haya en total.

**Por qué no re-leer el CSV completo al final:**

```python
# ANTES (causaba OOM con 36 GB):
self.df = pd.read_csv(ruta_salida, low_memory=False)

# AHORA:
self.ruta_csv = ruta_salida
self.df = pd.read_csv(ruta_salida, nrows=0)  # solo cabecera (esquema)
```

El CSV resultante pesa 36 GB. La clase de negocio decide cómo cargarlo
(con DuckDB, con Parquet, etc.). `DataLoader` solo guarda la ruta.

---

## 4. `option_price.py` — capa de negocio

### El problema de RAM

El dataset completo tiene 235M filas × 28 columnas. Con dtypes por defecto de
pandas serían ~100 GB en RAM. Con dtypes compactos se reduce a ~34 GB, que
sigue siendo demasiado para la mayoría de máquinas.

**Solución adoptada:** DuckDB + Parquet.
- **DuckDB** puede consultar ficheros CSV o Parquet directamente en disco sin
  cargarlos enteros en RAM. Solo trae a memoria el resultado de cada consulta.
- **Parquet** es un formato columnar comprimido: los 36 GB del CSV pasan a ~9 GB,
  y las consultas son 10-50× más rápidas que sobre CSV.
- **`cargar_parquet`** permite cargar un subconjunto filtrado (rango de fechas +
  columnas elegidas) directamente en `self.df` como pandas DataFrame normal.

### Constantes de clase

```python
_NAMES_IVYOPPRCD: list[str] = [
    'SecurityID', 'Date', 'OptionID', 'Exchange', 'Currency', 'Expiration',
    'Strike', 'CallPut', 'Symbol', 'Bid', 'Ask', 'Last', 'Volume',
    ...
]
```
Lista con los 28 nombres de columna en el orden exacto de los TXT.
Necesaria para los archivos 1996-2023 que no tienen cabecera.

```python
_DTYPE_IVYOPPRCD: dict[str, str] = {
    'SecurityID':        'int32',   # int64 (8B) → int32 (4B)
    'Exchange':          'int16',   # código de bolsa, int16 (máx 32767) suficiente
    'Expiration':        'int32',   # YYYYMMDD entero; se convierte a datetime tras carga
    'Strike':            'float32', # float64 (8B) → float32 (4B)
    'CallPut':           'category',# solo 'C' o 'P' → category: ~1B/fila vs ~50B object
    'Symbol':            'object',  # valores casi únicos, category no ayuda aquí
    ...
}
```

Estos dtypes se usan **durante la compilación TXT → CSV** para escribir datos
eficientes. En las consultas DuckDB sobre el fichero ya compilado, DuckDB infiere
los tipos directamente del CSV o Parquet.

**Ahorro de RAM con dtypes compactos:**

| Tipo default | Tipo compacto | Bytes/fila antes | Bytes/fila después |
|---|---|---|---|
| `float64` | `float32` | 8 | 4 |
| `int64` | `int32` | 8 | 4 |
| `int64` | `int16` | 8 | 2 |
| `object` | `category` | ~50 | ~1 |

Con 235M filas: **~100 GB default → ~34 GB compactos**.

---

### Métodos de carga

#### `cargar_datos`

```python
def cargar_datos(self, carpeta, guardar_en=None, limite=None):
    self._loader.compilar_carpeta(
        os.path.join(carpeta, "GI.ALL.IVYOPPRCD"),
        guardar_en = guardar_en,
        dtype      = self._DTYPE_IVYOPPRCD,
        names      = self._NAMES_IVYOPPRCD,
        limite     = limite,
    )
    self._ruta_csv = self._loader.ruta_csv
```

Compila los TXT al CSV y almacena la ruta. **No carga el CSV en RAM.**
Después de esto, el CSV está listo para consultas DuckDB o para exportar a Parquet.

`limite=5` permite probar con solo los 5 archivos más recientes sin tocar el código.

#### `cargar_csv`

```python
def cargar_csv(self, ruta):
    self._ruta_csv = ruta
    print(f"CSV listo ({tam_gb:.1f} GB en disco). Consultas vía DuckDB.")
```

Para cuando el CSV ya existe. Solo almacena la ruta; el CSV **no se carga en RAM**.
Las consultas posteriores usan DuckDB directamente sobre el fichero.

#### `exportar_parquet`

```python
def exportar_parquet(self, ruta):
    duckdb.sql(f"""
        COPY (SELECT * FROM read_csv_auto('{csv_p}', header=true))
        TO '{ruta_p}' (FORMAT PARQUET, COMPRESSION SNAPPY)
    """)
```

Convierte el CSV a Parquet usando DuckDB (mucho más rápido que pandas para
esta operación). Se ejecuta **una sola vez**. El Parquet resultante:
- Ocupa ~9 GB en disco (vs 36 GB del CSV)
- Se consulta 10-50× más rápido
- Permite filtrar por columnas y fechas antes de cargar en RAM

Requiere haber llamado antes a `cargar_datos()` o `cargar_csv()`.

#### `cargar_parquet` — método principal de uso habitual

```python
def cargar_parquet(self, ruta, desde=None, hasta=None, columnas=None):
    cols  = ', '.join(columnas) if columnas else '*'
    where = f"WHERE Date >= '{desde}' AND Date <= '{hasta}'"
    self.df = duckdb.sql(f"SELECT {cols} FROM '{ruta_p}' {where}").df()
    for fecha_col in ('Date', 'Expiration'):
        if fecha_col in self.df.columns:
            col = self.df[fecha_col]
            if pd.api.types.is_integer_dtype(col):
                self.df[fecha_col] = pd.to_datetime(col.astype(str), format='%Y%m%d', errors='coerce')
            else:
                self.df[fecha_col] = pd.to_datetime(col, format='mixed', errors='coerce')
    self._ruta_csv = ruta   # apunta al Parquet para métodos DuckDB
```

DuckDB lee solo las columnas y filas pedidas del Parquet (no lo carga entero),
materializa el resultado y lo convierte a un pandas DataFrame. A partir de
aquí `self.df` es un pandas DataFrame completamente normal.

La conversión de fechas usa `format='mixed'` para manejar los dos formatos
que coexisten en el Parquet (ver sección 5 para la explicación completa).

**Control de RAM mediante columnas y fechas:**

```
~1 GB por columna × 10 años de datos (estimación aproximada)
```

Ejemplos:
```python
# Solo volatilidad y gregas, 10 años → ~8 GB RAM
op.cargar_parquet(
    ruta     = "option_price.parquet",
    desde    = "2013-01-01",
    hasta    = "2023-12-31",
    columnas = ['SecurityID', 'Date', 'Expiration', 'Strike',
                'CallPut', 'ImpliedVolatility', 'Delta', 'Volume'],
)

# Todo el histórico, todas las columnas → ~34 GB RAM (solo si hay suficiente RAM)
op.cargar_parquet(ruta="option_price.parquet")
```

Una vez cargado, `op.df` se usa como cualquier DataFrame pandas:
```python
df = op.df
df.groupby('SecurityID')['Volume'].sum()
df[df['CallPut'] == 'C']
df[df['Date'].dt.year == 2020]
```

---

### Métodos de consulta DuckDB

Estos métodos consultan el fichero completo (CSV o Parquet) sin cargarlo en RAM.
Son útiles para obtener subconjuntos específicos o estadísticas del histórico completo.

#### `_fuente()` — detección automática de formato

```python
def _fuente(self) -> str:
    ruta = self._csv_path()
    if ruta.lower().endswith('.parquet'):
        return f"'{ruta}'"                          # DuckDB lee Parquet nativamente
    return f"read_csv_auto('{ruta}', header=true)"  # CSV con cabecera
```

Permite que todos los métodos DuckDB funcionen tanto si el fichero es CSV como
Parquet, sin que el usuario tenga que preocuparse por el formato.

#### `opciones_por_subyacente_fecha`

```python
sql = f"""
    SELECT *
    FROM {self._fuente()}
    WHERE SecurityID = {security_id}
      AND Date = '{fecha}'
    ORDER BY Expiration, Strike, CallPut
"""
return self._ejecutar(sql)
```

DuckDB escanea el fichero aplicando el filtro en disco. Solo las filas que
coinciden llegan a RAM como DataFrame.

#### `opciones_por_vencimiento`

```python
exp_int = int(expiration.replace('-', ''))   # 'YYYY-MM-DD' → YYYYMMDD int
```

`Expiration` se almacena como entero YYYYMMDD en el CSV/Parquet (el dtype
`int32` se preserva). La conversión `'2023-03-17'` → `20230317` se hace
en Python antes de construir la query SQL.

#### `vencimientos_disponibles` y `fechas_disponibles`

Consultas `SELECT DISTINCT` sobre el fichero completo. Devuelven listas de
strings en formato `'YYYY-MM-DD'`, coherentes con el formato de entrada
de los demás métodos.

#### `resumen`

```python
sql = f"""
    SELECT
        COUNT(*)                   AS filas,
        COUNT(DISTINCT SecurityID) AS subyacentes,
        ...
    FROM {self._fuente()}
"""
```

Agrega sobre todo el fichero (235M filas). Con Parquet tarda ~30 s;
con CSV puede tardar varios minutos.

#### `query`

```python
def query(self, sql: str) -> pd.DataFrame:
    sql_final = sql.replace('{csv}', self._fuente())
    return self._ejecutar(sql_final)
```

Permite SQL arbitrario usando `{csv}` como placeholder:
```python
op.query("SELECT SecurityID, SUM(Volume) AS vol FROM {csv} GROUP BY SecurityID")
```

---

### Métodos internos

#### `_verificar_datos`

```python
def _verificar_datos(self):
    if self._ruta_csv is None:
        raise RuntimeError("Llama primero a cargar_datos(), cargar_csv() o cargar_parquet().")
```

Todos los métodos de consulta DuckDB llaman a este método primero.
Garantiza un error claro si se olvida llamar a un método de carga.

#### `_ejecutar`

```python
def _ejecutar(self, sql: str) -> pd.DataFrame:
    return duckdb.sql(sql).df()
```

Punto único de ejecución DuckDB → pandas. Toda la interacción con DuckDB
pasa por aquí, facilitando cambios futuros (logging, caché, etc.).

---

## 5. `volatility_surface.py` — capa de negocio

El dataset de Volatility Surface es mucho más pequeño que Option Price
(normalmente cabe en RAM), por lo que la estrategia es diferente:
- `cargar_csv` carga el CSV **completo en RAM** con dtypes compactos (no OOM)
- `cargar_parquet` usa DuckDB para filtrar si se quiere, pero para VS normalmente
  se carga todo
- Todos los métodos de consulta operan directamente sobre `self.df` en pandas

```python
_DTYPE_IVYVSURFD = {
    'SecurityID': 'int32',
    'Days':       'int16',  # días al vencimiento; raramente >1000, int16 suficiente
    'Delta':      'int16',  # expresado como entero 0-100 en IvyDB (no 0.00-1.00)
    'CallPut':    'category',
    'ImpliedVol': 'float32',
    'Strike':     'float32',
    'Premium':    'float32',
    'Dispersion': 'float32',
    'Currency':   'category',
}
```

`Delta` es `int16` porque IvyDB almacena el delta como entero (25, 50, 75),
no como decimal (0.25, 0.50, 0.75).

### Métodos de carga

#### `cargar_datos` y `cargar_csv`

Igual que en `OptionPrice`, pero `_cargar_desde_csv` carga el CSV completo en RAM
(porque VS es pequeño). Sin conversión de `Expiration` (VS no tiene esa columna).

#### `exportar_parquet`

```python
def exportar_parquet(self, ruta):
    ruta_p = ruta.replace('\\', '/')
    duckdb.from_df(self.df).write_parquet(ruta_p, compression='snappy')
```

Para VS el DataFrame ya está en RAM, así que se exporta con `duckdb.from_df()`.
No requiere `pyarrow` instalado (DuckDB tiene su propio motor Parquet integrado).

#### `cargar_parquet`

```python
def cargar_parquet(self, ruta, desde=None, hasta=None, columnas=None):
    self.df = duckdb.sql(f"SELECT {cols} FROM '{ruta_p}' {where}").df()
    if 'Date' in self.df.columns:
        col = self.df['Date']
        if pd.api.types.is_integer_dtype(col):
            self.df['Date'] = pd.to_datetime(col.astype(str), format='%Y%m%d', errors='coerce')
        else:
            self.df['Date'] = pd.to_datetime(col, format='mixed', errors='coerce')
```

Igual que en OP pero sin conversión de `Expiration`. La conversión de `Date`
usa `format='mixed'` porque el Parquet almacena fechas como VARCHAR con dos
formatos mezclados según el período:

- **1996–feb 2023**: `YYYYMMDD` sin separadores (ej. `"20150901"`) — archivos TXT originales sin cabecera
- **mar 2023–2024**: `YYYY-MM-DD` con separadores (ej. `"2023-03-01"`) — archivos TXT con cabecera, donde DuckDB infiere el tipo como fecha y lo serializa en ISO

pandas ≥2.0 con `format='mixed'` parsea cada valor individualmente.
Sin este parámetro, pandas 3.x infiere el formato del primer elemento
del array y aplica ese mismo formato a todos, convirtiendo a `NaT`
el 95 % de las filas (todas las anteriores a marzo 2023).

### Métodos de consulta (pandas puro)

Todos los métodos de VS operan sobre `self.df` directamente porque el dataset
cabe en RAM:

#### `superficie`

```python
return sub.pivot_table(
    index='Days', columns='Delta', values='ImpliedVol', aggfunc='mean'
)
```

Devuelve una **matriz pivotada** Days × Delta. Cada celda es la IV media para
ese par (tenor, moneyness). Ideal para heatmaps o interpolación.

`aggfunc='mean'`: por si hay duplicados (calls y puts en el mismo punto),
se promedia en lugar de lanzar error.

#### `smile`

```python
sub = self.df[filtro][['Delta', 'CallPut', 'ImpliedVol', 'Strike']].copy()
return sub.sort_values('Delta').reset_index(drop=True)
```

Devuelve solo 4 columnas. Ordenado por Delta: de izquierda (puts OTM, delta bajo)
a derecha (calls OTM, delta alto).

---

## 6. `volatility_surface_eda.py` — análisis de superficie de volatilidad

### Carga con fallback automático

```python
if Path(PARQUET_RUTA).exists():
    vs.cargar_parquet(ruta=PARQUET_RUTA)   # más rápido
else:
    vs.cargar_csv(CSV_RUTA)                # fallback al CSV
```

Si el Parquet existe, se usa (más rápido). Si no, carga el CSV.
El script es autocontenido: funciona en cualquier estado del pipeline.

### Selección automática de subyacente

```python
sid   = int(vs.df['SecurityID'].value_counts().index[0])
fecha = vs.fechas_disponibles(sid)[-1]
```

`value_counts()` devuelve SecurityIDs ordenados por número de filas.
El más frecuente es el subyacente con más observaciones históricas (típicamente
el más líquido). `[-1]` toma la última fecha disponible.

### Gráfico: Smile de volatilidad

```python
palette = cm.viridis(np.linspace(0.1, 0.9, min(len(days_list), 6)))
for color, days in zip(palette, days_list[:6]):
    smile = vs.smile(sid, fecha, days)
    smile = smile[smile['ImpliedVol'] > MISSING]
```

`MISSING = -99.99` es el centinela de IvyDB para dato no calculado.
Se filtra `> MISSING` antes de graficar.

### Gráfico: Superficie como heatmap

```python
surf = vs.superficie(sid, fecha).replace(MISSING, np.nan)
im = ax.imshow(surf.values, aspect='auto', cmap='RdYlGn_r', origin='lower', vmin=0)
```

`replace(MISSING, np.nan)`: `imshow` renderiza NaN como transparente.
`cmap='RdYlGn_r'`: Rojo-Amarillo-Verde invertido (convención: rojo = vol alta).
`origin='lower'`: Days bajos abajo, Days altos arriba (orientación natural).

### Gráfico: Serie temporal ATM

```python
delta_atm = int(deltas_call[np.abs(deltas_call - 50).argmin()])
mask = (
    (vs.df['SecurityID'] == sid)      &
    (vs.df['Days']       == days_ref) &
    (vs.df['Delta']      == delta_atm) &
    (vs.df['CallPut']    == 'C')      &
    (vs.df['ImpliedVol'] >  MISSING)
)
serie = vs.df[mask][['Date', 'ImpliedVol']].set_index('Date').sort_index()
```

**Query vectorizada directa al DataFrame** en lugar de iterar con `smile()`.
Esto es 100-1000× más rápido porque pandas aplica el filtro en una sola
operación sobre todo el DataFrame.

`np.abs(deltas - 50).argmin()`: busca el delta disponible más cercano a 50
(definición estándar de ATM en coordenadas delta).

---

## 7. `option_price_eda.py` — análisis de precios de opciones

### Carga con fallback en tres niveles

```python
if Path(PARQUET_RUTA).exists():
    op.cargar_parquet(ruta=PARQUET_RUTA, desde=..., hasta=..., columnas=COLUMNAS)
elif Path(CSV_RUTA).exists():
    op.cargar_csv(CSV_RUTA)     # DuckDB sobre CSV
else:
    op.cargar_datos(...)        # compila desde TXT
```

**Parquet** (habitual): carga en `self.df` pandas el subconjunto filtrado.
**CSV** (fallback): no carga en RAM; las secciones 2-5 usan `op.query()`.
**Compilación** (primera vez): genera el CSV desde los TXT.

### Secciones 2-5: pandas si hay `self.df`, DuckDB si no

```python
if df is not None:
    sid = int(df.groupby('SecurityID')['Volume'].sum().idxmax())
else:
    sid = int(op.query("SELECT SecurityID FROM {csv} GROUP BY SecurityID ORDER BY SUM(Volume) DESC LIMIT 1").iloc[0]['SecurityID'])
```

Con Parquet cargado (`df is not None`): operaciones pandas puras.
Sin Parquet (solo CSV): `op.query()` para no cargar 36 GB en RAM.

### Gráfico: Top 10 por volumen

```python
top10 = (df.groupby('SecurityID')['Volume']
           .sum()
           .nlargest(10)
           .sort_values())
```

`sort_values()` después de `nlargest(10)` pone el más activo arriba en el
gráfico horizontal (el eje Y va de abajo a arriba).

### Gráfico: Put-Call Ratio

```python
cp = (df.groupby(['Date', 'CallPut'])['Volume']
        .sum()
        .unstack()
        .sort_index())
ratio = (cp['C'] / cp['P']).replace([np.inf, -np.inf], np.nan).dropna()
ax.axhline(1, ...)
```

`unstack()` convierte el índice multinivel `(Date, CallPut)` en columnas `C` y `P`.
`replace([np.inf, -np.inf], np.nan)`: si algún día no hay puts, la división da
`inf`; se reemplaza por NaN para no distorsionar el gráfico.
La línea horizontal en 1 marca el punto de equilibrio calls = puts.

### Gráficos: Smile y Term Structure

Usan `op.opciones_por_vencimiento()` que consulta DuckDB sobre el Parquet/CSV.
Son los únicos gráficos que no requieren que `self.df` esté cargado (funcionan
en modo CSV también).

---

## 8. Flujo completo de ejecución

### Paso 1 (una sola vez): compilar TXT → CSV

```
op.cargar_datos(carpeta=CARPETA, guardar_en=CSV)
    │
    └→ DataLoader.compilar_carpeta(GI.ALL.IVYOPPRCD/, ...)
            │
            ├→ GI.ALL.IVYOPPRCD_202402.txt  (cabecera)  → 2.2M filas → append CSV
            ├→ GI.ALL.IVYOPPRCD_202401.txt  (cabecera)  → 2.3M filas → append CSV
            ├→ GI.ALL.IVYOPPRCD_202312.txt  (sin cab.)  → 2.1M filas → append CSV
            ├→ ...  (335 archivos más)
            └→ GI.ALL.IVYOPPRCD_199601.txt  (sin cab.)  → 0.4M filas → append CSV
                    ↓
            option_price.csv  (36 GB en disco)
            self._ruta_csv = "...option_price.csv"
```

### Paso 2 (una sola vez): CSV → Parquet

```
op.cargar_csv(CSV)              # apunta _ruta_csv al CSV
op.exportar_parquet(PARQUET)    # DuckDB: CSV → Parquet (~9 GB, Snappy)
```

### Uso habitual: cargar Parquet → pandas

```
op.cargar_parquet(
    ruta     = PARQUET,
    desde    = "2010-01-01",
    hasta    = "2023-12-31",
    columnas = ['SecurityID', 'Date', 'Expiration', 'Strike',
                'CallPut', 'ImpliedVolatility', 'Delta', 'Volume'],
)
    │
    └→ DuckDB: SELECT cols FROM parquet WHERE Date BETWEEN ...
            ↓
        self.df  (pandas DataFrame, ~5 GB RAM para este ejemplo)
            ↓
        op.df.groupby(...)   # pandas puro, sin restricciones
```

### Pruebas rápidas

```
op.cargar_datos(..., limite=5)
    └→ Solo procesa los 5 archivos más recientes (~10M filas)
```

---

## 9. Decisiones de diseño y justificaciones

### ¿Por qué DuckDB en lugar de cargar el CSV con pandas?

El CSV de option_price pesa 36 GB. Cargarlo con pandas requiere ~34 GB de RAM
libre (con dtypes compactos) o ~100 GB (con dtypes por defecto). La mayoría de
máquinas no tienen esa RAM disponible.

DuckDB resuelve esto de dos maneras:
1. **Consultas sobre el fichero**: ejecuta SQL directamente sobre el CSV/Parquet
   en disco. Solo el resultado (miles de filas) llega a RAM.
2. **Carga filtrada**: `cargar_parquet` usa DuckDB para leer solo las columnas
   y fechas necesarias antes de materializar el DataFrame.

### ¿Por qué Parquet y no seguir con CSV?

| Aspecto | CSV | Parquet |
|---|---|---|
| Tamaño en disco | 36 GB | ~9 GB |
| Velocidad de consulta | lenta (escaneo lineal) | rápida (columnar + índices) |
| Selección de columnas | no (lee todo) | sí (solo lee columnas pedidas) |
| Filtro de filas | no | sí (predicate pushdown) |

El formato CSV es texto plano: para leer la columna `Volume` hay que leer
todas las columnas de todas las filas. Parquet almacena los datos por columna:
leer `Volume` solo accede al bloque de `Volume`, ignorando las otras 27 columnas.

### ¿Por qué `_fuente()` en lugar de hardcodear el formato?

```python
def _fuente(self) -> str:
    if ruta.lower().endswith('.parquet'):
        return f"'{ruta}'"
    return f"read_csv_auto('{ruta}', header=true)"
```

`_ruta_csv` puede apuntar tanto a un CSV como a un Parquet (tras `cargar_parquet`,
apunta al Parquet). `_fuente()` genera la expresión DuckDB correcta para cada
caso. Sin esto, todos los métodos DuckDB fallarían si se llama a `cargar_parquet`
porque intentarían usar `read_csv_auto` sobre un Parquet.

### ¿Por qué dos estrategias de `exportar_parquet` (DuckDB COPY en OP, duckdb.from_df en VS)?

- **OptionPrice**: el CSV de 36 GB **no está cargado en RAM**. Solo existe en disco.
  DuckDB `COPY` lo convierte directamente (CSV en disco → Parquet en disco) sin pasar
  por RAM.
- **VolatilitySurface**: el CSV **sí está cargado** en `self.df` (es pequeño).
  `duckdb.from_df(self.df).write_parquet()` serializa directamente desde el DataFrame
  sin pyarrow (DuckDB tiene su propio motor Parquet, no requiere dependencias adicionales).

### ¿Por qué `pd.concat` causa OOM pero la escritura incremental no?

`pd.concat(lista_de_dfs)` necesita:
1. Todos los DataFrames individuales en RAM simultáneamente
2. El DataFrame resultado (copia nueva)

Con 338 archivos × ~330 MB = pico de ~66 GB solo para los trozos, más el
resultado concatenado = ~100+ GB total.

La escritura incremental (`mode='a'` en `to_csv`) solo necesita un trozo
en RAM a la vez (~330 MB), independientemente del número total de archivos.

### ¿Por qué `Expiration` no se convierte durante la compilación?

`Date` se convierte en `DataLoader` (de `YYYYMMDD` entero a `YYYY-MM-DD`)
porque **todas** las tablas IvyDB tienen `Date`. Es lógica de ingesta genérica.

`Expiration` solo existe en `IVYOPPRCD`. Convertirla en `DataLoader` violaría
la separación de capas (la capa de ingesta no debe conocer el esquema de
Option Price). Por eso `Expiration` se convierte en `cargar_parquet` y en
`_cargar_desde_csv` de `OptionPrice`.

### ¿Por qué `MISSING = -99.99` en lugar de `NaN`?

IvyDB usa `-99.99` como centinela para datos no calculados. Los ficheros TXT
son texto plano; IvyDB eligió este valor numérico en lugar de celdas vacías.

Filtrar `> MISSING` es más robusto que `> -99.99` con float32, que podría
representar -99.99 como -99.98999786... Por eso en `option_price_eda.py`
se usa `> 0` para volatilidades (que siempre deben ser positivas).

### ¿Por qué el `__main__` tiene los pasos comentados?

```python
# ── Paso 1 (una sola vez): compilar TXT → CSV ──
# op.cargar_datos(...)

# ── Paso 2 (una sola vez): CSV → Parquet ──
# op.cargar_csv(CSV)
# op.exportar_parquet(PARQUET)

# ── Uso habitual ──
op.cargar_parquet(...)
```

Los pasos 1 y 2 son costosos (20-40 min y ~10 min respectivamente) y solo
se ejecutan una vez en la vida del proyecto. Dejarlos comentados evita
ejecutarlos por error. El uso habitual (cargar Parquet) está descomentado
y es lo que se ejecuta por defecto.
