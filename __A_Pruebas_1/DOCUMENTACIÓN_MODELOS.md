# Documentación - Modelos y tratamiento de la información.


## 1. OPCIONES: Cálculo de precios BSM y árbol binomial americano

A partir del módulo de volatilidad implícita de IvyDB, se construye un pipeline para asignar precio Black-Scholes-Merton a cada punto (volatilidad, strike, vencimiento) de la superficie. Se cubre la valoración de opciones **europeas** (fórmula analítica de Merton) y **americanas** (árbol binomial CRR), con soporte para dividendo continuo `q` e interpolación log-lineal de factores de descuento.

Divisas cubiertas: USD (333) y EUR (814).


---

## 2. IMPLEMENTACIÓN TÉCNICA

### 2.1 Arquitectura general del sistema

El sistema se organiza en tres capas:

```
Parquets IvyDB (Parquet → pandas DataFrame)
    ├── volatility_surface.parquet   →  VolatilitySurface.df
    ├── zero_curve.parquet           →  ZeroCurve.df
    └── index_dividend.parquet       →  IndexDividend.df
            ↓
    __3_Functions/interpolation.py
    ├── interpolate_rates(curve_df, expiry_days, currency, base)
    │       → log-lineal en DF, un security, una fecha de curva ya filtrada
    ├── interpolate_dividends(div_df, security_id, date, expiry_days, base)
    │       → log-lineal en DF, un security, una fecha
    ├── interpolate_rates_surface(curve_df, vol_fecha, fecha, currency, base)
    │       → filtra fecha + llama interpolate_rates → pd.Series r
    └── interpolate_dividends_surface(div_df, vol_fecha, fecha, base)
            → filtra fecha + bucle por SecurityID → pd.Series q
            ↓
    __3_Functions/valuation.py  →  clase model_valuation
    ├── __init__(curve_df, currency, div_df, base)
    │       → almacena configuración del modelo
    ├── price_BS(volatility_data)          → europeas (BSM)
    ├── price_american(volatility_data)    → americanas (CRR)
    ├── price(volatility_data)             → despacha por ExerciseStyle
    ├── pricing_error(priced_df)           → BS_Price vs Premium
    ├── greeks(volatility_data)            → Δ, Γ, ν, Θ, ρ
    ├── _filter_missing(vol_fecha)         → filtra MISSING y Strike=0
    ├── _recover_S(result, r)              → invierte Delta → d1, d2, S
    └── _crr_price(S, K, r, q, sigma, tau, is_call, n)  → árbol CRR escalar
            ↓
    DataFrame resultado:
        SecurityID | Date | Days | Delta | CallPut | ImpliedVol |
        Strike | Premium | Currency | r | q | Underlying | BS_Price / American_Price
```

**Principio de separación de responsabilidades:**
- `interpolation.py` solo conoce la matemática de interpolación. No sabe nada de estructuras de volatilidad.
- `valuation.py` solo conoce la lógica de pricing. Delega toda la interpolación al módulo anterior.
- La clase `model_valuation` encapsula la *configuración del modelo* (curva, divisa, dividendos) y expone *métodos de pricing* como `price_BS` o `price_american`.

---

### 2.2 Dependencias y configuración del entorno

#### 2.2.1 Librerías externas

```python
import pandas as pd
import numpy as np
from scipy.stats import norm
```

| Librería | Versión mínima | Uso principal |
|---------|---------------|--------------|
| `pandas` | 2.0 | DataFrames, `groupby`, `loc`, operaciones de fechas con `.dt` |
| `numpy` | 1.24 | Arrays numéricos, `interp`, `where`, operaciones vectoriales |
| `scipy` | 1.10 | `norm.ppf` (cuantil normal) y `norm.cdf` (CDF normal) |

#### 2.2.2 Configuración del path de Python

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Problema que resuelve:** los scripts están en `__3_Functions/` o `__4_Analysis/` pero necesitan importar módulos de carpetas hermanas como `__2_Files/`. Python busca módulos en los directorios listados en `sys.path`, que por defecto incluye el directorio del script pero no su carpeta padre.

**Cómo funciona paso a paso:**
- `__file__`: variable especial que contiene la ruta absoluta del fichero en ejecución, ej. `e:\PhD\Código\__3_Functions\valuation.py`
- `.parent`: directorio contenedor, `e:\PhD\Código\__3_Functions`
- `.parent.parent`: sube un nivel más, `e:\PhD\Código`
- `str(...)`: convierte el objeto `Path` a cadena de texto (formato que espera `sys.path`)
- `sys.path.insert(0, ...)`: inserta en la posición 0, dándole prioridad sobre otras rutas del sistema

**Por qué `insert(0, ...)` y no `append(...)`:** garantiza que si existe algún módulo con el mismo nombre instalado en el entorno, el local tiene precedencia.

```python
from __3_Functions.interpolation import interpolate_rates_surface, interpolate_dividends_surface
```

La notación de punto indica jerarquía de módulos: `__3_Functions` es el paquete, `interpolation` es el fichero dentro del paquete.

---

### 2.3 Constantes del módulo

```python
MISSING: float = -99.99
```

IvyDB usa `-99.99` como código universal para dato ausente en todas las columnas numéricas:
- `ImpliedVol`: volatilidad no calculable (opción ilíquida, precio fuera del modelo)
- `Strike`: fila sin información de strike
- `Rate` en dividendos: yield no disponible para esa expiración
- `Premium`: precio no calculable

Declarar `MISSING` como constante de clase (`class model_valuation: MISSING = -99.99`) en lugar del literal `-99.99` tiene dos ventajas:
1. Si IvyDB cambiara el código, solo hay que modificarlo en un lugar
2. `vol > self.MISSING` es más legible que `vol > -99.99`

En la clase, se declara como **atributo de clase** (no de instancia): es compartida por todas las instancias y accesible como `self.MISSING` o `model_valuation.MISSING`.

---

### 2.4 Carga de datos

```python
vs  = VolatilitySurface()
vs.cargar_parquet(ruta=PARQUET_RUTA)

zc  = ZeroCurve(sep='\t')
zc.cargar_parquet(ruta=str(Path(PARQUET_RUTA).parent / "zero_curve.parquet"))

id_ = IndexDividend()
id_.cargar_parquet(ruta=str(Path(PARQUET_RUTA).parent / "index_dividend.parquet"))
```

Cada clase del módulo `__2_Files` encapsula un dataset de IvyDB y expone `self.df` como pandas DataFrame.

**Tipos de datos tras la carga:**

| Columna | Tipo pandas | Detalle |
|---------|------------|---------|
| `SecurityID` | `int32` | ID numérico del subyacente |
| `Date` | `datetime64[ns]` | Fecha de observación |
| `Days` | `int16` | Días al vencimiento |
| `Delta` | `int16` | Delta en enteros de -100 a +100 |
| `CallPut` | `category` | Valores `'C'` y `'P'` |
| `ImpliedVol`, `Strike`, `Rate` | `float32` | Valores numéricos |
| `Currency` | `int16` | Código numérico de divisa |
| `Expiration` | `datetime64[ns]` | Fecha de vencimiento del dividendo |

**Construcción de rutas relativas:**
```python
str(Path(PARQUET_RUTA).parent / "zero_curve.parquet")
```
El operador `/` en objetos `Path` concatena rutas (equivalente a `os.path.join`). `str(...)` convierte a cadena para APIs que no aceptan objetos `Path`.

---

### 2.5 Función `interpolate_rates`

#### 2.5.1 Firma completa

```python
def interpolate_rates(
    curve_df: pd.DataFrame,
    expiry_days: pd.Series,
    currency: int,
    base: float = 360.0
) -> pd.DataFrame:
```

Recibe un fragmento de `curve_df` ya filtrado por fecha (o el DataFrame completo; el filtro por `currency` lo hace internamente). Devuelve un `pd.DataFrame` con columnas `Days` y `Rate`, preservando el índice de `expiry_days`.

#### 2.5.2 Filtrado y preparación de la curva

```python
curve = (
    curve_df[curve_df["Currency"] == currency]
    .sort_values("Days")
    .reset_index(drop=True)
)
```

**Filtrado booleano en pandas:** `curve_df["Currency"] == currency` genera una Series de `True`/`False`. Usarla entre `[...]` selecciona solo las filas `True`, operando en C internamente (eficiente).

**Encadenamiento de métodos:** los tres métodos se encadenan con el patrón de paréntesis externas para permitir salto de línea. `reset_index(drop=True)` reemplaza los índices originales (potencialmente no continuos tras el filtrado) por 0, 1, 2... `drop=True` descarta el índice antiguo en lugar de añadirlo como columna.

#### 2.5.3 Interpolación log-lineal de factores de descuento

```python
days  = curve["Days"].values.astype(float)
rates = curve["Rate"].values
log_df = -rates * days / base
```

**Fundamento matemático:** el factor de descuento para tasa `r` continuously compounded y plazo `t = days/base` años es:
```
DF(t) = e^(-r·t)   →   log(DF) = -r · days / base
```

**Por qué interpolar en log(DF) y no en r directamente:** la interpolación lineal de tipos `r` produce factores de descuento no necesariamente monótonos. La interpolación log-lineal de DF garantiza que el factor de descuento sea siempre decreciente y positivo, que es la propiedad fundamental de una curva consistente.

**`.values`:** devuelve el array numpy subyacente sin el índice de pandas. Necesario porque `np.interp` opera sobre arrays planos. `.astype(float)` evita problemas de precisión si `Days` está almacenado como `int16` o `int32`.

#### 2.5.4 Interpolación con `np.interp` y retorno

```python
t = expiry_days.values.astype(float)
log_df_interp = np.interp(t, days, log_df)
rates_interp  = np.where(t > 0, -log_df_interp * base / t, rates[0])

return pd.DataFrame(
    {"Days": expiry_days.values, "Rate": rates_interp},
    index=expiry_days.index,
)
```

`numpy.interp(x, xp, fp)`: interpolación lineal por tramos en 1D. Requiere `xp` ordenado ascendentemente. Para `x` fuera del rango: extrapolación plana (devuelve el primer o último valor de `fp`).

**Inversión:** `r = -log(DF) · base / t`. El caso `t = 0` es degenerado (división por cero); se retorna `rates[0]`.

**Preservación del índice:** al construir el DataFrame de retorno con `index=expiry_days.index`, el índice de `r` coincide con el de las columnas de `result` en `price_BS`. Sin esto, la operación `r - q` alinearía por índice y produciría `NaN` si los índices difieren.

---

### 2.6 Función `interpolate_dividends`

#### 2.6.1 Firma y tipo de retorno

```python
def interpolate_dividends(
    div_df: pd.DataFrame,
    security_id: int,
    date: pd.Timestamp,
    expiry_days: pd.Series,
    base: float = 360.0,
    MISSING: float = -99.99
) -> np.ndarray:
```

Devuelve un `numpy.ndarray` (no `pd.DataFrame`). Se asigna directamente a filas específicas de `result` usando `.loc`, y pandas acepta arrays numpy en esas asignaciones alineando por posición.

#### 2.6.2 Filtrado y copia defensiva

```python
sub = div_df[
    (div_df["SecurityID"] == security_id) &
    (div_df["Date"]       == date)        &
    (div_df["Rate"]       >  MISSING)
].copy()
```

**Operadores lógicos en pandas:** `&` (AND), `|` (OR), `~` (NOT) operan elemento a elemento sobre Series booleanas. No se pueden usar `and`, `or`, `not` de Python porque pandas no sabe reducir una Series a un único booleano. Los paréntesis son obligatorios por precedencia: `&` tiene mayor precedencia que `==`.

**`.copy()`:** el resultado del filtrado puede ser una vista (comparte memoria con el original) o una copia, dependiendo del tipo de indexación. Si se modifica después (`sub["Days"] = ...`), sin `.copy()` pandas puede lanzar `SettingWithCopyWarning`. La copia explícita elimina la ambigüedad.

#### 2.6.3 Retorno temprano y detección de flat yield

```python
if sub.empty:
    return np.zeros(len(expiry_days))

sub["Days"] = (sub["Expiration"] - date).dt.days.astype(float)
sub_pos = sub[sub["Days"] > 0].sort_values("Days")

if sub_pos.empty:
    flat_rate = sub["Rate"].iloc[-1]
    return np.full(len(expiry_days), flat_rate)
```

**`DataFrame.empty`:** propiedad booleana que devuelve `True` si el DataFrame no tiene filas. Más idiomático que `len(sub) == 0`.

**Cálculo de días:** `sub["Expiration"] - date` produce una Serie de `Timedelta`. El accessor `.dt` expone propiedades de duración temporal. `.dt.days` extrae la componente de días completos como entero.

**La convención de IvyDB para índices US:** para índices norteamericanos (S&P 500, Nasdaq 100, Russell 2000, etc.), OptionMetrics calcula un único dividend yield sin estructura temporal. La fecha de expiración se codifica como `1900-01-01`. Al calcular `Days = (1900-01-01 - fecha_actual)`, el resultado es ≈ -44.000 días (negativo). El filtro `Days > 0` descarta todas las filas, dejando `sub_pos` vacío.

**`iloc[-1]`:** acceso por posición entera al último elemento. `-1` es indexación negativa de Python (último elemento). `np.full(n, valor)` crea un array de `n` elementos todos iguales a `valor`.

#### 2.6.4 Interpolación con corrección de extrapolación izquierda

```python
days   = sub_pos["Days"].values
rates  = sub_pos["Rate"].values
log_df = -rates * days / base

t             = expiry_days.values.astype(float)
log_df_interp = np.interp(t, days, log_df)
q_interp      = np.where(t > 0, -log_df_interp * base / t, rates[0])
q_interp      = np.where(t < days[0], rates[0], q_interp)
return q_interp
```

La lógica de interpolación es idéntica a `interpolate_rates`. La diferencia clave es la última línea.

**El problema de la extrapolación izquierda:** cuando `t < days[0]` (el vencimiento pedido es más corto que el primer nodo de la term structure de dividendos), `np.interp` devuelve `log_df[0]`. Al invertir:
```
q = -log_df[0] · base / t  =  rates[0] · days[0] / t
```
Como `t << days[0]` (ej. t=30, days[0]=261), el cociente `days[0]/t ≈ 8.7` infla el rate de forma absurda (ej. 1.25% × 8.7 ≈ 10.9%).

**La solución:** una segunda pasada con `np.where` reemplaza el valor calculado por `rates[0]` para todas las posiciones donde `t < days[0]`. Esto implementa extrapolación constante en el espacio de tasas (no en log-DF), que es la convención habitual para dividendos fuera del rango de observación.

**Dos `np.where` en secuencia:** el primero calcula `q_interp` para todo el array. El segundo sobreescribe selectivamente las posiciones problemáticas. Es más legible que un `np.where` anidado y equivalentemente eficiente.

---

### 2.7 Función `interpolate_rates_surface`

```python
def interpolate_rates_surface(
    curve_df: pd.DataFrame,
    vol_fecha: pd.DataFrame,
    fecha: pd.Timestamp,
    currency: int,
    base: float = 360.0,
) -> pd.Series:
    curva_fecha = curve_df[curve_df["Date"] == fecha]
    return interpolate_rates(curva_fecha, vol_fecha["Days"], currency, base)["Rate"]
```

Función de nivel superior que combina:
1. El filtrado de `curve_df` por fecha
2. La llamada a `interpolate_rates` con los días de la superficie
3. La extracción de la columna `Rate` como `pd.Series`

**Por qué extraer `["Rate"]` aquí y no en `interpolate_rates`:** `interpolate_rates` devuelve un DataFrame con dos columnas (`Days` y `Rate`) porque puede ser útil en otros contextos. La función de superficie extrae solo lo que necesita `price_BS`: la Serie de tipos interpolados con el índice correcto.

**Separación de responsabilidades:** `interpolate_rates` es genérica (no sabe de superficies). `interpolate_rates_surface` es el adaptador que traduce entre la interfaz genérica y el pipeline de `model_valuation`.

---

### 2.8 Función `interpolate_dividends_surface`

```python
def interpolate_dividends_surface(
    div_df: pd.DataFrame | None,
    vol_fecha: pd.DataFrame,
    fecha: pd.Timestamp,
    base: float = 360.0,
) -> pd.Series:
    q = pd.Series(0.0, index=vol_fecha.index)
    if div_df is None:
        return q
    div_fecha = div_df[div_df["Date"] == fecha]
    for sid, grp in vol_fecha.groupby("SecurityID"):
        q.loc[grp.index] = interpolate_dividends(div_fecha, sid, fecha, grp["Days"], base)
    return q
```

**`pd.DataFrame | None`:** sintaxis de Python 3.10+ para tipos unión. Indica que `div_df` puede ser un DataFrame o `None`. La función maneja ambos casos: `None` devuelve ceros (q=0, sin dividendos).

**`pd.Series(0.0, index=vol_fecha.index)`:** inicializa una Serie de ceros con el mismo índice que `vol_fecha`. Esto garantiza:
- Las filas sin datos de dividendo conservan `q = 0`
- El resultado tiene el mismo índice que las demás columnas de `result` en `price_BS`

**Pre-filtrado por fecha:** `div_fecha = div_df[div_df["Date"] == fecha]` se hace una sola vez fuera del bucle. Si se hiciera dentro, se filtrarían O(securities) veces, cada vez recorriendo el DataFrame completo.

**`groupby("SecurityID")`:** dentro de una fecha puede haber múltiples securities (en una cartera de índices). Cada security tiene su propia curva de dividendos. El `groupby` agrupa las filas y pasa a `interpolate_dividends` solo los `Days` de ese security.

**`q.loc[grp.index]`:** asignación por etiquetas. `grp.index` contiene los índices de fila del grupo en `vol_fecha`. Esta sintaxis garantiza que el array devuelto por `interpolate_dividends` se asigne exactamente a las filas correctas, incluso si el orden no es el esperado.

---

### 2.9 Clase `model_valuation`: diseño y principios

#### 2.9.1 Motivación del diseño orientado a objetos

La clase encapsula la *configuración del modelo* (curva de tipos, divisa, dividendos, convención de días) que es invariante para todas las opciones de una sesión de pricing:

```python
model = model_valuation(
    curve_df = curva_3y,
    currency = 333,
    div_df   = divs_3y
)

europeas   = model.price_BS(datos_europeos)
americanas = model.price_american(datos_americanos)
todas      = model.price(datos_con_exercise_style)
```

**Alternativa: funciones sueltas con todos los parámetros en cada llamada.** Con el diseño anterior, `price_BS(vol, curve_df, currency, div_df, base)` requería pasar los mismos 4 parámetros en cada llamada. Con la clase, se configuran una vez en `__init__` y se reutilizan.

**Extensibilidad:** añadir nuevos modelos (`price_heston`, `price_sabr`, `price_local_vol`) solo requiere añadir métodos. No cambia la interfaz existente ni los parámetros de construcción.

#### 2.9.2 Atributo de clase vs atributo de instancia

```python
class model_valuation:
    MISSING: float = -99.99   # atributo de CLASE

    def __init__(self, curve_df, currency, div_df=None, base=360.0):
        self.curve_df = curve_df   # atributo de INSTANCIA
        self.currency = currency
        self.div_df   = div_df
        self.base     = base
```

- **Atributo de clase (`MISSING`):** definido fuera del `__init__`, existe en la propia clase y es compartido por todas las instancias. Se accede como `self.MISSING` o `model_valuation.MISSING`. Apropiado para constantes que no varían entre instancias.
- **Atributos de instancia** (`self.curve_df`, etc.): definidos en `__init__`, cada instancia tiene los suyos propios. Apropiado para datos de configuración que pueden diferir entre instancias (distintas curvas, distintas divisas).

#### 2.9.3 Convención de nomenclatura de métodos

| Prefijo | Significado | Ejemplos |
|---------|-------------|---------|
| sin prefijo | API pública, parte de la interfaz | `price_BS`, `price_american`, `greeks` |
| `_` | interno, detalle de implementación | `_filter_missing`, `_recover_S`, `_crr_price` |

Los métodos con `_` pueden cambiar sin afectar al código externo que usa la clase. Los métodos públicos forman el contrato de la API.

---

### 2.10 Método `__init__`

```python
def __init__(
    self,
    curve_df: pd.DataFrame,
    currency: int,
    div_df: pd.DataFrame | None = None,
    base: float = 360.0,
) -> None:
    self.curve_df = curve_df
    self.currency = currency
    self.div_df   = div_df
    self.base     = base
```

`__init__` es el **constructor** de Python. Se llama automáticamente al crear una instancia con `model_valuation(...)`. El parámetro `self` referencia a la instancia que se está creando.

**`-> None`:** anotación de retorno. Los constructores no retornan nada (devuelven implícitamente `None`). La anotación es solo informativa.

**`div_df: pd.DataFrame | None = None`:** parámetro opcional. Si no se proporciona, la instancia funciona sin dividendos (`q = 0` en todos los cálculos, vía `interpolate_dividends_surface`).

---

### 2.11 Método `price_BS`

#### 2.11.1 Estructura del bucle principal

```python
def price_BS(self, volatility_data: pd.DataFrame) -> pd.DataFrame:
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
        ...
        bloques.append(result)

    return pd.concat(bloques)
```

**`(self.curve_df["Date"] == fecha).any()`:** comprueba si existe alguna fila en `curve_df` para esa fecha. `.any()` devuelve `True` si al menos un elemento es `True`. Es la forma idiomática de comprobar existencia sin materializar un DataFrame intermedio.

**`vol_fecha.copy()`:** crea una copia independiente. Sin esto, añadir columnas (`result["r"] = ...`) modificaría el `vol_fecha` original, que es una vista del `volatility_data` original, produciendo `SettingWithCopyWarning`.

**`r.values` vs `r`:** al asignar `result["r"] = r.values`, se usa `.values` (array numpy) para que pandas alinee por posición y no por índice. Si se usara `r` directamente (Series con índice), la asignación también funcionaría porque los índices coinciden, pero `.values` hace explícita la intención.

**Patrón lista + concat (O(n) vs O(n²)):**
```python
# MAL: O(n²) — cada concat copia el acumulado
resultado = pd.DataFrame()
for ...:
    resultado = pd.concat([resultado, result])

# BIEN: O(n) — una sola concatenación al final
bloques = []
for ...:
    bloques.append(result)   # O(1) por iteración
return pd.concat(bloques)    # O(n) una sola vez
```
Con miles de fechas y millones de filas, la diferencia puede ser de minutos vs segundos.

#### 2.11.2 Inversión de Delta y recuperación de S

Este cálculo se realiza en `_recover_S` (ver §2.16). Las fórmulas implementadas:

```
d1 = Φ⁻¹(Δ/100 · e^(q·τ))           para calls  (Δ ≥ 0)
d1 = Φ⁻¹(Δ/100 · e^(q·τ) + 1)       para puts   (Δ < 0)
d2 = d1 − σ·√τ
S  = K · exp[d1·σ·√τ − (r−q+σ²/2)·τ]
```

#### 2.11.3 Fórmula BSM de Merton con máscara booleana

```python
disc_q    = np.exp(-result["q"] * (result["Days"] / self.base))
disc_r    = np.exp(-r           * (result["Days"] / self.base))
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
```

**Inicialización a `np.nan`:** si alguna fila falla en `norm.ppf` (Delta fuera de (0,1)), permanecerá como `NaN` y será detectable en análisis posteriores. Es más informativo que un cero.

**`mask_call = result["CallPut"] == "C"`:** debe ser pandas Series (no numpy array). Si fuera array, `.loc` lo aceptaría pero perdería la garantía de alineación por índice.

**`~mask_call`:** operador `~` es negación booleana elemento a elemento para Series de pandas. Equivale a `NOT`. No se puede usar `not mask_call` porque Python no sabe reducir una Serie a un único booleano.

**`S[mask_call]`:** indexar una pandas Series con una Series booleana del mismo índice selecciona los elementos donde la máscara es `True`, preservando las etiquetas originales.

**Fórmulas de Merton (1973) con dividendo continuo:**
```
Call: C = S·e^(-q·τ)·N(d1) - K·e^(-r·τ)·N(d2)
Put:  P = K·e^(-r·τ)·N(-d2) - S·e^(-q·τ)·N(-d1)
```

---

### 2.12 Método `price_american` y árbol CRR

#### 2.12.1 Estructura del método

```python
def price_american(self, volatility_data, n_steps=200):
    bloques = []
    for fecha, vol_fecha in volatility_data.groupby("Date"):
        ...
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
```

La preparación de datos (`_filter_missing`, `interpolate_rates_surface`, `interpolate_dividends_surface`, `_recover_S`) es idéntica a `price_BS`. La diferencia es el cálculo final: en lugar de la fórmula analítica, se llama a `_crr_price` para cada fila.

**`iterrows()`:** itera sobre las filas de un DataFrame devolviendo `(índice, Series)` en cada iteración. Es más lento que operaciones vectorizadas pero necesario aquí porque `_crr_price` recibe escalares (un árbol por opción).

**`result.at[idx, "American_Price"]`:** acceso y modificación por etiqueta de fila e índice de columna. Más eficiente que `.loc` para acceso escalar individual.

**`float(S[idx])`:** `S` es una pandas Series. `S[idx]` selecciona el valor por etiqueta (índice), devolviendo un escalar potencialmente de tipo `numpy.float32`. `float(...)` convierte a Python float nativo, que es el tipo esperado por `_crr_price`.

#### 2.12.2 Método estático `_crr_price`

```python
@staticmethod
def _crr_price(S, K, r, q, sigma, tau, is_call, n):
```

**`@staticmethod`:** decorador que indica que el método no accede a `self` (ni a la instancia ni a la clase). Puede llamarse como `self._crr_price(...)` o como `model_valuation._crr_price(...)`. Se usa porque `_crr_price` es una función matemática pura: dados unos inputs, produce un output sin consultar el estado del modelo.

**Construcción del árbol CRR (Cox-Ross-Rubinstein, 1979):**

```python
dt   = tau / n
u    = np.exp(sigma * np.sqrt(dt))   # factor de subida
d    = 1.0 / u                        # factor de bajada (árbol recombinante)
disc = np.exp(-r * dt)                # factor de descuento por paso
p    = (np.exp((r - q) * dt) - d) / (u - d)  # prob. riesgo-neutral subida
q_rn = 1.0 - p                        # prob. riesgo-neutral bajada
```

**Árbol recombinante:** la condición `d = 1/u` garantiza que un nodo de subida seguido de bajada llega al mismo precio que bajada seguida de subida. Sin esta propiedad, el árbol tendría `2^n` nodos en lugar de `(n+1)(n+2)/2`.

**Precios en el nodo final (vectorizado):**
```python
j   = np.arange(n + 1)
S_T = S * (u ** (n - j)) * (d ** j)
```

`j = 0, 1, ..., n` son los posibles números de bajadas. El precio en el nodo `(n, j)` es `S · u^(n-j) · d^j`. Al usar `np.arange`, todos los `n+1` nodos finales se calculan en paralelo.

**Inducción hacia atrás con early-exercise:**
```python
for i in range(n - 1, -1, -1):
    S_i          = S * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
    continuation = disc * (p * V[:i + 1] + q_rn * V[1:i + 2])
    intrinsic    = np.maximum(S_i - K, 0.0)   # o K - S_i para put
    V            = np.maximum(continuation, intrinsic)
```

Para cada paso `i` (de `n-1` hasta `0`):
- `continuation`: valor esperado descontado desde el siguiente paso
- `intrinsic`: payoff de ejercicio inmediato
- `V = max(continuation, intrinsic)`: el tenedor ejerce si el valor inmediato supera la continuación

**`range(n-1, -1, -1)`:** itera desde `n-1` hasta `0` inclusive. `range(a, b, c)` con `c < 0` produce una secuencia decreciente. El `-1` como límite superior (no incluido) garantiza que se procese el nodo `i = 0`.

**Complejidad:** O(n²) en tiempo y O(n) en memoria (se sobreescribe `V` en cada iteración, no se guarda el árbol completo).

---

### 2.13 Método `price`

```python
def price(self, volatility_data, exercise_style_col="ExerciseStyle"):
    if exercise_style_col not in volatility_data.columns:
        return self.price_BS(volatility_data)

    europeas   = volatility_data[volatility_data[exercise_style_col] == "E"]
    americanas = volatility_data[volatility_data[exercise_style_col] == "A"]

    partes = []
    if not europeas.empty:
        partes.append(self.price_BS(europeas))
    if not americanas.empty:
        partes.append(self.price_american(americanas))

    return pd.concat(partes).sort_index()
```

Método de entrada unificado que despacha según el estilo de ejercicio. Útil cuando la muestra mezcla europeas y americanas (ej. al cruzar con `option_price.parquet` que contiene la columna `ExerciseStyle`).

**`exercise_style_col not in volatility_data.columns`:** comprueba si la columna existe. Si no existe, se trata todo como europeo. `DataFrame.columns` es un Index de pandas; el operador `in` comprueba membresía en O(1) gracias al índice hash.

**`pd.concat(partes).sort_index()`:** concatena los dos grupos y reordena por el índice original, restaurando el orden de filas del input. `sort_index()` ordena por el índice entero del DataFrame, que preserva la posición original de cada opción.

---

### 2.14 Método `pricing_error`

```python
def pricing_error(self, priced_df, price_col="BS_Price"):
    df = priced_df[priced_df["Premium"] > self.MISSING].copy()
    df["dif"]     = df[price_col] - df["Premium"]
    df["dif_pct"] = df["dif"].abs() / df["Premium"].abs() * 100
    return (
        df.groupby(["Days", "Delta"])["dif_pct"]
        .agg(["mean", "std", "max"])
        .round(2)
    )
```

**Filtrado de Premium ausente:** antes de calcular el error porcentual, se excluyen las filas donde `Premium = -99.99`. Sin este filtro, la división produciría valores absurdos.

**`df["dif"].abs()`:** `abs()` en pandas Series devuelve el valor absoluto elemento a elemento.

**`groupby(["Days", "Delta"])`:** agrupación por dos columnas simultáneamente. El resultado es un multi-índice (Days, Delta) en el DataFrame de salida. Permite identificar patrones sistemáticos: ¿el error crece con la madurez? ¿con el delta?

**`.agg(["mean", "std", "max"])`:** aplica múltiples funciones de agregación a la vez. Devuelve un DataFrame con una columna por función.

**`.round(2)`:** redondea a 2 decimales para legibilidad.

---

### 2.15 Método `greeks`

```python
def greeks(self, volatility_data):
    bloques = []
    for fecha, vol_fecha in volatility_data.groupby("Date"):
        ...
        d1, d2, S = self._recover_S(result, r)
        q   = result["q"]
        K   = result["Strike"]
        t   = result["Days"] / self.base
        sig = result["ImpliedVol"]

        disc_q = np.exp(-q * t)
        disc_r = np.exp(-r * t)
        n_d1   = norm.pdf(d1)    # densidad normal en d1
        sqrt_t = np.sqrt(t)

        mask_call = result["CallPut"] == "C"

        # Delta
        result["Delta_calc"] = np.where(
            mask_call,
            disc_q * norm.cdf(d1),
            disc_q * (norm.cdf(d1) - 1),
        )
        # Gamma
        result["Gamma"] = disc_q * n_d1 / (S * sig * sqrt_t)
        # Vega (por 1% de σ)
        result["Vega"]  = S * disc_q * n_d1 * sqrt_t / 100
        # Theta (por día calendario)
        theta_common = -S * disc_q * n_d1 * sig / (2 * sqrt_t)
        result["Theta"] = np.where(
            mask_call,
            (theta_common - r*K*disc_r*norm.cdf( d2) + q*S*disc_q*norm.cdf( d1)) / self.base,
            (theta_common + r*K*disc_r*norm.cdf(-d2) - q*S*disc_q*norm.cdf(-d1)) / self.base,
        )
        # Rho (por 1% de r)
        result["Rho"] = np.where(
            mask_call,
             K * t * disc_r * norm.cdf( d2) / 100,
            -K * t * disc_r * norm.cdf(-d2) / 100,
        )
```

**`norm.pdf(d1)`:** función de densidad de la normal estándar: `n(d1) = e^(-d1²/2) / √(2π)`. Aparece en todas las griegas de segundo orden.

**Griegas analíticas de Merton:**

| Griega | Call | Put | Interpretación |
|--------|------|-----|----------------|
| Delta | `e^(-q·τ)·N(d1)` | `e^(-q·τ)·(N(d1)-1)` | Sensibilidad al precio del subyacente |
| Gamma | `e^(-q·τ)·n(d1)/(S·σ·√τ)` | igual | Sensibilidad de Delta al precio |
| Vega | `S·e^(-q·τ)·n(d1)·√τ / 100` | igual | Sensibilidad a 1% de volatilidad |
| Theta | fórmula completa | fórmula completa | Sensibilidad al paso del tiempo (por día) |
| Rho | `K·τ·e^(-r·τ)·N(d2) / 100` | `-K·τ·e^(-r·τ)·N(-d2) / 100` | Sensibilidad a 1% del tipo de interés |

**División por `self.base` en Theta:** convierte de "por año" a "por día calendario". Con `base=360`, `Theta_diario = Theta_anual / 360`.

---

### 2.16 Métodos internos `_filter_missing` y `_recover_S`

#### `_filter_missing`

```python
def _filter_missing(self, vol_fecha):
    return vol_fecha[
        (vol_fecha["ImpliedVol"] > self.MISSING) &
        (vol_fecha["Strike"]     > 0)
    ]
```

Descarta filas con datos ausentes antes de entrar en el cálculo. Sin este filtro, `norm.ppf` y la recuperación de S producirían `NaN` o valores absurdos para filas con `ImpliedVol = -99.99` o `Strike = 0`.

Se llama en `price_BS`, `price_american` y `greeks` al inicio de cada iteración de fecha. Al extraerla como método, el filtrado es consistente en todos los métodos sin duplicación.

#### `_recover_S`

```python
def _recover_S(self, result, r):
    delta = result["Delta"]
    sigma = result["ImpliedVol"]
    K     = result["Strike"]
    t     = result["Days"] / self.base
    q     = result["q"]

    adj = delta / 100 * np.exp(q * t)
    d1  = norm.ppf(np.where(delta >= 0, adj, adj + 1))
    d2  = d1 - sigma * np.sqrt(t)
    S   = K * np.exp(d1 * sigma * np.sqrt(t) - (r - q + sigma**2 / 2) * t)

    d1 = pd.Series(d1, index=result.index)
    d2 = pd.Series(d2, index=result.index)
    return d1, d2, S
```

**Inversión de Delta — derivación:** la Delta de Merton para call es `Δ = e^(-q·τ)·N(d1)`. Despejando `d1`:
```
N(d1) = Δ · e^(q·τ)
d1    = Φ⁻¹(Δ · e^(q·τ))
```
IvyDB reporta Delta en escala [-100, +100], por tanto `adj = Δ/100 · e^(q·τ)`. Para puts, Delta es negativa y `N(d1) = Δ/100 · e^(q·τ) + 1`.

**`norm.ppf(np.where(...))`:** `np.where` construye el argumento correcto para `norm.ppf` según call o put, vectorizando la condición. Aplicar `norm.ppf` una sola vez sobre el array combinado es más eficiente que dos llamadas separadas.

**`norm.ppf` requiere argumentos en (0,1):** si `adj` cae fuera de este rango (por Deltas extremas o datos ruidosos), el resultado es `NaN` o `inf`. Por eso se filtra previamente con `_filter_missing`.

**Conversión a pd.Series:** `norm.ppf` devuelve un numpy array. Convertir `d1` y `d2` a Series con `pd.Series(d1, index=result.index)` permite usarlos en operaciones con otras columnas de `result` con alineación automática por índice.

---

### 2.17 Módulo de validación: `validar_bs_inputs.py`

El módulo `__0_Validaciones/validar_bs_inputs.py` provee funciones para inspeccionar interactivamente los inputs interpolados de una fila del resultado de `price_BS`:

| Función | Propósito |
|---------|-----------|
| `validar_rate(row, curve_df)` | Muestra los nodos de curva usados para interpolar `r` y recomputa el valor |
| `validar_dividendo(row, div_df)` | Muestra los datos de dividendo, detecta flat yield vs term structure, recomputa `q` |
| `validar_fila(row, curve_df, div_df)` | Wrapper que llama a las dos anteriores con cabecera resumen |

**Uso típico:**
```python
from __0_Validaciones.validar_bs_inputs import validar_fila

# Inspeccionar una fila del resultado
validar_fila(ejemplo.iloc[0], curva_3y, divs_3y)

# Filtrar por security y vencimiento específicos
row = ejemplo[(ejemplo["SecurityID"] == 102480) & (ejemplo["Days"] == 365)].iloc[0]
validar_fila(row, curva_3y, divs_3y)
```

**Output de `validar_fila`:**
```
======================================================================
SecurityID=102480 | Date=2019-01-02 | Days=365 | CallPut=C | Delta=50 | Strike=2505.0
  ImpliedVol   = 0.184200
  q            = 0.019800
  BS_Price     = 142.310000
  Premium      = 141.500000
======================================================================

--- Tipo de interés (r) ---
Curva Zero  Currency=333 | Date=2019-01-02
  Days=  270  Rate=0.024100
  Days=  365  Rate=0.024800  ← izquierda
  Days=  548  Rate=0.025300  ← derecha
  [Days=365]  r interpolado = 0.024800
  r en row      = 0.024800  ✓

--- Dividend yield (q) ---
IndexDividend  SecurityID=102480 | Date=2019-01-02
  Tipo: FLAT YIELD (Expiration=1900-01-01, sentinel US index)
  Rate = 0.019800
  [Days=365]  q interpolado = 0.019800
  q en row      = 0.019800  ✓
```

---

### 2.18 Limitaciones conocidas y fuentes de error

| Limitación | Causa técnica | Causa de negocio | Impacto cuantificado |
|-----------|--------------|-----------------|---------------------|
| Error creciente con madurez | `e^(q·τ)` amplifica diferencias en `q` al crecer τ | Dividendos continuos vs discretos reales | ~0.7% para τ<1 año, ~4.7% para τ=2 años (deep ITM) |
| `q = 0` para algunos securities | SecurityID no presente en `IVYIDXDVD` | Solo cubre ~30 índices principales | Subvalora opciones en proporción a `q·τ` |
| Opciones americanas en la muestra | IvyDB mezcla `ExerciseStyle = A` y `E` | Sin filtro de estilo en datasets de vol surface | BSM subestima el precio americano (prima de ejercicio anticipado) |
| 14 fechas sin curva de tipos | Festivos US donde el mercado de bonos no cotiza | El mercado de opciones sí cotizó esos días | Esas fechas se omiten completamente del resultado |
| Extrapolación fuera del rango de la curva de tipos | `np.interp` usa extrapolación plana en log-DF | La curva zero cubre plazos estándar (30d–10y) | Error pequeño para opciones muy cortas o muy largas |
| Extrapolación izquierda en term structure de dividendos | `rates[0]·days[0]/t` diverge cuando `t << days[0]` | El primer nodo de dividendo puede estar a 3-6 meses | Corregido: se usa `rates[0]` directamente para `t < days[0]` |
| Redondeo de Delta en IvyDB | Delta reportada en enteros [-100, +100] | Resolución de 1 punto de Delta | Error en recuperación de S, mayor para opciones deep ITM donde `Φ⁻¹` tiene derivada alta |
| Lentitud de `price_american` | Bucle Python fila a fila (árbol por opción) | No hay vectorización analítica del árbol CRR | ~100x más lento que `price_BS` para misma muestra |
