# Carpeta `__8_Definición_Calculos` — Documentación

Pipeline para construir las **sensibilidades empíricas (delta / gamma)** del SP500 sobre la muestra de opciones con vencimiento **15–45 días**, diagnosticar cobertura y calidad, calcular niveles y **spreads ATM vs OTM** (empíricos y BS), y contrastarlos con **variables macro** (VIX, CFNAI, FF, retorno SPX).

Los scripts están numerados y se ejecutan en orden: cada uno lee el parquet que dejó el anterior y guarda el suyo en `/Volumes/data/OptionMetrics/OUTPUTS/` (mac) o `Y:\OUTPUTS\` (win).

---

## Esquema de dependencias

```
                 ┌───────────────────────────┐
                 │  __2_Files (loaders):     │
                 │  OptionPrice, ForwardPrice│
                 │  ZeroCurve                │
                 │  __3_Functions.interp     │
                 └────────────┬──────────────┘
                              │
   parquets OptionMetrics ────┤
   (option_price, forward,    │
    zero_curve, security_price)
                              ▼
   ┌──────────────────────────────────────────┐
   │ 01_limpieza_preliminar.py                │
   │  Filtro básico + Moneyness (spot) +      │
   │  MidPrice + horquilla                    │
   │  → opt_df_prueba.parquet                 │
   └──────────────┬───────────────────────────┘
                  │
                  ▼
   ┌──────────────────────────────────────────┐
   │ 02_panel_construction.py                 │
   │  Buckets maturity + moneyness            │
   │  Delta y Gamma empíricas (op1 contrato,  │
   │   op2 WA por bucket, op3 sobre precio    │
   │   agregado) por bucket de vencimiento    │
   │  → opt_df_empirical_greeks.parquet       │
   └──────────────┬──────────────┬────────────┘
                  │              │
      ┌───────────┘              └───────────────────────────┐
      ▼                                                      ▼
┌───────────────────────────────┐      ┌────────────────────────────────────┐
│ 03_panel_diagnosis.py         │      │ 05_Gamma_Spread.py                 │
│ Cobertura, rachas, series y   │      │ WA diaria (por OI y por $Vol) y    │
│ ridgeline por bucket. Diag.   │      │ spread ATM–{near, deep, very_deep} │
│ de signos / outliers (bloque  │      │ para gamma_emp y delta_emp         │
│ comentado replica 02).        │      │ → Agg_Greeks.csv/.parquet          │
│ (Sin output persistido).      │      └──────────────┬─────────────────────┘
└───────────────────────────────┘                     │
                                                      │
                                    ┌─────────────────┴──────────────────┐
                                    ▼                                    ▼
                        ┌──────────────────────────────┐     ┌───────────────────────────────┐
                        │ 06_gamma_spread_diagnosis.py │     │ 07_BS_spread_&_diagnosis.py   │
                        │ Continuidad / rachas del     │     │ Mismo cálculo que 05 pero     │
                        │ spread; serie mensual (últ.  │     │ sobre las griegas TEÓRICAS    │
                        │ día del mes); unión con FF y │     │ (Delta/Gamma BS)              │
                        │ TotalReturn SPX; regresión   │     │ → Agg_Greeks_BS.csv           │
                        │ predictiva IS con NW.        │     │  (mismo formato que 05, luego │
                        │ Lee Agg_Greeks o Agg_Greeks_ │     │   consumido por 06 con        │
                        │ BS según flag `empirical`.   │     │   flag empirical = False)     │
                        └──────────────────────────────┘     └───────────────────────────────┘

   ┌───────────────────────────────────────────────────────┐
   │ contrastes/04_contraste_macrovariables.py             │
   │ (rama paralela; consume opt_df_empirical_greeks)      │
   │ Descriptivos emp vs BS · series mensuales · corr /    │
   │ ACF / ADF · greek imbalance (Barbon 2020) · contraste │
   │ con VIX y CFNAI · descomposición vanna/charm          │
   └───────────────────────────────────────────────────────┘
```

`VERSIONES_SUCIAS/panel_construction.py` es una versión antigua de `02_`, se ignora.

---

## Detalle por script

### `01_limpieza_preliminar.py`

**Importa**
- Loaders del proyecto: `__2_Files.OptionPrice`, `ForwardPrice`, `ZeroCurve`; `__3_Functions.interpolation.interpolate_rates_surface`.
- Parquets: `option_price`, `forward_price_filtered`, `zero_curve`, `security_price` (SP500 SecurityID = 108105), 2003-01-02 → 2024-02-29.

**Hace**
- Downcast a `float32` de columnas de precio/griegas para reducir RAM.
- Convierte fechas, calcula `Days = Expiration − Date − AMSettlement`, `Strike/1000`, `MidPrice`, `horquilla`.
- Filtros: quita `IV == -99.99`, exige `Bid ≥ 0` y `Ask > Bid`.
- (Bloque comentado) opción 1: moneyness forward interpolando `ForwardPrice` y curva cero.
- Opción 2 activa: merge con `SP500 ClosePrice → SpotPrice`, calcula `Moneyness = Strike/SpotPrice`, `log_moneyness`, `flag_otm`.

**Obtiene** → `OUTPUTS/opt_df_prueba.parquet` (panel opción-día limpio con moneyness spot).

---

### `02_panel_construction.py`

**Importa**
- `opt_df_prueba.parquet` (salida de `01_`).

**Hace**
- Añade `Dummy_Bid` y `DolarVolume = Volume·MidPrice`.
- Asigna `maturity_bucket` con bordes `[0, 15, 45, 105, 183, 365, ∞]` y `moneyness_bucket` con `[0, 0.1, …, 2.0, ∞]` (paso 0.1).
- Define seis funciones para **delta / gamma empíricas**:
  - `op1` — nivel contrato: primera diferencia de `MidPrice` sobre primera diferencia de `SpotPrice`; para gamma, segunda diferencia (con factor 2 sobre `dS`) forzando misma `moneyness_bucket` en `t` y `t−1`.
  - `op2` — media ponderada por `OpenInterest` de las `op1` por `(Date, CallPut, moneyness_bucket)`.
  - `op3` — precio agregado (WA por OI) primero y luego diferencia por bucket.
- `calcular_greeks_empiricas` orquesta las tres y hace merges dejando `delta_emp`, `delta_emp_op2`, `delta_emp_op3`, `gamma_emp`, `gamma_emp_op2`, `gamma_emp_op3`.
- **Se ejecuta por cada `maturity_bucket` con filtro `Bid > 0`** y se concatena.

**Obtiene** → `OUTPUTS/opt_df_empirical_greeks.parquet` (panel opción-día con griegas empíricas en tres agregaciones).

---

### `03_panel_diagnosis.py`

**Importa**
- `opt_df_empirical_greeks.parquet` (mac) / `opt_df_prueba.parquet` (win). Filtra `Bid > 0`.

**Hace**
- Recupera los buckets como `pd.Interval` con `parse_interval`.
- `analisis_vencimiento(v_min, v_max)` — para un tramo de vencimiento:
  - Tabla por bucket de moneyness: `n_contracts`, sumas de OI y $Vol, medias de `Dummy_Bid`.
  - % de cobertura de días y **rachas** de continuidad (max/min/media).
  - Gráfico 3-paneles: bucket mínimo, máximo y n_buckets/día.
- `analisis_detallado(v_min, v_max, subperiodos)` — extiende con:
  - Métrica de **coexistencia** de `Bid=0` y `Bid>0` el mismo día/bucket.
  - Tabla y continuidad por subperiodo.
  - Gráficos `spread_cobertura` (fill_between m_min/m_max), `spread` (amplitud), `ridgeline` de nº contratos por bucket.
- Todo el bloque final con los cálculos de delta/gamma y `diagnostico_greek` (signos incorrectos, outliers, distribución por CP / bucket / cuartil de OI, días sin greek válida) está **comentado**: replica lo que ya hace `02_` con diagnóstico añadido.

**Obtiene** — Solo prints, tablas `tabulate` y figuras `matplotlib`; no persiste parquet.

---

### `05_Gamma_Spread.py`

**Importa**
- `opt_df_empirical_greeks.parquet`.

**Hace**
- Reconstruye los buckets y filtra `maturity_bucket = (15, 45]`; descarta las columnas `op2/op3` y `Moneyness_Forward`; `dropna`.
- `WA_diaria(df, variable, greek_emp, greek_teo)` → serie diaria por `CallPut` con la **media ponderada** de `greek_emp` (por `OpenInterest` o `DolarVolume`), la media del peso y `n_contratos`.
- `gamma_spread_left(df, variable, greek_emp, bucket_col="Moneyness")` — agrupa cada día en cuatro rangos: `very_deep [0, 0.5]`, `deep (0.5, 0.7]`, `near (0.7, 0.9]`, `ATM (0.9, 1.1]`; hace WA dentro de cada uno y calcula el **spread ATM − bucket** para cada zona.
- Se ejecuta para `gamma_emp` y `delta_emp`, y para pesos `OpenInterest` y `DolarVolume`; une todo por `(Date, CallPut, bucket)` / `(Date, CallPut)`, limpia columnas duplicadas y reordena.

**Obtiene** → `OUTPUTS/Agg_Greeks.csv` (parquet snappy) con niveles ATM, spreads gamma/delta, WA diarias y contadores.

---

### `06_gamma_spread_diagnosis.py`

**Importa**
- `Agg_Greeks` (si `empirical=True`) o `Agg_Greeks_BS` (si `False`).
- Fama-French `F-F_Research_Data_Factors.csv`, `security_price.parquet` (SPX) para retorno.

**Hace**
- **Continuidad del spread**: `analisis_continuidad_spread` replica el análisis de rachas de `03_` pero sobre la disponibilidad diaria del `spread_valor` por `(CallPut, bucket)`; visualiza con `grafico_continuidad_spread` (scatter tipo raster).
- **Panel mensual**: `serie_mensual` toma el último día de cotización de cada mes por variable (puts por defecto); `retorno_mensual_compuesto` compone `TotalReturn` diario del SPX.
- Une todas las series con `Mkt-RF`, `RF` y `R_SPX_m`.
- **Regresión predictiva IS**: `regresion_predictiva` corre OLS con errores **Newey-West** (lags = horizonte) para múltiples horizontes `h ∈ {1,2,3,12,24,36,48}` y variables `variables_mensuales`.

**Obtiene** — Prints de continuidad, tablas de regresión, figuras; no persiste parquet.

---

### `07_BS_spread_&_diagnosis.py`

**Importa**
- `opt_df_empirical_greeks.parquet` (usa las griegas **teóricas** `Delta`, `Gamma` que ya venían del proveedor).

**Hace**
- Es un **clon de `05_`** pero pasando `"Gamma"` y `"Delta"` (BS) en lugar de `gamma_emp` / `delta_emp` a `WA_diaria` y `gamma_spread_left`.
- Misma unión, limpieza y reordenación.

**Obtiene** → `OUTPUTS/Agg_Greeks_BS.csv` — mismo esquema que `Agg_Greeks.csv` pero con niveles y spreads BS; alimenta la rama `empirical=False` de `06_`.

---

### `contrastes/04_contraste_macrovariables.py`

**Importa**
- `opt_df_empirical_greeks.parquet`.
- Externos: `yfinance` (VIX), Fama-French, series CFNAI, `security_price` SPX.
- Estadística: `statsmodels` (ADF, ACF, PACF, ARIMA, OLS), `arch.arch_model`, `scipy.stats`.

**Hace** (organizado por “Análisis”)
- **A1** — Descriptivos por `CallPut` de `delta_emp/gamma_emp` vs `Delta/Gamma` BS con percentiles; recorte al [5%, 95%].
- **A2** — Series temporales mensuales:
  - `contrato_mediana_moneyness`: contrato diario más cercano a la mediana de moneyness (comparativa emp vs BS).
  - Media aritmética y **WA por OI** mensual (empírica vs BS).
- **A3** — Correlaciones y **autocorrelaciones**, ADF sobre las series mensuales.
- **A4** — Frecuencias mensuales alternativas de las griegas:
  - Método **0**: WA mensual por OI (ya en A2).
  - Método **A**: diferencia entre el primer y el último día del mes.
  - Método **Barbon (2020)**: `greek_net_exposure` — net exposure diario ponderado por OI y spot, puts en signo negativo, normalizado por el volumen medio de los `rolling_days` anteriores (semielasticidad al 1% del subyacente).
  - `diferencia_mensual` y `diagnostico_serie_temporal` (media, std, ADF, ACF, PACF, ARIMA fit) sobre cada versión.
- Pruebas con **ARCH/skew-t** sobre la innovación de gamma calls.
- **Contraste macro**: VIX (mensual, yfinance), CFNAI — series temporales dobles, correlaciones (heatmap seaborn), scatter con recta OLS, para Gamma1 y Delta1 y para el Net Exposure.
- **A5** — Descomposición **vanna / charm** de la brecha entre gamma realizada y gamma BS.

**Obtiene** — Prints, tablas `tabulate` y muchas figuras; no persiste parquet.

---

## Rutas de I/O (resumen)

| Script | Input | Output |
|---|---|---|
| `01` | `option_price.parquet`, `forward_price_filtered.parquet`, `zero_curve.parquet`, `security_price.parquet` | `opt_df_prueba.parquet` |
| `02` | `opt_df_prueba.parquet` | `opt_df_empirical_greeks.parquet` |
| `03` | `opt_df_empirical_greeks.parquet` | — (diagnóstico) |
| `05` | `opt_df_empirical_greeks.parquet` | `Agg_Greeks.csv` (parquet) |
| `07` | `opt_df_empirical_greeks.parquet` | `Agg_Greeks_BS.csv` (parquet) |
| `06` | `Agg_Greeks` / `Agg_Greeks_BS`, FF, SPX | — (regresión IS) |
| `contrastes/04` | `opt_df_empirical_greeks.parquet`, VIX, CFNAI, FF, SPX | — (contraste macro) |

Documentos previos: [`Descripción(01-03).md`](Descripción(01-03).md) y [`contrastes/Descripción(04).md`](contrastes/Descripción(04).md) recogen la motivación y los criterios de análisis; este README los complementa con la vista técnica del código.
