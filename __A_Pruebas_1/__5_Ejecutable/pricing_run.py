"""
pricing_run.py
==============
Ejecuta el pipeline de valoración para:
  - SecurityID 109764  →  árbol binomial CRR (opción americana)
  - SecurityID 108105  →  Black-Scholes-Merton (opción europea)

Produce un único DataFrame combinado con columnas homogéneas y lo
exporta a CSV en la carpeta de Downloads.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __2_Files.volatility_surface import VolatilitySurface
from __2_Files.zero_curve         import ZeroCurve
from __2_Files.index_dividend     import IndexDividend
from __3_Functions.valuation      import model_valuation


# ─── Configuración ────────────────────────────────────────────────────────────

PARQUET_DIR  = Path(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado")
OUTPUT_PATH  = Path(r"C:\Users\pablo.esparcia\Downloads\pricing_resultado.csv")

YEARS        = [2019, 2020]
CURRENCY     = 333          # USD

SID_AMERICANO = 109764
SID_EUROPEO   = 108105


# ─── 1. Carga de datos ────────────────────────────────────────────────────────

print("Cargando datos...")

vs  = VolatilitySurface()
vs.cargar_parquet(ruta=str(PARQUET_DIR / "volatility_surface.parquet"))

zc  = ZeroCurve(sep='\t')
zc.cargar_parquet(ruta=str(PARQUET_DIR / "zero_curve.parquet"))

id_ = IndexDividend()
id_.cargar_parquet(ruta=str(PARQUET_DIR / "index_dividend.parquet"))

print(f"  VolatilitySurface : {len(vs.df):,} filas")
print(f"  ZeroCurve         : {len(zc.df):,} filas")
print(f"  IndexDividend     : {len(id_.df):,} filas")


# ─── 2. Filtrado por años, divisa y securities ────────────────────────────────

mask_YEARS    = vs.df["Date"].dt.year.isin(YEARS)
mask_currency = vs.df["Currency"] == CURRENCY
mask_sids     = vs.df["SecurityID"].isin([SID_AMERICANO, SID_EUROPEO])

vol_filtrada  = vs.df[mask_YEARS & mask_currency & mask_sids]

curva_filtrada = zc.df[
    zc.df["Date"].dt.year.isin(YEARS) & (zc.df["Currency"] == CURRENCY)
]
divs_filtrada  = id_.df[id_.df["Date"].dt.year.isin(YEARS)]

print(f"\nOpciones a valorar   : {len(vol_filtrada):,} filas")
print(f"  SecurityID {SID_AMERICANO} (americana) : "
      f"{(vol_filtrada['SecurityID'] == SID_AMERICANO).sum():,} filas")
print(f"  SecurityID {SID_EUROPEO} (europea)   : "
      f"{(vol_filtrada['SecurityID'] == SID_EUROPEO).sum():,} filas")


# ─── 3. Modelo de valoración ──────────────────────────────────────────────────

model = model_valuation(
    curve_df = curva_filtrada,
    currency = CURRENCY,
    div_df   = divs_filtrada,
)


# ─── 4. Pricing ───────────────────────────────────────────────────────────────

# --- 4a. Americano (CRR binomial) ---
print(f"\nValorando {SID_AMERICANO} con árbol CRR (n=200)...")
datos_americano = vol_filtrada[vol_filtrada["SecurityID"] == SID_AMERICANO]
resultado_americano = model.price_american(datos_americano, n_steps=200)
resultado_americano = resultado_americano.rename(columns={"American_Price": "Precio_Modelo"})
resultado_americano["Modelo"] = "CRR-Americano"
print(f"  Filas resultado: {len(resultado_americano):,}")

# --- 4b. Europeo (BSM) ---
print(f"Valorando {SID_EUROPEO} con BSM...")
datos_europeo = vol_filtrada[vol_filtrada["SecurityID"] == SID_EUROPEO]
resultado_europeo = model.price_BS(datos_europeo)
resultado_europeo = resultado_europeo.rename(columns={"BS_Price": "Precio_Modelo"})
resultado_europeo["Modelo"] = "BSM-Europeo"
print(f"  Filas resultado: {len(resultado_europeo):,}")


# ─── 5. Combinación en un único DataFrame ─────────────────────────────────────

COLUMNAS_SALIDA = [
    "Modelo",
    "SecurityID",
    "Date",
    "Days",
    "CallPut",
    "Delta",
    "Strike",
    "ImpliedVol",
    "r",
    "q",
    "Underlying",
    "Precio_Modelo",
    "Premium",
]

resultado = (
    pd.concat([resultado_americano, resultado_europeo], ignore_index=True)
    .sort_values(["SecurityID", "Date", "Days", "CallPut", "Delta"])
    .reset_index(drop=True)
    [COLUMNAS_SALIDA]
)


# ─── 6. Columna de error ──────────────────────────────────────────────────────

MISSING = -99.99
mask_premium = resultado["Premium"] > MISSING
resultado["Error_abs"] = np.where(
    mask_premium,
    (resultado["Precio_Modelo"] - resultado["Premium"]).abs(),
    np.nan,
)
resultado["Error_pct"] = np.where(
    mask_premium,
    resultado["Error_abs"] / resultado["Premium"].abs() * 100,
    np.nan,
)

# ─── 7. Exportar CSV ──────────────────────────────────────────────────────────

resultado.to_csv(OUTPUT_PATH, index=False, float_format="%.6f")
print(f"\nCSV guardado en: {OUTPUT_PATH}")
