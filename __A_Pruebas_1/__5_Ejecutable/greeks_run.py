# In[]
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __2_Files.volatility_surface import VolatilitySurface
from __2_Files.forward_price import ForwardPrice

# In[]
# ─── Configuración ────────────────────────────────────────────────────────────

PARQUET_DIR  = Path(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado")
OUTPUT_PATH  = Path(r"C:\Users\pablo.esparcia\Downloads\pricing_resultado.csv")

YEARS        = list(range(2005, 2025))
YEARS        = [2015,2016]
CURRENCY     = 333          # USD
SID_EUROPEO  = 108105      #SP500

MISSING      = -99.99
# ─── 1. Carga de datos ────────────────────────────────────────────────────────

print("Cargando datos...")

vs  = VolatilitySurface()
vs.cargar_parquet(ruta=str(PARQUET_DIR / "volatility_surface.parquet"))

# ─── 2. Filtrado por años, divisa securities y missing values ────────────────────────────────

mask_YEARS    = vs.df["Date"].dt.year.isin(YEARS)
mask_currency = vs.df["Currency"] == CURRENCY
mask_sids     = vs.df["SecurityID"].isin([SID_EUROPEO])
# In[]
vol_filtrada  = vs.df[mask_YEARS & mask_currency & mask_sids]
vol_filtrada = vol_filtrada[(vol_filtrada["ImpliedVol"] > MISSING) & (vol_filtrada["Strike"] > 0)]

print(f"\nN úmero de filas   : {len(vol_filtrada):,}")

# In[]

# ─── 4. Cálculo de sensibilidades numéricas ───────────────────────────────────────────────────────────────
    #Para puts:
vol_filtrada_put = vol_filtrada[vol_filtrada["CallPut"] == "P"]    
for fecha, vol_fecha in vol_filtrada_put.groupby("Date"):
    print(fecha)



# ─── 7. Exportar CSV ──────────────────────────────────────────────────────────

# %%

PARQUET = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\forward_price.parquet"

fp = ForwardPrice()

fp.cargar_parquet(PARQUET,
    desde="2019-01-02",
    hasta="2019-01-02"
)
# %%
fp.df[fp.df["SecurityID"]==108105]


# %%
