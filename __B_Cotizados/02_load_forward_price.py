import pandas as pd
import duckdb
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __2_Files.forward_price import ForwardPrice



PARQUET = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\forward_price.parquet"
PARQUET_filtered = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\output\forward_price_filtered.parquet"
fp = ForwardPrice()
fp.cargar_parquet(PARQUET)

# In[]:

fp.df = fp.df[fp.df['SecurityID'] == 108105]

# creamos una columna de fecha de vencimiento, que es la fecha de negociación + el tiempo a vencimiento (en días)
fp.df["Date"] = pd.to_datetime(fp.df["Date"], format="%Y-%m-%d")
fp.df["Expiration"] = pd.to_datetime(fp.df["Expiration"], format="%Y-%m-%d")
fp.df["Days"] = ((fp.df["Expiration"] - fp.df["Date"]).dt.days - fp.df["AMSettlement"]) # en días (base 365 cuando se use)

# In[]:

# Priorizar AM settlement (AMSettlement=0) sobre PM (AMSettlement=1)
x = (
    fp.df
    .sort_values("AMSettlement", ascending=True)  # AM primero
    .drop_duplicates(subset=["SecurityID", "Date", "Expiration"])
)
print(x.head(20))

# %%

duckdb.from_df(x).write_parquet(PARQUET_filtered.replace('\\', '/'), compression='snappy')

# %%
