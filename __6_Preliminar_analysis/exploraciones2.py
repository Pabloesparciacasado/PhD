
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

PATH_DATA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\lpr_real_2023_C.parquet"

print("Cargando datos...")
opt_df = pd.read_parquet(PATH_DATA)

df = opt_df[~opt_df["n_eff"].isna()].copy()  # filtra solo donde se pudo estimar LPR

print(df.head())

print(f"Número de observaciones con LPR estimada: {len(df)}")
print(f"Numero de Nans en el calculo de derivadas: {opt_df['n_eff'].isna().sum()}")
print(f"Propocion de Nans en derivadas sobre el total: {opt_df['n_eff'].isna().mean():.2%}")