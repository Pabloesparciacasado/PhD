
import pandas as pd


opt_df = pd.read_parquet(r"C:\Users\pablo.esparcia\Documents\OptionMetrics\output\preliminares\opt_df_prueba.parquet")
### Aplicamos filtros básicos para incluir opciones que tengan al menos OI o Volumen:
opt_df_filtered = opt_df[(opt_df["OpenInterest"] > 0) | (opt_df["Volume"] > 0)].reset_index(drop=True)

#mostramos descriptivos de los contratos filtrados:
print("Número de contratos después del filtrado:", f"{len(opt_df_filtered):,.0f}")
print("Número de contratos después del filtrado:", f"{len(opt_df_filtered):,.0f}")
print("Número de contratos eliminados con el filtrado:", f"{len(opt_df) - len(opt_df_filtered):,.0f}" )

desc_strike = opt_df_filtered['Strike'].astype(float).describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99])
desc_moneyness = opt_df_filtered['Moneyness'].astype(float).describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99])
desc_OI  = opt_df_filtered['OpenInterest'].astype(float).describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99])
desc_volume = opt_df_filtered['Volume'].astype(float).describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99])
desc_midprice  = opt_df_filtered['MidPrice'].astype(float).describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99])


x = pd.concat([desc_strike, desc_moneyness, desc_OI, desc_volume, desc_midprice], axis=1)
x.columns = ['Strike', 'Moneyness', 'OpenInterest', 'Volume', 'MidPrice']

formatters = {
    'Strike':       '{:>14,.2f}'.format,
    'Moneyness':    '{:>12.4f}'.format,
    'OpenInterest': '{:>18,.0f}'.format,
    'Volume':       '{:>18,.0f}'.format,
    'MidPrice':     '{:>12.4f}'.format,
}
print(x.to_string(formatters=formatters))






