import pandas as pd

path = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\GI.ALL.IVYOPPRCD\GI.ALL.IVYOPPRCD_202402.txt"

df = pd.read_csv(
    path,
    sep=r"\t",
    engine="python",
    encoding="utf-8",
    on_bad_lines="warn"
)

print(df[df["SecurityID"]==108105]['Strike'].describe())
print(df.head())
print(df.columns)
print(df.shape)