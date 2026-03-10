import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from __2_Files.option_price import OptionPrice


# ─── Configuración ────────────────────────────────────────────────────────────

PARQUET_RUTA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\option_price.parquet"
CSV_RUTA     = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado\option_price.csv"
CARPETA      = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado"

# Columnas necesarias para este script
COLUMNAS = [
    'SecurityID', 'Date', 'Expiration', 'Strike', 'CallPut',
    'ImpliedVolatility', 'Delta', 'Bid', 'Ask', 'Volume', 'OpenInterest',
]


# ─── 1. Cargar datos ──────────────────────────────────────────────────────────

op = OptionPrice(sep='\t')

if Path(PARQUET_RUTA).exists():
    # Ruta habitual: Parquet ya generado → pandas puro
    op.cargar_parquet(
        ruta     = PARQUET_RUTA,
        desde    = "1996-01-01",
        hasta    = "2023-12-31",
        columnas = COLUMNAS,
    )
elif Path(CSV_RUTA).exists():
    # Fallback: CSV → DuckDB (sin carga en RAM)
    print("Parquet no encontrado — usando CSV vía DuckDB…")
    op.cargar_csv(CSV_RUTA)
else:
    # Primera vez: compilar desde TXT
    print("Compilando desde TXT (puede tardar varios minutos)…")
    op.cargar_datos(carpeta=CARPETA, guardar_en=CSV_RUTA)

df = op.df    # alias para trabajar más cómodamente


# ─── 2. Subyacente y fecha de referencia ──────────────────────────────────────

if df is not None:
    # Parquet cargado → pandas puro
    sid   = int(df.groupby('SecurityID')['Volume'].sum().idxmax())
    fecha = df['Date'].max()
    fecha = fecha.strftime('%Y-%m-%d') if hasattr(fecha, 'strftime') else str(fecha)
else:
    # CSV en DuckDB
    sid   = int(op.query("SELECT SecurityID FROM {csv} GROUP BY SecurityID ORDER BY SUM(Volume) DESC LIMIT 1").iloc[0]['SecurityID'])
    fecha = op.fechas_disponibles(sid)[-1]

print(f"\nSecurityID : {sid}")
print(f"Fecha ref  : {fecha}")


# ─── 3. Top 10 subyacentes por volumen acumulado ──────────────────────────────

if df is not None:
    top10 = (df.groupby('SecurityID')['Volume']
               .sum()
               .nlargest(10)
               .sort_values())
else:
    top10 = (op.query("SELECT SecurityID, SUM(Volume) AS vol FROM {csv} GROUP BY SecurityID ORDER BY vol DESC LIMIT 10")
               .set_index('SecurityID')['vol']
               .sort_values())

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(top10.index.astype(str), top10.values / 1e6,
               color='steelblue', alpha=0.85)
ax.bar_label(bars, fmt='%.1f M', padding=3, fontsize=8)
ax.set_xlabel('Volumen total (millones de contratos)')
ax.set_title('Top 10 subyacentes por volumen acumulado')
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('top10_volumen.png', dpi=150)
plt.show()


# ─── 4. Serie temporal de actividad diaria ────────────────────────────────────

if df is not None:
    serie = (df.groupby('Date')[['Volume', 'OpenInterest']]
               .sum()
               .sort_index())
else:
    serie = (op.query("SELECT Date, SUM(Volume) AS Volume, SUM(OpenInterest) AS OpenInterest FROM {csv} GROUP BY Date ORDER BY Date")
               .set_index('Date'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

ax1.plot(serie.index, serie['Volume'] / 1e6, linewidth=1, color='steelblue')
ax1.set_ylabel('Volumen (M contratos)')
ax1.set_title('Actividad diaria – mercado de opciones completo')
ax1.grid(True, alpha=0.3)

ax2.plot(serie.index, serie['OpenInterest'] / 1e6, linewidth=1, color='darkorange')
ax2.set_ylabel('Open Interest (M contratos)')
ax2.set_xlabel('Fecha')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('actividad_diaria.png', dpi=150)
plt.show()


# ─── 5. Put-Call Ratio diario (volumen) ───────────────────────────────────────

if df is not None:
    cp = (df.groupby(['Date', 'CallPut'])['Volume']
            .sum()
            .unstack()
            .sort_index())
else:
    cp_raw = op.query("SELECT Date, CallPut, SUM(Volume) AS Volume FROM {csv} GROUP BY Date, CallPut ORDER BY Date")
    cp     = cp_raw.pivot(index='Date', columns='CallPut', values='Volume')

ratio = (cp['C'] / cp['P']).replace([np.inf, -np.inf], np.nan).dropna()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(ratio.index, ratio, linewidth=1, color='purple')
ax.axhline(1, color='gray', linestyle='--', linewidth=0.8)
ax.set_title('Put-Call Ratio (volumen) – mercado completo')
ax.set_xlabel('Fecha')
ax.set_ylabel('C / P ratio')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('put_call_ratio.png', dpi=150)
plt.show()


# ─── 6. Smile de volatilidad implícita ────────────────────────────────────────

vencimientos = op.vencimientos_disponibles(sid, fecha)
print(f"\nVencimientos disponibles ({fecha}): {vencimientos}")

colores_call = cm.Blues(np.linspace(0.4, 0.9, min(len(vencimientos), 4)))
colores_put  = cm.Reds (np.linspace(0.4, 0.9, min(len(vencimientos), 4)))

fig, (ax_c, ax_p) = plt.subplots(1, 2, figsize=(13, 5))

for exp, cc, cp_color in zip(vencimientos[:4], colores_call, colores_put):
    for ax, tipo, color in ((ax_c, 'C', cc), (ax_p, 'P', cp_color)):
        try:
            df_exp = op.opciones_por_vencimiento(sid, fecha, exp, call_put=tipo)
            df_exp = df_exp[df_exp['ImpliedVolatility'] > 0].copy()
            if df_exp.empty:
                continue
            moneyness = df_exp['Strike'] / df_exp['Strike'].median()
            ax.plot(moneyness, df_exp['ImpliedVolatility'],
                    marker='o', ms=3, linewidth=1.2, color=color, label=exp)
        except ValueError:
            continue

for ax, titulo in ((ax_c, 'Calls'), (ax_p, 'Puts')):
    ax.set_title(f'{titulo}  –  SecurityID {sid}  ({fecha})')
    ax.set_xlabel('Strike / Strike mediano  (moneyness)')
    ax.set_ylabel('Implied Volatility')
    ax.legend(title='Vencimiento', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Smile de volatilidad implícita', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('smile_iv.png', dpi=150)
plt.show()


# ─── 7. Estructura de términos de IV ATM ──────────────────────────────────────

atm_iv: dict[int, float] = {}
for exp in vencimientos:
    try:
        df_exp = op.opciones_por_vencimiento(sid, fecha, exp, call_put='C')
        df_exp = df_exp[df_exp['ImpliedVolatility'] > 0].copy()
        if df_exp.empty:
            continue
        mid        = (df_exp['Bid'] + df_exp['Ask']) / 2
        atm_strike = df_exp.loc[(mid - mid.median()).abs().idxmin(), 'Strike']
        iv_atm     = float(df_exp[df_exp['Strike'] == atm_strike]['ImpliedVolatility'].mean())
        days       = (pd.Timestamp(exp) - pd.Timestamp(fecha)).days
        atm_iv[days] = iv_atm
    except (ValueError, KeyError):
        continue

if atm_iv:
    term = pd.Series(atm_iv).sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(term.index, term.values, marker='o', linewidth=1.5, color='teal')
    ax.set_xlabel('Días al vencimiento')
    ax.set_ylabel('IV ATM')
    ax.set_title(f'Estructura de términos de IV  –  SecurityID {sid}  ({fecha})')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('term_structure.png', dpi=150)
    plt.show()
