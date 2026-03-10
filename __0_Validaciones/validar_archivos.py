"""
Validación 1 — Número y continuidad de archivos TXT por producto.

Comprueba en cada subcarpeta GI.ALL.* de CARPETA:
  ✅ Número total de archivos TXT
  ⚠️  Meses faltantes en la secuencia YYYYMM
  ❌ Archivos duplicados para el mismo mes
  ⚠️  Archivos cuyo nombre no sigue el patrón esperado
"""

import os
import re
from pathlib import Path


# ─── Configuración ────────────────────────────────────────────────────────────

CARPETA = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado"

# ─── Utilidades ───────────────────────────────────────────────────────────────

_RE_FECHA = re.compile(r'_(\d{6})\.txt$', re.IGNORECASE)   # YYYYMM


def _extraer_yyyymm(nombre: str) -> str | None:
    m = _RE_FECHA.search(nombre)
    return m.group(1) if m else None


def _siguiente_mes(yyyymm: str) -> str:
    y, m = int(yyyymm[:4]), int(yyyymm[4:])
    m += 1
    if m > 12:
        m, y = 1, y + 1
    return f"{y:04d}{m:02d}"


def _meses_faltantes(fechas: list[str]) -> list[str]:
    """Devuelve los meses ausentes en la secuencia [min, max]."""
    if len(fechas) < 2:
        return []
    presentes = set(fechas)
    faltantes, actual = [], min(fechas)
    fin = max(fechas)
    while actual < fin:
        actual = _siguiente_mes(actual)
        if actual not in presentes:
            faltantes.append(actual)
    return faltantes


def _analizar_carpeta(ruta: Path) -> dict:
    txts = sorted(
        f for f in os.listdir(ruta)
        if f.lower().endswith('.txt') and os.path.isfile(ruta / f)
    )
    por_mes: dict[str, list[str]] = {}
    sin_fecha: list[str] = []
    for f in txts:
        yyyymm = _extraer_yyyymm(f)
        if yyyymm:
            por_mes.setdefault(yyyymm, []).append(f)
        else:
            sin_fecha.append(f)

    fechas     = sorted(por_mes)
    duplicados = {k: v for k, v in por_mes.items() if len(v) > 1}
    faltantes  = _meses_faltantes(fechas)

    return {
        'total':      len(txts),
        'fechas':     fechas,
        'rango':      (fechas[0], fechas[-1]) if fechas else ('—', '—'),
        'duplicados': duplicados,
        'faltantes':  faltantes,
        'sin_fecha':  sin_fecha,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    W = 72
    raiz       = Path(CARPETA)
    productos  = sorted(d for d in raiz.iterdir() if d.is_dir() and d.name.startswith('GI.'))

    if not productos:
        print(f"No se encontraron subcarpetas GI.* en:\n  {CARPETA}")
        return

    print(f"\n{'─' * W}")
    print(f"  VALIDACIÓN 1 — ARCHIVOS     {CARPETA}")
    print(f"{'─' * W}")
    print(f"  {'Producto':<30}  {'Archivos':>8}  {'Rango':>13}  Estado")
    print(f"{'─' * W}")

    total_warn = total_err = 0
    for prod in productos:
        r = _analizar_carpeta(prod)
        rango = f"{r['rango'][0]}–{r['rango'][1]}"

        avisos, errores = [], []
        if r['faltantes']:
            avisos.append(f"⚠  {len(r['faltantes'])} meses faltantes")
            total_warn += len(r['faltantes'])
        if r['sin_fecha']:
            avisos.append(f"⚠  {len(r['sin_fecha'])} sin fecha")
            total_warn += len(r['sin_fecha'])
        if r['duplicados']:
            errores.append(f"✗ {len(r['duplicados'])} duplicados")
            total_err += len(r['duplicados'])

        estado = '  '.join(errores + avisos) if (errores or avisos) else '✓ OK'
        print(f"  {prod.name:<30}  {r['total']:>8}  {rango:>13}  {estado}")

        if r['faltantes']:
            muestra = ', '.join(r['faltantes'][:10])
            sufijo  = f' … (+{len(r["faltantes"]) - 10} más)' if len(r['faltantes']) > 10 else ''
            print(f"      └─ Faltantes : {muestra}{sufijo}")
        for mes, archivos in r['duplicados'].items():
            print(f"      └─ Duplicado {mes}: {archivos}")
        for f in r['sin_fecha']:
            print(f"      └─ Sin fecha : {f}")

    print(f"{'─' * W}")
    print(f"  Productos analizados : {len(productos)}")
    print(f"  Avisos  (⚠)          : {total_warn}")
    print(f"  Errores (✗)          : {total_err}")
    print(f"{'─' * W}\n")


if __name__ == "__main__":
    main()
