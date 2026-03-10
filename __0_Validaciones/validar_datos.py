"""
Validación 3 — Integridad estructural de los datos (columnas y tabulaciones).

Comprueba para cada archivo TXT:
  · Número de columnas esperado (inferido de la primera fila del primer archivo).
  · Filas con un número distinto de tabuladores → dato desplazado o truncado.
  · Filas completamente vacías (solo salto de línea).
  · Codificación: avisa si hay caracteres no-UTF-8 (fallback a latin-1).

Configuración:
  CARPETA    — raíz de los datos (contiene las subcarpetas GI.ALL.*)
  PRODUCTO   — si se especifica (ej. "GI.ALL.IVYOPPRCD"), analiza solo ese.
               Si se deja en None, analiza todos los productos.
  MAX_ERRORES_MUESTRA — máximo de filas erróneas a imprimir por archivo.
"""

import os
from pathlib import Path


# ─── Configuración ────────────────────────────────────────────────────────────

CARPETA             = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado"
PRODUCTO            = None      # ej. "GI.ALL.IVYOPPRCD"  o  None para todos
MAX_ERRORES_MUESTRA = 5         # líneas erróneas a mostrar por archivo


# ─── Utilidades ───────────────────────────────────────────────────────────────

def _leer_lineas(ruta: str) -> tuple[list[str], str]:
    """Lee todas las líneas del fichero. Prueba UTF-8 primero, luego latin-1."""
    for enc in ('utf-8', 'latin-1'):
        try:
            with open(ruta, encoding=enc, errors='strict') as f:
                return f.readlines(), enc
        except UnicodeDecodeError:
            continue
    # Fallback total: ignora caracteres inválidos
    with open(ruta, encoding='utf-8', errors='replace') as f:
        return f.readlines(), 'utf-8 (con errores)'


def _analizar_archivo(ruta: str, cols_esperadas: int | None) -> dict:
    """
    Analiza un archivo TXT.
    Si cols_esperadas es None, lo infiere de la primera fila.
    Devuelve estadísticas y lista de filas erróneas.
    """
    lineas, encoding = _leer_lineas(ruta)

    if not lineas:
        return {
            'lineas': 0, 'cols': 0, 'encoding': encoding,
            'vacias': 0, 'erroneas': [], 'cols_esperadas': 0,
        }

    # Inferir columnas esperadas desde la primera línea no vacía
    primera = lineas[0].rstrip('\n\r')
    cols_inf = primera.count('\t') + 1
    if cols_esperadas is None:
        cols_esperadas = cols_inf

    vacias   = 0
    erroneas = []   # lista de (nº línea, tabs encontrados, extracto)

    for i, linea in enumerate(lineas, start=1):
        stripped = linea.rstrip('\n\r')
        if stripped == '':
            vacias += 1
            continue
        tabs = stripped.count('\t')
        cols = tabs + 1
        if cols != cols_esperadas:
            erroneas.append((i, cols, stripped[:80]))

    return {
        'lineas':          len(lineas),
        'cols':            cols_inf,
        'cols_esperadas':  cols_esperadas,
        'encoding':        encoding,
        'vacias':          vacias,
        'erroneas':        erroneas,
    }


def _analizar_producto(ruta_prod: Path) -> None:
    """Analiza todos los TXT de una carpeta de producto e imprime el resultado."""
    W    = 72
    txts = sorted(
        f for f in os.listdir(ruta_prod)
        if f.lower().endswith('.txt') and os.path.isfile(ruta_prod / f)
    )

    if not txts:
        print(f"  {ruta_prod.name:<30}  (sin archivos TXT)")
        return

    # Primera pasada: inferir columnas esperadas del primer archivo no vacío
    cols_esp = 0
    for f0 in txts:
        primer = _analizar_archivo(str(ruta_prod / f0), None)
        if primer['cols'] > 0:
            cols_esp = primer['cols']
            break
    if cols_esp == 0:
        print(f"  {ruta_prod.name:<30}  (no se pudo inferir nº de columnas)")
        return

    total_erroneas = total_vacias = archivos_con_error = 0
    detalles: list[tuple[str, dict]] = []

    for f in txts:
        r = _analizar_archivo(str(ruta_prod / f), cols_esp)
        total_erroneas += len(r['erroneas'])
        total_vacias   += r['vacias']
        if r['erroneas'] or r['vacias'] or r['lineas'] == 0:
            archivos_con_error += 1
            detalles.append((f, r))

    if not detalles:
        print(f"  {ruta_prod.name:<30}  {len(txts):>4} archivos  "
              f"{cols_esp} cols  ✓ todos OK")
        return

    # Hay problemas: imprimir detalle
    print(f"\n  ── {ruta_prod.name}  ({len(txts)} archivos, {cols_esp} cols esperadas) ──")
    print(f"  {'Archivo':<42}  {'Líneas':>8}  {'Err.col':>7}  {'Vacías':>6}  Estado")
    print(f"  {'─' * (W - 2)}")

    for nombre, r in detalles:
        n_err = len(r['erroneas'])
        estado = []
        if r['lineas'] == 0:
            estado.append('✗ VACÍO')
        if n_err:
            estado.append(f'✗ {n_err} filas col. incorrectas')
        if r['vacias']:
            estado.append(f'⚠ {r["vacias"]} líneas vacías')
        if r['encoding'] != 'utf-8':
            estado.append(f'⚠ encoding: {r["encoding"]}')

        print(f"  {nombre:<42}  {r['lineas']:>8,}  {n_err:>7}  {r['vacias']:>6}  "
              + '  '.join(estado))

        for linea_n, cols_enc, extracto in r['erroneas'][:MAX_ERRORES_MUESTRA]:
            print(f"      L{linea_n:>7}: {cols_enc} cols — {extracto!r}")
        if len(r['erroneas']) > MAX_ERRORES_MUESTRA:
            print(f"      … ({len(r['erroneas']) - MAX_ERRORES_MUESTRA} más)")

    print(f"  {'─' * (W - 2)}")
    print(f"  Filas erróneas total: {total_erroneas}   "
          f"Líneas vacías total: {total_vacias}   "
          f"Archivos afectados: {archivos_con_error}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    W    = 72
    raiz = Path(CARPETA)

    if PRODUCTO:
        productos = [raiz / PRODUCTO]
    else:
        productos = sorted(d for d in raiz.iterdir() if d.is_dir() and d.name.startswith('GI.'))

    if not productos:
        print(f"No se encontraron subcarpetas GI.* en:\n  {CARPETA}")
        return

    print(f"\n{'─' * W}")
    print(f"  VALIDACIÓN 3 — DATOS (columnas y tabulaciones)")
    print(f"  {CARPETA}")
    print(f"{'─' * W}")

    for prod in productos:
        if not prod.exists():
            print(f"  ⚠ No encontrado: {prod.name}")
            continue
        _analizar_producto(prod)

    print(f"\n{'─' * W}")
    print(f"  Fin de la validación.")
    print(f"{'─' * W}\n")


if __name__ == "__main__":
    main()
