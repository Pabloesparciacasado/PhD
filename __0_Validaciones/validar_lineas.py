"""
Validación 2 — Número de líneas por archivo TXT.

Comprueba para cada archivo TXT:
  ✓ OK    — líneas dentro del rango normal (± 50 % de la mediana)
  ⚠ POCO  — menos del 50 % de la mediana del producto (archivo corto)
  ✗ VACÍO — 0 líneas de datos (solo cabecera o completamente vacío)

Configuración:
  CARPETA  — raíz de los datos (contiene las subcarpetas GI.ALL.*)
  PRODUCTO — si se especifica (ej. "GI.ALL.IVYOPPRCD"), analiza solo ese producto.
             Si se deja en None, analiza todos los productos.
  UMBRAL   — fracción de la mediana por debajo de la cual se emite aviso (defecto 0.5)
"""

import os
from pathlib import Path


# ─── Configuración ────────────────────────────────────────────────────────────

CARPETA  = r"C:\Users\pablo.esparcia\Documents\OptionMetrics\Acumulado"
PRODUCTO = None      # ej. "GI.ALL.IVYOPPRCD"  o  None para todos
UMBRAL   = 0.5       # aviso si líneas < UMBRAL × mediana


# ─── Utilidades ───────────────────────────────────────────────────────────────

def _contar_lineas(ruta: str) -> int:
    """Cuenta líneas leyendo el fichero en binario (rápido, sin decodificar)."""
    with open(ruta, 'rb') as f:
        return f.read().count(b'\n')


def _mediana(valores: list[int]) -> float:
    if not valores:
        return 0.0
    s = sorted(valores)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)


def _analizar_producto(ruta_prod: Path) -> list[dict]:
    """Devuelve una lista de dicts {nombre, lineas} para cada TXT."""
    txts = sorted(
        f for f in os.listdir(ruta_prod)
        if f.lower().endswith('.txt') and os.path.isfile(ruta_prod / f)
    )
    return [
        {'nombre': f, 'lineas': _contar_lineas(str(ruta_prod / f))}
        for f in txts
    ]


# ─── Main ─────────────────────────────────────────────────────────────────────

def _imprimir_producto(nombre_prod: str, archivos: list[dict]) -> tuple[int, int]:
    """Imprime resumen del producto. Devuelve (n_avisos, n_errores)."""
    W       = 72
    lineas  = [a['lineas'] for a in archivos]
    med     = _mediana(lineas)
    umbral  = med * UMBRAL

    avisos = errores = 0
    filas_detalle: list[tuple[str, int, str]] = []

    for a in archivos:
        n = a['lineas']
        if n == 0:
            estado = '✗ VACÍO'
            errores += 1
        elif n < umbral:
            estado = f'⚠ POCO  ({n / med:.0%} de la mediana)'
            avisos += 1
        else:
            estado = '✓ OK'
        filas_detalle.append((a['nombre'], n, estado))

    total_nok = avisos + errores
    if total_nok == 0:
        # Resumen de una línea cuando todo está OK
        print(f"  {nombre_prod:<30}  {len(archivos):>4} archivos  "
              f"mediana {med:>10,.0f} líneas  ✓ todos OK")
        return 0, 0

    # Detalle si hay problemas
    print(f"\n  ── {nombre_prod}  ({len(archivos)} archivos, mediana {med:,.0f} líneas) ──")
    print(f"  {'Archivo':<42}  {'Líneas':>10}  Estado")
    print(f"  {'─' * (W - 2)}")
    for nombre, n, estado in filas_detalle:
        if estado != '✓ OK':
            print(f"  {nombre:<42}  {n:>10,}  {estado}")
    print(f"  {'─' * (W - 2)}")
    print(f"  Avisos: {avisos}   Errores: {errores}")

    return avisos, errores


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
    ambito = PRODUCTO if PRODUCTO else CARPETA
    print(f"  VALIDACIÓN 2 — LÍNEAS     {ambito}")
    print(f"  Umbral aviso: < {UMBRAL:.0%} de la mediana del producto")
    print(f"{'─' * W}")

    total_warn = total_err = 0
    for prod in productos:
        if not prod.exists():
            print(f"  ⚠ No encontrado: {prod.name}")
            continue
        archivos = _analizar_producto(prod)
        if not archivos:
            print(f"  {prod.name:<30}  (sin archivos TXT)")
            continue
        w, e = _imprimir_producto(prod.name, archivos)
        total_warn += w
        total_err  += e

    print(f"\n{'─' * W}")
    print(f"  Avisos  (⚠) : {total_warn}")
    print(f"  Errores (✗) : {total_err}")
    print(f"{'─' * W}\n")


if __name__ == "__main__":
    main()
