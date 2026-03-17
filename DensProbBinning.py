#!/usr/bin/env python3
"""
Calcula la distribución de probabilidad a partir de un vector columna de eventos enteros.
Traducción a Python del programa Fortran `DensProbBinning.for`.

Comportamiento original:
- Para valores `xi <= nobinning`, usa binning exacto.
- Para valores `xi > nobinning`, aplica binning exponencial en `ncajas` cajas.
- Escribe tres columnas en el archivo de salida:
    1) valor medio del bin
    2) probabilidad (densidad normalizada)
    3) distribución acumulada complementaria

Ejemplo de uso:
    python DensProbBinning.py \
        --input EMP/globalEMP.dat \
        --output 'P(gloc)nobinning_Wiki.dat' \
        --nobinning 2399632 \
        --ncajas 0 \
        --xf 2600000
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable


# Parámetros originales del código Fortran
DEFAULT_INPUT = "EMP/globalEMP.dat"
DEFAULT_OUTPUT = "P(gloc)nobinning_Wiki.dat"
DEFAULT_NOBINNING = 2_399_632
DEFAULT_NCAJAS = 0
DEFAULT_XF = 2_600_000.0


def _prefix_sum(diff: list[int | float], size: int) -> list[float]:
    """Reconstruye el vector acumulado a partir de un arreglo de diferencias."""
    out = [0.0] * size
    running = 0.0
    for i in range(size):
        running += diff[i]
        out[i] = running
    return out



def dens_prob_binning(
    input_file: str | Path = DEFAULT_INPUT,
    output_file: str | Path = DEFAULT_OUTPUT,
    nobinning: int = DEFAULT_NOBINNING,
    ncajas: int = DEFAULT_NCAJAS,
    xf: float = DEFAULT_XF,
) -> tuple[int, int, float]:
    """
    Procesa el archivo de entrada y escribe el archivo de salida.

    Devuelve:
        (nodes, ndegreemax, probabilidad_total)
    """
    input_file = Path(input_file)
    output_file = Path(output_file)

    # Traducción fiel de las variables del Fortran.
    x0 = float(nobinning) + 1.0
    xq = None
    if ncajas != 0:
        xq = (xf / x0) ** (1.0 / float(ncajas))

    # Conteos exactos para 0..nobinning
    nfreqnb = [0] * (nobinning + 1)

    # Arreglo de diferencias para reconstruir nfreqnbcum sin O(n^2)
    # nfreqnbcum(j) = número de nodos con xi >= j
    nfreqnbcum_diff = [0] * (nobinning + 2)

    # Información de cajas exponenciales (1..ncajas)
    xfreq_sum = [0.0] * (ncajas + 1)
    xfreq_count = [0.0] * (ncajas + 1)
    xfreqcum_diff = [0.0] * (ncajas + 2)

    ndegreemax = 0
    nodes = 0

    with input_file.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) < 2:
                raise ValueError(
                    f"Línea {line_number}: se esperaban al menos 2 columnas y se obtuvo: {stripped!r}"
                )

            # El Fortran lee: j, xi
            try:
                _j = float(parts[0])  # se conserva solo por fidelidad al formato de entrada
                xi = float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"Línea {line_number}: no se pudieron convertir los valores numéricos: {stripped!r}"
                ) from exc

            nodes += 1
            i = int(xi)
            if i > ndegreemax:
                ndegreemax = i

            if i <= nobinning:
                if i < 0:
                    raise ValueError(
                        f"Línea {line_number}: el valor xi={xi} produce un índice negativo ({i}), no soportado por el programa original."
                    )
                nfreqnb[i] += 1
                nfreqnbcum_diff[0] += 1
                nfreqnbcum_diff[i + 1] -= 1
            else:
                if ncajas == 0 or xq is None:
                    raise ValueError(
                        "Se encontró un valor mayor que 'nobinning' pero 'ncajas' es 0. "
                        "Esto deja el binning exponencial desactivado, igual que en el Fortran original. "
                        f"Valor conflictivo en la línea {line_number}: xi={xi}, nobinning={nobinning}."
                    )

                ncajai = int(math.log(xi / x0) / math.log(xq) + 1.0)
                if not (1 <= ncajai <= ncajas):
                    raise ValueError(
                        f"Línea {line_number}: la caja calculada ({ncajai}) está fuera del rango 1..{ncajas}. "
                        f"Revisa 'xf', 'ncajas' o los datos de entrada."
                    )

                xfreq_sum[ncajai] += xi
                xfreq_count[ncajai] += 1.0

                # En el Fortran: para todos j=0..nobinning, nfreqnbcum(j) += 1
                nfreqnbcum_diff[0] += 1
                nfreqnbcum_diff[nobinning + 1] -= 1

                # En el Fortran: para todos j=1..ncajai, xfreq(j,3) += 1
                xfreqcum_diff[1] += 1.0
                xfreqcum_diff[ncajai + 1] -= 1.0

    if nodes == 0:
        raise ValueError("El archivo de entrada no contiene datos válidos.")

    nfreqnbcum = _prefix_sum(nfreqnbcum_diff, nobinning + 1)
    xfreq_cum = _prefix_sum(xfreqcum_diff, ncajas + 1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    xsum = 0.0
    with output_file.open("w", encoding="utf-8") as out:
        for i in range(nobinning + 1):
            if nfreqnb[i] != 0:
                prob = float(nfreqnb[i]) / float(nodes)
                cum = float(nfreqnbcum[i]) / float(nodes)
                out.write(f"{float(i):.15g} {prob:.15g} {cum:.15g}\n")
                xsum += prob

        for i in range(1, ncajas + 1):
            l = int(x0 * (xq ** i)) - int(x0 * (xq ** (i - 1)))
            if xfreq_count[i] != 0:
                out.write(
                    f"{(xfreq_sum[i] / xfreq_count[i]):.15g} "
                    f"{(xfreq_count[i] / float(nodes) / float(l)):.15g} "
                    f"{(xfreq_cum[i] / float(nodes)):.15g}\n"
                )
                xsum += xfreq_count[i] / float(nodes)

    print(f"nodos {nodes}")
    print(f"ndegreemax {ndegreemax}")
    print(f"Probabilidad total= {xsum}")

    return nodes, ndegreemax, xsum



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calcula una distribución de probabilidad con binning exacto y exponencial."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Archivo de entrada (dos columnas: j xi)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Archivo de salida")
    parser.add_argument("--nobinning", type=int, default=DEFAULT_NOBINNING, help="Límite para binning exacto")
    parser.add_argument("--ncajas", type=int, default=DEFAULT_NCAJAS, help="Número de cajas exponenciales")
    parser.add_argument("--xf", type=float, default=DEFAULT_XF, help="Tamaño máximo usado para el binning exponencial")
    return parser



def main() -> None:
    args = build_parser().parse_args()
    dens_prob_binning(
        input_file=args.input,
        output_file=args.output,
        nobinning=args.nobinning,
        ncajas=args.ncajas,
        xf=args.xf,
    )


if __name__ == "__main__":
    main()
