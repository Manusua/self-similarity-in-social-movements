#!/usr/bin/env python3
"""
Lee archivos GEXF y genera archivos de texto con aristas y pesos.
Por cada arista se escribe: nodo_origen, nodo_destino, weight.
"""

import networkx as nx
from pathlib import Path

# Directorio base donde están los GEXF y donde se guardarán los .txt
BASE_DIR = Path(__file__).resolve().parent / "graphs" / "nodes_hashtag_anonymized" / "ch" / "2"

INPUT_OUTPUT = [
    ("CH_CTW.gexf", "CH_CTW_edges_weights.txt"),
    ("CH_other_TW.gexf", "CH_other_TW_edges_weights.txt"),
]


def gexf_to_edges_weights(gexf_path: Path, out_path: Path, delimiter: str = "\t") -> None:
    """
    Lee un GEXF, extrae aristas con su peso y escribe origen, destino y weight en out_path.
    """
    G = nx.read_gexf(gexf_path)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source{delimiter}target{delimiter}weight\n")
        for u, v, data in G.edges(data=True):
            weight = data.get("weight", "")
            f.write(f"{u}{delimiter}{v}{delimiter}{weight}\n")


def main() -> None:
    for gexf_name, out_name in INPUT_OUTPUT:
        gexf_path = BASE_DIR / gexf_name
        out_path = BASE_DIR / out_name
        if not gexf_path.exists():
            print(f"Advertencia: no existe {gexf_path}")
            continue
        gexf_to_edges_weights(gexf_path, out_path)
        print(f"Generado: {out_path}")


if __name__ == "__main__":
    main()
