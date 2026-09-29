"""Copia edgelists canónicos y genera manifest.csv."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OTHER_HOURS, RESULTS_DIR, WORK_ROOT  # noqa: E402


def edge_stats(edge_path: Path) -> tuple[int, int, str]:
    nodes: set[str] = set()
    n_edges = 0
    digest = hashlib.sha256()
    with open(edge_path, "rb") as handle:
        for raw in handle:
            digest.update(raw)
            line = raw.decode("utf-8").strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                nodes.add(parts[0])
                nodes.add(parts[1])
                n_edges += 1
    return len(nodes), n_edges, digest.hexdigest()[:16]


def prepare_all() -> pd.DataFrame:
    rows = []
    for spec in OTHER_HOURS:
        if not spec.source_edge.exists():
            raise FileNotFoundError(f"Edgelist no encontrado: {spec.source_edge}")

        spec.input_dir.mkdir(parents=True, exist_ok=True)
        if spec.input_edge.exists() and spec.input_edge.samefile(spec.source_edge):
            pass
        else:
            shutil.copy2(spec.source_edge, spec.input_edge)

        n_nodes, n_edges, edge_hash = edge_stats(spec.input_edge)
        rows.append(
            {
                "manifestacion": spec.manifestacion,
                "hour": spec.hour,
                "hour_window": spec.hour_window,
                "threshold": spec.threshold,
                "ctw_offset_h": spec.ctw_offset_h,
                "plot_label": spec.plot_label,
                "source_edge": str(spec.source_edge),
                "input_edge": str(spec.input_edge),
                "embedding_dir": str(spec.embedding_dir),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "edge_sha256_prefix": edge_hash,
            }
        )

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "manifest.csv", index=False)
    return df


if __name__ == "__main__":
    manifest = prepare_all()
    print(manifest.to_string(index=False))
    print(f"\nManifest guardado en {RESULTS_DIR / 'manifest.csv'}")
