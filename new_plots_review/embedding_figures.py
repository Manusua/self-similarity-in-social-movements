"""Generate SI-style embedding validation figures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import FIGURES_DIR, REPO_ROOT, RESULTS_DIR


def plot_pconn(edge_path: Path, coord_path: Path, title: str, out_path: Path) -> None:
    pconn = edge_path.with_suffix(".inf_pconn")
    if not pconn.exists():
        return
    data = pd.read_csv(pconn, sep=r"\s+", comment="#", header=None)
    if data.shape[1] < 3:
        return
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.loglog(data.iloc[:, 0], data.iloc[:, 2], "--", color="#555555", label="inferred")
    ax.loglog(data.iloc[:, 0], data.iloc[:, 1], "o", color="black", markersize=3, label="original")
    ax.set_xlabel(r"Rescaled distance $\chi$")
    ax.set_ylabel("Connection probability")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_all() -> None:
    inventory = pd.read_csv(RESULTS_DIR / "embedding_inventory.csv")
    for _, row in inventory.iterrows():
        folder = Path(row["embedding_dir"])
        gid = str(row["graph_id"])
        edge = folder / f"{gid}.edge"
        coord = folder / f"{gid}.inf_coord"
        if not edge.exists() or not coord.exists():
            continue
        title = f"{row['manifestacion'].upper()} {row['window_type']} ({gid})"
        out = FIGURES_DIR / f"embedding_pconn_{row['manifestacion']}_{row['window_type']}_{gid}.png"
        plot_pconn(edge, coord, title, out)

    dmercator_py = REPO_ROOT / "d-mercator" / "d-mercator.py"
    if dmercator_py.exists():
        for _, row in inventory.iterrows():
            if row["window_type"] not in {"CTW", "non-CTW"}:
                continue
            folder = Path(row["embedding_dir"])
            gid = str(row["graph_id"])
            edge = folder / f"{gid}.edge"
            if edge.exists():
                subprocess.run(
                    [sys.executable, str(dmercator_py), str(edge)],
                    check=False,
                    cwd=str(REPO_ROOT / "d-mercator"),
                )


if __name__ == "__main__":
    generate_all()
