"""Greedy routing / navigability utilities."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES_DIR, REPO_ROOT, RESULTS_DIR

sys.path.insert(0, str(REPO_ROOT))
from GRH_all import greedy_route, shortest_path_length  # noqa: E402

Position = Tuple[float, float]
Adjacency = List[List[int]]


def load_dmercator_embedding(folder: Path, graph_id: str):
    coord_path = folder / f"{graph_id}.inf_coord"
    edge_path = folder / f"{graph_id}.edge"
    labels: List[str] = []
    positions_raw: Dict[str, Position] = {}
    with open(coord_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            label = parts[0]
            theta_rad = float(parts[2])
            hyp_rad = float(parts[3])
            labels.append(label)
            positions_raw[label] = (hyp_rad, math.degrees(theta_rad))

    edges: List[Tuple[str, str]] = []
    with open(edge_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))

    label_to_idx = {label: idx + 1 for idx, label in enumerate(labels)}
    max_node = len(labels)
    adjacency: Adjacency = [[] for _ in range(max_node + 1)]
    for u, v in edges:
        if u in label_to_idx and v in label_to_idx:
            i, j = label_to_idx[u], label_to_idx[v]
            adjacency[i].append(j)
            adjacency[j].append(i)
    positions = {label_to_idx[k]: v for k, v in positions_raw.items()}
    return adjacency, positions, max_node


def compute_navigability(
    adjacency: Adjacency,
    positions: Dict[int, Position],
    *,
    diam: int = 1000,
    pair_step: int = 1,
) -> dict:
    n_nodes = len(adjacency) - 1
    failures = 0
    total_pairs = 0
    stretch_sum = 0.0
    distance_sum = 0.0
    stretch_sq_sum = 0.0
    distance_sq_sum = 0.0

    for source in range(1, n_nodes + 1):
        for target in range(1, n_nodes + 1, pair_step):
            if source == target:
                continue
            success, greedy_hops, greedy_distance_sum = greedy_route(
                source, target, adjacency, positions, diam=diam
            )
            if not success:
                failures += 1
            else:
                shortest = shortest_path_length(source, target, adjacency, diam=diam)
                if shortest and shortest > 0:
                    stretch_sum += greedy_hops / shortest
                    stretch_sq_sum += (greedy_hops / shortest) ** 2
                distance_sum += greedy_distance_sum
                distance_sq_sum += greedy_distance_sum**2
            total_pairs += 1

    successes = total_pairs - failures
    success_ratio = successes / total_pairs if total_pairs else float("nan")
    if successes > 0:
        avg_topo_stretch = stretch_sum / successes
        avg_stretch = distance_sum / successes
        std_topo_stretch = math.sqrt(max(stretch_sq_sum / successes - avg_topo_stretch**2, 0.0))
        std_stretch = math.sqrt(max(distance_sq_sum / successes - avg_stretch**2, 0.0))
    else:
        avg_topo_stretch = avg_stretch = std_topo_stretch = std_stretch = float("nan")

    return {
        "n_nodes": n_nodes,
        "n_pairs": total_pairs,
        "n_successes": successes,
        "n_failures": failures,
        "success_ratio": success_ratio,
        "avg_stretch": avg_stretch,
        "std_stretch": std_stretch,
        "avg_topo_stretch": avg_topo_stretch,
        "std_topo_stretch": std_topo_stretch,
    }


def evaluate_embedding(
    folder: Path,
    graph_id: str,
    *,
    label: str,
    window_type: str,
) -> dict:
    adjacency, positions, _ = load_dmercator_embedding(folder, graph_id)
    stats = compute_navigability(adjacency, positions, pair_step=1)
    stats.update({"label": label, "graph_id": graph_id, "folder": str(folder), "window_type": window_type})
    return stats


def plot_navigability_comparison(df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ct = df[df["window_type"].isin(["CTW", "CTW_sensitivity"])].copy()
    non = df[df["window_type"] == "non-CTW"].copy()
    ct = ct.drop_duplicates(subset=["manifestacion"], keep="first")
    non = non.drop_duplicates(subset=["manifestacion"], keep="first")
    labels = sorted(set(ct["manifestacion"]).union(set(non["manifestacion"])))
    ct_map = {row["manifestacion"]: row["success_ratio"] for _, row in ct.iterrows()}
    non_map = {row["manifestacion"]: row["success_ratio"] for _, row in non.iterrows()}
    ct_topo = {row["manifestacion"]: row["avg_topo_stretch"] for _, row in ct.iterrows()}
    non_topo = {row["manifestacion"]: row["avg_topo_stretch"] for _, row in non.iterrows()}
    x = np.arange(len(labels))
    axes[0].bar(x - 0.15, [ct_map.get(m, np.nan) for m in labels], width=0.3, label="CTW")
    axes[0].bar(x + 0.15, [non_map.get(m, np.nan) for m in labels], width=0.3, label="non-CTW")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.upper() for m in labels])
    axes[0].set_ylabel("Success ratio")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].bar(x - 0.15, [ct_topo.get(m, np.nan) for m in labels], width=0.3, label="CTW")
    axes[1].bar(x + 0.15, [non_topo.get(m, np.nan) for m in labels], width=0.3, label="non-CTW")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in labels])
    axes[1].set_ylabel("Topological stretch")
    axes[1].legend()

    fig.suptitle("Navigability — CTW vs non-CTW")
    fig.tight_layout()
    out = FIGURES_DIR / "navigability_ctw_vs_nonctw.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_embedding_metadata(folder: Path, graph_id: str) -> dict:
    coord = folder / f"{graph_id}.inf_coord"
    meta = {"graph_id": graph_id, "folder": str(folder)}
    if not coord.exists():
        return meta
    with open(coord, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") and ":" in line:
                key, val = line.replace("#", "").split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                meta[key] = val.strip()
    return meta
