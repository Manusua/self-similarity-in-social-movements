"""Carga embeddings D-Mercator S¹ para greedy routing."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

Position = Tuple[float, float]
Adjacency = List[List[int]]

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from GRH_all import dhyp  # noqa: E402


def _detect_coord_format(header_lines: list[str]) -> str:
    joined = " ".join(header_lines).lower()
    if "inf.theta" in joined:
        return "s1_theta"
    if "inf.pos.1" in joined or "inf.pos.1" in joined.replace(" ", ""):
        return "sd_pos"
    return "unknown"


def load_dmercator_embedding(folder: Path, graph_id: str) -> tuple[Adjacency, Dict[int, Position], int, str]:
    """Devuelve adjacency, positions (hyp_rad, theta_deg), n_nodes, coord_format."""
    coord_path = folder / f"{graph_id}.inf_coord"
    edge_path = folder / f"{graph_id}.edge"
    if not coord_path.exists():
        raise FileNotFoundError(coord_path)
    if not edge_path.exists():
        raise FileNotFoundError(edge_path)

    header_lines: list[str] = []
    labels: List[str] = []
    positions_raw: Dict[str, Position] = {}
    coord_format = "unknown"

    with open(coord_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                header_lines.append(stripped)
                continue
            parts = stripped.split()
            if len(parts) < 4:
                continue
            if coord_format == "unknown":
                coord_format = _detect_coord_format(header_lines)
            label = parts[0]
            if coord_format == "s1_theta":
                theta_rad = float(parts[2])
                hyp_rad = float(parts[3])
            elif coord_format == "sd_pos":
                # Formato S^D con Pos.1 como ángulo en radianes (d=1 fallback)
                hyp_rad = float(parts[2])
                theta_rad = float(parts[3])
            else:
                raise ValueError(f"Formato de coordenadas no reconocido en {coord_path}")
            labels.append(label)
            positions_raw[label] = (hyp_rad, math.degrees(theta_rad))

    label_to_idx = {label: idx + 1 for idx, label in enumerate(labels)}
    max_node = len(labels)
    adjacency: Adjacency = [[] for _ in range(max_node + 1)]

    with open(edge_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            u, v = parts[0], parts[1]
            if u in label_to_idx and v in label_to_idx:
                i, j = label_to_idx[u], label_to_idx[v]
                adjacency[i].append(j)
                adjacency[j].append(i)

    positions = {label_to_idx[k]: v for k, v in positions_raw.items()}
    return adjacency, positions, max_node, coord_format


def sanity_check_positions(positions: Dict[int, Position]) -> None:
    """Comprueba que las coordenadas producen distancias finitas."""
    nodes = sorted(positions.keys())
    if len(nodes) < 2:
        return
    a, b = nodes[0], nodes[1]
    x1, y1 = positions[a]
    x2, y2 = positions[b]
    dist = dhyp(x1, y1, x2, y2)
    if not math.isfinite(dist):
        raise ValueError("Distancia hiperbólica no finita en sanity check")
