"""Greedy routing: p_s y stretch topológico (todos los pares ordenados)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from GRH_all import greedy_route, shortest_path_length  # noqa: E402

from load_embedding import Adjacency, Position, load_dmercator_embedding, sanity_check_positions  # noqa: E402


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
    distance_stretch_sum = 0.0
    stretch_sq_sum = 0.0
    distance_stretch_sq_sum = 0.0
    identical_positions = 0

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
                if shortest is None or shortest <= 0:
                    failures += 1
                else:
                    topo = greedy_hops / shortest
                    stretch_sum += topo
                    stretch_sq_sum += topo**2

                    src_pos = positions.get(source, (0.0, 0.0))
                    tgt_pos = positions.get(target, (0.0, 0.0))
                    if src_pos[0] == tgt_pos[0] and src_pos[1] == tgt_pos[1]:
                        identical_positions += 1
                    else:
                        from GRH_all import dhyp  # noqa: WPS433

                        direct = dhyp(src_pos[0], src_pos[1], tgt_pos[0], tgt_pos[1])
                        if direct > 0:
                            dist_stretch = greedy_distance_sum / direct
                            distance_stretch_sum += dist_stretch
                            distance_stretch_sq_sum += dist_stretch**2
            total_pairs += 1

    successes = total_pairs - failures
    p_s = successes / total_pairs if total_pairs else float("nan")
    if successes > 0:
        avg_topo_stretch = stretch_sum / successes
        std_topo_stretch = math.sqrt(max(stretch_sq_sum / successes - avg_topo_stretch**2, 0.0))
        avg_stretch = distance_stretch_sum / successes
        std_stretch = math.sqrt(max(distance_stretch_sq_sum / successes - avg_stretch**2, 0.0))
    else:
        avg_topo_stretch = std_topo_stretch = avg_stretch = std_stretch = float("nan")

    return {
        "n_nodes": n_nodes,
        "n_pairs": total_pairs,
        "n_successes": successes,
        "n_failures": failures,
        "p_s": p_s,
        "avg_topo_stretch": avg_topo_stretch,
        "std_topo_stretch": std_topo_stretch,
        "avg_stretch": avg_stretch,
        "std_stretch": std_stretch,
        "identical_positions": identical_positions,
        "pair_step": pair_step,
    }


def evaluate_embedding_folder(folder: Path, graph_id: str, *, pair_step: int = 1, diam: int = 1000) -> dict:
    adjacency, positions, n_nodes, coord_format = load_dmercator_embedding(folder, graph_id)
    sanity_check_positions(positions)
    stats = compute_navigability(adjacency, positions, diam=diam, pair_step=pair_step)
    stats["coord_format"] = coord_format
    stats["folder"] = str(folder)
    stats["graph_id"] = graph_id
    return stats
