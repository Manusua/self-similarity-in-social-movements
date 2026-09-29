"""Controles de consistencia para embeddings y navegabilidad."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from extract_params import merge_params  # noqa: E402
from load_embedding import load_dmercator_embedding  # noqa: E402


def edge_hash(edge_path: Path) -> str:
    digest = hashlib.sha256()
    with open(edge_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def run_checks(spec, nav_row: dict, manifest_row: dict) -> list[dict]:
    checks: list[dict] = []
    graph_id = spec.graph_id
    emb_dir = spec.embedding_dir
    input_edge = spec.input_edge
    emb_edge = emb_dir / f"{graph_id}.edge"
    params = merge_params(emb_dir, graph_id)

    def add(check: str, expected, observed, passed: bool):
        checks.append(
            {
                "manifestacion": spec.manifestacion,
                "hour": spec.hour,
                "check": check,
                "expected": expected,
                "observed": observed,
                "pass": passed,
            }
        )

    add(
        "input_edge_exists",
        True,
        input_edge.exists(),
        input_edge.exists(),
    )
    add(
        "embedding_coord_exists",
        True,
        (emb_dir / f"{graph_id}.inf_coord").exists(),
        (emb_dir / f"{graph_id}.inf_coord").exists(),
    )
    add(
        "dmercator_log_complete",
        True,
        params.get("dmercator_complete"),
        bool(params.get("dmercator_complete")),
    )
    add(
        "coord_format_s1",
        "s1_theta",
        nav_row.get("coord_format"),
        nav_row.get("coord_format") == "s1_theta",
    )

    if emb_edge.exists() and input_edge.exists():
        same_hash = edge_hash(emb_edge) == edge_hash(input_edge)
        add("embedding_edge_matches_input", True, same_hash, same_hash)

    n_nodes = nav_row.get("n_nodes")
    expected_pairs = n_nodes * (n_nodes - 1) if n_nodes else None
    add(
        "n_pairs_all_ordered",
        expected_pairs,
        nav_row.get("n_pairs"),
        nav_row.get("n_pairs") == expected_pairs,
    )

    n_header = params.get("n_vertices_header")
    if n_header is not None and n_nodes is not None:
        add("n_nodes_matches_header", n_header, n_nodes, int(n_header) == int(n_nodes))

    avg_topo = nav_row.get("avg_topo_stretch")
    if avg_topo == avg_topo:  # not NaN
        add("avg_topo_stretch_ge_1", ">= 1", avg_topo, avg_topo >= 1.0)

    add(
        "manifest_nodes_match_loader",
        manifest_row.get("n_nodes"),
        n_nodes,
        manifest_row.get("n_nodes") == n_nodes,
    )

    return checks


def checks_to_dataframe(check_lists: list[list[dict]]) -> pd.DataFrame:
    flat = [item for sub in check_lists for item in sub]
    return pd.DataFrame(flat)
