"""Non-CTW selection and D-Mercator edge preparation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import pandas as pd

from config import (
    CH_PRIMARY_HOUR,
    CH_SENSITIVITY_HOUR,
    DMERCATOR_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_THRESHOLD,
    EXISTING_CTW_EMBEDDINGS,
    HOUR_WINDOW,
    HORA_CRITICA,
    MANIFESTATIONS,
    NONCTW_ACTIVITY_TOL,
    NONCTW_ACTIVITY_TOL_FALLBACK,
    NONCTW_EXCLUSION_HOURS,
    REPO_ROOT,
    RESULTS_DIR,
    VENDOR_DIR,
)
from temporal_plots import load_epsilon_series, load_activity_series


def select_nonctw_hour(manifestation: str) -> dict:
    cfg_hour = HORA_CRITICA[manifestation]
    hw = HOUR_WINDOW[manifestation]
    eps = load_epsilon_series(manifestation)
    activity = load_activity_series(manifestation)

    ct_hashtags = activity.loc[activity["hour"] == cfg_hour, "hashtags"].iloc[0]
    candidates = eps[
        (eps["hour"] < cfg_hour - NONCTW_EXCLUSION_HOURS)
        | (eps["hour"] > cfg_hour + NONCTW_EXCLUSION_HOURS)
    ].copy()
    candidates = candidates.merge(activity[["hour", "hashtags"]], on="hour", how="left")
    candidates["activity_ratio"] = candidates["hashtags"] / ct_hashtags

    def pick(tolerance: float) -> Optional[pd.Series]:
        pool = candidates[
            candidates["activity_ratio"].between(1 - tolerance, 1 + tolerance)
        ]
        if pool.empty:
            return None
        return pool.sort_values("epsilon_cco", ascending=False).iloc[0]

    chosen = pick(NONCTW_ACTIVITY_TOL)
    tol_used = NONCTW_ACTIVITY_TOL
    if chosen is None:
        chosen = pick(NONCTW_ACTIVITY_TOL_FALLBACK)
        tol_used = NONCTW_ACTIVITY_TOL_FALLBACK
    if chosen is None:
        chosen = candidates.sort_values("epsilon_cco", ascending=False).iloc[0]
        tol_used = float("nan")

    record = {
        "manifestacion": manifestation,
        "ctw_hour": cfg_hour,
        "nonctw_hour": int(chosen["hour"]),
        "hour_window": hw,
        "ctw_hashtags": ct_hashtags,
        "nonctw_hashtags": float(chosen["hashtags"]),
        "activity_ratio": float(chosen["activity_ratio"]),
        "activity_tolerance_used": tol_used,
        "epsilon_cco": float(chosen["epsilon_cco"]),
        "selection_rule": "max epsilon_cco outside ±12h with matched activity",
    }
    return record


def select_all_nonctw() -> pd.DataFrame:
    rows = [select_nonctw_hour(m) for m in MANIFESTATIONS]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "nonctw_selection.csv", index=False)
    return df


def prepare_gcc_edgelist(manifestation: str, hour: int, out_dir: Path, threshold: int = EMBEDDING_THRESHOLD) -> Path:
    hw = HOUR_WINDOW[manifestation]
    gexf = REPO_ROOT / "graphs" / "nodes_hashtag" / manifestation / str(hw) / f"{hour}.gexf"
    graph = nx.read_gexf(gexf)
    # Filter edges by weight threshold (same convention as d-mercator subfolders)
    to_remove = []
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 1))
        if weight < threshold:
            to_remove.append((u, v))
    graph.remove_edges_from(to_remove)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    largest = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest).copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    edge_path = out_dir / f"{hour}.edge"
    with open(edge_path, "w", encoding="utf-8") as handle:
        for u, v in graph.edges():
            handle.write(f"{u}\t{v}\n")
    return edge_path


def clone_dmercator_repo() -> Path:
    target = VENDOR_DIR / "d-mercator"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/networkgeometry/d-mercator.git", str(target)],
        check=True,
    )
    return target


def run_dmercator(edge_path: Path, dimension: int = 1, validation: bool = True) -> Path:
    repo = clone_dmercator_repo()
    output_root = edge_path.with_suffix("")

    mercator_bin = repo / "mercator"
    if not mercator_bin.exists():
        build_script = repo / "build.sh"
        if build_script.exists():
            subprocess.run(["bash", str(build_script), "-b", "Release"], check=True, cwd=str(repo))
    if mercator_bin.exists():
        local_cmd = [str(mercator_bin), "-d", str(dimension)]
        if validation:
            local_cmd.append("-v")
        local_cmd.append(str(edge_path))
        subprocess.run(local_cmd, check=True, cwd=str(edge_path.parent))
        return output_root

    docker_script = repo / "run_dmercator_docker.py"
    cmd = [sys.executable, str(docker_script), "-i", str(edge_path), "-d", str(dimension)]
    if validation:
        cmd.extend(["-v", "1"])
    subprocess.run(cmd, check=True, cwd=str(repo))
    return output_root


def copy_embedding_outputs(src_dir: Path, graph_id: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (
        ".edge",
        ".inf_coord",
        ".inf_log",
        ".inf_pconn",
        ".inf_theta_density",
        ".inf_vprop",
        ".inf_vstat",
        ".obs_vstat",
    ):
        src = src_dir / f"{graph_id}{suffix}"
        dst = dest_dir / src.name
        if src.exists() and src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    return dest_dir


def embedding_target(manifestation: str, hour: int, window_type: str) -> Path:
    return EMBEDDINGS_DIR / manifestation / window_type / str(hour) / str(EMBEDDING_THRESHOLD)


def ensure_embeddings(nonctw_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Reuse existing CTW for nat/9n
    for manifestation in ("nat", "9n"):
        folder, graph_id = EXISTING_CTW_EMBEDDINGS[manifestation]
        rows.append(
            {
                "manifestacion": manifestation,
                "hour": int(graph_id),
                "window_type": "CTW",
                "embedding_dir": str(folder),
                "graph_id": graph_id,
                "source": "reused_existing",
            }
        )

    # CH primary + sensitivity
    tasks = [
        ("ch", CH_PRIMARY_HOUR, "CTW", "compute"),
        ("ch", CH_SENSITIVITY_HOUR, "CTW_sensitivity", "reused_existing"),
    ]
    for manifestation, hour, wtype, mode in tasks:
        if mode == "reused_existing":
            folder, graph_id = EXISTING_CTW_EMBEDDINGS["ch_sensitivity"]
            rows.append(
                {
                    "manifestacion": manifestation,
                    "hour": hour,
                    "window_type": wtype,
                    "embedding_dir": str(folder),
                    "graph_id": graph_id,
                    "source": "reused_existing",
                }
            )
            continue
        dest = embedding_target(manifestation, hour, "CTW")
        coord = dest / f"{hour}.inf_coord"
        if coord.exists():
            rows.append(
                {
                    "manifestacion": manifestation,
                    "hour": hour,
                    "window_type": wtype,
                    "embedding_dir": str(dest),
                    "graph_id": str(hour),
                    "source": "cached",
                }
            )
            continue
        edge = prepare_gcc_edgelist(manifestation, hour, dest)
        run_dmercator(edge, dimension=1, validation=True)
        rows.append(
            {
                "manifestacion": manifestation,
                "hour": hour,
                "window_type": wtype,
                "embedding_dir": str(dest),
                "graph_id": str(hour),
                "source": "computed",
            }
        )

    for _, rec in nonctw_df.iterrows():
        manifestation = rec["manifestacion"]
        hour = int(rec["nonctw_hour"])
        dest = embedding_target(manifestation, hour, "non-CTW")
        coord = dest / f"{hour}.inf_coord"
        if coord.exists():
            rows.append(
                {
                    "manifestacion": manifestation,
                    "hour": hour,
                    "window_type": "non-CTW",
                    "embedding_dir": str(dest),
                    "graph_id": str(hour),
                    "source": "cached",
                }
            )
            continue
        edge = prepare_gcc_edgelist(manifestation, hour, dest)
        try:
            run_dmercator(edge, dimension=1, validation=True)
            source = "computed" if (dest / f"{hour}.inf_coord").exists() else "failed:no_inf_coord"
        except Exception as exc:  # noqa: BLE001
            source = f"failed:{exc}"
        rows.append(
            {
                "manifestacion": manifestation,
                "hour": hour,
                "window_type": "non-CTW",
                "embedding_dir": str(dest),
                "graph_id": str(hour),
                "source": source,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "embedding_inventory.csv", index=False)
    return out
