#!/usr/bin/env python3
"""Fill missing epsilon values incrementally for CTW±96 windows.

Skips hours that already have valid epsilon in Epsilon_values.txt.
Creates missing gexf/edge/kts prerequisites, computes epsilon, and merges results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from new_plots_review.epsilon_metrics import compute_epsilon_for_network


def gexf_a_edge(gexf_path: Path, out_path: Path, delimiter: str = "\t") -> Path:
    graph = nx.read_gexf(gexf_path)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        if graph.is_multigraph():
            for u, v, _k in graph.edges(keys=True):
                f.write(f"{u}{delimiter}{v}\n")
        else:
            for u, v in graph.edges():
                f.write(f"{u}{delimiter}{v}\n")
    return out_path


def _calc_avg_degree(graph: nx.Graph) -> float:
    return sum(dict(graph.degree).values()) / graph.number_of_nodes()


def _add_nodes_subgraph(graph: nx.Graph, threshold: float) -> nx.Graph:
    sub = nx.Graph()
    for node in graph.nodes():
        if graph.degree[node] > threshold:
            sub.add_node(node)
    return sub


def _add_edges_subgraph(graph: nx.Graph, sub: nx.Graph) -> nx.Graph:
    for node in sub.nodes():
        for neighbor in graph.neighbors(node):
            if neighbor in sub.nodes() and neighbor not in sub.neighbors(node):
                sub.add_edge(node, neighbor)
    return sub


def _add_hidden_variable(sub: nx.Graph) -> int | None:
    avg_deg = _calc_avg_degree(sub)
    if avg_deg == 0:
        return -1
    nx.set_node_attributes(
        sub,
        {node: sub.degree[node] / avg_deg for node in sub.nodes()},
        "internalDegree",
    )
    return None


def _thresh_normalization(graph: nx.Graph, threshold: float):
    sub = _add_nodes_subgraph(graph, threshold)
    if sub.number_of_nodes() == 0:
        return -1
    sub = _add_edges_subgraph(graph, sub)
    if _add_hidden_variable(sub) == -1:
        return -1
    return sub


def get_dict_hora_kt(
    manifestacion: str,
    hour: str | int,
    hour_window: int,
    graphs_folder: Path,
    max_umbral: int = 200,
) -> dict[float, float]:
    gexf_path = (
        graphs_folder
        / "nodes_hashtag"
        / manifestacion
        / str(hour_window)
        / f"{hour}.gexf"
    )
    graph = nx.read_gexf(gexf_path)
    result: dict[float, float] = {}
    for threshold in range(max_umbral):
        t = float(threshold)
        sub = _thresh_normalization(graph, t)
        if sub == -1:
            break
        result[t] = _calc_avg_degree(sub)
    return result

DATASET_CONFIG = {
    "ch": {"ctw": 394717, "hour_window": 2},
    "9n": {"ctw": 437037, "hour_window": 2},
    "nat": {"ctw": 429624, "hour_window": 1},
}

EPSILON_VALUES_HEADER = (
    "Network \t $\\epsilon_{\\ccdf}$  \t  $\\epsilon_{\\knn}$ \t $\\epsilon_{\\cco}$ \n"
)
EPSILON_MAX_HEADER = (
    "Network \t clustering\\_coeff \t $\\epsilon_{\\max}$ \n"
)


def load_existing_epsilon(results_dir: Path) -> tuple[dict[int, dict], dict[int, dict]]:
    """Load valid epsilon rows keyed by network hour."""
    values_path = results_dir / "Epsilon_values.txt"
    max_path = results_dir / "Epsilon_max_vs_clustering.txt"

    values: dict[int, dict] = {}
    max_rows: dict[int, dict] = {}

    if values_path.exists():
        df = pd.read_csv(values_path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        net_col = df.columns[0]
        metric_cols = df.columns[1:4]
        for _, row in df.iterrows():
            net = int(row[net_col])
            eps = [pd.to_numeric(row[c], errors="coerce") for c in metric_cols]
            if all(np.isfinite(e) for e in eps):
                values[net] = {
                    "epsilon_ccdf": float(eps[0]),
                    "epsilon_knn": float(eps[1]),
                    "epsilon_cco": float(eps[2]),
                }

    if max_path.exists():
        df = pd.read_csv(max_path, sep="\t")
        df.columns = [c.strip() for c in df.columns]
        net_col = df.columns[0]
        for _, row in df.iterrows():
            net = int(row[net_col])
            clust = pd.to_numeric(row[df.columns[1]], errors="coerce")
            eps_max = pd.to_numeric(row[df.columns[2]], errors="coerce")
            if np.isfinite(clust) and np.isfinite(eps_max):
                max_rows[net] = {
                    "avg_clustering": float(clust),
                    "epsilon_max": float(eps_max),
                }

    return values, max_rows


def load_kts(epsilon_dir: Path) -> dict[int, float]:
    kts_path = epsilon_dir / "kts.csv"
    if not kts_path.exists():
        return {}
    df = pd.read_csv(kts_path)
    return {
        int(row["hora"]): float(row["clave_max"])
        for _, row in df.iterrows()
        if pd.notna(row.get("clave_max"))
    }


def save_kts(epsilon_dir: Path, kts: dict[int, float]) -> None:
    rows = [{"hora": h, "clave_max": v} for h, v in sorted(kts.items())]
    pd.DataFrame(rows).to_csv(epsilon_dir / "kts.csv", index=False)


def create_single_gexf(
    df: pd.DataFrame,
    hour: int,
    manifestacion: str,
    hour_window: int,
    graphs_folder: Path,
) -> Path | None:
    """Create one hashtag graph gexf for [hour, hour+hour_window)."""
    gexf_dir = graphs_folder / "nodes_hashtag" / manifestacion / str(hour_window)
    gexf_dir.mkdir(parents=True, exist_ok=True)
    gexf_path = gexf_dir / f"{hour}.gexf"

    conditions = df["hour"] == hour
    for step in range(1, hour_window):
        conditions |= df["hour"] == hour + step
    df_hour = df[conditions]
    if df_hour.empty:
        return None

    g = nx.Graph()
    df_nodes = df_hour["hashtag"].unique()
    g.add_nodes_from(df_nodes)
    for node in df_nodes:
        df_node_edge = df_hour.loc[df_hour["hashtag"] == node, "user"]
        df_node_edge = df_node_edge.drop_duplicates()
        for edge in df_node_edge:
            df_edge = df_hour.loc[df_hour["user"] == edge, "hashtag"].drop_duplicates()
            for nd in df_edge:
                if nd != node:
                    if g.has_edge(node, nd):
                        g[node][nd]["weight"] += 1
                    else:
                        g.add_edge(node, nd, weight=1)

    for edge in g.edges():
        old_weight = g.edges[edge]["weight"]
        nx.set_edge_attributes(g, {edge: {"weight": old_weight / 2}})

    nx.write_gexf(g, gexf_path)
    return gexf_path


def ensure_prerequisites(
    hour: int,
    manifestacion: str,
    hour_window: int,
    df: pd.DataFrame,
    epsilon_dir: Path,
    graphs_folder: Path,
    kts: dict[int, float],
) -> float | None:
    """Ensure gexf, edge, and clave_max exist for one hour. Returns clave_max or None."""
    gexf_path = (
        graphs_folder
        / "nodes_hashtag"
        / manifestacion
        / str(hour_window)
        / f"{hour}.gexf"
    )
    if not gexf_path.exists():
        created = create_single_gexf(df, hour, manifestacion, hour_window, graphs_folder)
        if created is None:
            return None

    edge_dir = epsilon_dir / str(hour)
    edge_path = edge_dir / f"{hour}.edge"
    if not edge_path.exists():
        edge_dir.mkdir(parents=True, exist_ok=True)
        gexf_a_edge(gexf_path, edge_path)

    if hour not in kts or not np.isfinite(kts.get(hour, float("nan"))):
        d = get_dict_hora_kt(
            manifestacion,
            hour,
            hour_window=hour_window,
            graphs_folder=graphs_folder,
        )
        clave_max = max(d, key=d.get) if d else None
        if clave_max is None:
            return None
        kts[hour] = float(clave_max)

    return kts[hour]


def write_results(
    results_dir: Path,
    values: dict[int, dict],
    max_rows: dict[int, dict],
    target_hours: list[int],
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "Epsilon_values.txt", "w", encoding="utf-8") as f:
        f.write(EPSILON_VALUES_HEADER)
        for hour in sorted(target_hours):
            if hour not in values:
                continue
            v = values[hour]
            f.write(
                f"{hour} \t {v['epsilon_ccdf']:.4f} \t {v['epsilon_knn']:.4f} "
                f"\t {v['epsilon_cco']:.4f} \n"
            )

    with open(results_dir / "Epsilon_max_vs_clustering.txt", "w", encoding="utf-8") as f:
        f.write(EPSILON_MAX_HEADER)
        for hour in sorted(target_hours):
            if hour not in max_rows:
                continue
            m = max_rows[hour]
            f.write(
                f"{hour} \t {m['avg_clustering']:.4f} \t {m['epsilon_max']:.4f} \n"
            )


def process_dataset(
    manifestacion: str,
    *,
    half_window: int = 96,
    datasets_folder: Path | None = None,
    graphs_folder: Path | None = None,
    epsilon_root: Path | None = None,
) -> dict:
    cfg = DATASET_CONFIG[manifestacion]
    ctw = cfg["ctw"]
    hour_window = cfg["hour_window"]

    datasets_folder = datasets_folder or REPO_ROOT / "datasets" / "csvs"
    graphs_folder = graphs_folder or REPO_ROOT / "graphs"
    epsilon_dir = (epsilon_root or REPO_ROOT / "epsilon_sq") / manifestacion
    results_dir = epsilon_dir / "results"

    lo, hi = ctw - half_window, ctw + half_window + 1
    target_hours = list(range(lo, hi))

    df = pd.read_csv(datasets_folder / f"{manifestacion}.csv")
    available_hours = set(df["hour"].unique())
    hours_in_data = [h for h in target_hours if h in available_hours]
    hours_no_data = [h for h in target_hours if h not in available_hours]

    values, max_rows = load_existing_epsilon(results_dir)
    kts = load_kts(epsilon_dir)

    skipped = []
    computed = []
    failed = []
    prep_failed = []

    pending = [h for h in hours_in_data if h not in values]

    print(f"\n=== {manifestacion.upper()} CTW={ctw} window [{lo}, {hi}) ===")
    print(f"Target hours in data: {len(hours_in_data)}/{len(target_hours)}")
    print(f"Already valid: {len([h for h in hours_in_data if h in values])}")
    print(f"To compute: {len(pending)}")
    if hours_no_data:
        print(f"Skipped (no CSV data): {len(hours_no_data)} hours")

    for hour in tqdm(pending, desc=f"{manifestacion} epsilon"):
        clave_max = ensure_prerequisites(
            hour,
            manifestacion,
            hour_window,
            df,
            epsilon_dir,
            graphs_folder,
            kts,
        )
        if clave_max is None:
            prep_failed.append(hour)
            continue

        try:
            result = compute_epsilon_for_network(
                epsilon_dir,
                hour,
                clave_max,
            )
            if not all(
                np.isfinite(result[k])
                for k in ("epsilon_ccdf", "epsilon_knn", "epsilon_cco")
            ):
                failed.append(hour)
                continue

            values[hour] = {
                "epsilon_ccdf": result["epsilon_ccdf"],
                "epsilon_knn": result["epsilon_knn"],
                "epsilon_cco": result["epsilon_cco"],
            }
            max_rows[hour] = {
                "avg_clustering": result["avg_clustering"],
                "epsilon_max": result["epsilon_max"],
            }
            computed.append(hour)
        except Exception as exc:
            print(f"  FAILED hour {hour}: {exc}")
            failed.append(hour)

    for hour in hours_in_data:
        if hour in values:
            skipped.append(hour)

    save_kts(epsilon_dir, kts)
    write_results(results_dir, values, max_rows, hours_in_data)

    summary = {
        "manifestacion": manifestacion,
        "target_in_data": len(hours_in_data),
        "skipped": len(skipped),
        "computed": len(computed),
        "failed": failed,
        "prep_failed": prep_failed,
        "no_data": hours_no_data,
        "total_valid": len([h for h in hours_in_data if h in values]),
    }
    print(
        f"Done {manifestacion}: computed={summary['computed']}, "
        f"skipped={summary['skipped']}, failed={len(failed)}, "
        f"prep_failed={len(prep_failed)}, total_valid={summary['total_valid']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill missing epsilon values incrementally.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ch", "9n"],
        choices=list(DATASET_CONFIG),
    )
    parser.add_argument("--half-window", type=int, default=96)
    args = parser.parse_args()

    summaries = []
    for manifestacion in args.datasets:
        summaries.append(process_dataset(manifestacion, half_window=args.half_window))

    print("\n=== SUMMARY ===")
    for s in summaries:
        print(
            f"{s['manifestacion']}: valid={s['total_valid']}/{s['target_in_data']}, "
            f"computed={s['computed']}, failed={len(s['failed'])}, "
            f"prep_failed={len(s['prep_failed'])}"
        )


if __name__ == "__main__":
    main()
