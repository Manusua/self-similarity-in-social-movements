"""Canonical epsilon-squared metrics (extracted from Computing_epsilon2_values_annotated.py)."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

NBINS = 20
REFERENCE_THRESHOLD = 2


def read_graph(edge_dir: Path, network: str | int, threshold: int) -> nx.Graph:
    graph = nx.read_edgelist(edge_dir / f"{network}/{network}.edge", data=False)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() == 0:
        return graph
    largest = max(nx.connected_components(graph), key=len)
    if len(largest) != graph.number_of_nodes():
        graph = graph.subgraph(largest).copy()
    remove = [n for n, d in graph.degree() if d <= threshold]
    graph.remove_nodes_from(remove)
    graph.remove_nodes_from(list(nx.isolates(graph)))
    return graph


def read_original_graph(edge_dir: Path, network: str | int) -> nx.Graph:
    graph = nx.read_edgelist(edge_dir / f"{network}/{network}.edge", data=False)
    graph.remove_edges_from(nx.selfloop_edges(graph))
    if graph.number_of_nodes() == 0:
        return graph
    largest = max(nx.connected_components(graph), key=len)
    if len(largest) != graph.number_of_nodes():
        graph = graph.subgraph(largest).copy()
    return graph


def compute_metric_cco(graph: nx.Graph, out: dict) -> dict:
    degree_dict = dict(graph.degree())
    filtered = {n: c for n, c in nx.clustering(graph).items() if degree_dict[n] > 1}
    avg_clustering = sum(filtered.values()) / len(filtered) if filtered else 0.0
    avg_degree = sum(degree_dict.values()) / graph.number_of_nodes() if graph.number_of_nodes() else 0.0
    buckets: dict[float, list] = defaultdict(list)
    for node in graph.nodes():
        buckets[graph.degree(node) / avg_degree].append(node)
    for degree, nodes in buckets.items():
        clustering = nx.clustering(graph, nodes)
        mean_c = sum(clustering.values()) / len(clustering)
        out[degree].append(mean_c / avg_clustering if avg_clustering else 0.0)
    return out


def compute_metric_ccdf(graph: nx.Graph, out: dict) -> dict:
    if graph.number_of_nodes() == 0:
        return out
    avg_degree = sum(d for _, d in graph.degree()) / graph.number_of_nodes()
    buckets: dict[float, list] = defaultdict(list)
    for node in graph.nodes():
        buckets[graph.degree(node) / avg_degree].append(node)
    deg_g = {k: len(v) / graph.number_of_nodes() for k, v in buckets.items()}
    sorted_deg = {k: deg_g[k] for k in sorted(deg_g)}
    cs = np.cumsum(np.array(list(sorted_deg.values())))
    ccdf = np.insert(1 - cs, 0, 1)[:-1]
    for idx, degree in enumerate(sorted_deg):
        out[degree].append(float(ccdf[idx]))
    return out


def compute_metric_knn(graph: nx.Graph, out: dict) -> dict:
    avg_degree = sum(d for _, d in graph.degree()) / graph.number_of_nodes()
    avg_degree2 = sum(d**2 for _, d in graph.degree()) / graph.number_of_nodes()
    buckets: dict[float, list] = defaultdict(list)
    for node in graph.nodes():
        buckets[graph.degree(node) / avg_degree].append(node)
    avg_neigh = nx.average_neighbor_degree(graph)
    for degree, nodes in buckets.items():
        mean_val = sum(avg_neigh[n] for n in nodes) / len(nodes)
        out[degree].append(mean_val / (avg_degree2 / avg_degree))
    return out


def make_bins(intval1: dict, intval2: dict, nbins: int) -> np.ndarray:
    nonzero1 = [k for k in intval1 if k > 0]
    nonzero2 = [k for k in intval2 if k > 0]
    if not nonzero1 or not nonzero2:
        raise ValueError("Cannot build bins with empty positive support.")
    x0 = max(min(nonzero1), min(nonzero2))
    xf = min(max(nonzero1), max(nonzero2))
    xq = (xf / x0) ** (1 / nbins)
    x_exp = np.zeros(nbins + 1)
    x_exp[0] = x0
    for i in range(nbins):
        x_exp[i + 1] = x0 * (xq ** (i + 1))
    x_exp[-1] = xf
    return x_exp


def mean_bins(x_exp: np.ndarray, matrix: dict) -> np.ndarray:
    mean_values = np.zeros(len(x_exp) - 1)
    for i in range(1, len(x_exp)):
        values: list[np.ndarray] = []
        for key, entries in matrix.items():
            if i == len(x_exp) - 1:
                if x_exp[i - 1] <= key <= x_exp[i]:
                    values.append(np.ravel(np.array(entries)))
            elif x_exp[i - 1] <= key < x_exp[i]:
                values.append(np.ravel(np.array(entries)))
        if values:
            mean_values[i - 1] = float(np.mean(np.concatenate(values)))
    return mean_values


def compute_diff(ref: np.ndarray, other: np.ndarray) -> float:
    diffs = [
        ((a - b) / a) ** 2 if a > 0 else (a - b) ** 2
        for a, b in zip(ref, other)
    ]
    return float(sum(diffs) / len(other))


def average_clustering(graph: nx.Graph) -> float:
    degree_dict = dict(graph.degree())
    filtered = {n: c for n, c in nx.clustering(graph).items() if degree_dict[n] > 1}
    return sum(filtered.values()) / len(filtered) if filtered else 0.0


def generate_k_loop(k_t_max: int | float, n_max: int = 6, min_dif: int = 3) -> List[int]:
    n_max -= 1
    delta_k = max(min_dif, int(np.floor((int(k_t_max) - 2) / n_max)))
    n_curves = max(3, min(n_max, int(np.floor((int(k_t_max) - 2) / delta_k) + 1)))
    loop = [int(2 + i * delta_k) for i in range(1, int(n_curves))]
    if loop[-1] != int(k_t_max):
        if int(k_t_max) - loop[-1] >= min_dif:
            loop.append(int(k_t_max))
        else:
            loop[-1] = int(k_t_max)
    return loop


def compute_epsilon_for_network(
    edge_dir: Path,
    network: str | int,
    k_t_max: int | float,
    *,
    nbins: int = NBINS,
    threshold: int = REFERENCE_THRESHOLD,
    verbose: bool = False,
) -> Dict[str, float]:
    cco_ref: dict = defaultdict(list)
    ccdf_ref: dict = defaultdict(list)
    knn_ref: dict = defaultdict(list)

    graph_org = read_original_graph(edge_dir, network)
    avg_clust = average_clustering(graph_org)
    graph_ref = read_graph(edge_dir, network, threshold)
    compute_metric_cco(graph_ref, cco_ref)
    compute_metric_ccdf(graph_ref, ccdf_ref)
    compute_metric_knn(graph_ref, knn_ref)

    sorted_ccdf_ref = {k: ccdf_ref[k] for k in sorted(ccdf_ref)}
    sorted_cco_ref = {k: cco_ref[k] for k in sorted(cco_ref)}
    sorted_knn_ref = {k: knn_ref[k] for k in sorted(knn_ref)}

    eps_ccdf: list[float] = []
    eps_cco: list[float] = []
    eps_knn: list[float] = []

    for k_t in generate_k_loop(k_t_max):
        cco_k: dict = defaultdict(list)
        ccdf_k: dict = defaultdict(list)
        knn_k: dict = defaultdict(list)
        graph_k = read_graph(edge_dir, network, k_t)
        if graph_k.number_of_nodes() == 0:
            continue
        compute_metric_ccdf(graph_k, ccdf_k)
        compute_metric_cco(graph_k, cco_k)
        compute_metric_knn(graph_k, knn_k)
        sorted_ccdf = {k: ccdf_k[k] for k in sorted(ccdf_k)}
        sorted_cco = {k: cco_k[k] for k in sorted(cco_k)}
        sorted_knn = {k: knn_k[k] for k in sorted(knn_k)}

        x_ccdf = make_bins(sorted_ccdf, sorted_ccdf_ref, nbins)
        x_cco = make_bins(sorted_cco, sorted_cco_ref, nbins)
        x_knn = make_bins(sorted_knn, sorted_knn_ref, nbins)

        eps_ccdf.append(
            compute_diff(mean_bins(x_ccdf, sorted_ccdf_ref), mean_bins(x_ccdf, sorted_ccdf))
        )
        eps_cco.append(
            compute_diff(mean_bins(x_cco, sorted_cco_ref), mean_bins(x_cco, sorted_cco))
        )
        eps_knn.append(
            compute_diff(mean_bins(x_knn, sorted_knn_ref), mean_bins(x_knn, sorted_knn))
        )
        if verbose:
            print(network, k_t, eps_ccdf[-1], eps_cco[-1], eps_knn[-1])

    result = {
        "network": int(network),
        "epsilon_ccdf": float(np.mean(eps_ccdf)) if eps_ccdf else float("nan"),
        "epsilon_knn": float(np.mean(eps_knn)) if eps_knn else float("nan"),
        "epsilon_cco": float(np.mean(eps_cco)) if eps_cco else float("nan"),
        "avg_clustering": float(avg_clust),
    }
    result["epsilon_max"] = float(
        max(result["epsilon_ccdf"], result["epsilon_knn"], result["epsilon_cco"])
    )
    return result


def load_kts(manifestation: str, epsilon_root: Path | None = None) -> pd.DataFrame:
    root = epsilon_root or Path("epsilon_sq")
    return pd.read_csv(root / manifestation / "kts.csv")


def load_canonical_epsilon_table(manifestation: str, epsilon_root: Path | None = None) -> pd.DataFrame:
    root = epsilon_root or Path("epsilon_sq")
    path = root / manifestation / "results" / "Epsilon_values.txt"
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    if "Network" in df.columns:
        df = df.rename(columns={"Network": "network"})
    else:
        df = df.rename(columns={df.columns[0]: "network"})
    rename = {}
    for col in df.columns:
        low = col.lower()
        if "ccdf" in low:
            rename[col] = "epsilon_ccdf"
        elif "knn" in low:
            rename[col] = "epsilon_knn"
        elif "cco" in low:
            rename[col] = "epsilon_cco"
    df = df.rename(columns=rename)
    df["network"] = df["network"].astype(int)
    for col in ("epsilon_ccdf", "epsilon_knn", "epsilon_cco"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def compare_with_canonical(
    recomputed: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> pd.DataFrame:
    merged = recomputed.merge(canonical, on="network", suffixes=("_new", "_ref"))
    for metric in ("epsilon_ccdf", "epsilon_knn", "epsilon_cco"):
        merged[f"{metric}_abs_diff"] = (merged[f"{metric}_new"] - merged[f"{metric}_ref"]).abs()
        merged[f"{metric}_ok"] = np.isclose(
            merged[f"{metric}_new"], merged[f"{metric}_ref"], rtol=rtol, atol=atol
        )
    return merged
