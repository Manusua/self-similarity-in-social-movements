"""Community analyses: fixed modularity, AMI reuse, cluster-size entropy."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from ami_utils import adjusted_mutual_info_score

from config import (
    FIGURES_DIR,
    HOUR_RANGES,
    HOUR_WINDOW,
    HORA_CRITICA,
    LOUVAIN_SEED,
    MANIFESTATIONS,
    REPO_ROOT,
    RESULTS_DIR,
    get_manifestation_config,
)


def load_weighted_graph(manifestation: str, hour: int) -> nx.Graph | None:
    hw = HOUR_WINDOW[manifestation]
    path = REPO_ROOT / "graphs" / "nodes_hashtag" / manifestation / str(hw) / f"{hour}.gexf"
    if not path.exists():
        return None
    graph = nx.read_gexf(path)
    return graph


def louvain_partition(graph: nx.Graph, seed: int = LOUVAIN_SEED) -> Dict[str, int]:
    if graph.number_of_nodes() == 0:
        return {}
    communities = nx.community.louvain_communities(graph, seed=seed, weight="weight")
    partition: Dict[str, int] = {}
    for comm_id, comm in enumerate(communities):
        for node in comm:
            partition[str(node)] = comm_id
    return partition


def normalized_community_entropy(partition: Dict[str, int]) -> float:
    if not partition:
        return float("nan")
    sizes = Counter(partition.values())
    total = sum(sizes.values())
    probs = [count / total for count in sizes.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


def modularity_with_fixed_partition(
    graph: nx.Graph,
    partition_ref: Dict[str, int],
    nodes_ref: Iterable[str],
) -> float:
    if graph.number_of_edges() == 0:
        return 0.0
    next_comm = max(partition_ref.values(), default=-1) + 1
    node_to_comm: Dict[str, int] = {}
    for node in graph.nodes():
        node_s = str(node)
        if node_s in partition_ref:
            node_to_comm[node_s] = partition_ref[node_s]
        else:
            node_to_comm[node_s] = next_comm
            next_comm += 1
    communities: Dict[int, set] = {}
    for node, comm in node_to_comm.items():
        communities.setdefault(comm, set()).add(node)
    comm_list = list(communities.values())
    return float(nx.community.modularity(graph, comm_list, weight="weight"))


def labels_for_ami(
    partition_ref: Dict[str, int],
    partition_h: Dict[str, int],
    nodes_ref: Iterable[str],
) -> Tuple[List[int], List[int]]:
    labels_ref: List[int] = []
    labels_h: List[int] = []
    next_singleton = max(partition_h.values(), default=-1) + 1
    for node in nodes_ref:
        labels_ref.append(partition_ref[str(node)])
        if str(node) in partition_h:
            labels_h.append(partition_h[str(node)])
        else:
            labels_h.append(next_singleton)
            next_singleton += 1
    return labels_ref, labels_h


def pairwise_comembership(
    partition_ref: Dict[str, int],
    partition_h: Dict[str, int],
    nodes_ref: Iterable[str],
) -> float:
    nodes = list(nodes_ref)
    if len(nodes) < 2:
        return 0.0
    same_ref = 0
    same_both = 0
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = str(nodes[i]), str(nodes[j])
            if partition_ref[a] == partition_ref[b]:
                same_ref += 1
                if (
                    a in partition_h
                    and b in partition_h
                    and partition_h[a] == partition_h[b]
                ):
                    same_both += 1
    return same_both / same_ref if same_ref else 0.0


def list_hours(manifestation: str) -> List[int]:
    start, end = HOUR_RANGES[manifestation]
    return list(range(start, end + 1))


def compute_fixed_modularity_series(manifestation: str) -> pd.DataFrame:
    cfg = get_manifestation_config(manifestation)
    graph_ref = load_weighted_graph(manifestation, cfg.critical_hour)
    if graph_ref is None:
        raise FileNotFoundError(f"Missing critical graph for {manifestation}")
    partition_ref = louvain_partition(graph_ref)
    nodes_ref = sorted(partition_ref.keys(), key=str)

    rows = []
    for hour in list_hours(manifestation):
        graph_h = load_weighted_graph(manifestation, hour)
        if graph_h is None:
            q_fixed = np.nan
            q_reopt = np.nan
            ami = 0.0
            pairwise = 0.0
            entropy = np.nan
        else:
            q_fixed = modularity_with_fixed_partition(graph_h, partition_ref, nodes_ref)
            partition_h = louvain_partition(graph_h)
            q_reopt = float(
                nx.community.modularity(
                    graph_h,
                    nx.community.louvain_communities(graph_h, seed=LOUVAIN_SEED, weight="weight"),
                    weight="weight",
                )
            )
            labels_ref, labels_h = labels_for_ami(partition_ref, partition_h, nodes_ref)
            ami = float(adjusted_mutual_info_score(labels_ref, labels_h))
            pairwise = pairwise_comembership(partition_ref, partition_h, nodes_ref)
            entropy = normalized_community_entropy(partition_h)
        rows.append(
            {
                "manifestacion": manifestation,
                "hour": hour,
                "hour_rel": (hour - cfg.critical_hour) / cfg.hour_window,
                "q_fixed": q_fixed,
                "q_reoptimized": q_reopt,
                "ami_weighted": ami,
                "pairwise_weighted": pairwise,
                "community_entropy_norm": entropy,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"community_fixed_{manifestation}.csv", index=False)
    return df


def load_existing_ami(manifestation: str) -> pd.DataFrame:
    path = REPO_ROOT / "measures" / "ami_louvain_vs_critica.csv"
    df = pd.read_csv(path)
    return df[df["manifestacion"] == manifestation].copy()


def plot_community_comparison(manifestation: str) -> Path:
    fixed = pd.read_csv(RESULTS_DIR / f"community_fixed_{manifestation}.csv")
    ami_existing = load_existing_ami(manifestation)
    cfg = get_manifestation_config(manifestation)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(fixed["hour_rel"], fixed["q_fixed"], label="Q fixed (CTW partition)", linewidth=1.8)
    axes[0].plot(fixed["hour_rel"], fixed["q_reoptimized"], label="Q re-optimized", linewidth=1.5, alpha=0.8)
    axes[0].set_ylabel("Modularity Q")
    axes[0].legend()

    axes[1].plot(fixed["hour_rel"], fixed["ami_weighted"], label="AMI weighted GEXF", linewidth=1.8)
    axes[1].plot(ami_existing["hour_rel"], ami_existing["ami"], label="AMI binary edge (existing)", linewidth=1.2, alpha=0.7)
    axes[1].set_ylabel("AMI")
    axes[1].legend()

    axes[2].plot(fixed["hour_rel"], fixed["community_entropy_norm"], label="Normalized entropy", linewidth=1.8, color="#2ca02c")
    axes[2].set_ylabel("H / log K")
    axes[2].set_xlabel("Hours relative to CTW")
    axes[2].legend()
    for ax in axes:
        ax.axvline(0, color="black", linestyle="-.", linewidth=2)

    fig.suptitle(f"{manifestation.upper()} — fixed communities vs re-optimized")
    fig.tight_layout()
    out = FIGURES_DIR / f"community_fixed_{manifestation}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def window_entropy_diagnostic() -> pd.DataFrame:
    rows = []
    for manifestation in MANIFESTATIONS:
        cfg = get_manifestation_config(manifestation)
        df = pd.read_csv(REPO_ROOT / "datasets" / "csvs" / f"{manifestation}.csv")
        crit = cfg.critical_hour
        for window in range(1, 9):
            hours = list(range(crit - window + 1, crit + 1))
            sub = df[df["hour"].isin(hours)]
            if sub.empty:
                continue
            # Build aggregated co-occurrence graph for the block
            edges = sub.groupby(["hashtag", "user"]).size().reset_index(name="w")
            users_by_tag = edges.groupby("hashtag")["user"].apply(set).to_dict()
            tags = list(users_by_tag)
            graph = nx.Graph()
            graph.add_nodes_from(tags)
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    if users_by_tag[tags[i]] & users_by_tag[tags[j]]:
                        graph.add_edge(tags[i], tags[j])
            partition = louvain_partition(graph)
            rows.append(
                {
                    "manifestacion": manifestation,
                    "window_h": window,
                    "entropy_norm": normalized_community_entropy(partition),
                    "n_nodes": graph.number_of_nodes(),
                    "n_edges": graph.number_of_edges(),
                    "n_hashtags_activity": sub["hashtag"].nunique(),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "window_entropy_diagnostic.csv", index=False)
    return out
