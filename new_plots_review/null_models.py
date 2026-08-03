"""Null-model controls and epsilon figure generation."""

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from config import (
    FIGURES_DIR,
    HORA_CRITICA,
    MANIFESTATIONS,
    REPO_ROOT,
    RESULTS_DIR,
    get_manifestation_config,
)
from epsilon_metrics import (
    compare_with_canonical,
    compute_epsilon_for_network,
    load_canonical_epsilon_table,
    load_kts,
)


def configuration_model_rewire(graph: nx.Graph, seed: int = 42) -> nx.Graph:
    rng = random.Random(seed)
    stublist = [d for _, d in graph.degree()]
    rng.shuffle(stublist)
    null = nx.configuration_model(stublist, seed=seed)
    null = nx.Graph(null)
    null.remove_edges_from(nx.selfloop_edges(null))
    largest = max(nx.connected_components(null), key=len) if null.number_of_nodes() else []
    if len(largest) < null.number_of_nodes():
        null = null.subgraph(largest).copy()
    return null


def graph_metrics(graph: nx.Graph) -> Dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {
            "avg_path_length": np.nan,
            "global_efficiency": np.nan,
            "avg_clustering": np.nan,
        }
    if nx.is_connected(graph):
        apl = nx.average_shortest_path_length(graph)
        eff = nx.global_efficiency(graph)
    else:
        apl = np.nan
        eff = nx.global_efficiency(graph)
    return {
        "avg_path_length": float(apl) if apl == apl else np.nan,
        "global_efficiency": float(eff),
        "avg_clustering": float(nx.average_clustering(graph)),
    }


def write_temp_edgelist(graph: nx.Graph, base_dir: Path, hour: int) -> Path:
    network_dir = base_dir / str(hour)
    network_dir.mkdir(parents=True, exist_ok=True)
    edge_path = network_dir / f"{hour}.edge"
    with open(edge_path, "w", encoding="utf-8") as handle:
        for u, v in graph.edges():
            handle.write(f"{u}\t{v}\n")
    return base_dir


def epsilon_regression_check(manifestations: List[str] | None = None) -> pd.DataFrame:
    manifestations = manifestations or MANIFESTATIONS
    rows = []
    for manifestation in manifestations:
        edge_dir = REPO_ROOT / "epsilon_sq" / manifestation
        kts = load_kts(manifestation, REPO_ROOT / "epsilon_sq")
        canonical = load_canonical_epsilon_table(manifestation, REPO_ROOT / "epsilon_sq")
        sample_hours = [HORA_CRITICA[manifestation]]
        if manifestation == "nat":
            sample_hours.append(429624)
        sample_hours = sorted(set(sample_hours))
        for hour in sample_hours:
            row = kts[kts["hora"] == hour]
            if row.empty:
                continue
            kmax = row.iloc[0]["clave_max"]
            recomputed = pd.DataFrame([compute_epsilon_for_network(edge_dir, hour, kmax)])
            merged = compare_with_canonical(recomputed, canonical[canonical["network"] == hour])
            if merged.empty:
                continue
            rec = merged.iloc[0].to_dict()
            rec["manifestacion"] = manifestation
            rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "epsilon_regression_check.csv", index=False)
    return out


def ct_epsilon_summary(manifestations: List[str] | None = None) -> pd.DataFrame:
    manifestations = manifestations or MANIFESTATIONS
    rows = []
    for manifestation in manifestations:
        canonical = load_canonical_epsilon_table(manifestation, REPO_ROOT / "epsilon_sq")
        hour = HORA_CRITICA[manifestation]
        row = canonical[canonical["network"] == hour]
        if row.empty:
            continue
        rec = row.iloc[0].to_dict()
        rec["manifestacion"] = manifestation
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "epsilon_ctw_summary.csv", index=False)
    return out


def plot_epsilon_collapse_panels(manifestation: str) -> Path:
    hour = HORA_CRITICA[manifestation]
    edge_dir = REPO_ROOT / "epsilon_sq" / manifestation
    canonical = load_canonical_epsilon_table(manifestation, REPO_ROOT / "epsilon_sq")
    row = canonical[canonical["network"] == hour].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    metrics = ["epsilon_ccdf", "epsilon_cco", "epsilon_knn"]
    labels = [r"$\epsilon^2_{\mathrm{ccdf}}$", r"$\epsilon^2_{\mathrm{cco}}$", r"$\epsilon^2_{\mathrm{knn}}$"]
    values = [row[m] for m in metrics]
    axes[0].bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    axes[0].set_ylabel(r"$\epsilon^2$ at CTW")
    axes[0].set_title(f"{manifestation.upper()} CTW components")

    temporal = canonical.copy()
    for col in metrics:
        temporal[col] = temporal[col].astype(float)
    axes[1].plot(temporal["network"], temporal["epsilon_ccdf"], label=r"$\epsilon^2_{\mathrm{ccdf}}$", linewidth=1.5)
    axes[1].plot(temporal["network"], temporal["epsilon_cco"], label=r"$\epsilon^2_{\mathrm{cco}}$", linewidth=1.5)
    axes[1].plot(temporal["network"], temporal["epsilon_knn"], label=r"$\epsilon^2_{\mathrm{knn}}$", linewidth=1.5)
    axes[1].axvline(hour, color="black", linestyle="-.", linewidth=2)
    axes[1].set_xlabel("Hour index")
    axes[1].set_ylabel(r"$\epsilon^2$")
    axes[1].set_title("Temporal epsilon components (canonical pipeline)")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        f"{manifestation.upper()} CTW hour {hour} — "
        rf"$\epsilon^2_{{\mathrm{{cco}}}}={row['epsilon_cco']:.3f}$"
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"epsilon_collapse_{manifestation}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def rolling_mean_centrada(series: pd.Series, window: int = 3) -> pd.Series:
    """Centered moving average over consecutive TWs (latex_CTW.ipynb)."""
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=window, center=True, min_periods=1).mean()


def _format_local_xaxis(ax, tz_name: str, city_label: str) -> None:
    def formatter(x_val, _pos):
        dt_local = datetime.fromtimestamp(int(x_val) * 3600, tz=ZoneInfo("UTC")).astimezone(
            ZoneInfo(tz_name)
        )
        return dt_local.strftime("%d-%b %Hh")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.tick_params(axis="x", which="major", rotation=75)
    ax.set_xlabel(f"Local time ({city_label})")


def plot_epsilon_collapse_temporal_combined(
    manifestations: List[str] | None = None,
) -> Path:
    manifestations = manifestations or MANIFESTATIONS
    ma_window = 3
    city_labels = {
        "nat": "Buenos Aires",
        "9n": "Buenos Aires",
        "ch": "Paris",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    metrics = [
        ("epsilon_ccdf", r"$\epsilon^2_{\mathrm{ccdf}}$", "#1f77b4"),
        ("epsilon_cco", r"$\epsilon^2_{\mathrm{cco}}$", "#2ca02c"),
    ]

    for col, manifestation in enumerate(manifestations):
        ax = axes[col]
        cfg = get_manifestation_config(manifestation)
        hour = HORA_CRITICA[manifestation]
        canonical = load_canonical_epsilon_table(manifestation, REPO_ROOT / "epsilon_sq")
        temporal = canonical.copy()

        for metric, label, color in metrics:
            temporal[metric] = temporal[metric].astype(float)
            y_ma = rolling_mean_centrada(temporal[metric], window=ma_window)
            ax.plot(
                temporal["network"],
                y_ma,
                label=rf"{label} (MA{ma_window})",
                linewidth=1.8,
                color=color,
            )

        ax.axvline(hour, color="black", linestyle="-.", linewidth=2)
        ax.set_title(manifestation.upper())
        _format_local_xaxis(ax, cfg.timezone, city_labels[manifestation])
        if col == 0:
            ax.set_ylabel(r"$\epsilon^2$")
            ax.legend(fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / "epsilon_collapse_temporal_combined.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def null_model_analysis(manifestation: str, n_replicates: int = 20) -> pd.DataFrame:
    hour = HORA_CRITICA[manifestation]
    edge_dir = REPO_ROOT / "epsilon_sq" / manifestation
    graph = nx.read_edgelist(edge_dir / f"{hour}/{hour}.edge")
    graph.remove_edges_from(nx.selfloop_edges(graph))
    largest = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(largest).copy()

    observed_metrics = graph_metrics(graph)
    kts = load_kts(manifestation, REPO_ROOT / "epsilon_sq")
    kmax = kts[kts["hora"] == hour].iloc[0]["clave_max"]
    observed_eps = compute_epsilon_for_network(edge_dir, hour, kmax)

    rows = []
    temp_root = RESULTS_DIR / "null_temp" / manifestation / str(hour)
    for rep in range(n_replicates):
        null = configuration_model_rewire(graph, seed=1000 + rep)
        null_dir = temp_root / f"rep_{rep}"
        write_temp_edgelist(null, null_dir, hour)
        null_metrics = graph_metrics(null)
        null_eps = compute_epsilon_for_network(null_dir, hour, kmax)
        rows.append(
            {
                "manifestacion": manifestation,
                "replicate": rep,
                "type": "null_configuration",
                **{f"obs_{k}": v for k, v in observed_metrics.items()},
                **{f"null_{k}": v for k, v in null_metrics.items()},
                "obs_epsilon_cco": observed_eps["epsilon_cco"],
                "null_epsilon_cco": null_eps["epsilon_cco"],
                "obs_epsilon_ccdf": observed_eps["epsilon_ccdf"],
                "null_epsilon_ccdf": null_eps["epsilon_ccdf"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"null_model_{manifestation}.csv", index=False)
    return out


def plot_null_model_summary(manifestation: str) -> Path:
    df = pd.read_csv(RESULTS_DIR / f"null_model_{manifestation}.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    metrics = [
        ("epsilon_cco", r"$\epsilon^2_{\mathrm{cco}}$"),
        ("epsilon_ccdf", r"$\epsilon^2_{\mathrm{ccdf}}$"),
        ("avg_path_length", "Average path length"),
    ]
    for ax, (key, label) in zip(axes, metrics):
        obs_col = f"obs_{key}"
        null_col = f"null_{key}"
        if obs_col not in df.columns:
            obs_col = "obs_epsilon_cco" if key.startswith("epsilon") else obs_col
        if null_col not in df.columns:
            null_col = "null_epsilon_cco" if key.startswith("epsilon") else null_col
        obs_val = df[obs_col].iloc[0]
        ax.hist(df[null_col], bins=12, color="#bbbbbb", edgecolor="white")
        ax.axvline(obs_val, color="black", linewidth=2, label="Observed CTW")
        ax.set_title(label)
        ax.legend()
    fig.suptitle(f"{manifestation.upper()} — configuration-model null")
    fig.tight_layout()
    out = FIGURES_DIR / f"null_model_{manifestation}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out
