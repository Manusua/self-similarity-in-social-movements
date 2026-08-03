"""Shared configuration for reviewer-response analyses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Repository root (parent of this folder)
REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW_ROOT = Path(__file__).resolve().parent

FIGURES_DIR = REVIEW_ROOT / "figures"
RESULTS_DIR = REVIEW_ROOT / "results"
EMBEDDINGS_DIR = REVIEW_ROOT / "embeddings"
VENDOR_DIR = REVIEW_ROOT / "vendor"

DATASETS_DIR = REPO_ROOT / "datasets" / "csvs"
GRAPHS_DIR = REPO_ROOT / "graphs"
MEASURES_DIR = REPO_ROOT / "measures"
EPSILON_DIR = REPO_ROOT / "epsilon_sq"
DMERCATOR_DIR = REPO_ROOT / "d-mercator" / "graphs"

LOUVAIN_SEED = 42
MODULARITY_SEED = 123  # legacy seed in utils_graphs.get_mod_nest_coefficient
EMBEDDING_THRESHOLD = 2
EMBEDDING_DIMENSION = 1

HORA_CRITICA = {
    "nat": 429624,
    "9n": 437037,
    "ch": 394717,
}

HOUR_WINDOW = {
    "nat": 1,
    "9n": 2,
    "ch": 2,
}

TIMEZONE = {
    "nat": "America/Argentina/Buenos_Aires",
    "9n": "America/Argentina/Buenos_Aires",
    "ch": "Europe/Paris",
}

HOUR_RANGES = {
    "nat": (429576, 429672),
    "9n": (437037 - 72, 437037 + 72),
    "ch": (394717 - 48, 394717 + 48),
}

PLOT_SLICE = {
    "nat": {"inicio": 45, "final": 140},
    "9n": {"inicio": 0, "final": 140},
    "ch": {"inicio": 206, "final": 293},
}

# Grey = low-activity windows; orange = CTW candidate windows (hour indices)
ANNOTATION_WINDOWS: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
    "nat": {
        "grey": [(429580, 429587), (429604, 429611), (429628, 429635), (429652, 429659)],
        "orange": [(429620, 429628)],
    },
    "9n": {
        "grey": [(437025, 437032), (437048, 437055), (437070, 437077)],
        "orange": [(437033, 437041)],
    },
    "ch": {
        "grey": [(394680, 394687), (394700, 394707), (394730, 394737)],
        "orange": [(394712, 394720)],
    },
}

MANIFESTATIONS = list(HORA_CRITICA.keys())

# Existing CTW embeddings to reuse (NAT/9N); CH 394718 kept for sensitivity
EXISTING_CTW_EMBEDDINGS = {
    "nat": (DMERCATOR_DIR / "nat" / "429624" / str(EMBEDDING_THRESHOLD), "429624"),
    "9n": (DMERCATOR_DIR / "9n" / "437037" / str(EMBEDDING_THRESHOLD), "437037"),
    "ch_sensitivity": (DMERCATOR_DIR / "ch" / "394718" / str(EMBEDDING_THRESHOLD), "394718"),
}

CH_PRIMARY_HOUR = 394717
CH_SENSITIVITY_HOUR = 394718

NONCTW_EXCLUSION_HOURS = 12
NONCTW_ACTIVITY_TOL = 0.20
NONCTW_ACTIVITY_TOL_FALLBACK = 0.30


@dataclass(frozen=True)
class ManifestationConfig:
    name: str
    critical_hour: int
    hour_window: int
    timezone: str
    hour_range: Tuple[int, int]
    plot_slice: Dict[str, int]


def get_manifestation_config(name: str) -> ManifestationConfig:
    return ManifestationConfig(
        name=name,
        critical_hour=HORA_CRITICA[name],
        hour_window=HOUR_WINDOW[name],
        timezone=TIMEZONE[name],
        hour_range=HOUR_RANGES[name],
        plot_slice=PLOT_SLICE[name],
    )


def ensure_dirs() -> None:
    for path in (FIGURES_DIR, RESULTS_DIR, EMBEDDINGS_DIR, VENDOR_DIR):
        path.mkdir(parents=True, exist_ok=True)
