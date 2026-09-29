"""Configuración para parámetros D-Mercator y navegabilidad — ventanas Other TW."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(__file__).resolve().parent

INPUTS_DIR = WORK_ROOT / "inputs"
EMBEDDINGS_DIR = WORK_ROOT / "embeddings"
RESULTS_DIR = WORK_ROOT / "results"
VENDOR_DIR = WORK_ROOT / "vendor"
GRAPHS_FILTERED = REPO_ROOT / "graphs" / "nodes_filtered"

# Reutilizar vendor de new_plots_review si ya existe
FALLBACK_VENDOR = REPO_ROOT / "new_plots_review" / "vendor" / "d-mercator"

DMERCATOR_DIMENSION = 1
PAIR_STEP = 1
DIAM = 1000


@dataclass(frozen=True)
class OtherHourSpec:
    manifestacion: str
    hour: int
    hour_window: int
    threshold: int
    ctw_offset_h: int
    plot_label: str

    @property
    def graph_id(self) -> str:
        return str(self.hour)

    @property
    def source_edge(self) -> Path:
        return (
            GRAPHS_FILTERED
            / str(self.threshold)
            / self.manifestacion
            / str(self.hour_window)
            / f"{self.hour}.edge"
        )

    @property
    def input_dir(self) -> Path:
        return INPUTS_DIR / self.manifestacion / self.graph_id

    @property
    def input_edge(self) -> Path:
        return self.input_dir / f"{self.graph_id}.edge"

    @property
    def embedding_dir(self) -> Path:
        return EMBEDDINGS_DIR / self.manifestacion / self.graph_id


OTHER_HOURS: List[OtherHourSpec] = [
    OtherHourSpec(
        manifestacion="nat",
        hour=429600,
        hour_window=1,
        threshold=1,
        ctw_offset_h=-24,
        plot_label="No al tarifazo: Other TW",
    ),
    OtherHourSpec(
        manifestacion="9n",
        hour=436989,
        hour_window=2,
        threshold=3,
        ctw_offset_h=-48,
        plot_label="9n: Other TW",
    ),
    OtherHourSpec(
        manifestacion="ch",
        hour=394693,
        hour_window=2,
        threshold=1,
        ctw_offset_h=-24,
        plot_label="Charlie Hebdo: Other TW",
    ),
]

OTHER_HOURS_BY_KEY: Dict[str, OtherHourSpec] = {s.manifestacion: s for s in OTHER_HOURS}
