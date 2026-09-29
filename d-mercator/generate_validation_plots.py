#!/usr/bin/env python3
"""Generate validation plots for CTW and Other TW hours (NAT, 9N, CH)."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DMERCATOR_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = DMERCATOR_DIR / "validation_plots"


@dataclass(frozen=True)
class ValidationSpec:
    manifestacion: str
    hour: int
    window_type: str
    embedding_root: Path
    title: str

    @property
    def graph_id(self) -> str:
        return str(self.hour)

    @property
    def output_dir(self) -> Path:
        return VALIDATION_DIR / self.manifestacion / self.graph_id


SPECS = [
    ValidationSpec(
        manifestacion="nat",
        hour=429624,
        window_type="CTW",
        embedding_root=REPO_ROOT / "d-mercator" / "graphs" / "nat" / "429624" / "2" / "429624",
        title="No al tarifazo — CTW",
    ),
    ValidationSpec(
        manifestacion="nat",
        hour=429600,
        window_type="Other_TW",
        embedding_root=REPO_ROOT
        / "calculo_parameters_other_hours"
        / "embeddings"
        / "nat"
        / "429600"
        / "429600",
        title="No al tarifazo — Other TW",
    ),
    ValidationSpec(
        manifestacion="9n",
        hour=437037,
        window_type="CTW",
        embedding_root=REPO_ROOT / "d-mercator" / "graphs" / "9n" / "437037" / "2" / "437037",
        title="9n — CTW",
    ),
    ValidationSpec(
        manifestacion="9n",
        hour=436989,
        window_type="Other_TW",
        embedding_root=REPO_ROOT
        / "calculo_parameters_other_hours"
        / "embeddings"
        / "9n"
        / "436989"
        / "436989",
        title="9n — Other TW",
    ),
    ValidationSpec(
        manifestacion="ch",
        hour=394717,
        window_type="CTW",
        embedding_root=REPO_ROOT
        / "new_plots_review"
        / "embeddings"
        / "ch"
        / "CTW"
        / "394717"
        / "2"
        / "394717",
        title="Charlie Hebdo — CTW",
    ),
    ValidationSpec(
        manifestacion="ch",
        hour=394693,
        window_type="Other_TW",
        embedding_root=REPO_ROOT
        / "calculo_parameters_other_hours"
        / "embeddings"
        / "ch"
        / "394693"
        / "394693",
        title="Charlie Hebdo — Other TW",
    ),
]


def load_pdf_module():
    script = DMERCATOR_DIR / "pdf_d-mercator.py"
    spec = importlib.util.spec_from_file_location("pdf_dmercator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_files(root: Path) -> list[Path]:
    root_str = str(root)
    return [
        Path(root_str + ".edge"),
        Path(root_str + ".inf_coord"),
    ]


def generate_for_spec(spec: ValidationSpec, pdf_module) -> dict:
    root = spec.embedding_root
    graph_id = spec.graph_id
    missing = [str(p) for p in _required_files(root) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {spec.manifestacion} {graph_id}: {missing}")

    out_dir = spec.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{graph_id}_validation.pdf"
    png_path = out_dir / f"{graph_id}_theta_density.png"

    root_str = str(root)
    meta = pdf_module.generate_validation_pdf(
        root_str,
        root_str,
        str(pdf_path),
        title=spec.title,
        include_theta=True,
        include_angle_comparison=False,
    )
    pdf_module.plot_theta_density_standalone(str(png_path), root_str)

    return {
        "manifestacion": spec.manifestacion,
        "hour": spec.hour,
        "window_type": spec.window_type,
        "title": spec.title,
        "embedding_root": str(root),
        "output_dir": str(out_dir),
        "validation_pdf": str(pdf_path),
        "theta_density_png": str(png_path),
        "panels": ";".join(meta["panels"]),
        "skipped_panels": ";".join(meta["skipped_panels"])
        or "inferred_vs_original_theta: no .gen_coord (empirical network)",
        "has_vprop": meta["has_vprop"],
        "has_gen_coord": meta["has_gen_coord"],
        "pdf_size_bytes": pdf_path.stat().st_size if pdf_path.exists() else 0,
        "png_size_bytes": png_path.stat().st_size if png_path.exists() else 0,
    }


def main() -> None:
    pdf_module = load_pdf_module()
    rows = []
    for spec in SPECS:
        print(f"Generating {spec.manifestacion} {spec.hour} ({spec.window_type})...")
        rows.append(generate_for_spec(spec, pdf_module))

    df = pd.DataFrame(rows)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = VALIDATION_DIR / "manifest.csv"
    df.to_csv(manifest_path, index=False)
    print(f"\nManifest: {manifest_path}")
    print(df[["manifestacion", "hour", "window_type", "validation_pdf", "theta_density_png"]].to_string(index=False))


if __name__ == "__main__":
    main()
