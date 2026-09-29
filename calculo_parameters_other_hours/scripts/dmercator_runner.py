"""Ejecuta D-Mercator en dimensión 1 (S¹) sobre los edgelists Other TW."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    DMERCATOR_DIMENSION,
    FALLBACK_VENDOR,
    OTHER_HOURS,
    VENDOR_DIR,
)


def resolve_mercator_repo() -> Path:
    for candidate in (VENDOR_DIR / "d-mercator", FALLBACK_VENDOR):
        if candidate.exists():
            return candidate
    target = VENDOR_DIR / "d-mercator"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/networkgeometry/d-mercator.git", str(target)],
        check=True,
    )
    return target


def build_mercator(repo: Path) -> Path:
    mercator_bin = repo / "mercator"
    if mercator_bin.exists():
        return mercator_bin
    build_script = repo / "build.sh"
    if not build_script.exists():
        raise FileNotFoundError(f"No se encontró build.sh en {repo}")
    subprocess.run(["bash", str(build_script), "-b", "Release"], check=True, cwd=str(repo))
    if not mercator_bin.exists():
        raise RuntimeError(f"Compilación fallida: {mercator_bin} no existe")
    return mercator_bin


def embedding_complete(embedding_dir: Path, graph_id: str) -> bool:
    coord = embedding_dir / f"{graph_id}.inf_coord"
    log = embedding_dir / f"{graph_id}.inf_log"
    if not coord.exists() or not log.exists():
        return False
    text = log.read_text(encoding="utf-8", errors="replace")
    return "===========================================================================================" in text


def run_dmercator_on_edge(edge_path: Path, *, dimension: int = DMERCATOR_DIMENSION, validation: bool = True) -> None:
    repo = resolve_mercator_repo()
    mercator_bin = build_mercator(repo)
    cmd = [str(mercator_bin), "-d", str(dimension)]
    if validation:
        cmd.append("-v")
    cmd.append(str(edge_path))
    subprocess.run(cmd, check=True, cwd=str(edge_path.parent))


def copy_outputs(src_dir: Path, graph_id: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffixes = (
        ".edge",
        ".inf_coord",
        ".inf_log",
        ".inf_pconn",
        ".inf_theta_density",
        ".inf_vprop",
        ".inf_vstat",
        ".obs_vstat",
        ".validation.pdf",
    )
    for suffix in suffixes:
        src = src_dir / f"{graph_id}{suffix}"
        if src.exists():
            shutil.copy2(src, dest_dir / src.name)


def run_all(*, force: bool = False, specs: Iterable = OTHER_HOURS) -> list[dict]:
    results = []
    for spec in specs:
        embedding_dir = spec.embedding_dir
        graph_id = spec.graph_id
        coord = embedding_dir / f"{graph_id}.inf_coord"

        if not force and embedding_complete(embedding_dir, graph_id):
            results.append(
                {
                    "manifestacion": spec.manifestacion,
                    "hour": spec.hour,
                    "status": "cached",
                    "embedding_dir": str(embedding_dir),
                }
            )
            continue

        embedding_dir.mkdir(parents=True, exist_ok=True)
        edge_in_embedding = embedding_dir / f"{graph_id}.edge"
        if not edge_in_embedding.exists():
            shutil.copy2(spec.input_edge, edge_in_embedding)

        run_dmercator_on_edge(edge_in_embedding)
        if not embedding_complete(embedding_dir, graph_id):
            raise RuntimeError(f"D-Mercator no completó para {spec.manifestacion} {graph_id}")

        header = (embedding_dir / f"{graph_id}.inf_coord").read_text(encoding="utf-8", errors="replace").splitlines()
        if "Inf.Theta" not in " ".join(header[:20]):
            raise RuntimeError(
                f"Embedding {graph_id} no parece S¹ (falta Inf.Theta en cabecera). "
                "Verifique que se ejecutó con -d 1."
            )

        results.append(
            {
                "manifestacion": spec.manifestacion,
                "hour": spec.hour,
                "status": "computed",
                "embedding_dir": str(embedding_dir),
            }
        )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ejecutar D-Mercator S¹ para Other TW")
    parser.add_argument("--force", action="store_true", help="Recalcular aunque exista embedding")
    args = parser.parse_args()
    out = run_all(force=args.force)
    for row in out:
        print(row)
