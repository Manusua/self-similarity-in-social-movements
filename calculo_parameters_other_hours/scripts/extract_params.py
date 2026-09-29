"""Extrae beta, mu y metadatos de archivos .inf_coord / .inf_log."""

from __future__ import annotations

from pathlib import Path


def _parse_header_value(line: str) -> tuple[str, str] | None:
    if not line.startswith("#") or ":" not in line:
        return None
    key, val = line.replace("#", "", 1).split(":", 1)
    key = key.strip().lstrip("-").strip().lower().replace(" ", "_")
    return key, val.strip()


def parse_inf_coord_metadata(folder: Path, graph_id: str) -> dict:
    coord = folder / f"{graph_id}.inf_coord"
    meta: dict = {"graph_id": graph_id, "folder": str(folder)}
    if not coord.exists():
        return meta
    with open(coord, "r", encoding="utf-8") as handle:
        for line in handle:
            parsed = _parse_header_value(line)
            if parsed:
                key, val = parsed
                meta[key] = val
    return meta


def parse_inf_log_metadata(folder: Path, graph_id: str) -> dict:
    log = folder / f"{graph_id}.inf_log"
    meta: dict = {}
    if not log.exists():
        return meta
    text = log.read_text(encoding="utf-8", errors="replace")
    meta["dmercator_complete"] = "===========================================================================================" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- beta:"):
            meta["log_beta"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("- mu:"):
            meta["log_mu"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Nb vertices:"):
            meta["log_n_vertices"] = stripped.split(":", 1)[1].strip()
    return meta


def extract_numeric(meta: dict, *keys: str) -> float | int | None:
    for key in keys:
        if key not in meta:
            continue
        raw = str(meta[key]).replace(",", "")
        try:
            if "." in raw or "e" in raw.lower():
                return float(raw)
            return int(raw)
        except ValueError:
            continue
    return None


def merge_params(folder: Path, graph_id: str) -> dict:
    coord_meta = parse_inf_coord_metadata(folder, graph_id)
    log_meta = parse_inf_log_metadata(folder, graph_id)
    return {
        "beta": extract_numeric(coord_meta, "beta", "log_beta"),
        "mu": extract_numeric(coord_meta, "mu", "log_mu"),
        "radius_s1": extract_numeric(coord_meta, "radius_s1", "radius_s^d"),
        "radius_h2": extract_numeric(coord_meta, "radius_h2", "radius_h^d+1", "radius_h^2"),
        "kappa_min": extract_numeric(coord_meta, "kappa_min"),
        "n_vertices_header": extract_numeric(coord_meta, "nb._vertices", "nb_vertices"),
        "dmercator_complete": log_meta.get("dmercator_complete", False),
        "edgelist_file": coord_meta.get("edgelist_file", ""),
    }
