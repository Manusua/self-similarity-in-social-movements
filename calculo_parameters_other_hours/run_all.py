#!/usr/bin/env python3
"""Pipeline completo: inputs → D-Mercator S¹ → navegabilidad → CSV."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from config import DIAM, OTHER_HOURS, PAIR_STEP, RESULTS_DIR  # noqa: E402
from compute_navigability import evaluate_embedding_folder  # noqa: E402
from dmercator_runner import run_all as run_dmercator_all  # noqa: E402
from extract_params import merge_params  # noqa: E402
from prepare_inputs import prepare_all  # noqa: E402
from validate import checks_to_dataframe, run_checks  # noqa: E402


def build_summary_md(df: pd.DataFrame, checks: pd.DataFrame) -> str:
    lines = [
        "# Parámetros Other TW — resumen",
        "",
        f"Generado: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Definiciones",
        "",
        "- **p_s**: fracción de pares ordenados con greedy routing exitoso (`success_ratio`).",
        "- **avg_topo_stretch**: media de `hops_greedy / hops_BFS` sobre pares exitosos.",
        "- **avg_stretch**: media de `suma_distancias_greedy / distancia_hiperbólica_directa` sobre pares exitosos.",
        "- **beta, mu**: parámetros del modelo \\(\\mathbb{S}^1\\) inferidos por D-Mercator (dimensión 1).",
        "",
        "## Resultados",
        "",
    ]
    for _, row in df.iterrows():
        lines.extend(
            [
                f"### {row['manifestacion'].upper()} — hora {int(row['hour'])} ({row['plot_label']})",
                "",
                f"- Red: {int(row['n_nodes'])} nodos, {int(row['n_edges'])} aristas (umbral {int(row['threshold'])})",
                f"- β = {row['beta'] if row['beta'] is None else f'{float(row['beta']):.4g}'}, "
                f"μ = {row['mu'] if row['mu'] is None else f'{float(row['mu']):.6g}'}",
                f"- p_s = {float(row['p_s']):.4f}, avg_topo_stretch = {float(row['avg_topo_stretch']):.4f}, "
                f"avg_stretch = {float(row['avg_stretch']):.4f}",
                f"- Embedding: `{row['embedding_dir']}`",
                "",
            ]
        )

    failed = checks[~checks["pass"]]
    lines.append("## Validaciones")
    lines.append("")
    if failed.empty:
        lines.append("Todas las comprobaciones pasaron.")
    else:
        lines.append(f"**{len(failed)} comprobaciones fallidas:**")
        lines.append("")
        for _, r in failed.iterrows():
            lines.append(
                f"- {r['manifestacion']} {int(r['hour'])} — {r['check']}: "
                f"esperado {r['expected']}, observado {r['observed']}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcular parámetros Other TW")
    parser.add_argument("--force-dmercator", action="store_true", help="Recalcular embeddings D-Mercator")
    parser.add_argument("--skip-dmercator", action="store_true", help="Omitir D-Mercator (solo navegabilidad)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 Preparando inputs...")
    manifest = prepare_all()

    if not args.skip_dmercator:
        print("2/4 Ejecutando D-Mercator S¹...")
        dm_status = run_dmercator_all(force=args.force_dmercator)
        pd.DataFrame(dm_status).to_csv(RESULTS_DIR / "dmercator_status.csv", index=False)
    else:
        print("2/4 D-Mercator omitido (--skip-dmercator)")

    print("3/4 Calculando navegabilidad...")
    rows = []
    check_lists = []
    for spec in OTHER_HOURS:
        manifest_row = manifest[manifest["manifestacion"] == spec.manifestacion].iloc[0].to_dict()
        params = merge_params(spec.embedding_dir, spec.graph_id)
        nav = evaluate_embedding_folder(
            spec.embedding_dir,
            spec.graph_id,
            pair_step=PAIR_STEP,
            diam=DIAM,
        )
        row = {
            "manifestacion": spec.manifestacion,
            "hour": spec.hour,
            "hour_window": spec.hour_window,
            "threshold": spec.threshold,
            "ctw_offset_h": spec.ctw_offset_h,
            "plot_label": spec.plot_label,
            "input_edge": str(spec.input_edge),
            "embedding_dir": str(spec.embedding_dir),
            "n_nodes": nav["n_nodes"],
            "n_edges": manifest_row["n_edges"],
            "beta": params.get("beta"),
            "mu": params.get("mu"),
            "radius_s1": params.get("radius_s1"),
            "radius_h2": params.get("radius_h2"),
            "kappa_min": params.get("kappa_min"),
            "p_s": nav["p_s"],
            "n_pairs": nav["n_pairs"],
            "n_successes": nav["n_successes"],
            "n_failures": nav["n_failures"],
            "avg_topo_stretch": nav["avg_topo_stretch"],
            "std_topo_stretch": nav["std_topo_stretch"],
            "avg_stretch": nav["avg_stretch"],
            "std_stretch": nav["std_stretch"],
            "coord_format": nav["coord_format"],
            "pair_step": nav["pair_step"],
            "dmercator_complete": params.get("dmercator_complete"),
        }
        rows.append(row)
        check_lists.append(run_checks(spec, nav, manifest_row))

    df = pd.DataFrame(rows)
    checks_df = checks_to_dataframe(check_lists)

    print("4/4 Guardando resultados...")
    df.to_csv(RESULTS_DIR / "parameters_other_hours.csv", index=False)
    checks_df.to_csv(RESULTS_DIR / "validation_checks.csv", index=False)
    (RESULTS_DIR / "summary.md").write_text(build_summary_md(df, checks_df), encoding="utf-8")

    print(
        df[["manifestacion", "hour", "beta", "mu", "p_s", "avg_topo_stretch", "avg_stretch"]].to_string(
            index=False
        )
    )
    print(f"\nResultados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
