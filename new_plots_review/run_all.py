#!/usr/bin/env python3
"""Run all reviewer-response analyses and write outputs under new_plots_review/."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd

REVIEW_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REVIEW_ROOT))
sys.path.insert(0, str(REVIEW_ROOT.parent))

from community_analysis import (  # noqa: E402
    compute_fixed_modularity_series,
    plot_community_comparison,
    window_entropy_diagnostic,
)
from config import MANIFESTATIONS, REPO_ROOT, ensure_dirs  # noqa: E402
from dmercator_utils import ensure_embeddings, select_all_nonctw  # noqa: E402
from generate_responses import build_response_matrix  # noqa: E402
from navigability_utils import (  # noqa: E402
    evaluate_embedding,
    parse_embedding_metadata,
    plot_navigability_comparison,
)
from null_models import (  # noqa: E402
    ct_epsilon_summary,
    epsilon_regression_check,
    null_model_analysis,
    plot_epsilon_collapse_panels,
    plot_null_model_summary,
)
from temporal_plots import (  # noqa: E402
    plot_fig2_combined,
    plot_fig2_panel,
    plot_smoothing_comparison,
    smoothing_comparison,
)


def main() -> None:
    ensure_dirs()
    print("=== Step 1: Fig. 2 (local time) ===")
    for m in MANIFESTATIONS:
        plot_fig2_panel(m)
        smoothing_comparison(m)
        plot_smoothing_comparison(m)
    plot_fig2_combined()

    print("=== Step 2: Community / entropy / smoothing ===")
    window_entropy_diagnostic()
    for m in MANIFESTATIONS:
        compute_fixed_modularity_series(m)
        plot_community_comparison(m)

    print("=== Step 3: Epsilon / null models ===")
    ct_epsilon_summary()
    epsilon_regression_check()
    for m in MANIFESTATIONS:
        plot_epsilon_collapse_panels(m)
        null_model_analysis(m, n_replicates=10)
        plot_null_model_summary(m)

    print("=== Step 4: Non-CTW selection & embeddings ===")
    nonctw = select_all_nonctw()
    inventory = ensure_embeddings(nonctw)

    print("=== Step 5: Navigability ===")
    nav_rows = []
    for _, row in inventory.iterrows():
        if str(row["source"]).startswith("failed"):
            continue
        folder = Path(row["embedding_dir"])
        graph_id = str(row["graph_id"])
        coord = folder / f"{graph_id}.inf_coord"
        if not coord.exists():
            continue
        try:
            stats = evaluate_embedding(
                folder,
                graph_id,
                label=f"{row['manifestacion']}_{row['window_type']}",
                window_type=row["window_type"].replace("CTW_sensitivity", "CTW"),
            )
            stats.update(parse_embedding_metadata(folder, graph_id))
            stats["manifestacion"] = row["manifestacion"]
            nav_rows.append(stats)
        except Exception as exc:  # noqa: BLE001
            print(f"Navigability failed for {row}: {exc}")
            traceback.print_exc()

    nav_df = pd.DataFrame(nav_rows)
    nav_df.to_csv(REVIEW_ROOT / "results" / "navigability_summary.csv", index=False)
    if not nav_df.empty:
        plot_navigability_comparison(nav_df)

    print("=== Step 6: Reviewer responses ===")
    build_response_matrix()

    print("=== Done ===")
    print(f"Outputs in {REVIEW_ROOT}")


if __name__ == "__main__":
    main()
