"""Fig. 2 — temporal panels (All modality) with dual TW conventions.

Alignment follows ``latex_CTW.ipynb`` exactly:
  - activity x = ``np.sort(unique)[::hw]``
  - mod/nest y arrays from measures JSON sorted by key, zipped **by position**
    onto that x (no merge on hour)
  - epsilon plotted on its native ``Network`` hour index (MA3 centered)
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from config import FIGURES_DIR, REPO_ROOT, RESULTS_DIR, TIMEZONE, ensure_dirs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORA_CRITICA = {"nat": 429624, "9n": 437037, "ch": 394717}
# Comparison hours from latex_self_similarity.ipynb (hours before CTW)
OTRA_HORA = {
    "nat": HORA_CRITICA["nat"] - 24,
    "9n": HORA_CRITICA["9n"] - 48,
    "ch": HORA_CRITICA["ch"] - 72,
}

MANIFESTATION_ORDER = ["nat", "9n", "ch"]
MANIFESTATION_TITLES = {
    "nat": "No al Tarifazo",
    "9n": "9N",
    "ch": "Charlie Hebdo",
}
COLUMN_TITLES = {
    "nat": "No al Tarifazo. CTW: 04/01/2019 21h",
    "9n": "9n. CTW: 09/11/2019 18h",
    "ch": "Charlie Hebdo. CTW: 11/01/15 14h",
}

PANEL_LABELS_COLUMN = {
    "nat": ["a", "d", "g"],
    "9n": ["b", "e", "h"],
    "ch": ["c", "f", "i"],
}

PANEL_LABELS_COMBINED = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

# Manual annotation windows from latex_CTW.ipynb (hour indices, inclusive)
ANNOTATION_WINDOWS_ORIGINAL: Dict[str, Dict[str, List[Tuple[int, int]]]] = {
    "nat": {
        "grey": [(429580, 429587), (429604, 429611), (429628, 429635), (429652, 429659)],
        "orange": [(429620, 429628)],
    },
    "9n": {
        "grey": [(436996, 437003), (437020, 437027), (437044, 437051), (437068, 437075)],
        "orange": [(437033, 437043), (437059, 437062), (437013, 437018)],
    },
    "ch": {
        "grey": [(394680, 394687), (394704, 394711), (394728, 394735), (394752, 394754)],
        "orange": [(394716, 394724)],
    },
}

PLOT_SLICE = {
    "nat": {"inicio": 45, "final": 140},
    "9n": {"inicio": 0, "final": 140},
    "ch": {"inicio": 206, "final": 293},
}

X_LABEL_YEAR = {"nat": 2019, "9n": 2019, "ch": 2015}
CITY_LABEL = {
    "nat": "Buenos Aires",
    "9n": "Buenos Aires",
    "ch": "Paris",
}

# Line styling
LW_DEFAULT = 2.6
LW_NESTEDNESS = 3.4
LW_CTW = 2.8
COLOR_MOD = "#1f77b4"
COLOR_NEST = "#2ca02c"
COLOR_USERS = "orange"
COLOR_HASHTAGS = "magenta"
COLOR_EPS = "black"
COLOR_CTW = "green"
COLOR_OTRA = "red"
COLOR_GREY_BAND = "gray"
COLOR_ORANGE_BAND = "orange"

MA_WINDOW = 3


@dataclass(frozen=True)
class VariantConfig:
    name: str
    label: str
    hour_windows: Dict[str, int]
    epsilon_paths: Dict[str, Path]


VARIANTS: Dict[str, VariantConfig] = {
    "notebook": VariantConfig(
        name="notebook",
        label="Notebook reproduction (NAT=1 h, 9N=2 h, CH=2 h)",
        hour_windows={"nat": 1, "9n": 2, "ch": 2},
        epsilon_paths={
            "nat": REPO_ROOT / "epsilon_sq/nat/results/Epsilon_values.txt",
            "9n": REPO_ROOT / "epsilon_sq/9n/results/Epsilon_values.txt",
            "ch": REPO_ROOT / "epsilon_sq/ch/results/Epsilon_values.txt",
        },
    ),
    "manuscript": VariantConfig(
        name="manuscript",
        label="Manuscript convention (NAT=2 h, 9N=1 h, CH=2 h)",
        hour_windows={"nat": 2, "9n": 1, "ch": 2},
        epsilon_paths={
            "nat": REPO_ROOT / "epsilon_sq/nat/w2/results/Epsilon_values.txt",
            "9n": REPO_ROOT / "epsilon_sq/9n/w1/results/Epsilon_values.txt",
            "ch": REPO_ROOT / "epsilon_sq/ch/results/Epsilon_values.txt",
        },
    ),
}


# ---------------------------------------------------------------------------
# Data loading — latex_CTW.ipynb conventions
# ---------------------------------------------------------------------------


def rolling_mean_centrada(series: pd.Series, window: int = MA_WINDOW) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rolling(window=window, center=True, min_periods=1).mean()


def get_num_users_hashtags(
    df: pd.DataFrame, hour_window: int = 1
) -> Tuple[List[int], List[int]]:
    """Match ``utils_data_vis.get_num_users_hashtags`` / latex_CTW.ipynb."""
    num_users: List[int] = []
    num_hashtags: List[int] = []
    for hour in np.sort(df["hour"].unique())[::hour_window]:
        conditions = df["hour"] == hour
        for step in range(1, hour_window):
            conditions |= df["hour"] == hour + step
        df_hour = df[conditions]
        num_users.append(len(df_hour["user"].unique()))
        num_hashtags.append(len(df_hour["hashtag"].unique()))
    return num_users, num_hashtags


def load_activity(manifestation: str, hour_window: int) -> pd.DataFrame:
    """Activity on chronological ``sorted(unique)[::hw]`` (latex_CTW x-axis)."""
    csv_path = REPO_ROOT / "datasets" / "csvs" / f"{manifestation}.csv"
    df = pd.read_csv(csv_path, sep=",")
    hours = np.sort(df["hour"].unique())[::hour_window].astype(int)
    users, hashtags = get_num_users_hashtags(df, hour_window=hour_window)
    return pd.DataFrame({"hour": hours, "users": users, "hashtags": hashtags})


def load_modularity_nestedness(manifestation: str, hour_window: int) -> pd.DataFrame:
    """Bipartite mod/nest like ``get_mod_nest_coefficient`` (sort keys, return arrays)."""
    path = REPO_ROOT / "measures" / f"{manifestation}_{hour_window}_b.json"
    with open(path) as handle:
        payload = json.load(handle)
    # Same ordering as utils_graphs.get_mod_nest_coefficient: zip keys → sort
    data = []
    for hour_key, metrics in payload.items():
        data.append(
            (
                str(int(float(hour_key))),
                metrics.get("modularity", np.nan),
                metrics.get("nestedness", np.nan),
            )
        )
    data.sort()  # lexicographic on string hour keys (same as latex helpers)
    hours = [int(h) for h, _, _ in data]
    mods = [m for _, m, _ in data]
    nests = [n for _, _, n in data]
    return pd.DataFrame(
        {"measure_hour": hours, "modularity": mods, "nestedness": nests}
    )


def load_epsilon(path: Path) -> pd.DataFrame:
    """Epsilon on its native ``Network`` hour index (latex_CTW panels g–i)."""
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for col in df.columns:
        low = col.lower()
        if "network" in low:
            rename[col] = "hour"
        elif "ccdf" in low:
            rename[col] = "epsilon_ccdf"
        elif "knn" in low:
            rename[col] = "epsilon_knn"
        elif "cco" in low:
            rename[col] = "epsilon_cco"
    df = df.rename(columns=rename)
    if "hour" not in df.columns:
        df = df.rename(columns={df.columns[0]: "hour"})
    df["hour"] = df["hour"].astype(int)
    df["epsilon_cco"] = pd.to_numeric(df["epsilon_cco"], errors="coerce")
    df["epsilon_cco_ma3"] = rolling_mean_centrada(df["epsilon_cco"])
    return df.sort_values("hour").reset_index(drop=True)


def build_manifestation_frame(variant: VariantConfig, manifestation: str) -> pd.DataFrame:
    """Activity + mod/nest with latex_CTW **positional** zip (no hour merge).

    ``plot_mod_nestedness(hour_x, mod_sort)`` in latex_CTW plots
    ``hour_x[i]`` vs ``mod_sort[i]`` after the same ``inicio/final`` slice, even
    when measure keys differ from ``sorted(unique)[::hw]`` for ``hw > 1``.
    """
    hw = variant.hour_windows[manifestation]
    activity = load_activity(manifestation, hw)
    modnest = load_modularity_nestedness(manifestation, hw)
    if len(activity) != len(modnest):
        raise ValueError(
            f"{manifestation} hw={hw}: activity length {len(activity)} != "
            f"mod/nest length {len(modnest)} (latex_CTW positional zip requires equal length)"
        )
    out = activity.copy()
    out["modularity"] = modnest["modularity"].to_numpy()
    out["nestedness"] = modnest["nestedness"].to_numpy()
    out["measure_hour"] = modnest["measure_hour"].to_numpy()
    return out.reset_index(drop=True)


def slice_frame(frame: pd.DataFrame, manifestation: str, hour_window: int) -> pd.DataFrame:
    sl = PLOT_SLICE[manifestation]
    start = int(sl["inicio"] / hour_window)
    end = int(sl["final"] / hour_window)
    return frame.iloc[start:end].copy().reset_index(drop=True)


def slice_epsilon_to_view(eps: pd.DataFrame, hour_min: float, hour_max: float) -> pd.DataFrame:
    """Restrict native epsilon series to the activity/mod shared x window."""
    mask = (eps["hour"] >= hour_min) & (eps["hour"] <= hour_max)
    return eps.loc[mask].copy().reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _band_width_hours(start: int, end: int) -> int:
    return end - start + 1


def _apply_bands(ax, manifestation: str) -> None:
    cfg = ANNOTATION_WINDOWS_ORIGINAL[manifestation]
    for start, end in cfg.get("grey", []):
        ax.axvspan(start, end, color=COLOR_GREY_BAND, alpha=0.35, linewidth=0, zorder=0)
    for start, end in cfg.get("orange", []):
        ax.axvspan(start, end, color=COLOR_ORANGE_BAND, alpha=0.40, linewidth=0, zorder=0)


def _apply_ctw(ax, manifestation: str) -> None:
    ax.axvline(
        HORA_CRITICA[manifestation],
        linestyle="-.",
        color=COLOR_CTW,
        linewidth=LW_CTW,
        zorder=4,
    )


def _apply_otra_hora(ax, manifestation: str) -> None:
    ax.axvline(
        OTRA_HORA[manifestation],
        linestyle="--",
        color=COLOR_OTRA,
        linewidth=LW_CTW,
        zorder=4,
    )


def _format_xaxis(ax, manifestation: str, show_label: bool = True) -> None:
    tz_name = TIMEZONE[manifestation]
    city = CITY_LABEL[manifestation]
    year = X_LABEL_YEAR[manifestation]

    def _local_tick(x_val, _pos):
        local = dt.datetime.fromtimestamp(int(x_val) * 3600, tz=ZoneInfo("UTC")).astimezone(
            ZoneInfo(tz_name)
        )
        return local.strftime("%d-%b %Hh")

    ax.xaxis.set_major_formatter(plt.FuncFormatter(_local_tick))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.xaxis.set_minor_locator(MaxNLocator(nbins=40, integer=True))
    ax.tick_params(axis="x", which="major", labelsize=20, rotation=75)
    ax.tick_params(axis="x", which="minor", length=3)
    if show_label:
        ax.set_xlabel(f"Date and hour (local, {city}, {year})", fontsize=22)


def _style_axis(ax, panel_label: Optional[str] = None) -> None:
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.margins(x=0.01)
    ax.tick_params(axis="both", which="major", labelsize=20)
    if panel_label:
        ax.text(
            0.01,
            0.98,
            panel_label,
            transform=ax.transAxes,
            fontsize=24,
            fontweight="bold",
            va="top",
            ha="left",
        )


def _plot_activity(ax, sub: pd.DataFrame, panel_label: Optional[str]) -> None:
    x = sub["hour"].to_numpy()
    ax.plot(x, sub["users"], color=COLOR_USERS, linewidth=LW_DEFAULT, label="Unique users")
    ax.plot(
        x,
        sub["hashtags"],
        color=COLOR_HASHTAGS,
        linewidth=LW_DEFAULT,
        label="Unique hashtags",
    )
    ax.set_ylabel("Activity count", fontsize=20)
    _style_axis(ax, panel_label)


def _plot_mod_nest(ax, sub: pd.DataFrame, panel_label: Optional[str]) -> plt.Axes:
    # latex_CTW: plot(hour_x[slice], mod_sort[slice]) — positional, not key-merge
    x = sub["hour"].to_numpy()
    mod = sub["modularity"].to_numpy(dtype=float)
    nest = sub["nestedness"].to_numpy(dtype=float)

    ax.plot(x, mod, color=COLOR_MOD, linewidth=LW_DEFAULT, label="Modularity")
    ax.set_ylabel("Modularity", color=COLOR_MOD, fontsize=20)
    ax.tick_params(axis="y", labelcolor=COLOR_MOD)
    mod_lo, mod_hi = np.nanmin(mod), np.nanmax(mod)
    mod_pad = max((mod_hi - mod_lo) * 0.08, 0.01)
    ax.set_ylim(mod_lo - mod_pad, mod_hi + mod_pad)

    ax2 = ax.twinx()
    ax2.plot(x, nest, color=COLOR_NEST, linewidth=LW_NESTEDNESS, label="Nestedness")
    ax2.set_ylabel("Nestedness", color=COLOR_NEST, fontsize=20)
    ax2.tick_params(axis="y", labelcolor=COLOR_NEST)
    nest_lo, nest_hi = np.nanmin(nest), np.nanmax(nest)
    nest_pad = max((nest_hi - nest_lo) * 0.15, 0.002)
    ax2.set_ylim(nest_lo - nest_pad, nest_hi + nest_pad)
    _style_axis(ax, panel_label)
    return ax2


def _plot_epsilon_panel(ax, eps: pd.DataFrame, panel_label: Optional[str]) -> None:
    # latex_CTW: plot(df[Network], rolling_mean_centrada(eps_cco)) on native hours
    x = eps["hour"].to_numpy()
    ax.plot(
        x,
        eps["epsilon_cco_ma3"],
        color=COLOR_EPS,
        linewidth=LW_DEFAULT,
        label=rf"$\epsilon^2_{{\mathrm{{cco}}}}$ (MA{MA_WINDOW})",
    )
    ax.set_ylabel(r"$\epsilon^2_{\mathrm{cco}}$", fontsize=20)
    _style_axis(ax, panel_label)


def plot_fig2_column(
    variant: VariantConfig,
    manifestation: str,
    save: bool = True,
    show_legends: bool = None,
) -> Path:
    """Single 3×1 column (All modality) for one manifestation."""
    ensure_dirs()
    if show_legends is None:
        show_legends = manifestation == "ch"

    hw = variant.hour_windows[manifestation]
    frame = build_manifestation_frame(variant, manifestation)
    sub = slice_frame(frame, manifestation, hw)
    eps = load_epsilon(variant.epsilon_paths[manifestation])
    eps_view = slice_epsilon_to_view(eps, float(sub["hour"].min()), float(sub["hour"].max()))
    panel_labels = PANEL_LABELS_COLUMN[manifestation]

    plt.style.use("seaborn-v0_8-colorblind")
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 12), dpi=120)

    _plot_activity(axes[0], sub, panel_labels[0])
    nest_ax = _plot_mod_nest(axes[1], sub, panel_labels[1])
    _plot_epsilon_panel(axes[2], eps_view, panel_labels[2])

    for ax in axes:
        _apply_ctw(ax, manifestation)
        _apply_otra_hora(ax, manifestation)
        _apply_bands(ax, manifestation)

    # Keep shared view on the activity/mod slice (latex sharex behaviour)
    x0, x1 = float(sub["hour"].min()), float(sub["hour"].max())
    pad = 0.01 * (x1 - x0)
    for ax in axes:
        ax.set_xlim(x0 - pad, x1 + pad)

    _format_xaxis(axes[2], manifestation, show_label=True)

    if show_legends:
        axes[0].legend(loc=(0.66, 0.65), fontsize=16)
        axes[1].legend(loc=(0.73, 0.35), fontsize=16)
        nest_ax.legend(loc=(0.73, 0.15), fontsize=16)
        axes[2].legend(loc=(0.75, 0.35), fontsize=16)

    fig.suptitle(
        f"{MANIFESTATION_TITLES[manifestation]} — {variant.label}",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    out = FIGURES_DIR / f"fig2_{manifestation}_{variant.name}_all.png"
    if save:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fig2_combined(variant: VariantConfig, save: bool = True) -> Path:
    """Combined 3×3 figure with panels (a–i)."""
    ensure_dirs()
    plt.style.use("seaborn-v0_8-colorblind")
    fig, axes = plt.subplots(3, 3, figsize=(36, 12), dpi=120, sharex=False)

    for col, manifestation in enumerate(MANIFESTATION_ORDER):
        hw = variant.hour_windows[manifestation]
        frame = build_manifestation_frame(variant, manifestation)
        sub = slice_frame(frame, manifestation, hw)
        x = sub["hour"].to_numpy()
        eps = load_epsilon(variant.epsilon_paths[manifestation])
        eps_view = slice_epsilon_to_view(eps, float(sub["hour"].min()), float(sub["hour"].max()))

        ax_a, ax_b, ax_c = axes[0, col], axes[1, col], axes[2, col]
        labels = PANEL_LABELS_COLUMN[manifestation]

        # Row 0 — activity
        ax_a.plot(
            x,
            sub["users"],
            color=COLOR_USERS,
            linewidth=LW_DEFAULT,
            label="Unique users",
        )
        ax_a.plot(
            x,
            sub["hashtags"],
            color=COLOR_HASHTAGS,
            linewidth=LW_DEFAULT,
            label="Unique hashtags",
        )
        ax_a.set_title(COLUMN_TITLES[manifestation], fontsize=24)
        _style_axis(ax_a, labels[0])
        ax_a.tick_params(axis="x", labelbottom=False)
        _apply_ctw(ax_a, manifestation)
        _apply_otra_hora(ax_a, manifestation)
        _apply_bands(ax_a, manifestation)
        if col == 0:
            ax_a.legend(loc="upper right", fontsize=21, framealpha=0.9)

        # Row 1 — modularity / nestedness (positional zip, latex_CTW)
        mod = sub["modularity"].to_numpy(dtype=float)
        nest = sub["nestedness"].to_numpy(dtype=float)
        ax_b.plot(x, mod, color=COLOR_MOD, linewidth=LW_DEFAULT)
        ax_b.set_ylabel("Modularity" if col == 0 else "", fontsize=18, color=COLOR_MOD)
        ax_b.tick_params(axis="y", labelcolor=COLOR_MOD)
        ax_b.tick_params(axis="x", labelbottom=False)
        mod_lo, mod_hi = np.nanmin(mod), np.nanmax(mod)
        mod_pad = max((mod_hi - mod_lo) * 0.08, 0.01)
        ax_b.set_ylim(mod_lo - mod_pad, mod_hi + mod_pad)
        ax_b_t = ax_b.twinx()
        ax_b_t.plot(x, nest, color=COLOR_NEST, linewidth=LW_NESTEDNESS)
        ax_b_t.set_ylabel("Nestedness" if col == 2 else "", fontsize=18, color=COLOR_NEST)
        ax_b_t.tick_params(axis="y", labelcolor=COLOR_NEST)
        nest_lo, nest_hi = np.nanmin(nest), np.nanmax(nest)
        nest_pad = max((nest_hi - nest_lo) * 0.15, 0.002)
        ax_b_t.set_ylim(nest_lo - nest_pad, nest_hi + nest_pad)
        _style_axis(ax_b, labels[1])
        ax_b.tick_params(axis="x", labelbottom=False)
        _apply_ctw(ax_b, manifestation)
        _apply_otra_hora(ax_b, manifestation)
        _apply_bands(ax_b, manifestation)

        # Row 2 — epsilon on native Network hours
        ax_c.plot(
            eps_view["hour"],
            eps_view["epsilon_cco_ma3"],
            color=COLOR_EPS,
            linewidth=LW_DEFAULT,
        )
        ax_c.set_ylabel(r"$\epsilon^2_{\mathrm{cco}}$" if col == 0 else "", fontsize=18)
        _style_axis(ax_c, labels[2])
        _apply_ctw(ax_c, manifestation)
        _apply_otra_hora(ax_c, manifestation)
        _apply_bands(ax_c, manifestation)
        _format_xaxis(ax_c, manifestation, show_label=True)

        x0, x1 = float(sub["hour"].min()), float(sub["hour"].max())
        pad = 0.01 * (x1 - x0)
        for ax in (ax_a, ax_b, ax_c):
            ax.set_xlim(x0 - pad, x1 + pad)

    fig.tight_layout()
    out = FIGURES_DIR / f"fig2_combined_{variant.name}_all.png"
    if save:
        fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_variant_figures(variant_name: str) -> List[Path]:
    variant = VARIANTS[variant_name]
    paths = []
    for m in MANIFESTATION_ORDER:
        paths.append(plot_fig2_column(variant, m))
    paths.append(plot_fig2_combined(variant))
    return paths


# ---------------------------------------------------------------------------
# Diagnostics & documentation helpers
# ---------------------------------------------------------------------------


def band_diagnostics(variant: VariantConfig) -> pd.DataFrame:
    rows = []
    for manifestation in MANIFESTATION_ORDER:
        hw = variant.hour_windows[manifestation]
        eps = load_epsilon(variant.epsilon_paths[manifestation])
        ctw = HORA_CRITICA[manifestation]
        for band_type, bands in ANNOTATION_WINDOWS_ORIGINAL[manifestation].items():
            for start, end in bands:
                mask = (eps["hour"] >= start) & (eps["hour"] <= end)
                sub = eps.loc[mask].copy()
                if sub.empty:
                    rows.append(
                        {
                            "variant": variant.name,
                            "manifestation": manifestation,
                            "band_type": band_type,
                            "start_hour": start,
                            "end_hour": end,
                            "band_width_hours": _band_width_hours(start, end),
                            "hour_window_h": hw,
                            "n_tw_in_band": 0,
                            "min_hour": np.nan,
                            "min_epsilon2_cco_ma3": np.nan,
                            "ctw_hour": ctw,
                            "ctw_in_band": start <= ctw <= end,
                            "distance_to_ctw_h": abs(ctw - start),
                        }
                    )
                    continue
                idx = sub["epsilon_cco_ma3"].idxmin()
                min_row = sub.loc[idx]
                rows.append(
                    {
                        "variant": variant.name,
                        "manifestation": manifestation,
                        "band_type": band_type,
                        "start_hour": start,
                        "end_hour": end,
                        "band_width_hours": _band_width_hours(start, end),
                        "hour_window_h": hw,
                        "n_tw_in_band": len(sub),
                        "min_hour": int(min_row["hour"]),
                        "min_epsilon2_cco_ma3": float(min_row["epsilon_cco_ma3"]),
                        "ctw_hour": ctw,
                        "ctw_in_band": start <= ctw <= end,
                        "distance_to_ctw_h": abs(int(min_row["hour"]) - ctw),
                    }
                )
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / f"fig2_band_diagnostics_{variant.name}.csv"
    df.to_csv(out, index=False)
    return df


def window_width_summary() -> pd.DataFrame:
    rows = []
    for variant_name, variant in VARIANTS.items():
        for manifestation in MANIFESTATION_ORDER:
            hw = variant.hour_windows[manifestation]
            rows.append(
                {
                    "variant": variant_name,
                    "manifestation": manifestation,
                    "temporal_window_width_h": hw,
                    "major_tick_spacing_note": "MaxNLocator(nbins≈12) on hour index — label spacing only",
                    "candidate_band_widths_h": "; ".join(
                        f"{s}-{e} ({_band_width_hours(s, e)} h)"
                        for s, e in ANNOTATION_WINDOWS_ORIGINAL[manifestation]["orange"]
                    ),
                    "grey_band_widths_h": "; ".join(
                        f"{s}-{e} ({_band_width_hours(s, e)} h)"
                        for s, e in ANNOTATION_WINDOWS_ORIGINAL[manifestation]["grey"]
                    ),
                }
            )
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "fig2_window_width_summary.csv"
    df.to_csv(out, index=False)
    return df


FIG2_CAPTION = """
**FIG. 2: Detecting the Critical Temporal Window (CTW).**
(a–c) Number of unique users (orange) and unique hashtags (magenta) across temporal windows (TWs).
(d–f) Evolution of modularity (blue, left axis) and nestedness (green, right axis) in bipartite user–hashtag networks.
(g–i) Temporal evolution of $\\epsilon^2_{\\mathrm{cco}}(k)$, the clustering component of the DTR collapse-quality metric,
computed as a centered moving average over $N=3$ consecutive TWs.
Yellow/orange bands in all panels highlight manually selected candidate intervals associated with local minima of
$\\epsilon^2_{\\mathrm{cco}}(k)$, engagement peaks, and modular-to-nested transitions; grey bands mark low-activity
periods (01:00–08:00 local time). The identified CTW for each movement is marked by a **green** dash-dotted vertical line.
For No al Tarifazo (left) and Charlie Hebdo (right), a single clear CTW emerges. For 9N (centre), three candidate
segments show low $\\epsilon^2_{\\mathrm{cco}}(k)$ values, but only the second exhibits a clear modular-to-nested
transition in the bipartite networks, unambiguously identifying the CTW. All times are given in UTC.
""".strip()


METHODOLOGY_NOTE = """
### Methodological notes (Fig. 2)

1. **Candidate yellow/orange bands** were **not** generated by an automatic argmin algorithm. They are fixed hour-index
   ranges copied from `latex_CTW.ipynb`, selected by joint visual inspection of (i) local minima of the MA(3) series of
   $\\epsilon^2_{\\mathrm{cco}}$, (ii) activity peaks, and (iii) modular-to-nested transitions in the bipartite networks.
2. **Minimum identification:** for each band, the diagnostic table reports the hour and value of the minimum of the
   centered MA(3) series **within that band** (not a global minimum over the full timeline).
3. **Temporal window (TW) width:** each point aggregates hashtags posted within a TW of $W$ hours ($W=1$ or $2$ depending
   on the variant; see the window-width summary table). This is the **elementary sampling unit** of the figure.
4. **Band width vs. TW width vs. tick spacing:** a candidate band may span several TWs (e.g. 9 h = nine 1 h TWs or five
   2 h TWs). Major x-axis labels are spaced for readability (typically every ~8–10 h on the hour index) and **do not**
   indicate the aggregation width.
5. **Two TW conventions:** we provide both the notebook reproduction (NAT 1 h, 9N 2 h, CH 2 h) and the manuscript
   convention (NAT 2 h, 9N 1 h, CH 2 h) to make the internal inconsistency explicit without altering the manually selected bands.
6. **Series alignment (as in `latex_CTW.ipynb`):** activity uses chronological ``sorted(unique)[::W]`` as x;
   modularity/nestedness arrays from ``measures`` are sorted by their own keys and plotted **by positional index**
   against that x (no key merge). $\\epsilon^2_{\\mathrm{cco}}$ is plotted on the native ``Network`` hour column of
   the epsilon file.
""".strip()
