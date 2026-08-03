"""Temporal Fig. 2 and smoothing analyses."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from scipy.signal import savgol_filter
except ImportError:  # pragma: no cover
    def savgol_filter(values, window_length, polyorder):  # type: ignore
        import numpy as np
        series = np.asarray(values, dtype=float)
        if len(series) < window_length:
            return series
        half = window_length // 2
        out = series.copy()
        for i in range(len(series)):
            lo = max(0, i - half)
            hi = min(len(series), i + half + 1)
            out[i] = np.mean(series[lo:hi])
        return out
from zoneinfo import ZoneInfo

from config import (
    ANNOTATION_WINDOWS,
    FIGURES_DIR,
    HOUR_WINDOW,
    MANIFESTATIONS,
    PLOT_SLICE,
    REPO_ROOT,
    RESULTS_DIR,
    TIMEZONE,
    get_manifestation_config,
)


def hour_to_local_label(hour: int, tz_name: str) -> str:
    dt = datetime.fromtimestamp(int(hour) * 3600, tz=ZoneInfo("UTC")).astimezone(
        ZoneInfo(tz_name)
    )
    return dt.strftime("%Y-%m-%d %H:%M")


def build_local_axis(hours: Iterable[int], tz_name: str) -> List[str]:
    return [hour_to_local_label(h, tz_name) for h in hours]


def rolling_mean(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def rolling_median(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1).median()


def savgol_smooth(series: pd.Series, window: int = 5, polyorder: int = 2) -> pd.Series:
    values = series.to_numpy(dtype=float)
    if len(values) < window:
        return pd.Series(values, index=series.index)
    if window % 2 == 0:
        window += 1
    smoothed = savgol_filter(values, window_length=window, polyorder=polyorder)
    return pd.Series(smoothed, index=series.index)


def load_epsilon_series(manifestation: str) -> pd.DataFrame:
    path = REPO_ROOT / "epsilon_sq" / manifestation / "results" / "Epsilon_values.txt"
    df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    if "Network" in df.columns:
        df = df.rename(columns={"Network": "hour"})
    else:
        df = df.rename(columns={df.columns[0]: "hour"})
    rename = {}
    for col in df.columns:
        low = col.lower()
        if "ccdf" in low:
            rename[col] = "epsilon_ccdf"
        elif "knn" in low:
            rename[col] = "epsilon_knn"
        elif "cco" in low:
            rename[col] = "epsilon_cco"
    df = df.rename(columns=rename)
    df["hour"] = df["hour"].astype(int)
    return df.sort_values("hour")


def load_modularity_nestedness(manifestation: str) -> pd.DataFrame:
    import json

    hw = HOUR_WINDOW[manifestation]
    path = REPO_ROOT / "measures" / f"{manifestation}_{hw}_h.json"
    with open(path) as handle:
        payload = json.load(handle)
    rows = []
    for hour, metrics in payload.items():
        rows.append(
            {
                "hour": int(float(hour)),
                "modularity": metrics.get("modularity", np.nan),
                "nestedness": metrics.get("nestedness", np.nan),
            }
        )
    return pd.DataFrame(rows).sort_values("hour")


def get_num_users_hashtags(df: pd.DataFrame, hour_window: int = 1):
    import numpy as np

    num_users = []
    num_hashtags = []
    for hour in np.sort(df["hour"].unique())[::hour_window]:
        conditions = df["hour"] == hour
        for step in range(1, hour_window):
            conditions |= df["hour"] == hour + step
        df_hour = df[conditions]
        num_users.append(len(df_hour["user"].unique()))
        num_hashtags.append(len(df_hour["hashtag"].unique()))
    return num_users, num_hashtags


def load_activity_series(manifestation: str) -> pd.DataFrame:
    import numpy as np

    cfg = get_manifestation_config(manifestation)
    df = pd.read_csv(REPO_ROOT / "datasets" / "csvs" / f"{manifestation}.csv")
    users, hashtags = get_num_users_hashtags(df, hour_window=cfg.hour_window)
    hours = np.sort(df["hour"].unique())[:: cfg.hour_window]
    return pd.DataFrame(
        {"hour": hours.astype(int), "users": users, "hashtags": hashtags}
    )


def merge_temporal_frame(manifestation: str) -> pd.DataFrame:
    cfg = get_manifestation_config(manifestation)
    eps = load_epsilon_series(manifestation)
    modnest = load_modularity_nestedness(manifestation)
    activity = load_activity_series(manifestation)
    merged = activity.merge(modnest, on="hour", how="left").merge(eps, on="hour", how="left")
    merged["hour_rel"] = (merged["hour"] - cfg.critical_hour) / cfg.hour_window
    merged["local_time"] = merged["hour"].map(lambda h: hour_to_local_label(h, cfg.timezone))
    return merged


def _apply_window_shading(ax, manifestation: str, x_values: List[str], hours: np.ndarray) -> None:
    cfg = ANNOTATION_WINDOWS.get(manifestation, {})
    hour_to_x = {int(h): x for h, x in zip(hours, x_values)}
    for start, end in cfg.get("grey", []):
        xs = [hour_to_x.get(h) for h in range(start, end + 1) if h in hour_to_x]
        if len(xs) >= 2:
            ax.axvspan(xs[0], xs[-1], color="grey", alpha=0.25, linewidth=0)
    for start, end in cfg.get("orange", []):
        xs = [hour_to_x.get(h) for h in range(start, end + 1) if h in hour_to_x]
        if len(xs) >= 2:
            ax.axvspan(xs[0], xs[-1], color="orange", alpha=0.35, linewidth=0)


def plot_fig2_panel(manifestation: str, save: bool = True) -> Path:
    cfg = get_manifestation_config(manifestation)
    frame = merge_temporal_frame(manifestation)
    sl = PLOT_SLICE[manifestation]
    start_idx = int(sl["inicio"] / cfg.hour_window)
    end_idx = int(sl["final"] / cfg.hour_window)
    sub = frame.iloc[start_idx:end_idx].copy()
    hours = sub["hour"].to_numpy()
    x_labels = build_local_axis(hours, cfg.timezone)
    crit_label = hour_to_local_label(cfg.critical_hour, cfg.timezone)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    line_kw = {"linewidth": 1.8}

    ax0 = axes[0]
    ax0.plot(x_labels, sub["users"], color="orange", label="Unique users", **line_kw)
    ax0.plot(x_labels, sub["hashtags"], color="magenta", label="Unique hashtags", **line_kw)
    ax0.set_ylabel("Activity count")
    ax0.axvline(crit_label, color="black", linestyle="-.", linewidth=2.0)
    _apply_window_shading(ax0, manifestation, x_labels, hours)

    ax1 = axes[1]
    mod = sub["modularity"].to_numpy()
    nest = sub["nestedness"].to_numpy()
    ax1.plot(x_labels, mod, color="#1f77b4", **line_kw)
    ax1.set_ylabel("Modularity", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(min(mod) * 0.95, max(mod) * 1.02)

    ax1b = ax1.twinx()
    ax1b.plot(x_labels, nest, color="#2ca02c", **line_kw)
    ax1b.set_ylabel("Nestedness", color="#2ca02c")
    ax1b.tick_params(axis="y", labelcolor="#2ca02c")
    ax1b.set_ylim(min(nest) * 0.95, max(nest) * 1.02)
    ax1.axvline(crit_label, color="black", linestyle="-.", linewidth=2.0)
    _apply_window_shading(ax1, manifestation, x_labels, hours)

    ax2 = axes[2]
    eps = rolling_mean(sub["epsilon_cco"], window=3)
    ax2.plot(x_labels, eps, color="black", **line_kw)
    ax2.set_ylabel(r"$\epsilon^2_{\mathrm{cco}}$")
    ax2.set_xlabel(f"Date and hour ({cfg.timezone.replace('_', ' ')})")
    ax2.axvline(crit_label, color="black", linestyle="-.", linewidth=2.0)
    _apply_window_shading(ax2, manifestation, x_labels, hours)
    ax2.tick_params(axis="x", rotation=55)

    fig.suptitle(f"{manifestation.upper()} — temporal evolution (local time)", fontsize=14)
    fig.tight_layout()
    out = FIGURES_DIR / f"fig2_{manifestation}_local_time.png"
    if save:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_fig2_combined(save: bool = True) -> Path:
    paths = [plot_fig2_panel(m, save=False) for m in MANIFESTATIONS]
    del paths
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    panel_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]

    for col, manifestation in enumerate(MANIFESTATIONS):
        cfg = get_manifestation_config(manifestation)
        frame = merge_temporal_frame(manifestation)
        sl = PLOT_SLICE[manifestation]
        start_idx = int(sl["inicio"] / cfg.hour_window)
        end_idx = int(sl["final"] / cfg.hour_window)
        sub = frame.iloc[start_idx:end_idx].copy()
        hours = sub["hour"].to_numpy()
        x_labels = build_local_axis(hours, cfg.timezone)
        crit_label = hour_to_local_label(cfg.critical_hour, cfg.timezone)
        line_kw = {"linewidth": 1.8}

        ax_a = axes[0, col]
        ax_a.plot(x_labels, sub["users"], color="orange", **line_kw)
        ax_a.plot(x_labels, sub["hashtags"], color="magenta", **line_kw)
        ax_a.set_ylabel("Activity" if col == 0 else "")
        ax_a.axvline(crit_label, color="black", linestyle="-.", linewidth=2.0)
        _apply_window_shading(ax_a, manifestation, x_labels, hours)
        ax_a.set_title(manifestation.upper())

        ax_b = axes[1, col]
        mod = sub["modularity"].to_numpy()
        nest = sub["nestedness"].to_numpy()
        ax_b.plot(x_labels, mod, color="#1f77b4", **line_kw)
        ax_b.set_ylabel("Modularity" if col == 0 else "")
        ax_b.set_ylim(min(mod) * 0.95, max(mod) * 1.02)
        ax_b.axvline(crit_label, color="black", linestyle="-.", linewidth=2.0)
        _apply_window_shading(ax_b, manifestation, x_labels, hours)
        ax_b_t = ax_b.twinx()
        ax_b_t.plot(x_labels, nest, color="#2ca02c", **line_kw)
        ax_b_t.set_ylabel("Nestedness" if col == 2 else "")
        ax_b_t.set_ylim(min(nest) * 0.95, max(nest) * 1.02)

        ax_c = axes[2, col]
        eps = rolling_mean(sub["epsilon_cco"], window=3)
        ax_c.plot(x_labels, eps, color="black", **line_kw)
        ax_c.set_ylabel(r"$\epsilon^2_{\mathrm{cco}}$" if col == 0 else "")
        ax_c.set_xlabel(f"Local time ({cfg.timezone.split('/')[-1]})")
        ax_c.axvline(crit_label, color="black", linestyle="-.", linewidth=2.0)
        _apply_window_shading(ax_c, manifestation, x_labels, hours)
        ax_c.tick_params(axis="x", rotation=55)

    for idx, ax in enumerate(fig.axes[:9]):
        ax.text(0.02, 0.95, f"({panel_labels[idx]})", transform=ax.transAxes, fontsize=12, va="top")

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_combined_local_time.png"
    if save:
        fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def smoothing_comparison(manifestation: str) -> pd.DataFrame:
    frame = load_epsilon_series(manifestation)
    series = frame["epsilon_cco"]
    methods = {
        "raw": series,
        "rolling_mean_3": rolling_mean(series, 3),
        "rolling_median_3": rolling_median(series, 3),
        "savgol_5_2": savgol_smooth(series, window=5, polyorder=2),
    }
    out = pd.DataFrame({"hour": frame["hour"]})
    for name, values in methods.items():
        out[name] = values.to_numpy()
    out.to_csv(RESULTS_DIR / f"smoothing_{manifestation}.csv", index=False)

    corr = pd.DataFrame(
        {
            "method_a": [],
            "method_b": [],
            "pearson_r": [],
        }
    )
    names = list(methods)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows.append(
                {
                    "method_a": a,
                    "method_b": b,
                    "pearson_r": float(methods[a].corr(methods[b])),
                }
            )
    corr = pd.DataFrame(rows)
    corr.to_csv(RESULTS_DIR / f"smoothing_correlations_{manifestation}.csv", index=False)
    return out


def plot_smoothing_comparison(manifestation: str) -> Path:
    df = pd.read_csv(RESULTS_DIR / f"smoothing_{manifestation}.csv")
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, style in {
        "raw": ("grey", 1.0),
        "rolling_mean_3": ("black", 1.8),
        "rolling_median_3": ("#1f77b4", 1.5),
        "savgol_5_2": ("#d62728", 1.5),
    }.items():
        ax.plot(df["hour"], df[col], label=col, linewidth=style[1], color=style[0])
    cfg = get_manifestation_config(manifestation)
    ax.axvline(cfg.critical_hour, color="black", linestyle="-.", linewidth=2)
    ax.set_title(f"{manifestation.upper()} — smoothing comparison on $\\epsilon^2_{{cco}}$")
    ax.set_xlabel("Hour index")
    ax.set_ylabel(r"$\epsilon^2_{\mathrm{cco}}$")
    ax.legend()
    fig.tight_layout()
    out = FIGURES_DIR / f"smoothing_{manifestation}.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out
