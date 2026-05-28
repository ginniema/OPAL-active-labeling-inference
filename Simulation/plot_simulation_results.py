"""Plot cleaned simulation results."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plotting import (
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    METHOD_ORDER,
    _display_method_label,
    _finish_axis,
    set_theme_bw,
)
from utils import EFFECTIVE_N_COL, HUMAN_N_COL


ODDS_CASES = [
    ("odds_ratio_20_80_oracle", "(a) 20/80 split, oracle probabilities"),
    ("odds_ratio_20_80_estimated", "(b) 20/80 split, estimated probabilities"),
    ("odds_ratio_balanced_oracle", "(c) 50/50 split, oracle probabilities"),
    ("odds_ratio_balanced_estimated", "(d) 50/50 split, estimated probabilities"),
]

LEGACY_ODDS_FILES = {
    "odds_ratio_20_80_oracle.csv": "sim_hard_oracle_results.csv",
    "odds_ratio_20_80_estimated.csv": "sim_hard_results.csv",
    "odds_ratio_balanced_oracle.csv": "sim_easier_oracle_results.csv",
    "odds_ratio_balanced_estimated.csv": "sim_easier_results.csv",
}


def copy_legacy_odds_results(results_dir: Path) -> None:
    """Copy old 500-trial CSVs into the cleaned results naming scheme."""

    legacy_dir = REPO_ROOT.parent / "Simulation" / "odds ratio"
    if not legacy_dir.exists():
        raise FileNotFoundError(f"Legacy odds-ratio directory not found: {legacy_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)
    for clean_name, legacy_name in LEGACY_ODDS_FILES.items():
        shutil.copy2(legacy_dir / legacy_name, results_dir / clean_name)


def _ordered_methods_for_frames(frames: list[pd.DataFrame]) -> list[str]:
    present = set()
    for frame in frames:
        present.update(frame["estimator"].dropna().unique())
    return [method for method in METHOD_ORDER if method in present]


def _ess_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby([HUMAN_N_COL, "estimator"], observed=True)[EFFECTIVE_N_COL]
        .mean()
        .reset_index()
    )


def _draw_ess_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    methods: list[str],
    marker_size: float = 3.6,
    line_width: float = 1.25,
) -> None:
    for method in methods:
        sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
        if sub.empty:
            continue
        ax.plot(
            sub[HUMAN_N_COL],
            sub[EFFECTIVE_N_COL],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=marker_size,
            linewidth=line_width,
            label=_display_method_label(method),
        )
    ax.set_xlabel(HUMAN_N_COL)
    ax.set_ylabel(EFFECTIVE_N_COL)
    ax.locator_params(axis="x", nbins=4)
    _finish_axis(ax)


def plot_odds_ratio_ess_grid(
    results_dir: Path,
    plots_dir: Path,
    include_title: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot the four odds-ratio ESS cases in a compact paper figure."""

    frames = []
    for case_name, _ in ODDS_CASES:
        path = results_dir / f"{case_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing result file: {path}")
        frames.append(pd.read_csv(path))

    set_theme_bw(font_scale=0.88)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    methods = _ordered_methods_for_frames(frames)
    summaries = [_ess_summary(frame) for frame in frames]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), sharey=False)
    for index, (ax, summary, (_, panel_title)) in enumerate(zip(axes.ravel(), summaries, ODDS_CASES)):
        _draw_ess_panel(ax, summary, methods)
        ax.set_ylim(0, float(summary[EFFECTIVE_N_COL].max()) * 1.08)
        if index < 2:
            ax.set_xlabel("")
        if index % 2 == 1:
            ax.set_ylabel("")
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=2)

    handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=4.0,
            linewidth=1.5,
            label=_display_method_label(method),
        )
        for method in methods
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.6,
    )
    if include_title:
        fig.suptitle("Odds-ratio simulation effective sample size", y=0.985, fontsize=12, fontweight="normal")
        top = 0.925
    else:
        top = 0.965
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.18, top=top, wspace=0.28, hspace=0.42)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "odds_ratio_effective_sample_size.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / "odds_ratio_effective_sample_size.png", dpi=300, bbox_inches="tight")
    return fig, axes


def plot_kendall_tau_ess(results_dir: Path, plots_dir: Path) -> tuple[plt.Figure, plt.Axes]:
    """Plot Kendall tau ESS from ``kendall_tau_ess.csv``."""

    path = results_dir / "kendall_tau_ess.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    df = pd.read_csv(path)
    methods = [method for method in METHOD_ORDER + ["oracle"] if method in set(df["estimator"].unique())]

    set_theme_bw(font_scale=0.88)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    if "oracle" not in METHOD_COLORS:
        METHOD_COLORS["oracle"] = "#666666"
        METHOD_LINESTYLES["oracle"] = ":"
        METHOD_MARKERS["oracle"] = "^"

    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    _draw_ess_panel(ax, df, methods, marker_size=3.6, line_width=1.35)
    ax.legend(loc="best", frameon=False)
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "kendall_tau_effective_sample_size.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / "kendall_tau_effective_sample_size.png", dpi=300, bbox_inches="tight")
    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Simulation" / "results")
    parser.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "Simulation" / "plots")
    parser.add_argument("--copy-legacy-odds-results", action="store_true")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--skip-kendall", action="store_true")
    parser.add_argument("--include-odds-title", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.copy_legacy_odds_results:
        copy_legacy_odds_results(args.results_dir)
    if not args.skip_odds:
        fig, _ = plot_odds_ratio_ess_grid(args.results_dir, args.plots_dir, include_title=args.include_odds_title)
        plt.close(fig)
    if not args.skip_kendall:
        fig, _ = plot_kendall_tau_ess(args.results_dir, args.plots_dir)
        plt.close(fig)


if __name__ == "__main__":
    main()
