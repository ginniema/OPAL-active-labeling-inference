"""Plotting helpers for paper examples.

The default style mirrors ggplot2's theme_bw: white panels, a visible panel
border, light grid lines, and a colorblind-aware palette.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from utils import EFFECTIVE_N_COL, HUMAN_N_COL


METHOD_ORDER = [
    "active",
    "active + tuning",
    "spline",
    "spline + tuning",
    "uniform",
    "classical",
]

METHOD_COLORS = {
    "active": "#E69F00",
    "active + tuning": "#E69F00",
    "spline": "#009E73",
    "spline + tuning": "#009E73",
    "uniform": "#0072B2",
    "classical": "#CC79A7",
    "LLM only": "#D55E00",
}

METHOD_LINESTYLES = {
    "active": "-",
    "active + tuning": "--",
    "spline": "-",
    "spline + tuning": "--",
    "uniform": "-",
    "classical": "-",
    "LLM only": ":",
}

METHOD_MARKERS = {
    "active": "o",
    "active + tuning": "o",
    "spline": "s",
    "spline + tuning": "s",
    "uniform": "D",
    "classical": "X",
    "LLM only": "^",
}


def set_theme_bw(font_scale: float = 1.25) -> None:
    """Set a matplotlib style close to ggplot2 theme_bw."""

    base_size = 11 * font_scale
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "font.size": base_size,
            "axes.labelsize": base_size,
            "axes.titlesize": base_size * 1.05,
            "xtick.labelsize": base_size * 0.9,
            "ytick.labelsize": base_size * 0.9,
            "legend.fontsize": base_size * 0.85,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "xtick.color": "black",
            "ytick.color": "black",
            "text.color": "black",
            "axes.labelcolor": "black",
        }
    )


def _finish_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.8, colors="black")


def _save_show(fig: plt.Figure, path: str | Path | None, show: bool) -> None:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _ordered_methods(df: pd.DataFrame, methods: Iterable[str] | None = None) -> list[str]:
    if methods is None:
        methods = METHOD_ORDER
    present = set(df["estimator"].dropna().unique())
    return [method for method in methods if method in present]


def _add_budget_fraction_ticks(ax: plt.Axes, n_total: int | None, y_offset: float = -0.12) -> None:
    if n_total is None:
        return
    ticks = ax.get_xticks()
    for tick in ticks:
        if tick > 0:
            ax.text(
                tick,
                y_offset,
                f"({tick / n_total:.0%})",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
            )


def plot_effective_sample_size(
    df: pd.DataFrame,
    path: str | Path | None = None,
    n_total: int | None = None,
    show_iqr: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot mean effective sample size by budget and estimator."""

    set_theme_bw()
    methods = _ordered_methods(df)
    plot_df = df[df["estimator"].isin(methods)].copy()
    summary = (
        plot_df.groupby([HUMAN_N_COL, "estimator"], observed=True)[EFFECTIVE_N_COL]
        .agg(mean="mean", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method in methods:
        sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
        if sub.empty:
            continue
        ax.plot(
            sub[HUMAN_N_COL],
            sub["mean"],
            label=method,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=5.5,
        )
        if show_iqr:
            ax.fill_between(
                sub[HUMAN_N_COL].to_numpy(),
                sub["q25"].to_numpy(),
                sub["q75"].to_numpy(),
                color=METHOD_COLORS[method],
                alpha=0.16,
                linewidth=0,
            )

    ax.set_xlabel(HUMAN_N_COL)
    ax.set_ylabel(EFFECTIVE_N_COL)
    ax.legend(loc="best", ncol=2)
    _add_budget_fraction_ticks(ax, n_total)
    _finish_axis(ax)
    fig.tight_layout()
    _save_show(fig, path, show)
    return fig, ax


def plot_coverage(
    df: pd.DataFrame,
    alpha: float,
    path: str | Path | None = None,
    n_total: int | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot empirical CI coverage by budget and estimator."""

    set_theme_bw()
    methods = _ordered_methods(df)
    plot_df = df[df["estimator"].isin(methods)].copy()
    summary = (
        plot_df.groupby([HUMAN_N_COL, "estimator"], observed=True)["coverage"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method in methods:
        sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
        if sub.empty:
            continue
        ax.plot(
            sub[HUMAN_N_COL],
            sub["coverage"],
            label=method,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=5.5,
        )

    ax.axhline(1 - alpha, color="#666666", linestyle="--", linewidth=1)
    ax.set_xlabel(HUMAN_N_COL)
    ax.set_ylabel("coverage")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", ncol=2)
    _add_budget_fraction_ticks(ax, n_total)
    _finish_axis(ax)
    fig.tight_layout()
    _save_show(fig, path, show)
    return fig, ax


def plot_monte_carlo_variance(
    df: pd.DataFrame,
    path: str | Path | None = None,
    use_log: bool = True,
    n_total: int | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot empirical Monte Carlo variance of point estimates."""

    set_theme_bw()
    variance_col = "mc log variance" if use_log else "mc variance"
    ylabel = "MC variance of log odds-ratio estimate" if use_log else "MC variance of odds-ratio estimate"
    methods = _ordered_methods(df)
    summary = (
        df[df["estimator"].isin(methods)]
        .groupby([HUMAN_N_COL, "estimator"], observed=True)[variance_col]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method in methods:
        sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
        if sub.empty:
            continue
        ax.plot(
            sub[HUMAN_N_COL],
            sub[variance_col],
            label=method,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=5.5,
        )

    ax.set_xlabel(HUMAN_N_COL)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.legend(loc="best", ncol=2)
    _add_budget_fraction_ticks(ax, n_total)
    _finish_axis(ax)
    fig.tight_layout()
    _save_show(fig, path, show)
    return fig, ax


def plot_intervals(
    df: pd.DataFrame,
    true_value: float,
    path: str | Path | None = None,
    estimand_label: str = "odds ratio",
    n_index: int = -1,
    num_intervals: int = 5,
    seed: int = 614,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Draw sampled confidence intervals for one budget slice."""

    set_theme_bw()
    methods = _ordered_methods(df)
    budgets = np.sort(df[HUMAN_N_COL].dropna().unique())
    n_human = budgets[n_index]
    rng = np.random.default_rng(seed)

    rows_to_plot = []
    for method in methods:
        sub = df[(df["estimator"] == method) & (df[HUMAN_N_COL] == n_human)]
        if sub.empty:
            continue
        take = min(num_intervals, len(sub))
        sampled = sub.iloc[rng.choice(len(sub), take, replace=False)]
        for _, row in sampled.iterrows():
            rows_to_plot.append((method, row["lb"], row["ub"]))
    rows_to_plot = rows_to_plot[::-1]

    fig_h = max(3.5, 0.28 * len(rows_to_plot))
    fig, ax = plt.subplots(figsize=(7, fig_h))
    ax.axvline(true_value, color="#666666", linestyle="--", linewidth=1)

    bar_height = 0.18
    hatch = {"active + tuning": "///", "spline + tuning": "///"}
    for y, (method, lb, ub) in enumerate(rows_to_plot):
        rect = mpatches.Rectangle(
            (lb, y - bar_height / 2),
            ub - lb,
            bar_height,
            facecolor=METHOD_COLORS[method],
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch.get(method, ""),
            alpha=0.78,
        )
        ax.add_patch(rect)

    ax.set_ylim(-0.6, len(rows_to_plot) - 0.4)
    ax.set_yticks([])
    ax.set_xlabel(estimand_label)
    ax.grid(axis="y", visible=False)
    _finish_axis(ax)
    fig.tight_layout()
    _save_show(fig, path, show)
    return fig, ax


def save_legend(path: str | Path, show: bool = False) -> tuple[plt.Figure, plt.Axes]:
    """Save a standalone legend for the BRCA competitor methods."""

    set_theme_bw()
    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            markersize=6,
            label=method,
        )
        for method in METHOD_ORDER
    ]
    fig, ax = plt.subplots(figsize=(8.5, 1.4))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=3)
    fig.tight_layout()
    _save_show(fig, path, show)
    return fig, ax
