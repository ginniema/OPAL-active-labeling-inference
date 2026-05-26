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

from utils import EFFECTIVE_N_COL, HUMAN_N_COL, monte_carlo_variance_table


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
    error_bars: str = "sd",
    error_style: str = "bar",
    show_se: bool = False,
    show_iqr: bool | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot mean effective sample size by budget and estimator.

    By default, this shows one Monte Carlo standard deviation around the mean,
    matching the Robust Sampling paper's ESS convention. Use
    ``error_bars="none"`` for a clean mean curve, ``error_bars="se"`` for
    standard errors, or ``error_style="ribbon"`` for shaded bands instead of
    vertical bars. ``show_se`` and ``show_iqr`` are retained as deprecated
    aliases for older notebooks.
    """

    if show_se:
        error_bars = "se"
    if show_iqr is not None:
        error_bars = "sd" if show_iqr else "none"
    if error_bars not in {"none", "se", "sd"}:
        raise ValueError("error_bars must be one of: 'none', 'se', 'sd'")
    if error_style not in {"bar", "ribbon"}:
        raise ValueError("error_style must be one of: 'bar', 'ribbon'")

    set_theme_bw()
    methods = _ordered_methods(df)
    plot_df = df[df["estimator"].isin(methods)].copy()
    summary = (
        plot_df.groupby([HUMAN_N_COL, "estimator"], observed=True)[EFFECTIVE_N_COL]
        .agg(mean="mean", sd="std", count="count")
        .reset_index()
    )
    summary["se"] = summary["sd"] / np.sqrt(summary["count"])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    for method in methods:
        sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
        if sub.empty:
            continue
        if error_bars != "none" and error_style == "bar":
            yerr = sub["se"].fillna(0) if error_bars == "se" else sub["sd"].fillna(0)
            ax.errorbar(
                sub[HUMAN_N_COL],
                sub["mean"],
                yerr=yerr,
                label=method,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=5.5,
                capsize=3,
                elinewidth=1,
                capthick=1,
            )
        else:
            ax.plot(
                sub[HUMAN_N_COL],
                sub["mean"],
                label=method,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=5.5,
            )
            if error_bars != "none" and error_style == "ribbon":
                yerr = sub["se"].fillna(0) if error_bars == "se" else sub["sd"].fillna(0)
                lower = np.maximum(sub["mean"].to_numpy() - yerr.to_numpy(), 0)
                upper = sub["mean"].to_numpy() + yerr.to_numpy()
                ax.fill_between(
                    sub[HUMAN_N_COL].to_numpy(),
                    lower,
                    upper,
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
    coverage_col: str = "coverage",
    ylabel: str | None = None,
    title: str | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot empirical CI coverage by budget and estimator."""

    set_theme_bw()
    if coverage_col not in df.columns:
        raise ValueError(f"Coverage column not found: {coverage_col}")
    if ylabel is None:
        ylabel = coverage_col.replace("_", " ")
    methods = _ordered_methods(df)
    plot_df = df[df["estimator"].isin(methods)].copy()
    summary = (
        plot_df.groupby([HUMAN_N_COL, "estimator"], observed=True)[coverage_col]
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
            sub[coverage_col],
            label=method,
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=5.5,
        )

    ax.axhline(1 - alpha, color="#666666", linestyle="--", linewidth=1)
    ax.set_xlabel(HUMAN_N_COL)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", ncol=2)
    _add_budget_fraction_ticks(ax, n_total)
    _finish_axis(ax)
    fig.tight_layout()
    _save_show(fig, path, show)
    return fig, ax


def plot_finite_population_coverage(
    df: pd.DataFrame,
    alpha: float,
    path: str | Path | None = None,
    n_total: int | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot coverage using finite-population-calibrated intervals."""

    return plot_coverage(
        df,
        alpha=alpha,
        path=path,
        n_total=n_total,
        coverage_col="finite population coverage",
        ylabel="finite-population calibrated coverage",
        title="Coverage against fixed empirical population",
        show=show,
    )


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


def make_monte_carlo_variance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact table of MC variance components."""

    summary = monte_carlo_variance_table(df)
    columns = [
        HUMAN_N_COL,
        "estimator",
        "point_estimate_mean",
        "point_estimate_variance",
        "lb_mean",
        "lb_variance",
        "ub_mean",
        "ub_variance",
        "interval_width_mean",
        "interval_width_variance",
        "coverage",
        "finite_population_coverage",
        "finite_population_interval_width",
        "finite_population_variance_inflation",
    ]
    return summary[columns].sort_values([HUMAN_N_COL, "estimator"]).reset_index(drop=True)


def plot_monte_carlo_variance_components(
    df: pd.DataFrame,
    path: str | Path | None = None,
    n_total: int | None = None,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot MC variance of estimate, interval width, and interval endpoints."""

    set_theme_bw(font_scale=1.05)
    summary = monte_carlo_variance_table(df)
    methods = _ordered_methods(df)
    components = [
        ("point_estimate_variance", "estimate"),
        ("interval_width_variance", "CI width"),
        ("lb_variance", "left endpoint"),
        ("ub_variance", "right endpoint"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes_flat = axes.ravel()
    for ax, (variance_col, title) in zip(axes_flat, components):
        for method in methods:
            sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
            if sub.empty:
                continue
            values = sub[variance_col].mask(sub[variance_col] <= 0)
            ax.plot(
                sub[HUMAN_N_COL],
                values,
                label=method,
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=4.8,
            )
        ax.set_title(f"Var({title})")
        ax.set_xlabel(HUMAN_N_COL)
        ax.set_ylabel("empirical MC variance")
        ax.set_yscale("log")
        _finish_axis(ax)

    for ax in axes_flat[2:]:
        _add_budget_fraction_ticks(ax, n_total, y_offset=-0.18)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_show(fig, path, show)
    return fig, axes


def save_monte_carlo_variance_table(
    df: pd.DataFrame,
    path: str | Path,
    max_rows: int | None = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Render the MC variance component table to an image file."""

    set_theme_bw(font_scale=0.85)
    table = make_monte_carlo_variance_table(df)
    if max_rows is not None:
        table = table.head(max_rows)

    display_table = table.copy()
    numeric_cols = display_table.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col == HUMAN_N_COL:
            display_table[col] = display_table[col].map(lambda x: f"{x:.0f}")
        elif col.endswith("coverage") or col == "coverage":
            display_table[col] = display_table[col].map(lambda x: f"{x:.2f}")
        elif "variance" in col:
            display_table[col] = display_table[col].map(lambda x: f"{x:.3g}")
        else:
            display_table[col] = display_table[col].map(lambda x: f"{x:.3f}")

    fig_h = max(2.5, 0.28 * (len(display_table) + 1))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.axis("off")
    table_artist = ax.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        loc="center",
        cellLoc="center",
    )
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(8)
    table_artist.scale(1.0, 1.25)
    for (row, _), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#BDBDBD")
        if row == 0:
            cell.set_facecolor("#F2F2F2")
            cell.get_text().set_weight("bold")
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
    """Save a standalone legend for the competitor methods."""

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
