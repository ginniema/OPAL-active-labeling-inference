"""Plotting helpers for paper examples.
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


ESS_MULTIPLIER_COL = "ESS multiplier over classical"

METHOD_ORDER = [
    "active",
    "active + tuning",
    "active+",
    "spline",
    "spline + tuning",
    "spline+",
    "uniform",
    "classical",
]

METHOD_LABELS = {
    "active+": "active + tuning",
    "spline": "OPAL",
    "spline + tuning": "OPAL + tuning",
    "spline+": "OPAL + tuning",
}

METHOD_COLORS = {
    "active": "#E69F00",
    "active + tuning": "#E69F00",
    "active+": "#E69F00",
    "spline": "#009E73",
    "spline + tuning": "#009E73",
    "spline+": "#009E73",
    "uniform": "#0072B2",
    "classical": "#CC79A7",
    "LLM only": "#D55E00",
}

METHOD_LINESTYLES = {
    "active": "-",
    "active + tuning": "--",
    "active+": "--",
    "spline": "-",
    "spline + tuning": "--",
    "spline+": "--",
    "uniform": "-",
    "classical": "-",
    "LLM only": ":",
}

METHOD_MARKERS = {
    "active": "o",
    "active + tuning": "o",
    "active+": "o",
    "spline": "s",
    "spline + tuning": "s",
    "spline+": "s",
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


def _display_method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _multiplier_key_cols(df: pd.DataFrame) -> list[str]:
    key_cols = [HUMAN_N_COL]
    for col in ("trial", "num_trial"):
        if col in df.columns:
            key_cols.append(col)
            break
    return key_cols


def make_effective_sample_size_multiplier(
    df: pd.DataFrame,
    baseline_method: str = "classical",
) -> pd.DataFrame:
    """Return ESS divided by the matched classical ESS.

    The denominator is matched by budget and, when available, Monte Carlo
    replicate. This makes the classical multiplier exactly one for every
    matched row and compares each method against the same replicate.
    """

    key_cols = _multiplier_key_cols(df)
    baseline = (
        df[df["estimator"] == baseline_method]
        .groupby(key_cols, observed=True)[EFFECTIVE_N_COL]
        .mean()
        .reset_index()
        .rename(columns={EFFECTIVE_N_COL: "_baseline_ess"})
    )
    if baseline.empty:
        raise ValueError(f"No baseline ESS rows found for estimator: {baseline_method}")

    out = df.merge(baseline, on=key_cols, how="inner")
    out[ESS_MULTIPLIER_COL] = out[EFFECTIVE_N_COL] / out["_baseline_ess"].replace(0, np.nan)
    return out.drop(columns=["_baseline_ess"])


def _add_budget_fraction_ticks(ax: plt.Axes, n_total: int | None, y_offset: float = -0.12) -> None:
    if n_total is None:
        return
    x_min, x_max = ax.get_xlim()
    ticks = ax.get_xticks()
    for tick in ticks:
        if tick > 0 and x_min <= tick <= x_max:
            ax.text(
                tick,
                y_offset,
                f"({tick / n_total:.0%})",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
            )


def _finish_budget_figure(fig: plt.Figure, n_total: int | None) -> None:
    fig.tight_layout()
    if n_total is not None:
        fig.subplots_adjust(bottom=0.22)


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
    ``error_bars="none"`` for a just the mean curve, ``error_bars="se"`` for
    standard errors, or ``error_style="ribbon"`` for shaded bands instead of
    vertical bars. 
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
                label=_display_method_label(method),
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
                label=_display_method_label(method),
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
    _add_budget_fraction_ticks(ax, n_total, y_offset=-0.18)
    _finish_axis(ax)
    _finish_budget_figure(fig, n_total)
    _save_show(fig, path, show)
    return fig, ax


def plot_effective_sample_size_multiplier(
    df: pd.DataFrame,
    path: str | Path | None = None,
    n_total: int | None = None,
    error_bars: str = "sd",
    error_style: str = "bar",
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot ESS as a multiplier over matched classical ESS."""

    if error_bars not in {"none", "se", "sd"}:
        raise ValueError("error_bars must be one of: 'none', 'se', 'sd'")
    if error_style not in {"bar", "ribbon"}:
        raise ValueError("error_style must be one of: 'bar', 'ribbon'")

    set_theme_bw()
    multiplier_df = make_effective_sample_size_multiplier(df)
    methods = _ordered_methods(multiplier_df)
    plot_df = multiplier_df[multiplier_df["estimator"].isin(methods)].copy()
    summary = (
        plot_df.groupby([HUMAN_N_COL, "estimator"], observed=True)[ESS_MULTIPLIER_COL]
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
                label=_display_method_label(method),
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
                label=_display_method_label(method),
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

    ax.axhline(1, color="#666666", linestyle="--", linewidth=1)
    ax.set_xlabel(HUMAN_N_COL)
    ax.set_ylabel(ESS_MULTIPLIER_COL)
    ax.legend(loc="best", ncol=2)
    _add_budget_fraction_ticks(ax, n_total, y_offset=-0.18)
    _finish_axis(ax)
    _finish_budget_figure(fig, n_total)
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
    summary = _coverage_summary(df, coverage_col, methods)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    _draw_coverage_panel(ax, summary, methods, coverage_col, alpha)
    ax.set_xlabel(HUMAN_N_COL)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best", ncol=2)
    _add_budget_fraction_ticks(ax, n_total, y_offset=-0.18)
    _finish_axis(ax)
    _finish_budget_figure(fig, n_total)
    _save_show(fig, path, show)
    return fig, ax


def _coverage_summary(
    df: pd.DataFrame,
    coverage_col: str,
    methods: list[str],
) -> pd.DataFrame:
    plot_df = df[df["estimator"].isin(methods)].copy()
    return (
        plot_df.groupby([HUMAN_N_COL, "estimator"], observed=True)[coverage_col]
        .mean()
        .reset_index()
    )


def _draw_coverage_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    methods: list[str],
    coverage_col: str,
    alpha: float,
    marker_size: float = 5.5,
    line_width: float | None = None,
    ylim: tuple[float, float] = (0.0, 1.05),
) -> None:
    for method in methods:
        sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
        if sub.empty:
            continue
        ax.plot(
            sub[HUMAN_N_COL],
            sub[coverage_col],
            label=_display_method_label(method),
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=marker_size,
            linewidth=line_width,
        )

    ax.axhline(1 - alpha, color="#666666", linestyle="--", linewidth=1)
    ax.set_ylim(*ylim)


def plot_coverage_comparison(
    df: pd.DataFrame,
    alpha: float,
    path: str | Path | None = None,
    n_total: int | None = None,
    title: str = "CheXpert Coverage",
    panel_titles: tuple[str, str] = (
        "(a) Raw coverage",
        "(b) Finite-population corrected coverage",
    ),
    figsize: tuple[float, float] = (7.2, 3.7),
    ylim: tuple[float, float] = (0.75, 1.05),
    show_fraction_ticks: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot raw and finite-population-calibrated coverage in one figure."""

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
    methods = _ordered_methods(df)
    panels = [
        ("coverage", panel_titles[0]),
        ("finite population coverage", panel_titles[1]),
    ]
    missing = [col for col, _ in panels if col not in df.columns]
    if missing:
        raise ValueError(f"Coverage columns not found: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, (coverage_col, panel_title) in zip(axes, panels):
        summary = _coverage_summary(df, coverage_col, methods)
        _draw_coverage_panel(
            ax,
            summary,
            methods,
            coverage_col,
            alpha,
            marker_size=3.6,
            line_width=1.35,
            ylim=ylim,
        )
        ax.set_xlabel(HUMAN_N_COL)
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=2)
        tick_upper = min(ylim[1], 1.0)
        ax.set_yticks(np.arange(ylim[0], tick_upper + 1e-9, 0.05))
        ax.locator_params(axis="x", nbins=5)
        if show_fraction_ticks:
            _add_budget_fraction_ticks(ax, n_total, y_offset=-0.20)
        _finish_axis(ax)
    axes[0].set_ylabel("coverage")

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
    fig.suptitle(title, y=0.94, fontsize=12.0, fontweight="normal")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.36, top=0.84, wspace=0.10)
    _save_show(fig, path, show)
    return fig, axes


def plot_effective_sample_size_and_finite_population_coverage(
    df: pd.DataFrame,
    alpha: float,
    path: str | Path | None = None,
    n_total: int | None = None,
    panel_titles: tuple[str, str] = (
        "(a) Effective sample size",
        "(b) Finite-population corrected coverage",
    ),
    figsize: tuple[float, float] = (7.2, 3.4),
    coverage_ylim: tuple[float, float] = (0.75, 1.05),
    show_fraction_ticks: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot ESS and finite-population coverage side by side."""

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
    methods = _ordered_methods(df)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ess_summary = (
        df[df["estimator"].isin(methods)]
        .groupby([HUMAN_N_COL, "estimator"], observed=True)[EFFECTIVE_N_COL]
        .mean()
        .reset_index()
    )
    for method in methods:
        sub = ess_summary[ess_summary["estimator"] == method].sort_values(HUMAN_N_COL)
        if sub.empty:
            continue
        axes[0].plot(
            sub[HUMAN_N_COL],
            sub[EFFECTIVE_N_COL],
            label=_display_method_label(method),
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=3.8,
            linewidth=1.35,
        )
    axes[0].set_xlabel(HUMAN_N_COL)
    axes[0].set_ylabel(EFFECTIVE_N_COL)
    axes[0].set_title(panel_titles[0], loc="left", fontweight="normal", pad=2)
    axes[0].locator_params(axis="x", nbins=5)
    _finish_axis(axes[0])

    coverage_col = "finite population coverage"
    if coverage_col not in df.columns:
        raise ValueError(f"Coverage column not found: {coverage_col}")
    coverage_summary = _coverage_summary(df, coverage_col, methods)
    _draw_coverage_panel(
        axes[1],
        coverage_summary,
        methods,
        coverage_col,
        alpha,
        marker_size=3.8,
        line_width=1.35,
        ylim=coverage_ylim,
    )
    axes[1].set_xlabel(HUMAN_N_COL)
    axes[1].set_ylabel("coverage")
    axes[1].set_title(panel_titles[1], loc="left", fontweight="normal", pad=2)
    tick_upper = min(coverage_ylim[1], 1.0)
    axes[1].set_yticks(np.arange(coverage_ylim[0], tick_upper + 1e-9, 0.05))
    axes[1].locator_params(axis="x", nbins=5)
    _finish_axis(axes[1])

    if show_fraction_ticks:
        for ax in axes:
            _add_budget_fraction_ticks(ax, n_total, y_offset=-0.20)

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
    bottom = 0.38 if show_fraction_ticks else 0.32
    legend_y = 0.02
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, legend_y),
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=bottom, top=0.92, wspace=0.24)
    _save_show(fig, path, show)
    return fig, axes


def plot_batch_sequential_effective_sample_size_and_finite_population_coverage(
    batch_df: pd.DataFrame,
    sequential_df: pd.DataFrame,
    alpha: float,
    path: str | Path | None = None,
    n_total_batch: int | None = None,
    n_total_sequential: int | None = None,
    panel_titles: tuple[str, str, str, str] = (
        "(a) Batch effective sample size",
        "(b) Batch corrected coverage",
        "(c) Sequential effective sample size",
        "(d) Sequential corrected coverage",
    ),
    figsize: tuple[float, float] = (7.2, 6.25),
    coverage_ylim: tuple[float, float] = (0.75, 1.05),
    show_fraction_ticks: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot batch and sequential ESS/corrected coverage in a four-panel figure."""

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

    methods = [
        method
        for method in METHOD_ORDER
        if method in set(batch_df["estimator"].dropna().unique())
        or method in set(sequential_df["estimator"].dropna().unique())
    ]
    coverage_col = "finite population coverage"
    for name, frame in [("batch", batch_df), ("sequential", sequential_df)]:
        missing = [
            col
            for col in [HUMAN_N_COL, "estimator", EFFECTIVE_N_COL, coverage_col]
            if col not in frame.columns
        ]
        if missing:
            raise ValueError(f"{name} data frame is missing required columns: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    plot_specs = [
        (batch_df, axes[0, 0], "ess", panel_titles[0]),
        (batch_df, axes[0, 1], "coverage", panel_titles[1]),
        (sequential_df, axes[1, 0], "ess", panel_titles[2]),
        (sequential_df, axes[1, 1], "coverage", panel_titles[3]),
    ]

    ess_max = 0.0
    for frame in [batch_df, sequential_df]:
        summary = (
            frame[frame["estimator"].isin(methods)]
            .groupby([HUMAN_N_COL, "estimator"], observed=True)[EFFECTIVE_N_COL]
            .mean()
            .reset_index()
        )
        if not summary.empty:
            ess_max = max(ess_max, float(summary[EFFECTIVE_N_COL].max()))

    for frame, ax, panel_kind, panel_title in plot_specs:
        if panel_kind == "ess":
            summary = (
                frame[frame["estimator"].isin(methods)]
                .groupby([HUMAN_N_COL, "estimator"], observed=True)[EFFECTIVE_N_COL]
                .mean()
                .reset_index()
            )
            for method in methods:
                sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
                if sub.empty:
                    continue
                ax.plot(
                    sub[HUMAN_N_COL],
                    sub[EFFECTIVE_N_COL],
                    label=_display_method_label(method),
                    color=METHOD_COLORS[method],
                    linestyle=METHOD_LINESTYLES[method],
                    marker=METHOD_MARKERS[method],
                    markersize=3.4,
                    linewidth=1.25,
                )
            if ess_max > 0:
                ax.set_ylim(0, ess_max * 1.08)
            ax.set_ylabel(EFFECTIVE_N_COL)
        else:
            summary = _coverage_summary(frame, coverage_col, methods)
            _draw_coverage_panel(
                ax,
                summary,
                methods,
                coverage_col,
                alpha,
                marker_size=3.4,
                line_width=1.25,
                ylim=coverage_ylim,
            )
            tick_upper = min(coverage_ylim[1], 1.0)
            ax.set_yticks(np.arange(coverage_ylim[0], tick_upper + 1e-9, 0.05))
            ax.set_ylabel("coverage")

        ax.set_xlabel(HUMAN_N_COL)
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=2)
        ax.locator_params(axis="x", nbins=4)
        _finish_axis(ax)

    if show_fraction_ticks:
        for ax in axes[0, :]:
            _add_budget_fraction_ticks(ax, n_total_batch, y_offset=-0.22)
        for ax in axes[1, :]:
            _add_budget_fraction_ticks(ax, n_total_sequential, y_offset=-0.22)

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
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.18, top=0.965, wspace=0.26, hspace=0.42)
    _save_show(fig, path, show)
    return fig, axes


def plot_batch_sequential_effective_sample_size(
    batch_df: pd.DataFrame,
    sequential_df: pd.DataFrame,
    path: str | Path | None = None,
    n_total_batch: int | None = None,
    n_total_sequential: int | None = None,
    panel_titles: tuple[str, str] = (
        "(a) Batch",
        "(b) Sequential",
    ),
    figsize: tuple[float, float] = (7.2, 3.4),
    show_fraction_ticks: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot batch and sequential effective sample size side by side."""

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
    methods = [
        method
        for method in METHOD_ORDER
        if method in set(batch_df["estimator"].dropna().unique())
        or method in set(sequential_df["estimator"].dropna().unique())
    ]
    for name, frame in [("batch", batch_df), ("sequential", sequential_df)]:
        missing = [
            col
            for col in [HUMAN_N_COL, "estimator", EFFECTIVE_N_COL]
            if col not in frame.columns
        ]
        if missing:
            raise ValueError(f"{name} data frame is missing required columns: {missing}")

    summaries = []
    ess_max = 0.0
    for frame in [batch_df, sequential_df]:
        summary = (
            frame[frame["estimator"].isin(methods)]
            .groupby([HUMAN_N_COL, "estimator"], observed=True)[EFFECTIVE_N_COL]
            .mean()
            .reset_index()
        )
        summaries.append(summary)
        if not summary.empty:
            ess_max = max(ess_max, float(summary[EFFECTIVE_N_COL].max()))

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, summary, panel_title in zip(axes, summaries, panel_titles):
        for method in methods:
            sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
            if sub.empty:
                continue
            ax.plot(
                sub[HUMAN_N_COL],
                sub[EFFECTIVE_N_COL],
                label=_display_method_label(method),
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=3.8,
                linewidth=1.35,
            )
        if ess_max > 0:
            ax.set_ylim(0, ess_max * 1.08)
        ax.set_xlabel(HUMAN_N_COL)
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=2)
        ax.locator_params(axis="x", nbins=4)
        _finish_axis(ax)
    axes[0].set_ylabel(EFFECTIVE_N_COL)

    if show_fraction_ticks:
        _add_budget_fraction_ticks(axes[0], n_total_batch, y_offset=-0.20)
        _add_budget_fraction_ticks(axes[1], n_total_sequential, y_offset=-0.20)

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
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.6,
    )
    bottom = 0.38 if show_fraction_ticks else 0.32
    fig.subplots_adjust(left=0.075, right=0.995, bottom=bottom, top=0.91, wspace=0.10)
    _save_show(fig, path, show)
    return fig, axes


def plot_batch_sequential_finite_population_coverage(
    batch_df: pd.DataFrame,
    sequential_df: pd.DataFrame,
    alpha: float,
    path: str | Path | None = None,
    n_total_batch: int | None = None,
    n_total_sequential: int | None = None,
    panel_titles: tuple[str, str] = (
        "(a) Batch",
        "(b) Sequential",
    ),
    figsize: tuple[float, float] = (7.2, 3.4),
    coverage_ylim: tuple[float, float] = (0.75, 1.05),
    show_fraction_ticks: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot batch and sequential finite-population corrected coverage."""

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
    methods = [
        method
        for method in METHOD_ORDER
        if method in set(batch_df["estimator"].dropna().unique())
        or method in set(sequential_df["estimator"].dropna().unique())
    ]
    coverage_col = "finite population coverage"
    for name, frame in [("batch", batch_df), ("sequential", sequential_df)]:
        missing = [
            col
            for col in [HUMAN_N_COL, "estimator", coverage_col]
            if col not in frame.columns
        ]
        if missing:
            raise ValueError(f"{name} data frame is missing required columns: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    for ax, frame, panel_title in zip(axes, [batch_df, sequential_df], panel_titles):
        summary = _coverage_summary(frame, coverage_col, methods)
        _draw_coverage_panel(
            ax,
            summary,
            methods,
            coverage_col,
            alpha,
            marker_size=3.8,
            line_width=1.35,
            ylim=coverage_ylim,
        )
        tick_upper = min(coverage_ylim[1], 1.0)
        ax.set_yticks(np.arange(coverage_ylim[0], tick_upper + 1e-9, 0.05))
        ax.set_xlabel(HUMAN_N_COL)
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=2)
        ax.locator_params(axis="x", nbins=4)
        _finish_axis(ax)
    axes[0].set_ylabel("coverage")

    if show_fraction_ticks:
        _add_budget_fraction_ticks(axes[0], n_total_batch, y_offset=-0.20)
        _add_budget_fraction_ticks(axes[1], n_total_sequential, y_offset=-0.20)

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
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.6,
    )
    bottom = 0.38 if show_fraction_ticks else 0.32
    fig.subplots_adjust(left=0.075, right=0.995, bottom=bottom, top=0.91, wspace=0.10)
    _save_show(fig, path, show)
    return fig, axes


def plot_batch_sequential_coverage_comparison(
    batch_df: pd.DataFrame,
    sequential_df: pd.DataFrame,
    alpha: float,
    path: str | Path | None = None,
    n_total_batch: int | None = None,
    n_total_sequential: int | None = None,
    panel_titles: tuple[str, str, str, str] = (
        "(a) Batch raw coverage",
        "(b) Batch corrected coverage",
        "(c) Sequential raw coverage",
        "(d) Sequential corrected coverage",
    ),
    figsize: tuple[float, float] = (7.2, 5.8),
    coverage_ylim: tuple[float, float] = (0.75, 1.05),
    show_fraction_ticks: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot raw and finite-population corrected coverage for batch/sequential runs."""

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
    methods = [
        method
        for method in METHOD_ORDER
        if method in set(batch_df["estimator"].dropna().unique())
        or method in set(sequential_df["estimator"].dropna().unique())
    ]
    panels = [
        (batch_df, "coverage", panel_titles[0], n_total_batch),
        (batch_df, "finite population coverage", panel_titles[1], n_total_batch),
        (sequential_df, "coverage", panel_titles[2], n_total_sequential),
        (sequential_df, "finite population coverage", panel_titles[3], n_total_sequential),
    ]
    for name, frame in [("batch", batch_df), ("sequential", sequential_df)]:
        missing = [
            col
            for col in [
                HUMAN_N_COL,
                "estimator",
                "coverage",
                "finite population coverage",
            ]
            if col not in frame.columns
        ]
        if missing:
            raise ValueError(f"{name} data frame is missing required columns: {missing}")

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
    for index, (ax, (frame, coverage_col, panel_title, n_total)) in enumerate(
        zip(axes.ravel(), panels)
    ):
        summary = _coverage_summary(frame, coverage_col, methods)
        _draw_coverage_panel(
            ax,
            summary,
            methods,
            coverage_col,
            alpha,
            marker_size=3.6,
            line_width=1.25,
            ylim=coverage_ylim,
        )
        tick_upper = min(coverage_ylim[1], 1.0)
        ax.set_yticks(np.arange(coverage_ylim[0], tick_upper + 1e-9, 0.05))
        ax.set_xlabel(HUMAN_N_COL if index >= 2 else "")
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=2)
        ax.locator_params(axis="x", nbins=4)
        if show_fraction_ticks:
            _add_budget_fraction_ticks(ax, n_total, y_offset=-0.23)
        _finish_axis(ax)
    axes[0, 0].set_ylabel("coverage")
    axes[1, 0].set_ylabel("coverage")

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
        bbox_to_anchor=(0.5, 0.035),
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.6,
    )
    bottom = 0.30 if show_fraction_ticks else 0.24
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=bottom,
        top=0.965,
        wspace=0.10,
        hspace=0.45,
    )
    _save_show(fig, path, show)
    return fig, axes


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


def plot_sequential_effective_sample_size(
    df: pd.DataFrame,
    path: str | Path | None = None,
    n_human: int | float | None = None,
    max_iterations: int | None = 50,
    iteration_col: str = "num_trial",
    title: str | None = "Effective Sample Size Across Sequential Runs",
    xlabel: str = "Monte Carlo replicate",
    legend: str = "side",
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot trial-by-trial ESS for one sequential budget.

    The archived sequential simulations store each Monte Carlo replicate as a
    row. This plot fixes one budget and shows the replicate-level ESS traces,
    which is useful for visualizing stability across repeated runs.
    """

    set_theme_bw(font_scale=1.2)
    if iteration_col not in df.columns:
        raise ValueError(f"Iteration column not found: {iteration_col}")
    if legend not in {"side", "bottom", "best", "none"}:
        raise ValueError("legend must be one of: 'side', 'bottom', 'best', 'none'")

    methods = _ordered_methods(df)
    plot_df = df[df["estimator"].isin(methods)].copy()
    if n_human is None:
        n_human = float(np.nanmax(plot_df[HUMAN_N_COL]))
    plot_df = plot_df[plot_df[HUMAN_N_COL] == n_human]

    iterations = np.sort(plot_df[iteration_col].dropna().unique())
    if max_iterations is not None:
        iterations = iterations[:max_iterations]
        plot_df = plot_df[plot_df[iteration_col].isin(iterations)]

    summary = (
        plot_df.groupby([iteration_col, "estimator"], observed=True)[EFFECTIVE_N_COL]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for method in methods:
        sub = summary[summary["estimator"] == method].sort_values(iteration_col)
        if sub.empty:
            continue
        ax.plot(
            sub[iteration_col],
            sub[EFFECTIVE_N_COL],
            label=_display_method_label(method),
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Effective Samples")
    if title is not None:
        ax.set_title(title)
    if legend == "side":
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, borderaxespad=0)
        tight_rect = (0, 0, 0.78, 1)
    elif legend == "bottom":
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
        tight_rect = (0, 0.08, 1, 1)
    elif legend == "best":
        ax.legend(loc="best", ncol=2)
        tight_rect = (0, 0, 1, 1)
    else:
        tight_rect = (0, 0, 1, 1)
    _finish_axis(ax)
    fig.tight_layout(rect=tight_rect)
    _save_show(fig, path, show)
    return fig, ax


def plot_sequential_effective_sample_size_distribution(
    df: pd.DataFrame,
    path: str | Path | None = None,
    n_human: int | float | None = None,
    max_iterations: int | None = None,
    iteration_col: str = "num_trial",
    title: str | None = "Sequential Effective Sample Size Distribution",
    seed: int = 614,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Show the raw spread of sequential ESS at one budget."""

    set_theme_bw(font_scale=1.05)
    methods = _ordered_methods(df)
    plot_df = df[df["estimator"].isin(methods)].copy()
    if n_human is None:
        n_human = float(np.nanmax(plot_df[HUMAN_N_COL]))
    plot_df = plot_df[plot_df[HUMAN_N_COL] == n_human]

    if max_iterations is not None:
        iterations = np.sort(plot_df[iteration_col].dropna().unique())[:max_iterations]
        plot_df = plot_df[plot_df[iteration_col].isin(iterations)]

    data = []
    labels = []
    colors = []
    for method in methods:
        values = plot_df.loc[plot_df["estimator"] == method, EFFECTIVE_N_COL].dropna().to_numpy()
        if len(values) == 0:
            continue
        data.append(values)
        labels.append(_display_method_label(method))
        colors.append(METHOD_COLORS[method])

    fig_h = max(4.0, 0.48 * len(data) + 1.7)
    fig, ax = plt.subplots(figsize=(7.8, fig_h))
    positions = np.arange(len(data))
    box = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=(5, 95),
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_edgecolor(color)
    for key in ("whiskers", "caps", "medians"):
        for artist in box[key]:
            artist.set_color("#333333")
            artist.set_linewidth(1.1)

    rng = np.random.default_rng(seed)
    for y, (values, color) in enumerate(zip(data, colors)):
        jitter = rng.normal(0, 0.035, size=len(values))
        ax.scatter(values, np.full(len(values), y) + jitter, color=color, alpha=0.35, s=14, linewidth=0)

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Effective Samples")
    if title is not None:
        ax.set_title(title)
    ax.grid(axis="y", visible=False)
    _finish_axis(ax)
    fig.tight_layout()
    _save_show(fig, path, show)
    return fig, ax


def plot_sequential_variance_components(
    df: pd.DataFrame,
    path: str | Path | None = None,
    n_total: int | None = None,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot across-run sequential variance for estimates and interval pieces.

    The plotted values are empirical variances across Monte Carlo replicates;
    each panel uses a log-scaled y-axis.
    """

    set_theme_bw(font_scale=1.0)
    methods = _ordered_methods(df)
    components = [
        ("log point estimate", "log estimate"),
        ("point estimate", "estimate"),
        ("lb", "left endpoint"),
        ("ub", "right endpoint"),
        ("interval width", "CI width"),
        (EFFECTIVE_N_COL, "ESS"),
    ]
    missing = [col for col, _ in components if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for sequential variance plot: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), sharex=True)
    axes_flat = axes.ravel()
    for ax, (value_col, title) in zip(axes_flat, components):
        summary = (
            df[df["estimator"].isin(methods)]
            .groupby([HUMAN_N_COL, "estimator"], observed=True)[value_col]
            .var()
            .reset_index(name="empirical_variance")
        )
        for method in methods:
            sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
            if sub.empty:
                continue
            values = sub["empirical_variance"].mask(sub["empirical_variance"] <= 0)
            ax.plot(
                sub[HUMAN_N_COL],
                values,
                label=_display_method_label(method),
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=4.6,
            )
        ax.set_title(f"Var({title})")
        ax.set_xlabel(HUMAN_N_COL)
        ax.set_ylabel("empirical variance")
        ax.set_yscale("log")
        _finish_axis(ax)

    for ax in axes_flat[3:]:
        _add_budget_fraction_ticks(ax, n_total, y_offset=-0.18)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_show(fig, path, show)
    return fig, axes


def plot_sequential_endpoint_variance_components(
    df: pd.DataFrame,
    path: str | Path | None = None,
    n_total: int | None = None,
    panel_titles: tuple[str, str] = (
        "(a) Var(left endpoint)",
        "(b) Var(right endpoint)",
    ),
    figsize: tuple[float, float] = (7.2, 3.4),
    show_fraction_ticks: bool = False,
    show: bool = True,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot sequential across-run variance for left and right CI endpoints."""

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
    methods = _ordered_methods(df)
    components = [("lb", panel_titles[0]), ("ub", panel_titles[1])]
    missing = [col for col, _ in components if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for sequential endpoint variance plot: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    for ax, (value_col, panel_title) in zip(axes, components):
        summary = (
            df[df["estimator"].isin(methods)]
            .groupby([HUMAN_N_COL, "estimator"], observed=True)[value_col]
            .var()
            .reset_index(name="empirical_variance")
        )
        for method in methods:
            sub = summary[summary["estimator"] == method].sort_values(HUMAN_N_COL)
            if sub.empty:
                continue
            values = sub["empirical_variance"].mask(sub["empirical_variance"] <= 0)
            ax.plot(
                sub[HUMAN_N_COL],
                values,
                label=_display_method_label(method),
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                marker=METHOD_MARKERS[method],
                markersize=3.8,
                linewidth=1.35,
            )
        ax.set_xlabel(HUMAN_N_COL)
        ax.set_ylabel("empirical variance")
        ax.set_yscale("log")
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=2)
        ax.locator_params(axis="x", nbins=4)
        if show_fraction_ticks:
            _add_budget_fraction_ticks(ax, n_total, y_offset=-0.20)
        _finish_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.02),
        columnspacing=1.4,
        handlelength=1.8,
        handletextpad=0.6,
    )
    bottom = 0.38 if show_fraction_ticks else 0.32
    fig.subplots_adjust(left=0.09, right=0.995, bottom=bottom, top=0.91, wspace=0.26)
    _save_show(fig, path, show)
    return fig, axes


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
            label=_display_method_label(method),
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=5.5,
        )

    ax.set_xlabel(HUMAN_N_COL)
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.legend(loc="best", ncol=2)
    _add_budget_fraction_ticks(ax, n_total, y_offset=-0.18)
    _finish_axis(ax)
    _finish_budget_figure(fig, n_total)
    _save_show(fig, path, show)
    return fig, ax


def make_monte_carlo_variance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a table of MC variance components."""

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
                label=_display_method_label(method),
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
    """Convert the MC variance component table to an image file."""

    set_theme_bw(font_scale=0.85)
    table = make_monte_carlo_variance_table(df)
    if max_rows is not None:
        table = table.head(max_rows)

    display_table = table.copy()
    if "estimator" in display_table.columns:
        display_table["estimator"] = display_table["estimator"].map(_display_method_label)
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
    hatch = {"active + tuning": "///", "active+": "///", "spline + tuning": "///", "spline+": "///"}
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
    handles = []
    seen_labels = set()
    for method in METHOD_ORDER:
        label = _display_method_label(method)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        handles.append(
            Line2D(
                [0],
                [0],
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                markersize=6,
                label=label,
            )
        )
    fig, ax = plt.subplots(figsize=(8.5, 1.4))
    ax.axis("off")
    ax.legend(handles=handles, loc="center", ncol=3)
    fig.tight_layout()
    _save_show(fig, path, show)
    return fig, ax
