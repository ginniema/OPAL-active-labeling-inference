"""Plot cleaned simulation results."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Simulation.odds_ratio_simulation import generate_odds_ratio_population
from plotting import (
    METHOD_COLORS,
    METHOD_LINESTYLES,
    METHOD_MARKERS,
    METHOD_ORDER,
    _display_method_label,
    _finish_axis,
    set_theme_bw,
)
from utils import (
    EFFECTIVE_N_COL,
    HUMAN_N_COL,
    active_sampling_probabilities,
    sampling_probs_spline_inv,
)


ODDS_CASES = [
    ("odds_ratio_20_80_oracle", "(a) 20/80 split, oracle probabilities"),
    ("odds_ratio_20_80_estimated", "(b) 20/80 split, estimated probabilities"),
    ("odds_ratio_balanced_oracle", "(c) 50/50 split, oracle probabilities"),
    ("odds_ratio_balanced_estimated", "(d) 50/50 split, estimated probabilities"),
]

EXTREME_ODDS_CASES = [
    ("odds_ratio_extreme_oracle", "(a) Oracle probabilities"),
    ("odds_ratio_extreme_estimated", "(b) Estimated probabilities"),
]

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


def plot_extreme_imbalance_ess(
    results_dir: Path,
    plots_dir: Path,
    include_title: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot 95/5 extreme imbalance ESS for oracle and estimated probabilities."""

    frames = []
    for case_name, _ in EXTREME_ODDS_CASES:
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
    y_max = max(float(summary[EFFECTIVE_N_COL].max()) for summary in summaries)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    for ax, summary, (_, panel_title) in zip(axes, summaries, EXTREME_ODDS_CASES):
        _draw_ess_panel(ax, summary, methods)
        ax.set_ylim(0, y_max * 1.08)
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=2)
    axes[1].set_ylabel("")

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
    top = 0.82
    if include_title:
        fig.suptitle("Extreme imbalance: 95/5 split", y=0.965, fontsize=12, fontweight="normal")
        top = 0.80
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.32, top=top, wspace=0.10)

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "odds_ratio_extreme_imbalance_effective_sample_size.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / "odds_ratio_extreme_imbalance_effective_sample_size.png", dpi=300, bbox_inches="tight")
    return fig, axes


def plot_uniform_mixing_lambda(
    results_dir: Path,
    plots_dir: Path,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the selected method-uniform mixing weight by budget."""

    path = results_dir / "uniform_mixing_lambda.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    df = pd.read_csv(path)

    set_theme_bw(font_scale=0.88)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    methods = [method for method in ("active", "spline") if method in set(df["estimator"])]

    fig, ax = plt.subplots(figsize=(4.0, 2.85))
    for method in methods:
        sub = df[df["estimator"] == method].sort_values("budget")
        ax.plot(
            sub["budget"],
            sub["lambda_method_weight"],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markersize=3.8,
            linewidth=1.35,
            label=_display_method_label(method),
        )
    ax.set_xlabel("budget")
    ax.set_ylabel(r"$\lambda^*$")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="best", frameon=False)
    _finish_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.19, top=0.98)

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "uniform_mixing_lambda.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / "uniform_mixing_lambda.png", dpi=300, bbox_inches="tight")
    return fig, ax


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


def _setting2_policy_curve(
    n_train: int,
    *,
    n_total: int = 10_000,
    p_easy: float = 0.8,
    frac_human: float = 0.10,
    seed: int = 614,
    group_value: int = 1,
) -> pd.DataFrame:
    """Return active and OPAL policy curves for Setting 2.

    ``group_value=1`` matches the majority/easy group from the original
    imbalanced Setting 2 visualization.
    """

    x, y, z, _ = generate_odds_ratio_population(n_samples=n_total, p_easy=p_easy, seed=seed)
    covariates = np.column_stack([x, z])
    cov_train, cov_test, y_train, _ = train_test_split(
        covariates,
        y,
        train_size=n_train,
        random_state=42,
        stratify=z,
    )
    z_train = cov_train[:, 1].astype(int)
    z_test = cov_test[:, 1].astype(int)

    model = LogisticRegression(random_state=0, max_iter=1_000).fit(cov_train, y_train)
    yhat = model.predict_proba(cov_test)[:, 1]
    uncertainty = np.minimum(yhat, 1 - yhat)
    spline_score = yhat * (1 - yhat)
    uncertainty1 = uncertainty[z_test == 1]
    uncertainty0 = uncertainty[z_test == 0]
    spline_score1 = spline_score[z_test == 1]
    spline_score0 = spline_score[z_test == 0]
    mu1 = float(np.mean(y_train[z_train == 1]))
    mu0 = float(np.mean(y_train[z_train == 0]))

    p1_opal, p0_opal, _, _ = sampling_probs_spline_inv(
        spline_score1,
        spline_score0,
        spline_score1,
        spline_score0,
        mu1,
        mu0,
        n_total * frac_human,
        num_knots=5,
        degree=3,
        split_budget_evenly=False,
    )
    if p1_opal is None or p0_opal is None:
        raise RuntimeError(f"OPAL policy optimization failed for n_train={n_train}.")

    if group_value == 1:
        group_uncertainty = uncertainty1
        opal_probability = p1_opal
    elif group_value == 0:
        group_uncertainty = uncertainty0
        opal_probability = p0_opal
    else:
        raise ValueError("group_value must be 0 or 1")

    active_probability = active_sampling_probabilities(
        group_uncertainty,
        target_mean_probability=frac_human,
        tau=0.0,
    )
    active_mixed_probability = active_sampling_probabilities(
        group_uncertainty,
        target_mean_probability=frac_human,
        tau=0.5,
    )
    order = np.argsort(group_uncertainty)
    return pd.DataFrame(
        {
            "uncertainty": group_uncertainty[order],
            "active": active_probability[order],
            "active + uniform mixing": active_mixed_probability[order],
            "OPAL": opal_probability[order],
            "n_train": n_train,
            "p_easy": p_easy,
        }
    )


def plot_setting2_labeling_policy(
    plots_dir: Path,
    *,
    train_sizes: tuple[int, int] = (100, 500),
    frac_human: float = 0.10,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Replicate the Setting 2 active-vs-OPAL policy figure."""

    set_theme_bw(font_scale=0.88)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
        }
    )
    curves = [_setting2_policy_curve(n_train, frac_human=frac_human) for n_train in train_sizes]
    y_max = max(
        float(curve[["active", "active + uniform mixing", "OPAL"]].to_numpy().max())
        for curve in curves
    ) * 1.12

    fig, axes = plt.subplots(1, len(train_sizes), figsize=(6.4, 2.35), sharey=True)
    if len(train_sizes) == 1:
        axes = np.array([axes])
    for panel_index, (ax, curve, n_train) in enumerate(zip(axes, curves, train_sizes)):
        ax.plot(
            curve["uncertainty"],
            curve["OPAL"],
            color=METHOD_COLORS["spline"],
            label="OPAL",
            linewidth=2.0,
        )
        ax.plot(
            curve["uncertainty"],
            curve["active"],
            color=METHOD_COLORS["active"],
            label="active",
            linewidth=2.0,
        )
        ax.plot(
            curve["uncertainty"],
            curve["active + uniform mixing"],
            color=METHOD_COLORS["active"],
            linestyle="--",
            label="active + uniform mixing",
            linewidth=2.0,
        )
        ax.set_title(rf"$n_{{\mathrm{{train}}}} = {n_train}$", fontweight="normal", pad=4)
        ax.set_xlabel("uncertainty score")
        ax.set_xlim(0, 0.52)
        ax.set_ylim(0, y_max)
        if panel_index == 0:
            ax.legend(loc="upper left", frameon=True, borderpad=0.3, handlelength=1.2)
        _finish_axis(ax)
    axes[0].set_ylabel(r"labeling probability $\pi(u)$")
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.22, top=0.84, wspace=0.13)

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "odds_ratio_setting2_labeling_policy.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / "odds_ratio_setting2_labeling_policy.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_setting_policy_comparison(
    plots_dir: Path,
    *,
    n_train: int = 500,
    frac_human: float = 0.10,
    show: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Compare imbalanced and balanced odds-ratio policy curves."""

    set_theme_bw(font_scale=0.88)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
        }
    )
    panel_specs = [
        (0.8, "(a) Imbalanced setting: 80/20 split"),
        (0.5, "(b) Balanced setting: 50/50 split"),
    ]
    curves = [
        _setting2_policy_curve(n_train, p_easy=p_easy, frac_human=frac_human)
        for p_easy, _ in panel_specs
    ]
    y_max = max(
        float(curve[["active", "active + uniform mixing", "OPAL"]].to_numpy().max())
        for curve in curves
    ) * 1.12

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.35), sharey=True)
    for ax, curve, (_, panel_title) in zip(axes, curves, panel_specs):
        ax.plot(
            curve["uncertainty"],
            curve["OPAL"],
            color=METHOD_COLORS["spline"],
            label="OPAL",
            linewidth=2.0,
        )
        ax.plot(
            curve["uncertainty"],
            curve["active"],
            color=METHOD_COLORS["active"],
            label="active",
            linewidth=2.0,
        )
        ax.plot(
            curve["uncertainty"],
            curve["active + uniform mixing"],
            color=METHOD_COLORS["active"],
            linestyle="--",
            label=r"active + uniform mixing",
            linewidth=2.0,
        )
        ax.set_title(panel_title, loc="left", fontweight="normal", pad=4)
        ax.set_xlabel("uncertainty score")
        ax.set_xlim(0, 0.52)
        ax.set_ylim(0, y_max)
        _finish_axis(ax)
    axes[0].set_ylabel(r"labeling probability $\pi(u)$")
    fig.suptitle(rf"$n_{{\mathrm{{train}}}} = {n_train}$", y=0.99, fontsize=11, fontweight="normal")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.015),
        frameon=False,
        columnspacing=1.4,
        handlelength=2.0,
        handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.105, right=0.995, bottom=0.34, top=0.78, wspace=0.13)

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "odds_ratio_setting_policy_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(plots_dir / "odds_ratio_setting_policy_comparison.png", dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Simulation" / "results")
    parser.add_argument("--plots-dir", type=Path, default=REPO_ROOT / "Simulation" / "plots")
    parser.add_argument("--skip-odds", action="store_true")
    parser.add_argument("--skip-kendall", action="store_true")
    parser.add_argument("--plot-extreme", action="store_true")
    parser.add_argument("--plot-uniform-mixing", action="store_true")
    parser.add_argument("--plot-policy", action="store_true")
    parser.add_argument("--plot-policy-comparison", action="store_true")
    parser.add_argument("--policy-n-train", type=int, default=500)
    parser.add_argument("--include-odds-title", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_odds:
        fig, _ = plot_odds_ratio_ess_grid(args.results_dir, args.plots_dir, include_title=args.include_odds_title)
        plt.close(fig)
    if args.plot_extreme:
        fig, _ = plot_extreme_imbalance_ess(
            args.results_dir,
            args.plots_dir,
            include_title=args.include_odds_title,
        )
        plt.close(fig)
    if not args.skip_kendall:
        fig, _ = plot_kendall_tau_ess(args.results_dir, args.plots_dir)
        plt.close(fig)
    if args.plot_uniform_mixing:
        fig, _ = plot_uniform_mixing_lambda(args.results_dir, args.plots_dir)
        plt.close(fig)
    if args.plot_policy:
        fig, _ = plot_setting2_labeling_policy(args.plots_dir)
        plt.close(fig)
    if args.plot_policy_comparison:
        fig, _ = plot_setting_policy_comparison(args.plots_dir, n_train=args.policy_n_train)
        plt.close(fig)


if __name__ == "__main__":
    main()
