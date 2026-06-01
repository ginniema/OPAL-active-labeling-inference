"""Tune method-uniform mixing weights for the odds-ratio simulation.

For each budget, we estimate the Monte Carlo variance obtained by mixing either active or OPAL sampling with
uniform sampling and chooses the method weight that minimizes the empirical variance of the log odds-ratio estimate on the labeled pilot set.

The mixing convention is
    pi_lambda = lambda * pi_method + (1 - lambda) * pi_uniform,
so lambda=1 is the original method and lambda=0 is pure uniform sampling.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Simulation.odds_ratio_simulation import generate_odds_ratio_population
from utils import (
    HUMAN_N_COL,
    active_sampling_probabilities,
    binary_uncertainty,
    inverse_probability_weights,
    mix_probabilities_with_uniform,
    normalize_probabilities_to_budget,
    sampling_probs_spline_inv,
)


GroupSplit = Literal["20_80", "extreme", "balanced"]
ProbabilitySource = Literal["estimated", "oracle"]
Estimator = Literal["active", "spline"]


@dataclass(frozen=True)
class UniformMixingConfig:
    n_inference: int = 10_000
    n_train: int = 500
    p_easy: float = 0.8
    probability_source: ProbabilitySource = "estimated"
    budgets: tuple[float, ...] = tuple(np.linspace(0.01, 0.20, 20))
    lambda_grid: tuple[float, ...] = tuple(np.linspace(0.0, 1.0, 21))
    num_trials: int = 500
    seed: int = 614
    num_knots: int = 5
    degree: int = 3
    show_progress: bool = True

    @property
    def n_total(self) -> int:
        return self.n_inference + self.n_train


@dataclass(frozen=True)
class PilotPopulation:
    y0: np.ndarray
    y1: np.ndarray
    yhat0: np.ndarray
    yhat1: np.ndarray
    policy_score0: np.ndarray
    policy_score1: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y0) + len(self.y1)


def p_easy_for_group_split(group_split: GroupSplit) -> float:
    if group_split == "20_80":
        return 0.8
    if group_split == "extreme":
        return 0.95
    if group_split == "balanced":
        return 0.5
    raise ValueError(f"Unknown group split: {group_split}")


def _split_by_group(values: np.ndarray, group1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    group1 = np.asarray(group1, dtype=bool)
    return values[~group1], values[group1]


def prepare_pilot_population(config: UniformMixingConfig) -> PilotPopulation:
    """Generate the labeled pilot set used to tune lambda."""

    x, y, z, true_probability = generate_odds_ratio_population(
        n_samples=config.n_total,
        p_easy=config.p_easy,
        seed=config.seed,
    )
    covariates = np.column_stack([x, z])
    cov_train, _, y_train, _, prob_train, _ = train_test_split(
        covariates,
        y,
        true_probability,
        train_size=config.n_train,
        random_state=42,
        stratify=z,
    )

    model = LogisticRegression(random_state=0, max_iter=1_000)
    model.fit(cov_train, y_train)
    yhat_train = model.predict_proba(cov_train)[:, 1]
    group1_train = cov_train[:, 1].astype(int) == 1

    y0, y1 = _split_by_group(y_train, group1_train)
    yhat0, yhat1 = _split_by_group(yhat_train, group1_train)
    prob0, prob1 = _split_by_group(prob_train, group1_train)

    if config.probability_source == "oracle":
        policy_score0 = prob0 * (1.0 - prob0)
        policy_score1 = prob1 * (1.0 - prob1)
    else:
        policy_score0 = yhat0 * (1.0 - yhat0)
        policy_score1 = yhat1 * (1.0 - yhat1)

    return PilotPopulation(
        y0=y0,
        y1=y1,
        yhat0=yhat0,
        yhat1=yhat1,
        policy_score0=policy_score0,
        policy_score1=policy_score1,
    )


def _normalize_pair_to_budget(
    p0: np.ndarray,
    p1: np.ndarray,
    expected_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    joined = normalize_probabilities_to_budget(np.concatenate([p0, p1]), expected_budget)
    return joined[: len(p0)], joined[len(p0) :]


def method_probabilities(
    pilot: PilotPopulation,
    estimator: Estimator,
    budget: float,
    config: UniformMixingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pilot-set probabilities for one base method and budget."""

    expected_budget = pilot.n * budget
    if estimator == "active":
        p0 = active_sampling_probabilities(
            binary_uncertainty(pilot.yhat0),
            target_mean_probability=budget,
            tau=0.0,
        )
        p1 = active_sampling_probabilities(
            binary_uncertainty(pilot.yhat1),
            target_mean_probability=budget,
            tau=0.0,
        )
        return _normalize_pair_to_budget(p0, p1, expected_budget)

    if estimator == "spline":
        mu0 = float(np.mean(pilot.y0))
        mu1 = float(np.mean(pilot.y1))
        p1, p0, _, _ = sampling_probs_spline_inv(
            pilot.policy_score1,
            pilot.policy_score0,
            pilot.policy_score1,
            pilot.policy_score0,
            mu1,
            mu0,
            expected_budget,
            num_knots=config.num_knots,
            degree=config.degree,
            split_budget_evenly=False,
        )
        if p0 is None or p1 is None:
            raise RuntimeError(f"OPAL policy optimization failed for budget={budget:.3f}")
        return _normalize_pair_to_budget(p0, p1, expected_budget)

    raise ValueError(f"Unknown estimator: {estimator}")


def mixed_probabilities(
    p0_method: np.ndarray,
    p1_method: np.ndarray,
    budget: float,
    method_weight: float,
    expected_budget: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix a method policy with uniform and preserve the total budget."""

    p0 = mix_probabilities_with_uniform(p0_method, budget, method_weight)
    p1 = mix_probabilities_with_uniform(p1_method, budget, method_weight)
    return _normalize_pair_to_budget(p0, p1, expected_budget)


def estimate_stability_metrics(
    pilot: PilotPopulation,
    p0: np.ndarray,
    p1: np.ndarray,
    num_trials: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Monte Carlo estimate of point-estimate stability metrics."""

    def finite_mean_sd(values: np.ndarray) -> tuple[float, float, float, float]:
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return np.nan, np.nan, np.nan, 0.0
        if len(finite) == 1:
            return float(finite[0]), np.nan, np.nan, 1.0
        mean = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1))
        variance = float(np.var(finite, ddof=1))
        return mean, sd, variance, len(finite) / len(values)

    def log_odds_ratio_estimate_and_variance(
        weights0: np.ndarray,
        weights1: np.ndarray,
    ) -> tuple[float, float]:
        eta0 = pilot.yhat0 + (pilot.y0 - pilot.yhat0) * weights0
        eta1 = pilot.yhat1 + (pilot.y1 - pilot.yhat1) * weights1
        mu0_hat = float(np.clip(np.mean(eta0), 1e-10, 1.0 - 1e-10))
        mu1_hat = float(np.clip(np.mean(eta1), 1e-10, 1.0 - 1e-10))
        log_point = np.log(mu1_hat / (1.0 - mu1_hat)) - np.log(mu0_hat / (1.0 - mu0_hat))

        n0 = len(pilot.y0)
        n1 = len(pilot.y1)
        n = n0 + n1
        var_mu0_hat = np.var(eta0)
        var_mu1_hat = np.var(eta1)
        var0 = var_mu0_hat / ((mu0_hat * (1.0 - mu0_hat)) ** 2)
        var1 = var_mu1_hat / ((mu1_hat * (1.0 - mu1_hat)) ** 2)
        scaled_variance = (n / n0) * var0 + (n / n1) * var1
        return float(log_point), float(scaled_variance)

    log_points = np.empty(num_trials)
    variances = np.empty(num_trials)
    for trial in range(num_trials):
        _, weights0 = inverse_probability_weights(p0, rng)
        _, weights1 = inverse_probability_weights(p1, rng)
        log_points[trial], variances[trial] = log_odds_ratio_estimate_and_variance(
            weights0,
            weights1,
        )

    log_point_mean, log_point_sd, log_point_variance, finite_fraction = finite_mean_sd(log_points)
    variance_mean, variance_sd, variance_variance, _ = finite_mean_sd(variances)
    return {
        "mean log point estimate": log_point_mean,
        "sd log point estimate": log_point_sd,
        "mc log point estimate variance": log_point_variance,
        "finite estimate fraction": finite_fraction,
        "mean variance estimate": variance_mean,
        "sd variance estimate": variance_sd,
        "mc variance estimate variance": variance_variance,
    }


def run_uniform_mixing_tuning(
    config: UniformMixingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tune lambda over all budgets for active and OPAL."""

    pilot = prepare_pilot_population(config)
    methods: tuple[Estimator, ...] = ("active", "spline")
    budget_iterable = list(config.budgets)
    if config.show_progress:
        from tqdm.auto import tqdm

        budget_iterable = tqdm(budget_iterable, desc="uniform-mixing budgets")

    grid_rows = []
    selected_rows = []
    for budget_index, budget in enumerate(budget_iterable):
        expected_budget = pilot.n * float(budget)
        for method_index, estimator in enumerate(methods):
            p0_method, p1_method = method_probabilities(pilot, estimator, float(budget), config)
            method_rows = []
            for lambda_index, method_weight in enumerate(config.lambda_grid):
                p0_mix, p1_mix = mixed_probabilities(
                    p0_method,
                    p1_method,
                    float(budget),
                    float(method_weight),
                    expected_budget,
                )
                seed_sequence = np.random.SeedSequence(
                    [config.seed, budget_index, method_index, lambda_index]
                )
                metrics = estimate_stability_metrics(
                    pilot,
                    p0_mix,
                    p1_mix,
                    config.num_trials,
                    np.random.default_rng(seed_sequence),
                )
                row = {
                    "budget": float(budget),
                    HUMAN_N_COL: int(round(expected_budget)),
                    "estimator": estimator,
                    "lambda_method_weight": float(method_weight),
                    "uniform_weight": float(1.0 - method_weight),
                    "num_trials": config.num_trials,
                    "n_train": config.n_train,
                    "probability_source": config.probability_source,
                    **metrics,
                }
                grid_rows.append(row)
                method_rows.append(row)

            method_df = pd.DataFrame(method_rows)
            best_idx = method_df["mc log point estimate variance"].idxmin()
            selected_row = method_df.loc[best_idx].to_dict()
            selected_row["selection_metric"] = "mc log point estimate variance"
            selected_rows.append(selected_row)

    return pd.DataFrame(selected_rows), pd.DataFrame(grid_rows)


def write_outputs(
    selected: pd.DataFrame,
    grid: pd.DataFrame,
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(results_dir / "uniform_mixing_lambda.csv", index=False)
    grid.to_csv(results_dir / "uniform_mixing_lambda_grid.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["20_80", "extreme", "balanced"], default="20_80")
    parser.add_argument("--probability-source", choices=["estimated", "oracle"], default="estimated")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--n-inference", type=int, default=10_000)
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--budget-min", type=float, default=0.01)
    parser.add_argument("--budget-max", type=float, default=0.20)
    parser.add_argument("--budget-count", type=int, default=20)
    parser.add_argument("--lambda-count", type=int, default=21)
    parser.add_argument("--seed", type=int, default=614)
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Simulation" / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = UniformMixingConfig(
        n_inference=1_000 if args.quick else args.n_inference,
        n_train=120 if args.quick else args.n_train,
        p_easy=p_easy_for_group_split(args.group),
        probability_source=args.probability_source,
        budgets=tuple(
            np.linspace(
                args.budget_min,
                args.budget_max,
                3 if args.quick else args.budget_count,
            )
        ),
        lambda_grid=tuple(np.linspace(0.0, 1.0, 5 if args.quick else args.lambda_count)),
        num_trials=10 if args.quick else args.trials,
        seed=args.seed,
        show_progress=not args.no_progress,
    )
    selected, grid = run_uniform_mixing_tuning(config)
    write_outputs(selected, grid, args.results_dir)


if __name__ == "__main__":
    main()
