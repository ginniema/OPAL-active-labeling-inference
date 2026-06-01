"""Odds-ratio simulation cases for the paper appendix.

This replaces the duplicated exploratory notebooks from
``../Simulation/odds ratio`` with one parameterized script. The four default
cases are:

- 20/80 hard/easy group split with oracle sampling probabilities.
- 20/80 hard/easy group split with estimated sampling probabilities.
- 50/50 hard/easy group split with oracle sampling probabilities.
- 50/50 hard/easy group split with estimated sampling probabilities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import (
    binary_odds_ratio_truth,
    binary_uncertainty,
    monte_carlo_variance_table,
    run_odds_ratio_monte_carlo,
    summarize_monte_carlo,
)


ProbabilitySource = Literal["oracle", "estimated"]
GroupSplit = Literal["20_80", "extreme", "balanced"]


@dataclass(frozen=True)
class OddsRatioSimulationCase:
    group_split: GroupSplit
    probability_source: ProbabilitySource
    p_easy: float

    @property
    def name(self) -> str:
        return f"odds_ratio_{self.group_split}_{self.probability_source}"

    @property
    def panel_label(self) -> str:
        split_labels = {
            "20_80": "20/80 split",
            "extreme": "95/5 split",
            "balanced": "50/50 split",
        }
        split_label = split_labels[self.group_split]
        source_label = "oracle probabilities" if self.probability_source == "oracle" else "estimated probabilities"
        return f"{split_label}, {source_label}"


@dataclass(frozen=True)
class OddsRatioRunConfig:
    n_inference: int = 10_000
    n_train: int = 200
    num_trials: int = 500
    alpha: float = 0.10
    tau: float = 0.5
    seed: int = 614
    fracs_human_min: float = 0.01
    fracs_human_max: float = 0.20
    fracs_human_count: int = 20
    active_budget: Literal["proportional", "even"] = "proportional"
    split_spline_budget_evenly: bool = False
    show_progress: bool = True

    @property
    def n_total(self) -> int:
        return self.n_inference + self.n_train

    @property
    def fracs_human(self) -> np.ndarray:
        return np.linspace(self.fracs_human_min, self.fracs_human_max, self.fracs_human_count)


def generate_odds_ratio_population(
    n_samples: int,
    p_easy: float,
    seed: int = 614,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the hard/easy group simulation population.

    ``Z=1`` is the easier majority group when ``p_easy > 0.5``. ``Z=0`` is the
    harder mixture group.
    """

    rng = np.random.default_rng(seed)
    z = rng.binomial(n=1, p=p_easy, size=n_samples)
    x = np.zeros(n_samples)
    hard = z == 0
    easy = z == 1
    x[hard] = rng.choice([-2, 0, 2], size=np.sum(hard)) + rng.normal(0, 0.5, np.sum(hard))
    x[easy] = rng.normal(0, 1, np.sum(easy))
    probability = conditional_expectation(x, z)
    y = rng.binomial(n=1, p=probability)
    return x, y, z, probability


def conditional_expectation(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """True ``P(Y=1|X,Z)`` for the odds-ratio simulation."""

    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=int)
    hard_probability = np.clip(0.3 + 0.2 * expit(x), 0.05, 0.6)
    easy_probability = expit(1.5 * x)
    return np.where(z == 1, easy_probability, hard_probability)


def prepare_population(
    case: OddsRatioSimulationCase,
    config: OddsRatioRunConfig,
) -> dict[str, np.ndarray | float]:
    """Generate data, fit the nuisance model, and return inference arrays."""

    x, y, z, true_probability = generate_odds_ratio_population(
        n_samples=config.n_total,
        p_easy=case.p_easy,
        seed=config.seed,
    )
    covariates = np.column_stack([x, z])
    cov_train, cov_inference, y_train, y_inference, prob_train, prob_inference = train_test_split(
        covariates,
        y,
        true_probability,
        train_size=config.n_train,
        random_state=42,
        stratify=z,
    )
    z_train = cov_train[:, 1].astype(int)
    z_inference = cov_inference[:, 1].astype(int)

    model = LogisticRegression(random_state=0, max_iter=1_000)
    model.fit(cov_train, y_train)
    yhat = model.predict_proba(cov_inference)[:, 1]

    group1 = z_inference == 1
    y0 = y_inference[~group1]
    y1 = y_inference[group1]
    yhat0 = yhat[~group1]
    yhat1 = yhat[group1]
    prob0 = prob_inference[~group1]
    prob1 = prob_inference[group1]

    true_odds_ratio, true_variance = binary_odds_ratio_truth(y0, y1)
    mu0_pilot = float(np.mean(y_train[z_train == 0])) if np.any(z_train == 0) else float(np.mean(y0))
    mu1_pilot = float(np.mean(y_train[z_train == 1])) if np.any(z_train == 1) else float(np.mean(y1))
    return {
        "y0": y0,
        "y1": y1,
        "yhat0": yhat0,
        "yhat1": yhat1,
        "prob0": prob0,
        "prob1": prob1,
        "true_odds_ratio": true_odds_ratio,
        "true_variance": true_variance,
        "mu0_pilot": mu0_pilot,
        "mu1_pilot": mu1_pilot,
    }


def run_case(
    case: OddsRatioSimulationCase,
    config: OddsRatioRunConfig,
) -> pd.DataFrame:
    """Run one odds-ratio case."""

    population = prepare_population(case, config)
    uncertainty0 = None
    uncertainty1 = None
    spline_score0 = None
    spline_score1 = None
    if case.probability_source == "oracle":
        prob0 = np.asarray(population["prob0"], dtype=float)
        prob1 = np.asarray(population["prob1"], dtype=float)
        uncertainty0 = binary_uncertainty(prob0)
        uncertainty1 = binary_uncertainty(prob1)
        spline_score0 = prob0 * (1 - prob0)
        spline_score1 = prob1 * (1 - prob1)

    return run_odds_ratio_monte_carlo(
        y0=np.asarray(population["y0"]),
        yhat0=np.asarray(population["yhat0"]),
        y1=np.asarray(population["y1"]),
        yhat1=np.asarray(population["yhat1"]),
        fracs_human=config.fracs_human,
        alpha=config.alpha,
        num_trials=config.num_trials,
        true_odds_ratio=float(population["true_odds_ratio"]),
        true_variance=float(population["true_variance"]),
        mu0_pilot=float(population["mu0_pilot"]),
        mu1_pilot=float(population["mu1_pilot"]),
        tau=config.tau,
        seed=config.seed,
        split_spline_budget_evenly=config.split_spline_budget_evenly,
        uncertainty0=uncertainty0,
        uncertainty1=uncertainty1,
        spline_score0=spline_score0,
        spline_score1=spline_score1,
        active_budget=config.active_budget,
        show_progress=config.show_progress,
    )


def default_cases(group: str, probability_source: str) -> list[OddsRatioSimulationCase]:
    groups = {
        "20_80": OddsRatioSimulationCase("20_80", "estimated", 0.8),
        "extreme": OddsRatioSimulationCase("extreme", "estimated", 0.95),
        "balanced": OddsRatioSimulationCase("balanced", "estimated", 0.5),
    }
    selected_groups = list(groups) if group == "all" else [group]
    selected_sources = ["oracle", "estimated"] if probability_source == "both" else [probability_source]
    cases: list[OddsRatioSimulationCase] = []
    for group_name in selected_groups:
        for source in selected_sources:
            base = groups[group_name]
            cases.append(
                OddsRatioSimulationCase(
                    group_split=base.group_split,
                    probability_source=source,  # type: ignore[arg-type]
                    p_easy=base.p_easy,
                )
            )
    return cases


def write_case_outputs(df: pd.DataFrame, case: OddsRatioSimulationCase, output_dir: Path) -> None:
    """Write raw, summary, and MC-variance tables for one case."""

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / f"{case.name}.csv", index=False)
    summarize_monte_carlo(df).to_csv(output_dir / f"{case.name}_summary.csv", index=False)
    monte_carlo_variance_table(df).to_csv(output_dir / f"{case.name}_mc_variance.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["all", "20_80", "extreme", "balanced"], default="all")
    parser.add_argument("--probability-source", choices=["both", "oracle", "estimated"], default="both")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--n-inference", type=int, default=10_000)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--active-budget", choices=["proportional", "even"], default="proportional")
    parser.add_argument("--split-spline-budget-evenly", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=614)
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run.")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Simulation" / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OddsRatioRunConfig(
        n_inference=1_000 if args.quick else args.n_inference,
        n_train=80 if args.quick else args.n_train,
        num_trials=3 if args.quick else args.trials,
        seed=args.seed,
        fracs_human_count=3 if args.quick else 20,
        active_budget=args.active_budget,
        split_spline_budget_evenly=args.split_spline_budget_evenly,
        show_progress=not args.no_progress,
    )
    for case in default_cases(args.group, args.probability_source):
        df = run_case(case, config)
        write_case_outputs(df, case, args.results_dir)


if __name__ == "__main__":
    main()
