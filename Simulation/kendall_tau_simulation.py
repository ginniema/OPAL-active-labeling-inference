"""Kendall's tau active-labeling simulation.

This script cleans up the original exploratory code from ``../Kendall's tau``.
It keeps the Kendall-specific data-generating process and one-step estimator
local to the simulation, while reusing the shared plotting column names and
spline-basis helper from the repo root.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import cvxpy as cp
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils import EFFECTIVE_N_COL, HUMAN_N_COL, setup_spline_basis


@dataclass(frozen=True)
class KendallSimulationConfig:
    n: int = 20_000
    d: int = 12
    n_seed: int = 2_000
    gamma: float = 1.1
    sigma_s: float = 0.7
    trials: int = 500
    alpha: float = 0.10
    seed: int = 614
    mix_active: float = 0.1
    num_knots: int = 6
    include_oracle: bool = False
    include_seed_in_inference: bool = False
    include_classical: bool = True
    rho_min: float = 0.05
    rho_max: float = 0.20
    rho_count: int = 20


def generate_kendall_data(
    n: int = 30_000,
    d: int = 12,
    gamma: float = 1.1,
    sigma_s: float = 0.7,
    seed: int = 614,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate covariates, observed score, binary label, and true label prob."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    s_star = (
        1.1 * x[:, 0]
        - 0.9 * x[:, 1]
        + 0.7 * x[:, 2]
        + 0.6 * np.sin(x[:, 3])
        - 0.6 * x[:, 4] * x[:, 5]
        + 0.4 * (x[:, 6] > 0)
        - 0.4 * x[:, 7] * (x[:, 8] > 0)
    )
    score = s_star + rng.normal(0, sigma_s, size=n)
    label_prob = expit(-0.3 + gamma * s_star + 0.4 * np.sin(x[:, 0]) - 0.5 * x[:, 2] * x[:, 3])
    y = rng.binomial(1, label_prob)
    return x, score, y, label_prob


def fit_mu_on_seed(
    x: np.ndarray,
    y: np.ndarray,
    n_seed: int = 2_000,
    seed: int = 614,
    calibrate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``mu_hat=P(Y=1|X)`` on a seed set and predict for all units."""

    rng = np.random.default_rng(seed)
    seed_idx = rng.choice(len(y), size=min(n_seed, len(y)), replace=False)
    is_seed = np.zeros(len(y), dtype=bool)
    is_seed[seed_idx] = True

    model = LogisticRegression(max_iter=1_000)
    model.fit(x[is_seed], y[is_seed])
    mu_hat = model.predict_proba(x)[:, 1]

    if calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3)
        calibrator.fit(mu_hat[is_seed], y[is_seed])
        mu_hat = calibrator.transform(mu_hat)
    return is_seed, np.clip(mu_hat, 1e-3, 1 - 1e-3)


def kendall_ab_from_score(score: np.ndarray, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the ``A_i`` and ``B_i`` components for Kendall's tau-a."""

    score = np.asarray(score, dtype=float)
    mu = np.asarray(mu, dtype=float)
    n = len(score)

    order = np.argsort(score, kind="mergesort")
    score_sorted = score[order]
    mu_sorted = mu[order]
    prefix_mu = np.concatenate([[0.0], np.cumsum(mu_sorted)])
    total_mu = prefix_mu[-1]

    a_sorted = np.empty(n, dtype=float)
    b_sorted = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and score_sorted[j] == score_sorted[i]:
            j += 1
        mid_rank = 0.5 * (i + (j - 1)) + 1.0
        a_sorted[i:j] = 2.0 * mid_rank / n - 1.0
        mu_less = prefix_mu[i]
        mu_greater = total_mu - prefix_mu[j]
        b_sorted[i:j] = (mu_less - mu_greater) / n
        i = j

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(n)
    return a_sorted[inverse_order], b_sorted[inverse_order]


def kendall_tau_aipw_from_components(
    y: np.ndarray,
    selected: np.ndarray,
    probabilities: np.ndarray,
    mu_hat: np.ndarray,
    a_hat: np.ndarray,
    b_hat: np.ndarray,
    alpha: float,
) -> tuple[float, float, float, float]:
    """Return tau estimate, lower CI, upper CI, and scaled EIF variance."""

    selected = np.asarray(selected, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    term = selected / probabilities * (y - mu_hat) * a_hat + mu_hat * a_hat - b_hat
    tau_hat = float(np.mean(term))
    scaled_variance = float(np.mean((term - tau_hat) ** 2))
    half_width = norm.ppf(1 - alpha / 2) * np.sqrt(scaled_variance / len(y))
    return tau_hat, tau_hat - half_width, tau_hat + half_width, scaled_variance


def classical_kendall_influence(score: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Return label-only Kendall tau-a and U-statistic influence values."""

    score = np.asarray(score)
    y = np.asarray(y, dtype=int)
    m = len(y)
    if m < 2:
        return np.nan, np.full(m, np.nan)

    order = np.argsort(score, kind="mergesort")
    score_sorted = score[order]
    y_sorted = y[order]
    pos_prefix = np.concatenate([[0], np.cumsum(y_sorted)])
    neg_prefix = np.concatenate([[0], np.cumsum(1 - y_sorted)])
    total_pos = int(pos_prefix[-1])
    total_neg = int(neg_prefix[-1])

    hbar_sorted = np.empty(m, dtype=float)
    concordant = 0
    discordant = 0
    i = 0
    while i < m:
        j = i
        while j < m and score_sorted[j] == score_sorted[i]:
            j += 1

        pos_less = int(pos_prefix[i])
        neg_less = int(neg_prefix[i])
        pos_tied = int(pos_prefix[j] - pos_prefix[i])
        neg_tied = int(neg_prefix[j] - neg_prefix[i])
        pos_greater = total_pos - pos_less - pos_tied
        neg_greater = total_neg - neg_less - neg_tied

        block_y = y_sorted[i:j]
        concordant += int(block_y.sum() * neg_less)
        discordant += int((1 - block_y).sum() * pos_less)

        values = np.where(
            block_y == 1,
            neg_less - neg_greater,
            pos_greater - pos_less,
        )
        hbar_sorted[i:j] = values / (m - 1)
        i = j

    tau_hat = float((concordant - discordant) / (m * (m - 1) / 2))
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(m)
    hbar = hbar_sorted[inverse_order]
    influence = 2.0 * (hbar - tau_hat)
    return tau_hat, influence


def classical_kendall_tau_ci(
    score: np.ndarray,
    y: np.ndarray,
    selected: np.ndarray,
    alpha: float,
    population_n: int,
) -> tuple[float, float, float, float]:
    """Label-only Kendall tau-a CI on a uniform human-label sample."""

    selected = np.asarray(selected, dtype=bool)
    selected_score = score[selected]
    selected_y = y[selected]
    point, influence = classical_kendall_influence(selected_score, selected_y)
    if len(selected_y) < 3 or not np.isfinite(point):
        return point, np.nan, np.nan, np.nan

    selected_n = len(selected_y)
    influence_variance = float(np.var(influence, ddof=1))
    scaled_variance = population_n * influence_variance / selected_n
    half_width = norm.ppf(1 - alpha / 2) * np.sqrt(scaled_variance / population_n)
    return point, point - half_width, point + half_width, scaled_variance


def c_tau_kendall(score: np.ndarray, mu_hat: np.ndarray) -> np.ndarray:
    """Policy-design weight ``A_i^2 mu_hat_i (1-mu_hat_i)``."""

    a_hat, _ = kendall_ab_from_score(score, mu_hat)
    return a_hat**2 * mu_hat * (1 - mu_hat)


def scaled_active_probabilities(
    score: np.ndarray,
    rho: float,
    eps: float,
    mix_uniform: float,
    iterations: int = 60,
) -> np.ndarray:
    """Scale nonnegative scores to have mean ``rho``, with clipping."""

    score = np.maximum(np.asarray(score, dtype=float), 0.0)
    if np.isclose(np.mean(score), 0):
        probabilities = np.full(len(score), rho)
    else:
        lo, hi = 1e-12, 1e6
        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            if np.clip(mid * score, eps, 1.0).mean() >= rho:
                hi = mid
            else:
                lo = mid
        probabilities = np.clip(lo * score, eps, 1.0)
    if mix_uniform > 0:
        probabilities = (1 - mix_uniform) * probabilities + mix_uniform * rho
    return np.clip(probabilities, eps, 1.0)


def oracle_probabilities_from_c(c_weight: np.ndarray, rho: float, eps: float) -> np.ndarray:
    """Oracle Neyman allocation proportional to ``sqrt(c_i)``."""

    return scaled_active_probabilities(np.sqrt(np.maximum(c_weight, 0.0)), rho, eps, mix_uniform=0.0)


def spline_policy_inverse(
    basis_score: np.ndarray,
    c_weight: np.ndarray,
    rho: float,
    eps: float,
    num_knots: int = 6,
    degree: int = 3,
    solver_names: tuple[str, ...] = (cp.ECOS, "CLARABEL", cp.SCS),
) -> np.ndarray:
    """Spline policy for one population with ``log(1/pi)=B(score) theta``."""

    basis, _ = setup_spline_basis(basis_score, num_knots=num_knots, degree=degree, name="u")
    theta = cp.Variable(basis.shape[1])
    log_inv_probability = basis @ theta
    inv_probability = cp.exp(log_inv_probability)
    probability = cp.exp(-log_inv_probability)
    problem = cp.Problem(
        cp.Minimize(cp.sum(cp.multiply(c_weight, inv_probability))),
        [
            cp.sum(probability) <= rho * len(c_weight),
            log_inv_probability >= 0.0,
            log_inv_probability <= np.log(1.0 / eps),
        ],
    )

    installed = set(cp.installed_solvers())
    for solver_name in solver_names:
        if solver_name not in installed:
            continue
        try:
            kwargs = {}
            if solver_name == cp.ECOS:
                kwargs.update({"max_iters": 300, "abstol": 1e-8, "reltol": 1e-7})
            elif solver_name == cp.SCS:
                kwargs.update({"eps": 1e-5})
            problem.solve(solver=solver_name, **kwargs)
        except Exception:
            continue
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
            return np.clip(np.exp(-(basis @ theta.value)), eps, 1.0)
    raise RuntimeError(f"Spline policy optimization failed with status {problem.status}.")


def tau_true_full(score: np.ndarray, y: np.ndarray) -> float:
    """Exact finite-population Kendall tau-a for binary labels."""

    score = np.asarray(score)
    y = np.asarray(y)
    n = len(y)
    order = np.argsort(score, kind="mergesort")
    score_sorted = score[order]
    y_sorted = y[order]
    pos_prefix = np.cumsum(y_sorted)
    neg_prefix = np.cumsum(1 - y_sorted)

    concordant = 0
    discordant = 0
    i = 0
    while i < n:
        j = i
        while j < n and score_sorted[j] == score_sorted[i]:
            j += 1
        pos_below = pos_prefix[i - 1] if i > 0 else 0
        neg_below = neg_prefix[i - 1] if i > 0 else 0
        block_y = y_sorted[i:j]
        concordant += int(block_y.sum() * neg_below)
        discordant += int((1 - block_y).sum() * pos_below)
        i = j
    return float((concordant - discordant) / (n * (n - 1) / 2))


def summarize_kendall_results(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return Monte Carlo performance summary and ESS summary."""

    summary = (
        results.groupby(["rho", HUMAN_N_COL, "estimator"], observed=True)
        .agg(
            point_estimate_mean=("point estimate", "mean"),
            point_estimate_variance=("point estimate", "var"),
            lb_mean=("lb", "mean"),
            lb_variance=("lb", "var"),
            ub_mean=("ub", "mean"),
            ub_variance=("ub", "var"),
            interval_width_mean=("interval width", "mean"),
            interval_width_variance=("interval width", "var"),
            coverage=("coverage", "mean"),
            variance_estimate=("variance estimate", "mean"),
            n_trials=("trial", "count"),
        )
        .reset_index()
    )

    ess_rows = []
    reference_variance_col = "classical reference variance"
    population_n_col = "population n"
    has_classical_reference = (
        reference_variance_col in results.columns
        and population_n_col in results.columns
        and results[reference_variance_col].notna().any()
    )
    if has_classical_reference:
        reference_variance = float(results[reference_variance_col].dropna().iloc[0])
        population_n = int(results[population_n_col].dropna().iloc[0])
        for (rho, n_human), budget_summary in summary.groupby(["rho", HUMAN_N_COL], observed=True):
            for _, row in budget_summary.iterrows():
                if row["estimator"] == "classical":
                    ess = float(n_human)
                else:
                    ess = float(population_n * reference_variance / row["variance_estimate"])
                ess_rows.append(
                    {
                        "rho": rho,
                        HUMAN_N_COL: n_human,
                        "estimator": row["estimator"],
                        EFFECTIVE_N_COL: ess,
                    }
                )
    else:
        for (rho, n_human), budget_summary in summary.groupby(["rho", HUMAN_N_COL], observed=True):
            uniform_variance = budget_summary.loc[
                budget_summary["estimator"] == "uniform", "variance_estimate"
            ]
            if uniform_variance.empty:
                continue
            base_variance = float(uniform_variance.iloc[0])
            for _, row in budget_summary.iterrows():
                ess_rows.append(
                    {
                        "rho": rho,
                        HUMAN_N_COL: n_human,
                        "estimator": row["estimator"],
                        EFFECTIVE_N_COL: float(n_human * base_variance / row["variance_estimate"]),
                    }
                )
    return summary, pd.DataFrame(ess_rows)


def run_kendall_simulation(config: KendallSimulationConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the Kendall tau simulation and return raw, summary, and ESS tables."""

    x, score, y, _ = generate_kendall_data(
        n=config.n,
        d=config.d,
        gamma=config.gamma,
        sigma_s=config.sigma_s,
        seed=config.seed,
    )
    is_seed, mu_hat_all = fit_mu_on_seed(x, y, n_seed=config.n_seed, seed=config.seed)

    inference_mask = np.ones(len(y), dtype=bool) if config.include_seed_in_inference else ~is_seed
    score = score[inference_mask]
    y = y[inference_mask].astype(float)
    mu_hat = mu_hat_all[inference_mask]
    n_inference = len(y)

    a_hat, b_hat = kendall_ab_from_score(score, mu_hat)
    c_hat = c_tau_kendall(score, mu_hat)
    normalized_c = (c_hat - c_hat.min()) / (c_hat.max() - c_hat.min() + 1e-12)
    tau_star = tau_true_full(score, y)
    _, classical_full_influence = classical_kendall_influence(score, y)
    classical_reference_variance = float(np.var(classical_full_influence, ddof=1))

    rng = np.random.default_rng(config.seed + 1)
    rows = []
    rho_values = np.linspace(config.rho_min, config.rho_max, config.rho_count)
    for budget_index, rho in enumerate(rho_values):
        eps = max(0.02, rho / 5)
        policies = {
            "uniform": np.full(n_inference, rho),
            "active": scaled_active_probabilities(normalized_c, rho, eps, config.mix_active),
            "spline": spline_policy_inverse(
                normalized_c,
                c_hat,
                rho,
                eps,
                num_knots=config.num_knots,
            ),
        }
        if config.include_oracle:
            policies["oracle"] = oracle_probabilities_from_c(c_hat, rho, eps)

        for trial in range(config.trials):
            uniforms = rng.random(n_inference)
            selected_by_estimator = {}
            for estimator, probabilities in policies.items():
                selected = uniforms < probabilities
                selected_by_estimator[estimator] = selected
                point, lb, ub, variance = kendall_tau_aipw_from_components(
                    y,
                    selected,
                    probabilities,
                    mu_hat,
                    a_hat,
                    b_hat,
                    alpha=config.alpha,
                )
                rows.append(
                    {
                        "trial": trial,
                        "rho": rho,
                        HUMAN_N_COL: int(round(rho * n_inference)),
                        "estimator": estimator,
                        "point estimate": point,
                        "lb": lb,
                        "ub": ub,
                        "interval width": ub - lb,
                        "variance estimate": variance,
                        "coverage": bool(lb <= tau_star <= ub),
                        "true tau": tau_star,
                        "budget_index": budget_index,
                        "include seed in inference": config.include_seed_in_inference,
                        "classical reference variance": classical_reference_variance,
                        "population n": n_inference,
                    }
                )

            if config.include_classical:
                point, lb, ub, variance = classical_kendall_tau_ci(
                    score,
                    y,
                    selected_by_estimator["uniform"],
                    alpha=config.alpha,
                    population_n=n_inference,
                )
                rows.append(
                    {
                        "trial": trial,
                        "rho": rho,
                        HUMAN_N_COL: int(round(rho * n_inference)),
                        "estimator": "classical",
                        "point estimate": point,
                        "lb": lb,
                        "ub": ub,
                        "interval width": ub - lb,
                        "variance estimate": variance,
                        "coverage": bool(lb <= tau_star <= ub) if np.isfinite(lb) and np.isfinite(ub) else False,
                        "true tau": tau_star,
                        "budget_index": budget_index,
                        "include seed in inference": config.include_seed_in_inference,
                        "classical reference variance": classical_reference_variance,
                        "population n": n_inference,
                    }
                )

    raw = pd.DataFrame(rows)
    summary, ess = summarize_kendall_results(raw)
    return raw, summary, ess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--n", type=int, default=20_000)
    parser.add_argument("--n-seed", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=614)
    parser.add_argument("--include-oracle", action="store_true")
    parser.add_argument(
        "--include-classical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the label-only classical baseline on the uniform sample.",
    )
    parser.add_argument(
        "--include-seed-in-inference",
        action="store_true",
        help="Use the archived convention where seed-fit labels remain in the Monte Carlo population.",
    )
    parser.add_argument("--quick", action="store_true", help="Small smoke-test run.")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "Simulation" / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = KendallSimulationConfig(
        n=3_000 if args.quick else args.n,
        n_seed=300 if args.quick else args.n_seed,
        trials=5 if args.quick else args.trials,
        seed=args.seed,
        include_oracle=args.include_oracle,
        include_seed_in_inference=args.include_seed_in_inference,
        include_classical=args.include_classical,
        rho_count=3 if args.quick else 20,
    )
    raw, summary, ess = run_kendall_simulation(config)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.results_dir / "kendall_tau_results.csv", index=False)
    summary.to_csv(args.results_dir / "kendall_tau_summary.csv", index=False)
    ess.to_csv(args.results_dir / "kendall_tau_ess.csv", index=False)


if __name__ == "__main__":
    main()
