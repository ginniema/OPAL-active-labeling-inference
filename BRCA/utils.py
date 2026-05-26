"""Shared numerical helpers for the active-inference examples.

The functions in this module are intentionally not tied to a particular
dataset. BRCA-specific data loading and presentation code should stay in the
notebook or in a dataset-specific module.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable

import cvxpy as cp
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm


HUMAN_N_COL = r"$n_{\mathrm{human}}$"
EFFECTIVE_N_COL = r"$n_{\mathrm{effective}}$"


@dataclass(frozen=True)
class OddsRatioResult:
    """Confidence interval output for a two-group odds ratio."""

    point_estimate: float
    log_point_estimate: float
    lb: float
    ub: float
    variance: float


def _as_array(x: Iterable[float]) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _clip_probabilities(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    return np.clip(_as_array(x), eps, 1.0 - eps)


def _clip_mean(x: float, eps: float = 1e-10) -> float:
    return float(np.clip(x, eps, 1.0 - eps))


def binary_uncertainty(yhat: np.ndarray) -> np.ndarray:
    """Return min(p, 1-p) for binary probability predictions."""

    yhat = _clip_probabilities(yhat)
    return np.minimum(yhat, 1.0 - yhat)


def setup_spline_basis(
    values: np.ndarray,
    num_knots: int = 5,
    degree: int = 3,
    name: str = "u",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Create a B-spline basis with fallbacks for small or degenerate inputs."""

    values = _as_array(values)
    if len(values) == 0:
        return np.zeros((0, 0)), None

    unique_values = np.unique(values)
    if len(unique_values) < num_knots + 1:
        num_knots = max(0, len(unique_values) - degree - 1)

    if len(unique_values) == 1:
        formula = f"bs({name}, df={degree + 1}, degree={degree}, include_intercept=True) - 1"
        try:
            basis = dmatrix(formula, {name: values}, return_type="dataframe")
            return basis.values, None
        except Exception:
            return np.ones((len(values), 1)), None

    if num_knots > 0:
        knot_quantiles = np.linspace(0, 1, num_knots + 2)[1:-1]
        knots = np.unique(np.quantile(values, knot_quantiles))
        if len(knots) == 0:
            num_knots = 0
            knots = np.array([])
    else:
        knots = np.array([])

    if num_knots > 0:
        formula = f"bs({name}, knots=knots, degree={degree}, include_intercept=True) - 1"
    else:
        formula = f"bs({name}, df={degree + 1}, degree={degree}, include_intercept=True) - 1"

    try:
        basis = dmatrix(formula, {name: values}, return_type="dataframe")
    except Exception:
        try:
            fallback_df = max(degree + 1, num_knots + degree + 1)
            formula = f"bs({name}, df={fallback_df}, degree={degree}, include_intercept=True) - 1"
            basis = dmatrix(formula, {name: values}, return_type="dataframe")
            knots = None
        except Exception:
            return np.ones((len(values), 1)), None

    if basis.shape[1] == 0:
        return np.ones((len(values), 1)), None
    return basis.values, knots


def sampling_probs_spline_inv(
    u1_all: np.ndarray,
    u0_all: np.ndarray,
    b1: np.ndarray,
    b0: np.ndarray,
    mu1: float,
    mu0: float,
    n_budget: float,
    num_knots: int = 5,
    degree: int = 3,
    verbose_solver: bool = False,
    split_budget_evenly: bool = True,
    solver_names: tuple[str, ...] = (cp.ECOS, "CLARABEL", cp.SCS),
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None, dict | None]:
    """Optimize spline-parameterized inverse sampling probabilities.

    The optimization models log(1 / p_i) as a spline in a score. When
    ``split_budget_evenly`` is true, half the expected sampling budget is
    assigned to each group, matching the current BRCA experiment.
    """

    start_time = time.perf_counter()
    u1_all = _as_array(u1_all)
    u0_all = _as_array(u0_all)
    b1 = _as_array(b1)
    b0 = _as_array(b0)

    n1 = len(u1_all)
    n0 = len(u0_all)
    if n1 == 0 and n0 == 0:
        return np.array([]), np.array([]), None, None
    if n_budget <= 0:
        return np.zeros(n1), np.zeros(n0), None, None

    B1, knots1 = setup_spline_basis(b1, num_knots, degree, name="u1") if n1 else (np.zeros((0, 0)), None)
    B0, knots0 = setup_spline_basis(b0, num_knots, degree, name="u0") if n0 else (np.zeros((0, 0)), None)
    n_basis1 = B1.shape[1]
    n_basis0 = B0.shape[1]

    if n_basis1 == 0 and n_basis0 == 0 and (n1 > 0 or n0 > 0):
        return None, None, None, None

    theta1 = cp.Variable(n_basis1, name="theta1") if n_basis1 > 0 else None
    theta0 = cp.Variable(n_basis0, name="theta0") if n_basis0 > 0 else None
    p1 = None
    p0 = None

    objective_terms = []
    budget_terms1 = []
    budget_terms0 = []
    constraints = []

    if n1 > 0 and n_basis1 > 0 and theta1 is not None and 0 < mu1 < 1:
        weights1 = u1_all / (n1**2 * mu1**2 * (1 - mu1) ** 2)
        log_inv_p1 = B1 @ theta1
        inv_p1 = cp.exp(log_inv_p1)
        p1 = cp.exp(-log_inv_p1)
        objective_terms.append(cp.sum(cp.multiply(weights1, inv_p1)))
        budget_terms1.append(cp.sum(p1))
        constraints.append(log_inv_p1 >= 0)

    if n0 > 0 and n_basis0 > 0 and theta0 is not None and 0 < mu0 < 1:
        weights0 = u0_all / (n0**2 * mu0**2 * (1 - mu0) ** 2)
        log_inv_p0 = B0 @ theta0
        inv_p0 = cp.exp(log_inv_p0)
        p0 = cp.exp(-log_inv_p0)
        objective_terms.append(cp.sum(cp.multiply(weights0, inv_p0)))
        budget_terms0.append(cp.sum(p0))
        constraints.append(log_inv_p0 >= 0)

    if not objective_terms:
        return None, None, None, None

    if split_budget_evenly:
        if budget_terms1:
            constraints.append(cp.sum(budget_terms1) <= n_budget / 2)
        if budget_terms0:
            constraints.append(cp.sum(budget_terms0) <= n_budget / 2)
    else:
        constraints.append(cp.sum(budget_terms1 + budget_terms0) <= n_budget)

    problem = cp.Problem(cp.Minimize(cp.sum(objective_terms)), constraints)
    solved = False
    installed_solvers = set(cp.installed_solvers())
    for solver_name in solver_names:
        if solver_name not in installed_solvers:
            continue
        try:
            solver_kwargs = {"verbose": verbose_solver}
            if solver_name == cp.ECOS:
                solver_kwargs.update({"max_iters": 200, "abstol": 1e-8, "reltol": 1e-7})
            elif solver_name == cp.SCS:
                solver_kwargs.update({"eps": 1e-5})
            problem.solve(solver=solver_name, **solver_kwargs)
        except Exception:
            continue

        if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            solved = True
            break

    if not solved:
        print(f"Spline optimization failed after {time.perf_counter() - start_time:.2f}s.")
        return None, None, None, None

    final_probs1 = np.zeros(n1)
    final_probs0 = np.zeros(n0)
    final_theta1 = None
    final_theta0 = None

    if p1 is not None and theta1 is not None and theta1.value is not None:
        final_theta1 = np.asarray(theta1.value)
        final_probs1 = np.clip(np.exp(-(B1 @ final_theta1)), 0, 1)
    if p0 is not None and theta0 is not None and theta0.value is not None:
        final_theta0 = np.asarray(theta0.value)
        final_probs0 = np.clip(np.exp(-(B0 @ final_theta0)), 0, 1)

    spline_params1 = (
        {"coeffs": final_theta1, "knots": knots1, "degree": degree, "n_basis": n_basis1}
        if final_theta1 is not None
        else None
    )
    spline_params0 = (
        {"coeffs": final_theta0, "knots": knots0, "degree": degree, "n_basis": n_basis0}
        if final_theta0 is not None
        else None
    )
    return final_probs1, final_probs0, spline_params1, spline_params0


def opt_mean_tuning(
    y: np.ndarray,
    yhat: np.ndarray,
    weights: np.ndarray,
    sampling_ratio: np.ndarray,
) -> float:
    """Estimate the scalar tuning parameter used by the rectified mean."""

    y = _as_array(y)
    yhat = _as_array(yhat)
    weights = _as_array(weights)
    sampling_ratio = _as_array(sampling_ratio)
    denom = np.mean(yhat**2 * sampling_ratio)
    if np.isclose(denom, 0):
        return 0.0
    return float(np.clip(np.mean(y * yhat * weights * sampling_ratio) / denom, 0, 1))


def odds_ratio_estimate_ci(
    y0: np.ndarray,
    yhat0: np.ndarray,
    y1: np.ndarray,
    yhat1: np.ndarray,
    weights0: np.ndarray,
    weights1: np.ndarray,
    alpha: float,
    lhat0: float = 1.0,
    lhat1: float = 1.0,
) -> OddsRatioResult:
    """Estimate a two-group odds ratio and a Wald CI with rectified labels."""

    y0 = _as_array(y0)
    y1 = _as_array(y1)
    yhat0 = _as_array(yhat0)
    yhat1 = _as_array(yhat1)
    weights0 = _as_array(weights0)
    weights1 = _as_array(weights1)

    n0 = y0.shape[0]
    n1 = y1.shape[0]
    n = n0 + n1
    mu0_hat = _clip_mean(np.mean(lhat0 * yhat0 + (y0 - lhat0 * yhat0) * weights0))
    mu1_hat = _clip_mean(np.mean(lhat1 * yhat1 + (y1 - lhat1 * yhat1) * weights1))

    log_point = np.log(mu1_hat / (1 - mu1_hat)) - np.log(mu0_hat / (1 - mu0_hat))
    var_mu0_hat = np.var(lhat0 * yhat0 + (y0 - lhat0 * yhat0) * weights0)
    var_mu1_hat = np.var(lhat1 * yhat1 + (y1 - lhat1 * yhat1) * weights1)
    var0 = var_mu0_hat / ((mu0_hat * (1 - mu0_hat)) ** 2)
    var1 = var_mu1_hat / ((mu1_hat * (1 - mu1_hat)) ** 2)
    p0 = n0 / n
    p1 = n1 / n
    variance = (1 / p0) * var0 + (1 / p1) * var1
    width_log = norm.ppf(1 - alpha / 2) * np.sqrt(variance / n)

    return OddsRatioResult(
        point_estimate=float(np.exp(log_point)),
        log_point_estimate=float(log_point),
        lb=float(np.exp(log_point - width_log)),
        ub=float(np.exp(log_point + width_log)),
        variance=float(variance),
    )


def odds_ratio_ci(*args, **kwargs) -> tuple[float, float, float]:
    """Backward-compatible wrapper returning ``lb, ub, variance``."""

    result = odds_ratio_estimate_ci(*args, **kwargs)
    return result.lb, result.ub, result.variance


def classical_odds_ratio_estimate_ci(
    y0: np.ndarray,
    y1: np.ndarray,
    selected0: np.ndarray,
    selected1: np.ndarray,
    alpha: float,
    total_n: int | None = None,
) -> OddsRatioResult:
    """Classical odds-ratio CI using only selected human labels."""

    y0_selected = _as_array(y0)[np.asarray(selected0, dtype=bool)]
    y1_selected = _as_array(y1)[np.asarray(selected1, dtype=bool)]
    if total_n is None:
        total_n = len(y0_selected) + len(y1_selected)

    if len(y0_selected) == 0 or len(y1_selected) == 0:
        return OddsRatioResult(np.nan, np.nan, np.nan, np.nan, np.nan)

    mu0 = _clip_mean(np.mean(y0_selected))
    mu1 = _clip_mean(np.mean(y1_selected))
    log_point = np.log(mu1 / (1 - mu1)) - np.log(mu0 / (1 - mu0))

    counts = [
        np.sum(y0_selected == 0),
        np.sum(y0_selected == 1),
        np.sum(y1_selected == 0),
        np.sum(y1_selected == 1),
    ]
    if any(count == 0 for count in counts):
        return OddsRatioResult(float(np.exp(log_point)), float(log_point), np.nan, np.nan, np.nan)

    selected_scale_variance = sum(1 / count for count in counts)
    variance = selected_scale_variance * total_n
    width_log = norm.ppf(1 - alpha / 2) * np.sqrt(variance / total_n)
    return OddsRatioResult(
        point_estimate=float(np.exp(log_point)),
        log_point_estimate=float(log_point),
        lb=float(np.exp(log_point - width_log)),
        ub=float(np.exp(log_point + width_log)),
        variance=float(variance),
    )


def inverse_probability_weights(
    probabilities: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw Bernoulli labels and return ``selection_mask, mask / p``."""

    probabilities = _clip_probabilities(probabilities)
    selected = rng.binomial(1, probabilities).astype(bool)
    return selected, selected.astype(float) / probabilities


def active_sampling_probabilities(
    uncertainty: np.ndarray,
    frac_human: float,
    tau: float = 0.1,
) -> np.ndarray:
    """Probability rule proportional to prediction uncertainty with uniform mixing."""

    uncertainty = _as_array(uncertainty)
    if np.isclose(np.mean(uncertainty), 0):
        base_probs = np.full(len(uncertainty), frac_human)
    else:
        base_probs = frac_human * uncertainty / np.mean(uncertainty)
    return np.clip((1 - tau) * base_probs + tau * frac_human, 0, 1)


def effective_sample_size(true_variance: float, variance_estimate: float, n: int) -> float:
    if not np.isfinite(variance_estimate) or np.isclose(variance_estimate, 0):
        return np.nan
    return float((true_variance / variance_estimate) * n)


def _result_row(
    result: OddsRatioResult,
    estimator: str,
    n_human: int,
    frac_human: float,
    trial: int,
    true_odds_ratio: float,
    true_variance: float,
    n: int,
) -> dict:
    return {
        "trial": trial,
        "frac_human": frac_human,
        HUMAN_N_COL: n_human,
        "estimator": estimator,
        "point estimate": result.point_estimate,
        "log point estimate": result.log_point_estimate,
        "variance estimate": result.variance,
        "lb": result.lb,
        "ub": result.ub,
        "interval width": result.ub - result.lb,
        "coverage": bool(result.lb <= true_odds_ratio <= result.ub),
        EFFECTIVE_N_COL: effective_sample_size(true_variance, result.variance, n),
    }


def add_monte_carlo_variance(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = (HUMAN_N_COL, "estimator"),
) -> pd.DataFrame:
    """Attach empirical Monte Carlo variance columns to every result row."""

    out = df.copy()
    grouped = out.groupby(list(group_cols), observed=True)
    out["mc variance"] = grouped["point estimate"].transform(lambda x: x.var(ddof=1))
    out["mc sd"] = np.sqrt(out["mc variance"])
    out["mc log variance"] = grouped["log point estimate"].transform(lambda x: x.var(ddof=1))
    out["mc log sd"] = np.sqrt(out["mc log variance"])
    return out


def summarize_monte_carlo(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = (HUMAN_N_COL, "estimator"),
) -> pd.DataFrame:
    """One-row-per-method summary of CI performance and MC stability."""

    summary = (
        df.groupby(list(group_cols), observed=True)
        .agg(
            point_estimate_mean=("point estimate", "mean"),
            point_estimate_var=("point estimate", "var"),
            log_point_estimate_mean=("log point estimate", "mean"),
            log_point_estimate_var=("log point estimate", "var"),
            coverage=("coverage", "mean"),
            interval_width=("interval width", "mean"),
            variance_estimate=("variance estimate", "mean"),
            effective_n=(EFFECTIVE_N_COL, "mean"),
        )
        .reset_index()
    )
    return summary


def run_odds_ratio_monte_carlo(
    y0: np.ndarray,
    yhat0: np.ndarray,
    y1: np.ndarray,
    yhat1: np.ndarray,
    fracs_human: Iterable[float],
    alpha: float,
    num_trials: int,
    true_odds_ratio: float,
    true_variance: float,
    mu0_pilot: float | None = None,
    mu1_pilot: float | None = None,
    tau: float = 0.1,
    seed: int | None = None,
    num_knots: int = 5,
    degree: int = 3,
    split_spline_budget_evenly: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run the two-group odds-ratio Monte Carlo comparison used by BRCA."""

    y0 = _as_array(y0)
    y1 = _as_array(y1)
    yhat0 = _clip_probabilities(yhat0)
    yhat1 = _clip_probabilities(yhat1)
    n0 = len(y0)
    n1 = len(y1)
    n = n0 + n1
    mu0_pilot = _clip_mean(np.mean(y0) if mu0_pilot is None else mu0_pilot)
    mu1_pilot = _clip_mean(np.mean(y1) if mu1_pilot is None else mu1_pilot)

    rng = np.random.default_rng(seed)
    uncertainty0 = binary_uncertainty(yhat0)
    uncertainty1 = binary_uncertainty(yhat1)
    score0 = yhat0 * (1 - yhat0)
    score1 = yhat1 * (1 - yhat1)

    iterator = list(fracs_human)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="human budget")

    rows = []
    for frac_human in iterator:
        frac_human = float(frac_human)
        n_human = int(frac_human * n)
        p0_active = active_sampling_probabilities(uncertainty0, frac_human, tau=tau)
        p1_active = active_sampling_probabilities(uncertainty1, frac_human, tau=tau)

        p1_spline, p0_spline, _, _ = sampling_probs_spline_inv(
            score1,
            score0,
            score1,
            score0,
            mu1_pilot,
            mu0_pilot,
            n * frac_human,
            num_knots=num_knots,
            degree=degree,
            split_budget_evenly=split_spline_budget_evenly,
        )
        if p0_spline is None or p1_spline is None:
            raise RuntimeError(f"Spline sampling optimization failed for frac_human={frac_human}.")
        p0_spline = _clip_probabilities(p0_spline)
        p1_spline = _clip_probabilities(p1_spline)

        trial_iterator = range(num_trials)
        if show_progress:
            from tqdm.auto import tqdm

            trial_iterator = tqdm(trial_iterator, desc=f"trials {frac_human:.3f}", leave=False)

        for trial in trial_iterator:
            _, weights_active0 = inverse_probability_weights(p0_active, rng)
            _, weights_active1 = inverse_probability_weights(p1_active, rng)
            active_ratio0 = (1 - p0_active) / p0_active
            active_ratio1 = (1 - p1_active) / p1_active
            lam0_active = opt_mean_tuning(y0, yhat0, weights_active0, active_ratio0)
            lam1_active = opt_mean_tuning(y1, yhat1, weights_active1, active_ratio1)

            _, weights_spline0 = inverse_probability_weights(p0_spline, rng)
            _, weights_spline1 = inverse_probability_weights(p1_spline, rng)
            spline_ratio0 = (1 - p0_spline) / p0_spline
            spline_ratio1 = (1 - p1_spline) / p1_spline
            lam0_spline = opt_mean_tuning(y0, yhat0, weights_spline0, spline_ratio0)
            lam1_spline = opt_mean_tuning(y1, yhat1, weights_spline1, spline_ratio1)

            method_results = [
                (
                    "active",
                    odds_ratio_estimate_ci(
                        y0, yhat0, y1, yhat1, weights_active0, weights_active1, alpha
                    ),
                ),
                (
                    "spline",
                    odds_ratio_estimate_ci(
                        y0, yhat0, y1, yhat1, weights_spline0, weights_spline1, alpha
                    ),
                ),
                (
                    "active + tuning",
                    odds_ratio_estimate_ci(
                        y0,
                        yhat0,
                        y1,
                        yhat1,
                        weights_active0,
                        weights_active1,
                        alpha,
                        lhat0=lam0_active,
                        lhat1=lam1_active,
                    ),
                ),
                (
                    "spline + tuning",
                    odds_ratio_estimate_ci(
                        y0,
                        yhat0,
                        y1,
                        yhat1,
                        weights_spline0,
                        weights_spline1,
                        alpha,
                        lhat0=lam0_spline,
                        lhat1=lam1_spline,
                    ),
                ),
            ]

            selected0_uniform, weights_uniform0 = inverse_probability_weights(
                np.full(n0, frac_human), rng
            )
            selected1_uniform, weights_uniform1 = inverse_probability_weights(
                np.full(n1, frac_human), rng
            )
            method_results.append(
                (
                    "uniform",
                    odds_ratio_estimate_ci(
                        y0, yhat0, y1, yhat1, weights_uniform0, weights_uniform1, alpha
                    ),
                )
            )
            method_results.append(
                (
                    "classical",
                    classical_odds_ratio_estimate_ci(
                        y0, y1, selected0_uniform, selected1_uniform, alpha, total_n=n
                    ),
                )
            )

            for estimator, result in method_results:
                rows.append(
                    _result_row(
                        result,
                        estimator,
                        n_human,
                        frac_human,
                        trial,
                        true_odds_ratio,
                        true_variance,
                        n,
                    )
                )

    return add_monte_carlo_variance(pd.DataFrame(rows))
