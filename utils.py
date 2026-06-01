"""Shared numerical helpers for the active-inference examples.

The functions in this module are intentionally not tied to a particular
dataset. Example-specific data loading and presentation code should stay in
the notebooks or in dataset-specific modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Literal

import cvxpy as cp
import numpy as np
import pandas as pd
from patsy import dmatrix
from scipy.stats import norm


HUMAN_N_COL = r"$n_{\mathrm{human}}$"
EFFECTIVE_N_COL = r"$n_{\mathrm{effective}}$"
MC_VARIANCE_TARGETS = {
    "point estimate": "point_estimate",
    "log point estimate": "log_point_estimate",
    "lb": "lb",
    "ub": "ub",
    "interval width": "interval_width",
    "finite population lb": "finite_population_lb",
    "finite population ub": "finite_population_ub",
    "finite population interval width": "finite_population_interval_width",
}


@dataclass(frozen=True)
class OddsRatioResult:
    """Confidence interval output for a two-group odds ratio."""

    point_estimate: float
    log_point_estimate: float
    lb: float
    ub: float
    variance: float
    mu0_hat: float = np.nan
    mu1_hat: float = np.nan


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
    split_budget_evenly: bool = False,
    solver_names: tuple[str, ...] = (cp.ECOS, "CLARABEL", cp.SCS),
) -> tuple[np.ndarray | None, np.ndarray | None, dict | None, dict | None]:
    """Optimize spline-parameterized inverse sampling probabilities.

    The optimization models log(1 / p_i) as a spline in a score. By default,
    the expected sampling budget is shared across groups proportionally through
    one joint budget constraint. Set ``split_budget_evenly=True`` only when a
    half-budget-per-group design is desired.
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
        mu0_hat=mu0_hat,
        mu1_hat=mu1_hat,
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
        return OddsRatioResult(
            float(np.exp(log_point)),
            float(log_point),
            np.nan,
            np.nan,
            np.nan,
            mu0_hat=mu0,
            mu1_hat=mu1,
        )

    selected_scale_variance = sum(1 / count for count in counts)
    variance = selected_scale_variance * total_n
    width_log = norm.ppf(1 - alpha / 2) * np.sqrt(variance / total_n)
    return OddsRatioResult(
        point_estimate=float(np.exp(log_point)),
        log_point_estimate=float(log_point),
        lb=float(np.exp(log_point - width_log)),
        ub=float(np.exp(log_point + width_log)),
        variance=float(variance),
        mu0_hat=mu0,
        mu1_hat=mu1,
    )


def finite_population_log_odds_variance(
    y0: np.ndarray,
    yhat0: np.ndarray,
    y1: np.ndarray,
    yhat1: np.ndarray,
    probabilities0: np.ndarray,
    probabilities1: np.ndarray,
    selected0: np.ndarray,
    selected1: np.ndarray,
    lhat0: float = 1.0,
    lhat1: float = 1.0,
    mu0_hat: float | None = None,
    mu1_hat: float | None = None,
) -> float:
    """Finite-population HT variance for a prediction-assisted log odds ratio.

    This is the Appendix I.2 calibration: groupwise HT variances of
    ``R_i = Y_i - a_i`` with ``a_i = lhat * Yhat_i`` are combined through the
    Delta method. The returned variance is on the log-odds-ratio scale.
    """

    y0 = _as_array(y0)
    y1 = _as_array(y1)
    yhat0 = _as_array(yhat0)
    yhat1 = _as_array(yhat1)
    probabilities0 = _clip_probabilities(probabilities0)
    probabilities1 = _clip_probabilities(probabilities1)
    selected0 = np.asarray(selected0, dtype=bool)
    selected1 = np.asarray(selected1, dtype=bool)

    a0 = lhat0 * yhat0
    a1 = lhat1 * yhat1
    if mu0_hat is None:
        mu0_hat = np.mean(a0 + selected0.astype(float) / probabilities0 * (y0 - a0))
    if mu1_hat is None:
        mu1_hat = np.mean(a1 + selected1.astype(float) / probabilities1 * (y1 - a1))
    mu0_hat = _clip_mean(mu0_hat)
    mu1_hat = _clip_mean(mu1_hat)

    residual0 = y0 - a0
    residual1 = y1 - a1
    v0_ht = (
        np.sum(selected0 * (1 - probabilities0) / (probabilities0**2) * residual0**2)
        / (len(y0) ** 2)
    )
    v1_ht = (
        np.sum(selected1 * (1 - probabilities1) / (probabilities1**2) * residual1**2)
        / (len(y1) ** 2)
    )

    d0 = -1 / (mu0_hat * (1 - mu0_hat))
    d1 = 1 / (mu1_hat * (1 - mu1_hat))
    return float(d0**2 * v0_ht + d1**2 * v1_ht)


def finite_population_classical_log_odds_variance(
    y0: np.ndarray,
    y1: np.ndarray,
    probabilities0: np.ndarray,
    probabilities1: np.ndarray,
    selected0: np.ndarray,
    selected1: np.ndarray,
    mu0_hat: float | None = None,
    mu1_hat: float | None = None,
) -> float:
    """Approximate finite-population variance for the classical sample-only CI."""

    y0 = _as_array(y0)
    y1 = _as_array(y1)
    probabilities0 = _clip_probabilities(probabilities0)
    probabilities1 = _clip_probabilities(probabilities1)
    selected0 = np.asarray(selected0, dtype=bool)
    selected1 = np.asarray(selected1, dtype=bool)

    if mu0_hat is None:
        mu0_hat = _clip_mean(np.mean(y0[selected0]))
    else:
        mu0_hat = _clip_mean(mu0_hat)
    if mu1_hat is None:
        mu1_hat = _clip_mean(np.mean(y1[selected1]))
    else:
        mu1_hat = _clip_mean(mu1_hat)

    residual0 = y0 - mu0_hat
    residual1 = y1 - mu1_hat
    v0_ht = (
        np.sum(selected0 * (1 - probabilities0) / (probabilities0**2) * residual0**2)
        / (len(y0) ** 2)
    )
    v1_ht = (
        np.sum(selected1 * (1 - probabilities1) / (probabilities1**2) * residual1**2)
        / (len(y1) ** 2)
    )
    d0 = -1 / (mu0_hat * (1 - mu0_hat))
    d1 = 1 / (mu1_hat * (1 - mu1_hat))
    return float(d0**2 * v0_ht + d1**2 * v1_ht)


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
    target_mean_probability: float,
    tau: float = 0.1,
) -> np.ndarray:
    """Probability rule proportional to uncertainty with uniform mixing.

    ``target_mean_probability`` is the desired average label probability within
    the group. For most batch examples this is the overall human-label fraction;
    for an even group split it can be group specific.
    """

    uncertainty = _as_array(uncertainty)
    if np.isclose(np.mean(uncertainty), 0):
        base_probs = np.full(len(uncertainty), target_mean_probability)
    else:
        base_probs = target_mean_probability * uncertainty / np.mean(uncertainty)
    return np.clip((1 - tau) * base_probs + tau * target_mean_probability, 0, 1)


def normalize_probabilities_to_budget(
    probabilities: np.ndarray,
    target_sum: float,
    eps: float = 1e-10,
    max_iter: int = 80,
) -> np.ndarray:
    """Rescale probabilities to a target expected sample size.

    The output is the clipped vector ``clip(c * probabilities, eps, 1)`` whose
    sum is as close as possible to ``target_sum``. This is useful after
    convexly mixing two policies when solver tolerances leave a small budget
    mismatch.
    """

    probabilities = np.clip(_as_array(probabilities), eps, 1.0)
    n = len(probabilities)
    if n == 0:
        return probabilities
    target_sum = float(np.clip(target_sum, n * eps, n))

    def scaled_sum(scale: float) -> float:
        return float(np.clip(scale * probabilities, eps, 1.0).sum())

    lower = 0.0
    upper = 1.0
    while scaled_sum(upper) < target_sum and upper < 1e12:
        upper *= 2.0

    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        if scaled_sum(mid) < target_sum:
            lower = mid
        else:
            upper = mid
    return np.clip(upper * probabilities, eps, 1.0)


def mix_probabilities_with_uniform(
    method_probabilities: np.ndarray,
    uniform_probability: float,
    method_weight: float,
    target_sum: float | None = None,
) -> np.ndarray:
    """Convexly mix a policy with uniform sampling.

    ``method_weight=1`` returns the method policy and ``method_weight=0``
    returns uniform sampling. If ``target_sum`` is supplied, the mixed
    probabilities are renormalized to that expected budget.
    """

    method_probabilities = _clip_probabilities(method_probabilities)
    method_weight = float(np.clip(method_weight, 0.0, 1.0))
    uniform_probability = float(np.clip(uniform_probability, 0.0, 1.0))
    mixed = method_weight * method_probabilities + (1.0 - method_weight) * uniform_probability
    mixed = _clip_probabilities(mixed)
    if target_sum is not None:
        mixed = normalize_probabilities_to_budget(mixed, target_sum)
    return mixed


class _BinnedResidualRegressor:
    """Fast one-dimensional smoother used by sequential examples."""

    def __init__(self, n_bins: int = 12):
        self.n_bins = n_bins
        self.edges_: np.ndarray | None = None
        self.values_: np.ndarray | None = None
        self.global_mean_: float = np.nan

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_BinnedResidualRegressor":
        x = np.asarray(x, dtype=float)
        if x.ndim > 1:
            x = x[:, 0]
        y = _as_array(y)
        self.global_mean_ = float(np.mean(y)) if len(y) else 0.0
        if len(np.unique(x)) <= 1:
            self.edges_ = np.array([])
            self.values_ = np.array([self.global_mean_])
            return self

        quantiles = np.linspace(0, 1, self.n_bins + 1)
        edges = np.unique(np.quantile(x, quantiles))
        if len(edges) <= 2:
            self.edges_ = np.array([])
            self.values_ = np.array([self.global_mean_])
            return self

        bin_ids = np.clip(np.searchsorted(edges[1:-1], x, side="right"), 0, len(edges) - 2)
        values = np.full(len(edges) - 1, self.global_mean_)
        for bin_id in range(len(values)):
            in_bin = bin_ids == bin_id
            if np.any(in_bin):
                values[bin_id] = float(np.mean(y[in_bin]))
        self.edges_ = edges
        self.values_ = values
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.values_ is None:
            raise RuntimeError("Binned residual regressor has not been fit.")
        x = np.asarray(x, dtype=float)
        if x.ndim > 1:
            x = x[:, 0]
        if self.edges_ is None or len(self.edges_) == 0:
            return np.full(len(x), self.values_[0])
        bin_ids = np.clip(np.searchsorted(self.edges_[1:-1], x, side="right"), 0, len(self.values_) - 1)
        return self.values_[bin_ids]


def _fit_residual_uncertainty_model(
    features: np.ndarray,
    residuals: np.ndarray,
    *,
    model: Literal["binned", "gradient_boosting"] = "binned",
    seed: int | None = None,
    n_bins: int = 12,
):
    features = np.asarray(features, dtype=float)
    residuals = _as_array(residuals)
    if model == "binned":
        return _BinnedResidualRegressor(n_bins=n_bins).fit(features, residuals)
    if model == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor

        fit_model = GradientBoostingRegressor(
            random_state=seed,
            max_depth=2,
            n_estimators=100,
        )
        fit_model.fit(features, residuals)
        return fit_model
    raise ValueError("model must be one of: 'binned', 'gradient_boosting'")


def _predict_residual_uncertainty(model, features: np.ndarray) -> np.ndarray:
    predictions = np.asarray(model.predict(np.asarray(features, dtype=float)), dtype=float)
    return np.sqrt(np.maximum(predictions, 1e-8))


def binary_odds_ratio_truth(y0: np.ndarray, y1: np.ndarray) -> tuple[float, float]:
    """Return the empirical odds ratio and scaled classical log-OR variance."""

    y0 = _as_array(y0)
    y1 = _as_array(y1)
    n = len(y0) + len(y1)
    mu0 = _clip_mean(np.mean(y0))
    mu1 = _clip_mean(np.mean(y1))
    odds_ratio = (mu1 / (1 - mu1)) / (mu0 / (1 - mu0))
    counts = [
        np.sum(y0 == 0),
        np.sum(y0 == 1),
        np.sum(y1 == 0),
        np.sum(y1 == 1),
    ]
    if any(count == 0 for count in counts):
        return float(odds_ratio), np.nan
    scaled_variance = sum(1 / count for count in counts) * n
    return float(odds_ratio), float(scaled_variance)


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
    alpha: float,
    true_odds_ratio: float,
    true_variance: float,
    n: int,
    finite_population_log_variance: float | None = None,
) -> dict:
    z_value = norm.ppf(1 - alpha / 2)
    if (
        finite_population_log_variance is None
        or not np.isfinite(finite_population_log_variance)
        or finite_population_log_variance <= 0
    ):
        fp_log_width = np.nan
        fp_lb = np.nan
        fp_ub = np.nan
        fp_coverage = np.nan
        variance_inflation = np.nan
        implied_usual_coverage = np.nan
    else:
        fp_log_width = z_value * np.sqrt(finite_population_log_variance)
        fp_lb = float(np.exp(result.log_point_estimate - fp_log_width))
        fp_ub = float(np.exp(result.log_point_estimate + fp_log_width))
        fp_coverage = bool(fp_lb <= true_odds_ratio <= fp_ub)
        usual_log_variance = result.variance / n
        variance_inflation = float(usual_log_variance / finite_population_log_variance)
        implied_usual_coverage = float(2 * norm.cdf(z_value * np.sqrt(variance_inflation)) - 1)

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
        "finite population log variance estimate": finite_population_log_variance,
        "finite population scaled variance estimate": (
            finite_population_log_variance * n if finite_population_log_variance is not None else np.nan
        ),
        "finite population lb": fp_lb,
        "finite population ub": fp_ub,
        "finite population interval width": fp_ub - fp_lb,
        "finite population coverage": fp_coverage,
        "finite population variance inflation": variance_inflation,
        "implied usual finite population coverage": implied_usual_coverage,
    }


def add_monte_carlo_variance(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = (HUMAN_N_COL, "estimator"),
) -> pd.DataFrame:
    """Attach empirical Monte Carlo variance columns to every result row."""

    out = df.copy()
    grouped = out.groupby(list(group_cols), observed=True)
    for source_col, safe_name in MC_VARIANCE_TARGETS.items():
        if source_col not in out.columns:
            continue
        variance_col = f"mc {safe_name} variance"
        sd_col = f"mc {safe_name} sd"
        out[variance_col] = grouped[source_col].transform(lambda x: x.var(ddof=1))
        out[sd_col] = np.sqrt(out[variance_col])

    # Backward-compatible aliases used by the first cleaned plotting draft.
    if "mc point_estimate variance" in out.columns:
        out["mc variance"] = out["mc point_estimate variance"]
        out["mc sd"] = out["mc point_estimate sd"]
    if "mc log_point_estimate variance" in out.columns:
        out["mc log variance"] = out["mc log_point_estimate variance"]
        out["mc log sd"] = out["mc log_point_estimate sd"]
    return out


def monte_carlo_variance_table(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = (HUMAN_N_COL, "estimator"),
) -> pd.DataFrame:
    """Summarize empirical MC variance for estimates and interval quantities."""

    out = df.copy()
    required_targets = ["point estimate", "lb", "ub", "interval width", "coverage"]
    missing_targets = [col for col in required_targets if col not in out.columns]
    if missing_targets:
        raise ValueError(f"Missing columns required for MC variance summary: {missing_targets}")
    for optional_col in [
        "finite population coverage",
        "finite population interval width",
        "finite population variance inflation",
    ]:
        if optional_col not in out.columns:
            out[optional_col] = np.nan

    summary = (
        out.groupby(list(group_cols), observed=True)
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
            finite_population_coverage=("finite population coverage", "mean"),
            finite_population_interval_width=("finite population interval width", "mean"),
            finite_population_variance_inflation=("finite population variance inflation", "mean"),
        )
        .reset_index()
    )
    return summary


def summarize_monte_carlo(
    df: pd.DataFrame,
    group_cols: tuple[str, ...] = (HUMAN_N_COL, "estimator"),
) -> pd.DataFrame:
    """One-row-per-method summary of CI performance and MC stability."""

    out = df.copy()
    for optional_col in [
        "finite population coverage",
        "finite population interval width",
        "finite population variance inflation",
        "finite population log variance estimate",
    ]:
        if optional_col not in out.columns:
            out[optional_col] = np.nan

    summary = (
        out.groupby(list(group_cols), observed=True)
        .agg(
            point_estimate_mean=("point estimate", "mean"),
            point_estimate_var=("point estimate", "var"),
            log_point_estimate_mean=("log point estimate", "mean"),
            log_point_estimate_var=("log point estimate", "var"),
            lb_mean=("lb", "mean"),
            lb_var=("lb", "var"),
            ub_mean=("ub", "mean"),
            ub_var=("ub", "var"),
            interval_width_var=("interval width", "var"),
            coverage=("coverage", "mean"),
            finite_population_coverage=("finite population coverage", "mean"),
            finite_population_interval_width=("finite population interval width", "mean"),
            finite_population_variance_inflation=("finite population variance inflation", "mean"),
            interval_width=("interval width", "mean"),
            variance_estimate=("variance estimate", "mean"),
            finite_population_log_variance_estimate=("finite population log variance estimate", "mean"),
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
    split_spline_budget_evenly: bool = False,
    uncertainty0: np.ndarray | None = None,
    uncertainty1: np.ndarray | None = None,
    spline_score0: np.ndarray | None = None,
    spline_score1: np.ndarray | None = None,
    active_budget: Literal["proportional", "even"] = "proportional",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run a two-group odds-ratio Monte Carlo comparison.

    ``uncertainty0``/``uncertainty1`` control the active sampling rule. If they
    are omitted, binary prediction uncertainty ``min(p, 1-p)`` is used.

    ``spline_score0``/``spline_score1`` control both the spline basis and the
    objective weights for the variance-minimizing sampler. If they are omitted,
    ``p * (1-p)`` is used.

    ``active_budget="proportional"`` gives each group the same expected label
    fraction. ``"even"`` spends half of the total expected labels in each group,
    matching the older Stance workflow.
    """

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
    if uncertainty0 is None:
        uncertainty0 = binary_uncertainty(yhat0)
    else:
        uncertainty0 = _as_array(uncertainty0)
    if uncertainty1 is None:
        uncertainty1 = binary_uncertainty(yhat1)
    else:
        uncertainty1 = _as_array(uncertainty1)
    if spline_score0 is None:
        spline_score0 = yhat0 * (1 - yhat0)
    else:
        spline_score0 = _as_array(spline_score0)
    if spline_score1 is None:
        spline_score1 = yhat1 * (1 - yhat1)
    else:
        spline_score1 = _as_array(spline_score1)

    if active_budget not in {"proportional", "even"}:
        raise ValueError("active_budget must be 'proportional' or 'even'")

    iterator = list(fracs_human)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="human budget")

    rows = []
    for frac_human in iterator:
        frac_human = float(frac_human)
        n_human = int(frac_human * n)
        if active_budget == "even":
            target0 = frac_human * n / (2 * n0)
            target1 = frac_human * n / (2 * n1)
        else:
            target0 = frac_human
            target1 = frac_human
        p0_active = active_sampling_probabilities(uncertainty0, target0, tau=tau)
        p1_active = active_sampling_probabilities(uncertainty1, target1, tau=tau)

        p1_spline, p0_spline, _, _ = sampling_probs_spline_inv(
            spline_score1,
            spline_score0,
            spline_score1,
            spline_score0,
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
            selected_active0, weights_active0 = inverse_probability_weights(p0_active, rng)
            selected_active1, weights_active1 = inverse_probability_weights(p1_active, rng)
            active_ratio0 = (1 - p0_active) / p0_active
            active_ratio1 = (1 - p1_active) / p1_active
            lam0_active = opt_mean_tuning(y0, yhat0, weights_active0, active_ratio0)
            lam1_active = opt_mean_tuning(y1, yhat1, weights_active1, active_ratio1)

            selected_spline0, weights_spline0 = inverse_probability_weights(p0_spline, rng)
            selected_spline1, weights_spline1 = inverse_probability_weights(p1_spline, rng)
            spline_ratio0 = (1 - p0_spline) / p0_spline
            spline_ratio1 = (1 - p1_spline) / p1_spline
            lam0_spline = opt_mean_tuning(y0, yhat0, weights_spline0, spline_ratio0)
            lam1_spline = opt_mean_tuning(y1, yhat1, weights_spline1, spline_ratio1)

            active_result = odds_ratio_estimate_ci(
                y0, yhat0, y1, yhat1, weights_active0, weights_active1, alpha
            )
            active_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                p0_active,
                p1_active,
                selected_active0,
                selected_active1,
                lhat0=1,
                lhat1=1,
                mu0_hat=active_result.mu0_hat,
                mu1_hat=active_result.mu1_hat,
            )

            spline_result = odds_ratio_estimate_ci(
                y0, yhat0, y1, yhat1, weights_spline0, weights_spline1, alpha
            )
            spline_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                p0_spline,
                p1_spline,
                selected_spline0,
                selected_spline1,
                lhat0=1,
                lhat1=1,
                mu0_hat=spline_result.mu0_hat,
                mu1_hat=spline_result.mu1_hat,
            )

            active_tuned_result = odds_ratio_estimate_ci(
                y0,
                yhat0,
                y1,
                yhat1,
                weights_active0,
                weights_active1,
                alpha,
                lhat0=lam0_active,
                lhat1=lam1_active,
            )
            active_tuned_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                p0_active,
                p1_active,
                selected_active0,
                selected_active1,
                lhat0=lam0_active,
                lhat1=lam1_active,
                mu0_hat=active_tuned_result.mu0_hat,
                mu1_hat=active_tuned_result.mu1_hat,
            )

            spline_tuned_result = odds_ratio_estimate_ci(
                y0,
                yhat0,
                y1,
                yhat1,
                weights_spline0,
                weights_spline1,
                alpha,
                lhat0=lam0_spline,
                lhat1=lam1_spline,
            )
            spline_tuned_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                p0_spline,
                p1_spline,
                selected_spline0,
                selected_spline1,
                lhat0=lam0_spline,
                lhat1=lam1_spline,
                mu0_hat=spline_tuned_result.mu0_hat,
                mu1_hat=spline_tuned_result.mu1_hat,
            )

            method_results = [
                ("active", active_result, active_fp_var),
                ("spline", spline_result, spline_fp_var),
                ("active + tuning", active_tuned_result, active_tuned_fp_var),
                ("spline + tuning", spline_tuned_result, spline_tuned_fp_var),
            ]

            p0_uniform = np.full(n0, frac_human)
            p1_uniform = np.full(n1, frac_human)
            selected0_uniform, weights_uniform0 = inverse_probability_weights(
                p0_uniform, rng
            )
            selected1_uniform, weights_uniform1 = inverse_probability_weights(
                p1_uniform, rng
            )
            uniform_result = odds_ratio_estimate_ci(
                y0, yhat0, y1, yhat1, weights_uniform0, weights_uniform1, alpha
            )
            uniform_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                p0_uniform,
                p1_uniform,
                selected0_uniform,
                selected1_uniform,
                lhat0=1,
                lhat1=1,
                mu0_hat=uniform_result.mu0_hat,
                mu1_hat=uniform_result.mu1_hat,
            )
            method_results.append(
                ("uniform", uniform_result, uniform_fp_var)
            )

            classical_result = classical_odds_ratio_estimate_ci(
                y0, y1, selected0_uniform, selected1_uniform, alpha, total_n=n
            )
            classical_fp_var = finite_population_classical_log_odds_variance(
                y0,
                y1,
                p0_uniform,
                p1_uniform,
                selected0_uniform,
                selected1_uniform,
                mu0_hat=classical_result.mu0_hat,
                mu1_hat=classical_result.mu1_hat,
            )
            method_results.append(
                ("classical", classical_result, classical_fp_var)
            )

            for estimator, result, finite_population_log_variance in method_results:
                rows.append(
                    _result_row(
                        result,
                        estimator,
                        n_human,
                        frac_human,
                        trial,
                        alpha,
                        true_odds_ratio,
                        true_variance,
                        n,
                        finite_population_log_variance=finite_population_log_variance,
                    )
                )

    return add_monte_carlo_variance(pd.DataFrame(rows))


def run_sequential_odds_ratio_monte_carlo(
    y: np.ndarray,
    yhat: np.ndarray,
    group1: np.ndarray,
    uncertainty_features: np.ndarray,
    fracs_human: Iterable[float],
    alpha: float,
    num_trials: int,
    true_odds_ratio: float | None = None,
    true_variance: float | None = None,
    burnin_steps: int = 50,
    retrain_steps: int = 250,
    tau: float = 0.1,
    seed: int | None = None,
    num_knots: int = 5,
    degree: int = 3,
    uncertainty_model: Literal["binned", "gradient_boosting"] = "binned",
    uncertainty_bins: int = 12,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run a sequential active-labeling odds-ratio Monte Carlo comparison.

    Unlike the archived sequential CSVs, this runner computes the exact
    finite-population calibration inside each replicate while the selected
    labels and their first-order sampling probabilities are still available.
    """

    y = _as_array(y)
    yhat = _clip_probabilities(yhat)
    group1 = np.asarray(group1, dtype=bool)
    uncertainty_features = np.asarray(uncertainty_features, dtype=float)
    if uncertainty_features.ndim == 1:
        uncertainty_features = uncertainty_features.reshape(-1, 1)

    if not (len(y) == len(yhat) == len(group1) == len(uncertainty_features)):
        raise ValueError("y, yhat, group1, and uncertainty_features must have the same length")
    if burnin_steps <= 0:
        raise ValueError("burnin_steps must be positive")
    if retrain_steps <= 0:
        raise ValueError("retrain_steps must be positive")
    if burnin_steps >= len(y):
        raise ValueError("burnin_steps must be smaller than the population size")

    y0, y1 = y[~group1], y[group1]
    yhat0, yhat1 = yhat[~group1], yhat[group1]
    n0 = len(y0)
    n1 = len(y1)
    n = len(y)
    if n0 == 0 or n1 == 0:
        raise ValueError("Both groups must contain observations")

    if true_odds_ratio is None or true_variance is None:
        computed_odds_ratio, computed_variance = binary_odds_ratio_truth(y0, y1)
        if true_odds_ratio is None:
            true_odds_ratio = computed_odds_ratio
        if true_variance is None:
            true_variance = computed_variance

    mu0_init = _clip_mean(np.mean(y0[: min(burnin_steps, n0)]))
    mu1_init = _clip_mean(np.mean(y1[: min(burnin_steps, n1)]))
    residuals = (y - yhat) ** 2
    base_seed = 0 if seed is None else int(seed)

    iterator = list(fracs_human)
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(iterator, desc="sequential human budget")

    rows = []
    for budget_index, frac_human in enumerate(iterator):
        frac_human = float(frac_human)
        n_human = int(frac_human * n)
        remaining_expected_budget = n * frac_human - burnin_steps
        if remaining_expected_budget <= 0:
            raise ValueError("Each budget must exceed burnin_steps / n")
        frac_human_adjusted = remaining_expected_budget / (n - burnin_steps)

        trial_iterator = range(num_trials)
        if show_progress:
            from tqdm.auto import tqdm

            trial_iterator = tqdm(
                trial_iterator,
                desc=f"sequential trials {frac_human:.3f}",
                leave=False,
            )

        for trial in trial_iterator:
            seed_sequence = np.random.SeedSequence([base_seed, budget_index, trial])
            rng = np.random.default_rng(seed_sequence)

            selected_active = np.zeros(n, dtype=bool)
            selected_spline = np.zeros(n, dtype=bool)
            probabilities_active = np.ones(n)
            probabilities_spline = np.ones(n)
            weights_active = np.zeros(n)
            weights_spline = np.zeros(n)

            selected_active[:burnin_steps] = True
            selected_spline[:burnin_steps] = True
            weights_active[:burnin_steps] = 1.0
            weights_spline[:burnin_steps] = 1.0

            active_uncertainty = np.full(n, np.nan)
            spline_probabilities_current = np.full(n, frac_human_adjusted)

            for t in range(burnin_steps, n):
                if (t - burnin_steps) % retrain_steps == 0:
                    active_model = _fit_residual_uncertainty_model(
                        uncertainty_features[selected_active],
                        residuals[selected_active],
                        model=uncertainty_model,
                        seed=base_seed + trial,
                        n_bins=uncertainty_bins,
                    )
                    active_uncertainty = _predict_residual_uncertainty(
                        active_model,
                        uncertainty_features,
                    )

                    spline_model = _fit_residual_uncertainty_model(
                        uncertainty_features[selected_spline],
                        residuals[selected_spline],
                        model=uncertainty_model,
                        seed=base_seed + trial,
                        n_bins=uncertainty_bins,
                    )
                    spline_uncertainty = _predict_residual_uncertainty(
                        spline_model,
                        uncertainty_features,
                    )
                    p1_spline, p0_spline, _, _ = sampling_probs_spline_inv(
                        spline_uncertainty[group1],
                        spline_uncertainty[~group1],
                        spline_uncertainty[group1],
                        spline_uncertainty[~group1],
                        mu1_init,
                        mu0_init,
                        remaining_expected_budget,
                        num_knots=num_knots,
                        degree=degree,
                        split_budget_evenly=False,
                    )
                    if p0_spline is None or p1_spline is None:
                        raise RuntimeError(
                            f"Spline sampling optimization failed for frac_human={frac_human}, trial={trial}."
                        )
                    spline_probabilities_current[group1] = _clip_probabilities(p1_spline)
                    spline_probabilities_current[~group1] = _clip_probabilities(p0_spline)

                mean_uncertainty = float(np.mean(active_uncertainty))
                if np.isclose(mean_uncertainty, 0) or not np.isfinite(mean_uncertainty):
                    active_probability = frac_human_adjusted
                else:
                    active_probability = active_uncertainty[t] / mean_uncertainty * frac_human_adjusted
                active_probability = float(
                    np.clip(
                        (1 - tau) * active_probability + tau * frac_human_adjusted,
                        1e-10,
                        1.0,
                    )
                )
                spline_probability = float(np.clip(spline_probabilities_current[t], 1e-10, 1.0))

                probabilities_active[t] = active_probability
                probabilities_spline[t] = spline_probability
                selected_active[t] = bool(rng.binomial(1, active_probability))
                selected_spline[t] = bool(rng.binomial(1, spline_probability))
                weights_active[t] = selected_active[t] / active_probability
                weights_spline[t] = selected_spline[t] / spline_probability

            weights_active0 = weights_active[~group1]
            weights_active1 = weights_active[group1]
            weights_spline0 = weights_spline[~group1]
            weights_spline1 = weights_spline[group1]
            probabilities_active0 = probabilities_active[~group1]
            probabilities_active1 = probabilities_active[group1]
            probabilities_spline0 = probabilities_spline[~group1]
            probabilities_spline1 = probabilities_spline[group1]
            selected_active0 = selected_active[~group1]
            selected_active1 = selected_active[group1]
            selected_spline0 = selected_spline[~group1]
            selected_spline1 = selected_spline[group1]
            active_ratio0 = (1 - probabilities_active0) / probabilities_active0
            active_ratio1 = (1 - probabilities_active1) / probabilities_active1
            spline_ratio0 = (1 - probabilities_spline0) / probabilities_spline0
            spline_ratio1 = (1 - probabilities_spline1) / probabilities_spline1

            lam0_active = opt_mean_tuning(y0, yhat0, weights_active0, active_ratio0)
            lam1_active = opt_mean_tuning(y1, yhat1, weights_active1, active_ratio1)
            lam0_spline = opt_mean_tuning(y0, yhat0, weights_spline0, spline_ratio0)
            lam1_spline = opt_mean_tuning(y1, yhat1, weights_spline1, spline_ratio1)

            active_result = odds_ratio_estimate_ci(
                y0, yhat0, y1, yhat1, weights_active0, weights_active1, alpha
            )
            active_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                probabilities_active0,
                probabilities_active1,
                selected_active0,
                selected_active1,
                lhat0=1,
                lhat1=1,
                mu0_hat=active_result.mu0_hat,
                mu1_hat=active_result.mu1_hat,
            )

            active_tuned_result = odds_ratio_estimate_ci(
                y0,
                yhat0,
                y1,
                yhat1,
                weights_active0,
                weights_active1,
                alpha,
                lhat0=lam0_active,
                lhat1=lam1_active,
            )
            active_tuned_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                probabilities_active0,
                probabilities_active1,
                selected_active0,
                selected_active1,
                lhat0=lam0_active,
                lhat1=lam1_active,
                mu0_hat=active_tuned_result.mu0_hat,
                mu1_hat=active_tuned_result.mu1_hat,
            )

            spline_result = odds_ratio_estimate_ci(
                y0, yhat0, y1, yhat1, weights_spline0, weights_spline1, alpha
            )
            spline_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                probabilities_spline0,
                probabilities_spline1,
                selected_spline0,
                selected_spline1,
                lhat0=1,
                lhat1=1,
                mu0_hat=spline_result.mu0_hat,
                mu1_hat=spline_result.mu1_hat,
            )

            spline_tuned_result = odds_ratio_estimate_ci(
                y0,
                yhat0,
                y1,
                yhat1,
                weights_spline0,
                weights_spline1,
                alpha,
                lhat0=lam0_spline,
                lhat1=lam1_spline,
            )
            spline_tuned_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                probabilities_spline0,
                probabilities_spline1,
                selected_spline0,
                selected_spline1,
                lhat0=lam0_spline,
                lhat1=lam1_spline,
                mu0_hat=spline_tuned_result.mu0_hat,
                mu1_hat=spline_tuned_result.mu1_hat,
            )

            uniform_probabilities0 = np.full(n0, frac_human)
            uniform_probabilities1 = np.full(n1, frac_human)
            selected_uniform0 = rng.binomial(1, uniform_probabilities0).astype(bool)
            selected_uniform1 = rng.binomial(1, uniform_probabilities1).astype(bool)
            weights_uniform0 = selected_uniform0.astype(float) / uniform_probabilities0
            weights_uniform1 = selected_uniform1.astype(float) / uniform_probabilities1

            uniform_result = odds_ratio_estimate_ci(
                y0, yhat0, y1, yhat1, weights_uniform0, weights_uniform1, alpha
            )
            uniform_fp_var = finite_population_log_odds_variance(
                y0,
                yhat0,
                y1,
                yhat1,
                uniform_probabilities0,
                uniform_probabilities1,
                selected_uniform0,
                selected_uniform1,
                lhat0=1,
                lhat1=1,
                mu0_hat=uniform_result.mu0_hat,
                mu1_hat=uniform_result.mu1_hat,
            )

            classical_result = classical_odds_ratio_estimate_ci(
                y0, y1, selected_uniform0, selected_uniform1, alpha, total_n=n
            )
            classical_fp_var = finite_population_classical_log_odds_variance(
                y0,
                y1,
                uniform_probabilities0,
                uniform_probabilities1,
                selected_uniform0,
                selected_uniform1,
                mu0_hat=classical_result.mu0_hat,
                mu1_hat=classical_result.mu1_hat,
            )

            method_results = [
                ("active", active_result, active_fp_var),
                ("active + tuning", active_tuned_result, active_tuned_fp_var),
                ("spline", spline_result, spline_fp_var),
                ("spline + tuning", spline_tuned_result, spline_tuned_fp_var),
                ("uniform", uniform_result, uniform_fp_var),
                ("classical", classical_result, classical_fp_var),
            ]
            for estimator, result, finite_population_log_variance in method_results:
                row = _result_row(
                    result,
                    estimator,
                    n_human,
                    frac_human,
                    trial,
                    alpha,
                    true_odds_ratio,
                    true_variance,
                    n,
                    finite_population_log_variance=finite_population_log_variance,
                )
                row["num_trial"] = trial
                row["burnin_steps"] = burnin_steps
                row["retrain_steps"] = retrain_steps
                row["uncertainty_model"] = uncertainty_model
                rows.append(row)

    return add_monte_carlo_variance(pd.DataFrame(rows))
