import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pyreadstat
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import re
import xgboost as xgb
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys
from scipy.stats import norm, bernoulli, beta, expon
from ppi_py.datasets import load_dataset
import matplotlib.patheffects as pe
from utils import make_width_coverage_plot, make_width_coverage_split_plot, make_budget_plot
import warnings; warnings.simplefilter('ignore')
import cvxpy as cp
from structuremap.processing import get_smooth_score
from IPython.display import clear_output
from patsy import dmatrix # For generating spline basis
import time
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches


# setup_spline_basis
def setup_spline_basis(uncertainties, num_knots=5, degree=3, name='u'):
    """
    Sets up B-spline basis functions for given uncertainties. 
    """
    if len(uncertainties) == 0:
        return np.zeros((0, 0)), None # Handle empty input
        
    # Ensure enough unique points relative to knots/degree
    unique_uncertainties = np.unique(uncertainties)
    required_unique = degree + 1 
    if len(unique_uncertainties) < required_unique:
         print(f"Warning: Need at least {required_unique} unique uncertainty points for degree {degree} spline, "
               f"found {len(unique_uncertainties)}. Reducing degree or complexity might be needed.")

    if len(unique_uncertainties) < num_knots + 1:
        num_knots = max(0, len(unique_uncertainties) - degree - 1) # Try to leave enough points between knots


    # Define knot locations (e.g., quantiles)
    knot_quantiles = np.linspace(0, 1, num_knots + 2)[1:-1] 
    
    # Handle case where all uncertainties are the same
    if len(unique_uncertainties) == 1:
         knots = np.array([]) # No internal knots
         num_knots = 0 # Force num_knots to 0
         # Use df instead of knots for constant case
         formula = f"bs({name}, df={degree+1}, degree={degree}, include_intercept=True)"
         data = {name: uncertainties}
         try:
              basis_matrix = dmatrix(formula, data, return_type='dataframe')
              return basis_matrix.values, None
         except Exception as e:
             print(f"Patsy error even with df for constant uncertainty: {e}. Returning basic constant basis.")
             # Fallback to a truly constant basis (vector of ones)
             return np.ones((len(uncertainties), 1)), None


    # Proceed with knot calculation for non-constant cases
    if num_knots > 0:
         knots = np.quantile(uncertainties, knot_quantiles)
         knots = np.unique(knots) # Ensure knots are unique
         # Check if knots collapsed or are too close to boundaries
         min_u, max_u = np.min(uncertainties), np.max(uncertainties)
         # Remove knots too close to the boundary? Sometimes helps patsy.
         # knots = knots[(knots > min_u) & (knots < max_u)] 
         if len(knots) < num_knots:
              # print(f"Warning: Number of unique knots reduced to {len(knots)}.")
              pass # Let patsy handle fewer knots
         if len(knots) == 0 and num_knots > 0:
              print("Warning: All internal knots collapsed. Proceeding without internal knots.")
              num_knots = 0 # Update num_knots if they vanished

    else: # num_knots is 0
         knots = np.array([])

    # Create the design matrix using patsy
    # Use bs() for B-splines
    if num_knots > 0:
        formula = f"bs({name}, knots=knots, degree=degree, include_intercept=True)"
    else: # No internal knots, use df
         df = degree + 1 # Number of basis functions for degree d, no internal knots
         formula = f"bs({name}, df={df}, degree={degree}, include_intercept=True)"

    data = {name: uncertainties}
    
    try:
        basis_matrix = dmatrix(formula, data, return_type='dataframe')
    except Exception as e:
         print(f"Patsy error: {e}. Trying simpler df if knots were used.")
         try:
             # Fallback if specific knot creation failed
             df = num_knots + degree + 1 # Total degrees of freedom expected
             formula = f"bs({name}, df={df}, degree={degree}, include_intercept=True)"
             basis_matrix = dmatrix(formula, data, return_type='dataframe')
             knots = None # Knots aren't explicitly defined in this case
         except Exception as e2:
             print(f"Patsy fallback failed: {e2}. Returning basic constant basis.")
             return np.ones((len(uncertainties), 1)), None


    # Check if basis matrix has expected shape
    expected_cols = num_knots + degree + 1 # Intercept is included in the count here
    if basis_matrix.shape[1] == 0 and expected_cols > 0 :
         print(f"Warning: Patsy generated an empty basis matrix ({basis_matrix.shape}). Check inputs/parameters. Returning constant basis.")
         return np.ones((len(uncertainties), 1)), None
         
    return basis_matrix.values, knots # Return as numpy array


def sampling_probs_spline_inv(u1_all, u0_all, b1, b0, mu1, mu0, n_budget, num_knots=5, degree=3, verbose_solver=False):
    """
    Compute optimal sampling probabilities parametrized by splines on INVERSE probabilities.
    MODIFIED TO PRIORITIZE MOSEK SOLVER.

    Models log(1/p_i) = B(u_i) @ theta, such that 1/p_i >= 1.

    Args:
        u1_all (np.ndarray): Individual uncertainty scores for group 1 points.
        u0_all (np.ndarray): Individual uncertainty scores for group 0 points.
        b1 (np.ndarray): individual scores (potentially diff from uncertainty scores) for constructing spline basis for group 1 points.
        b0 (np.ndarray): individual scores ("") for constructing spline basis for group 0 points.
        mu1 (float): Estimated mean for group 1 (used for scaling objective).
        mu0 (float): Estimated mean for group 0 (used for scaling objective).
        n_budget (float): Total expected sample budget.
        num_knots (int): Number of internal knots for the splines.
        degree (int): Degree of the spline polynomials.
        verbose_solver (bool): If True, show detailed solver output.

    Returns:
        tuple: (probs1, probs0, spline_params1, spline_params0)
               probs1 (np.ndarray): Sampling probabilities for group 1 points.
               probs0 (np.ndarray): Sampling probabilities for group 0 points.
               spline_params1 (dict): Info about group 1 spline (coeffs, knots, degree).
               spline_params0 (dict): Info about group 0 spline (coeffs, knots, degree).
        Returns (None, None, None, None) if optimization fails.
    """
    overall_start_time = time.perf_counter()
    n1 = len(u1_all)
    n0 = len(u0_all)

    if n1 == 0 and n0 == 0:
        print("Warning: Both groups are empty.")
        return np.array([]), np.array([]), None, None
    if n_budget <= 0:
         print("Warning: Budget is non-positive. Returning zero probabilities.")
         return np.zeros(n1), np.zeros(n0), None, None

    # --- Setup Spline Basis (Same as before) ---
    B1, knots1 = np.zeros((n1, 0)), None
    B0, knots0 = np.zeros((n0, 0)), None
    n_basis1 = 0
    n_basis0 = 0

    if n1 > 0:
        try:
             #B1, knots1 = setup_spline_basis(u1_all, num_knots, degree, name='u1')
             B1, knots1 = setup_spline_basis(b1, num_knots, degree, name = 'u1')
             n_basis1 = B1.shape[1]
             if n_basis1 == 0: raise ValueError("Basis matrix is empty")
        except Exception as e:
            print(f"Could not create spline basis for group 1: {e}. Check uncertainty values.")
            n_basis1 = 0 # Ensure it's zero
    if n0 > 0:
        try:
            B0, knots0 = setup_spline_basis(b0, num_knots, degree, name = 'u0')
            #B0, knots0 = setup_spline_basis(u0_all, num_knots, degree, name='u0')
            n_basis0 = B0.shape[1]
            if n_basis0 == 0: raise ValueError("Basis matrix is empty")
        except Exception as e:
             print(f"Could not create spline basis for group 0: {e}. Check uncertainty values.")
             n_basis0 = 0 # Ensure it's zero

    if n_basis1 == 0 and n_basis0 == 0 and (n1 > 0 or n0 > 0):
         print("Error: Could not create spline basis for either group. Cannot optimize.")
         return None, None, None, None

    # --- CVXPY Optimization
    theta1 = cp.Variable(n_basis1, name="theta1") if n_basis1 > 0 else None
    theta0 = cp.Variable(n_basis0, name="theta0") if n_basis0 > 0 else None

    log_x1, log_x0 = None, None
    x1, x0 = None, None
    p1, p0 = None, None

    constraints = []
    objective_terms = []
    budget_terms = []

    # Group 1 calculations
    if n1 > 0 and n_basis1 > 0 and theta1 is not None and mu1 > 0 and mu1 < 1:
        w1 = u1_all / (n1**2 * (mu1**2) * (1 - mu1)**2)
        log_x1 = B1 @ theta1
        x1 = cp.exp(log_x1)
        p1 = cp.exp(-log_x1)
        objective_terms.append(cp.sum(cp.multiply(w1, x1)))
        budget_terms.append(cp.sum(p1))
        #budget_terms.append(-log_x1)
        constraints.append(log_x1 >= 0)
    elif n1 > 0:
         print(f"Warning: Skipping group 1 contributions (n1={n1}, n_basis1={n_basis1}, mu1={mu1})")

    # Group 0 calculations
    if n0 > 0 and n_basis0 > 0 and theta0 is not None and mu0 > 0 and mu0 < 1:
        w0 = u0_all / (n0**2 * (mu0**2) * (1 - mu0)**2)
        log_x0 = B0 @ theta0
        x0 = cp.exp(log_x0)
        p0 = cp.exp(-log_x0)
        objective_terms.append(cp.sum(cp.multiply(w0, x0)))
        budget_terms.append(cp.sum(p0))
        #budget_terms.append(-log_x0)
        constraints.append(log_x0 >= 0)
    elif n0 > 0:
        print(f"Warning: Skipping group 0 contributions (n0={n0}, n_basis0={n_basis0}, mu0={mu0})")

    if not objective_terms:
         print("Error: No valid objective terms could be constructed.")
         return None, None, None, None

    # Objective function: Minimize sum of weighted inverse probabilities (w_i * x_i)
    objective = cp.Minimize(cp.sum(objective_terms))

    # Budget Constraint: sum(p_i) <= n_budget
    if budget_terms: # Only add budget constraint if there are probabilities to sum
        constraints.append(cp.sum(budget_terms) <= n_budget)
        #log_xs = cp.multiply(np.ones(len(budget_terms), budget_terms))
        #constraints.append(cp.log_sum_exp(-cp.hstack(budget_terms)) <= cp.log(n_budget))
    else:
         print("Warning: No variables to constrain budget.")

    problem = cp.Problem(objective, constraints)

    # --- Solve the problem with MOSEK, falling back to ECOS, then SCS ---
    # Define solver priority list
    solver_list = [cp.MOSEK, cp.ECOS, cp.SCS]
    solved = False

    for solver_name in solver_list: # solver_name is 'MOSEK', 'ECOS', or 'SCS'

        #print(f"Attempting to solve with {solver_name}...")
        try:
            # --- Solver Specific Arguments & Options ---
            # Start with common arguments
            solver_kwargs = {
                'verbose': verbose_solver,
                # General tolerance arguments might be added here,
                # but specific ones are often better handled below.
            }

            # --- Configure solver-specific arguments ---

            if solver_name == cp.ECOS: # Or just 'ECOS'
                 # ECOS takes arguments directly
                 solver_kwargs['max_iters'] = 200
                 solver_kwargs['abstol'] = 1e-8 # Example specific tolerances
                 solver_kwargs['reltol'] = 1e-7

            elif solver_name == cp.SCS: # Or just 'SCS'
                 # SCS takes arguments directly
                 solver_kwargs['eps'] = 1e-5 # Example specific tolerance

            # --- Solve ---
            # Unpack the prepared kwargs dictionary. If solver is MOSEK,
            # this will now directly pass MSK_DPAR... arguments.
            #print(f"Passing kwargs to {solver_name}: {solver_kwargs}") # Debug print
            problem.solve(solver=solver_name, **solver_kwargs, mosek_params={"MSK_IPAR_NUM_THREADS": 1})
            final_status = problem.status

            # --- Check Status ---
            # If solved optimally or inaccurately, stop trying other solvers
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                #print(f"{solver_name} finished with status: {problem.status}")
                solved = True
                break # Exit the loop
            else:
                print(f"{solver_name} failed with status: {problem.status}")
                # Continue to the next solver in the list

        except (cp.SolverError, ValueError, ImportError) as e:
             # ImportError can happen if solver seems installed but has runtime issues (e.g., MOSEK license)
            print(f"{solver_name} solver encountered an error: {e}")
            # Continue to the next solver in the list

    if not solved:
        print("All attempted solvers failed to find an optimal or inaccurate solution.")
        # Ensure problem status reflects the failure if loop finishes without success
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
             problem.status = "solver_error"


    # --- Check for solution (rest of the code is the same) ---
    overall_end_time = time.perf_counter()
    if problem.status in ["optimal", "optimal_inaccurate"]:
        #print(f"Optimization successful (Final Status: {problem.status}). Objective value: {problem.value}")
        #print(f"Total Time: {overall_end_time - overall_start_time:.4f} seconds") # Added total time

        final_probs1 = np.zeros(n1)
        final_probs0 = np.zeros(n0)
        final_theta1 = None
        final_theta0 = None

        # Retrieve results safely (check if variables and .value exist)
        if p1 is not None and theta1 is not None and hasattr(theta1, 'value') and theta1.value is not None:
            final_theta1 = theta1.value
            try:
                 log_x1_opt = B1 @ final_theta1
                 final_probs1 = np.exp(-log_x1_opt)
                 final_probs1 = np.clip(final_probs1, 0, 1)
            except Exception as calc_e:
                 print(f"Error calculating final probabilities for group 1: {calc_e}")
                 final_probs1 = np.zeros(n1) # Fallback
                 final_theta1 = None # Invalidate theta if calc failed

        if p0 is not None and theta0 is not None and hasattr(theta0, 'value') and theta0.value is not None:
            final_theta0 = theta0.value
            try:
                log_x0_opt = B0 @ final_theta0
                final_probs0 = np.exp(-log_x0_opt)
                final_probs0 = np.clip(final_probs0, 0, 1)
            except Exception as calc_e:
                 print(f"Error calculating final probabilities for group 0: {calc_e}")
                 final_probs0 = np.zeros(n0) # Fallback
                 final_theta0 = None # Invalidate theta if calc failed

        # Store spline parameters
        spline_params1 = {'coeffs': final_theta1, 'knots': knots1, 'degree': degree, 'n_basis': n_basis1} if n_basis1 > 0 and final_theta1 is not None else None
        spline_params0 = {'coeffs': final_theta0, 'knots': knots0, 'degree': degree, 'n_basis': n_basis0} if n_basis0 > 0 and final_theta0 is not None else None

        # Sanity check budget constraint
        actual_expected_samples = np.sum(final_probs1) + np.sum(final_probs0)
        # Use a slightly larger tolerance for feasibility checks after solve
        # Check against 1% deviation or a small absolute value
        upper_bound = max(n_budget * 1.01, n_budget + 1e-4)
        lower_bound = min(n_budget * 0.99, n_budget - 1e-4)

        if actual_expected_samples > upper_bound:
             print(f"Warning: Calculated expected samples ({actual_expected_samples:.4f}) "
                   f"significantly exceed budget ({n_budget:.4f}). Status: {problem.status}")
        # Check for underutilization only if budget likely should have been tight (objective > 0)
        elif actual_expected_samples < lower_bound and abs(problem.value) > 1e-6:
              print(f"Warning: Calculated expected samples ({actual_expected_samples:.4f}) "
                   f"is significantly under budget ({n_budget:.4f}). Solution might be suboptimal if budget wasn't tight.")


        return final_probs1, final_probs0, spline_params1, spline_params0

    else:
        print(f"Problem not solved optimally. Final Status: {problem.status}")
        print(f"Total Time: {overall_end_time - overall_start_time:.4f} seconds")
        return None, None, None, None


def opt_mean_tuning(Y, Yhat, weights, sampling_ratio): #lambda star is sum of Y * f where f is Yhat * weights * sampling ratio
    return np.clip(np.mean(Y*Yhat*weights*sampling_ratio)/np.mean(Yhat**2*sampling_ratio), 0, 1)

def odds_ratio_ci(Y0, Yhat0, Y1, Yhat1, weights0, weights1, alpha, lhat0=None, lhat1=None):
    n0 = Y0.shape[0]
    n1 = Y1.shape[0]    
    mu0_hat = np.mean(lhat0*Yhat0 + (Y0 - lhat0*Yhat0)*weights0)
    mu1_hat = np.mean(lhat1*Yhat1 + (Y1 - lhat1*Yhat1)*weights1)
    pointest_log = np.log(mu1_hat/(1-mu1_hat)) - np.log(mu0_hat/(1-mu0_hat))
    var_mu0_hat = np.var(lhat0*Yhat0 + (Y0 - lhat0*Yhat0)*weights0)
    var_mu1_hat = np.var(lhat1*Yhat1 + (Y1 - lhat1*Yhat1)*weights1)
    var0 = var_mu0_hat/((mu0_hat*(1-mu0_hat))**2)
    var1 = var_mu1_hat/((mu1_hat*(1-mu1_hat))**2)
    p0 = n0/(n0+n1)
    p1 = n1/(n0+n1)
    var = 1/p0*var0 + 1/p1*var1
    width_log = norm.ppf(1-alpha/2)*np.sqrt(var/(n0+n1))
    return np.exp(pointest_log - width_log), np.exp(pointest_log + width_log), var


def train_tree(X, Y, eta=0.001, max_depth=3, objective='reg:squarederror', boost_rounds=2000):
    dtrain = xgb.DMatrix(X, label=Y)
    tree = xgb.train({'eta': eta, 'max_depth': max_depth, 'objective': objective}, dtrain, boost_rounds)
    return tree


def tree_predict(tree, X):
    return tree.predict(xgb.DMatrix(X))