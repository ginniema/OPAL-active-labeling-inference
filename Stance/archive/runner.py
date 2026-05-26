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
from helpers import *

# get our arguments
num_trials = 100
frac_humans = np.linspace(0.2, 0.5, 10)
settings = []
for num_trial in range(num_trials):
    for frac_human in frac_humans:
        settings.append((num_trial, frac_human))
num_trial, frac_human = settings[int(sys.argv[1])]


#### LOAD THE DATA

data = pd.read_csv('data/stance_dataset.csv')
data = data.sample(frac=1).reset_index(drop=True) # shuffle data

affirming_devices = ['uncover', 'realize', 'know', 'understand', 'learn', 'concede',
'remember', 'recall', 'discover', 'show', 'reveal', 'see',
'forget', 'find', 'point out', 'indicate', 'acknowledge',
'admit', 'realize', 'notice', 'certify', 'verify', 'corroborate', 'affirm', 'confirm', 'agree', 'conclude',
'proven', 'settled', 'conclusive', 'definitive',
'famed', 'unequivocal', 'skilful', 'notable', 'strong', 'famous', 'Nobel', 'skillful',
'Nobelist', 'Nobel Laureate', 'Nobel prize winner',
'Nobel prize winning', 'prize winning', 'award',
'winning', 'distinguished', 'well-grounded', 'esteemed', 'proficient', 'key', 'evidence', 'noted', 'top',
'preeminent', 'breakthrough', 'significant', 'intelligent', 'of import', 'celebrated', 'novel', 'recent',
'major', 'landmark', 'important', 'distinguished',
'renowned', 'peer-reviewed', 'expert', 'leading',
'thousand', '1000', 'hundred', '100', 'unanimous', 'diverse',
'substantial', 'many', 'multiple', 'dozen', 'numerous']

escaped_terms = [re.escape(term) for term in affirming_devices]
pattern = '|'.join(escaped_terms)
data['contains_affirming_device'] = data['sentence'].str.contains(pattern, case=False, regex=True)

Yhat_string = data["label_gpt4o"].to_numpy()
confidence = data["confidence_in_prediction_gpt-4o"].to_numpy()
nan_indices = list(np.where(pd.isna(confidence))[0]) + list(np.where(pd.isna(Yhat_string))[0])
good_indices = list(set(range(len(data))) - set(nan_indices))
confidence = confidence[good_indices]
device = data['contains_affirming_device'].to_numpy()[good_indices]
Yhat_string = Yhat_string[good_indices]
Y_string = data["MACE_pred"].to_numpy()[good_indices]
n = len(Yhat_string)
dict = {"A" : 1, "B" : 0, "C" : 0, "agrees": 1, "neutral" : 0, "disagrees": 0}
Yhat = np.array([dict[Yhat_string[i]] for i in range(n)])
Y = np.array([dict[Y_string[i]] for i in range(n)])
confidence = confidence.reshape(len(confidence),1)

Y1 = Y[device]
Y0 = Y[~device]
Yhat1 = Yhat[device]
Yhat0 = Yhat[~device]
confidence1 = confidence[device]
confidence0 = confidence[~device]
n0 = len(Y0)
n1 = len(Y1)

mu0 = Y0.mean()
mu1 = Y1.mean()
true_odds_ratio = (mu1 / (1 - mu1)) / (mu0 / (1 - mu0))
true_var = (1/np.sum(Y0==0) + 1/np.sum(Y0==1) + 1/np.sum(Y1==0) + 1/np.sum(Y1==1))*n

#### HELPER FUNCTION TO RUN ONE TRIAL:

def one_trial(num_trial, frac_human):
    np.random.seed(num_trial)
    alpha = 0.1
    num_methods = 6
    retrain_steps, burnin_steps = 50, 50 
    temp_df = pd.DataFrame({
        "lb": np.zeros(num_methods),
        "ub": np.zeros(num_methods),
        "interval width": np.zeros(num_methods),
        "coverage": np.zeros(num_methods),
        "estimator": [""] * num_methods,
        "$n_{\mathrm{human}}$": np.zeros(num_methods),
        "$n_{\mathrm{effective}}$": np.zeros(num_methods)
    })
    tau = 0.1
    frac_human_adjusted = (frac_human*n - burnin_steps)/(n - burnin_steps) # remove burnin_steps samples from available budget for both classes
    mu1_init, mu0_init = (Y1[:burnin_steps]).mean(), (Y0[:burnin_steps]).mean()

    tree = train_tree(confidence[:burnin_steps], ((Y - Yhat)[:burnin_steps])**2)
    uncertainties_active = np.sqrt(tree_predict(tree, confidence))
    #uncertainties_spline = uncertainties_active.copy()
    #uncertainty1 = np.sqrt(tree_predict(tree, confidence1_test))
    #uncertainty0 = np.sqrt(tree_predict(tree, confidence0_test))
    #uncertainty = np.concatenate([uncertainty1, uncertainty0])
    avg_uncertainty = np.mean(uncertainties_active)
    u1, u0 = uncertainties_active[device], uncertainties_active[~device]

    # need to get probabilities for spline version:
    p1_spline, p0_spline, params1_inv, params0_inv = sampling_probs_spline_inv(
        u1, u0, u1, u0, mu1_init, mu0_init, (n - burnin_steps) * frac_human_adjusted, num_knots=5, degree=3)
    prob_spline = np.zeros(n)
    prob_spline[device], prob_spline[~device] = p1_spline, p0_spline
    weights_active = np.zeros(n)
    weights_spline = np.zeros(n)
    sampling_ratio = np.zeros(n)
    sampling_ratio_spline = np.zeros(n)
    weights_active[:burnin_steps] = 1 # will label the first burnin_steps 
    weights_spline[:burnin_steps] = 1 # will label the first burnin_steps
    
    for t in range(burnin_steps, n):
        if ((t-burnin_steps) % retrain_steps == 0):
            # re-train for active and spline
            obs_inds = np.where(weights_active)
            obs_inds_spline = np.where(weights_spline)
            tree_active = train_tree(confidence[obs_inds], ((Y - Yhat)[obs_inds])**2)
            tree_spline = train_tree(confidence[obs_inds_spline], ((Y - Yhat)[obs_inds_spline])**2)
            uncertainties_active = np.sqrt(tree_predict(tree_active, confidence))
            avg_uncertainty = np.mean(uncertainties_active)

            uncertainties_spline = np.sqrt(tree_predict(tree_spline, confidence))
            u1, u0 = uncertainties_spline[device], uncertainties_spline[~device]
            p1_spline, p0_spline, params1_inv, params0_inv = sampling_probs_spline_inv(
                u1, u0, u1, u0, mu1_init, mu0_init, (n - burnin_steps) * frac_human_adjusted, num_knots=5, degree=3)
            prob_spline[device], prob_spline[~device] = p1_spline, p0_spline
            
        # for active
        sampling_prob = uncertainties_active[t]/avg_uncertainty*frac_human_adjusted
        sampling_prob = np.clip((1-tau)*sampling_prob + tau*frac_human_adjusted, 0, 1)
        sampling_ratio[t] = (1-sampling_prob)/sampling_prob
        weights_active[t] = bernoulli.rvs(sampling_prob)/sampling_prob

        # for spline
        sampling_prob_spline = prob_spline[t]
        sampling_ratio_spline[t] = (1-sampling_prob_spline)/sampling_prob_spline
        weights_spline[t] = bernoulli.rvs(sampling_prob_spline)/sampling_prob_spline
        
    weights_active0 = weights_active[~device]
    weights_active1 = weights_active[device]
    weights_spline0 = weights_spline[~device]
    weights_spline1 = weights_spline[device]
    sampling_ratio0 = sampling_ratio[~device]
    sampling_ratio1 = sampling_ratio[device]
    sampling_ratio_spline0 = sampling_ratio_spline[~device]
    sampling_ratio_spline1 = sampling_ratio_spline[device]

    lam0 = opt_mean_tuning(Y0, Yhat0, weights_active0, sampling_ratio0)
    lam1 = opt_mean_tuning(Y1, Yhat1, weights_active1, sampling_ratio1)
    l, u, varhat = odds_ratio_ci(Y0, Yhat0, Y1, Yhat1, weights_active0, weights_active1, alpha, lhat0=1, lhat1=1)
    coverage = (true_odds_ratio >= l)*(true_odds_ratio <= u)
    temp_df.loc[0] = l, u, u-l, coverage, "active", int(n*frac_human), (true_var/varhat)*n

    l, u, varhat = odds_ratio_ci(Y0, Yhat0, Y1, Yhat1, weights_active0, weights_active1, alpha, lhat0=lam0, lhat1=lam1)
    coverage = (true_odds_ratio >= l)*(true_odds_ratio <= u)
    temp_df.loc[1] = l, u, u-l, coverage, "active+", int(n*frac_human), (true_var/varhat)*n

    lam0_spline = opt_mean_tuning(Y0, Yhat0, weights_spline0, sampling_ratio_spline0)
    lam1_spline = opt_mean_tuning(Y1, Yhat1, weights_spline1, sampling_ratio_spline1)
    l, u, varhat = odds_ratio_ci(Y0, Yhat0, Y1, Yhat1, weights_spline0, weights_spline1, alpha, lhat0=1, lhat1=1)
    coverage = (true_odds_ratio >= l)*(true_odds_ratio <= u)
    temp_df.loc[2] = l, u, u-l, coverage, "spline", int(n*frac_human), (true_var/varhat)*n
    l, u, varhat = odds_ratio_ci(Y0, Yhat0, Y1, Yhat1, weights_spline0, weights_spline1, alpha, lhat0=lam0_spline, lhat1=lam1_spline)
    coverage = (true_odds_ratio >= l)*(true_odds_ratio <= u)
    temp_df.loc[3] = l, u, u-l, coverage, "spline+", int(n*frac_human), (true_var/varhat)*n

    xi_unif0 = bernoulli.rvs([frac_human] * n0)
    xi_unif1 = bernoulli.rvs([frac_human] * n1)
    l, u, varhat = odds_ratio_ci(Y0, Yhat0, Y1, Yhat1, xi_unif0/frac_human, xi_unif1/frac_human, alpha, lhat0=1, lhat1=1)
    coverage = (true_odds_ratio >= l)*(true_odds_ratio <= u)
    temp_df.loc[4] = l, u, u-l, coverage, "uniform", int(n*frac_human), (true_var/varhat)*n
    
    mu0 = Y0[np.where(xi_unif0)].mean()
    mu1 = Y1[np.where(xi_unif1)].mean()
    odds_ratio_est = np.log((mu1 / (1 - mu1)) / (mu0 / (1 - mu0)))
    varhat = (1/np.sum(Y0[np.where(xi_unif0)]==0) + 1/np.sum(Y0[np.where(xi_unif0)]==1) + 1/np.sum(Y1[np.where(xi_unif1)]==0) + 1/np.sum(Y1[np.where(xi_unif1)]==1))
    l = np.exp(odds_ratio_est - norm.ppf(1-alpha/2)*np.sqrt(varhat))
    u = np.exp(odds_ratio_est + norm.ppf(1-alpha/2)*np.sqrt(varhat))
    coverage = (true_odds_ratio >= l)*(true_odds_ratio <= u)
    temp_df.loc[5] = l, u, u-l, coverage, "classical", int(n*frac_human), (true_var/varhat)
    
    # save our file
    # print("YAY")
    temp_df.to_csv(f"logs/frac-human={frac_human}_num-trial={num_trial}.csv", index=False)

# run the actual function
one_trial(num_trial, frac_human)