# OPAL Active Labeling Inference

Reproducible code examples for the OPAL active-labeling inference paper.

## Shared helpers

- `utils.py`: shared numerical helpers and the batch/sequential odds-ratio Monte Carlo runners.
- `plotting.py`: plotting helpers.

In plotting.py, effective sample size plots include `mean +/- 1` Monte Carlo standard deviation
bars and also quantile ribbons. Passing `error_bars="none"` to `plot_effective_sample_size` gives just mean curves. 
Use `plot_effective_sample_size_multiplier` to show ESS relative to the matched
classical replicate, which puts the classical baseline at exactly 1.

The code includes traditional Wald-style intervals as well as finite-population-calibrated coverage used for evaluation
against the fixed empirical population, following the paper's Appendix I.2
calibration.

## Additional Examples
- `BRCA/BRCA_active_clean.ipynb`: BRCA TNBA workflow.
- `CheXpert/CheXpert_active_clean.ipynb`: Cardiomegaly AP-vs-PA workflow.
- `Alphafold/Alphafold_active_clean.ipynb`: phosphorylation-vs-nonphosphorylation workflow.
- `Stance/Stance_active_clean.ipynb`: stance batch workflow.
- `Stance/Stance_sequential_clean.ipynb`: sequential Stance stability workflow that reruns the sequential Monte Carlo and computes exact finite-population calibrated coverage during each replicate.
- `Simulation/`: simulation appendix code for Kendall's tau and synthetic odds-ratio examples.
Data for all examples can be found in `Data/<name of example>`

Result CSVs keep the original estimator IDs such
as `spline` and `spline+`. The plotting helpers display these methods as
`OPAL` and `OPAL + tuning` in figures and rendered tables.

The cleaned simulation scripts write generated CSVs to `Simulation/results/`
and figures to `Simulation/plots/`. See `Simulation/README.md` for the full
commands, including the four odds-ratio cases crossing 20/80 vs 50/50
hard/easy group composition with oracle vs estimated sampling probabilities.

## Setup

```bash
uv sync
```

or install the dependencies from `pyproject.toml` with your preferred Python
environment manager.
