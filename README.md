# OPAL Active Labeling Inference

Reproducible code examples for the OPAL active-labeling inference paper.

## Shared code

- `utils.py`: shared numerical helpers and the batch odds-ratio Monte Carlo runner.
- `plotting.py`: shared ggplot2 `theme_bw`-style plotting helpers.

The effective sample size plot uses `mean +/- 1` Monte Carlo standard deviation
bars by default, matching the Robust Sampling paper convention. Pass
`error_bars="none"` to `plot_effective_sample_size` for a clean mean curve,
`show_se=True` for standard error bars, or `error_style="ribbon"` for shaded
bands instead of vertical bars.

## BRCA

The BRCA example is in `BRCA/`.

- `BRCA/BRCA_active_clean.ipynb`: cleaned notebook for the paper workflow.
- `BRCA/archive/BRCA_active_original.ipynb`: preserved copy of the exploratory notebook.
- `BRCA/utils.py`: shared numerical helpers that are not BRCA-data specific.
- `BRCA/plotting.py`: shared plotting helpers with a ggplot2 `theme_bw`-style look.
- `Data/BRCA/`: compact tabular inputs used by the notebook.

The BRCA Monte Carlo output includes both the usual superpopulation-style Wald
coverage and the finite-population-calibrated coverage used for evaluation
against the fixed empirical population, following the paper's Appendix I.2
calibration.

Run the BRCA notebook from `BRCA/` so its local imports resolve. The newer
cleaned notebooks for CheXpert, AlphaFold, and Stance add the repo root to
`sys.path`, so they can be run from the repo root or their example directory.

## Additional Examples

- `CheXpert/CheXpert_active_clean.ipynb`: cleaned Cardiomegaly AP-vs-PA workflow.
- `Alphafold/Alphafold_active_clean.ipynb`: cleaned phosphorylation-vs-nonphosphorylation workflow.
- `Stance/Stance_active_clean.ipynb`: cleaned stance batch workflow.
- `Stance/Stance_sequential_clean.ipynb`: cleaned sequential Stance stability workflow using the archived sequential Monte Carlo output.

Each example has `archive/` copies of the exploratory source files and writes
generated outputs to its local `plots/` and `results/` directories. Compact
inputs live under `Data/<Example>/`.

For historical consistency, result CSVs keep the original estimator IDs such
as `spline` and `spline+`. The plotting helpers display these methods as
`OPAL` and `OPAL + tuning` in figures and rendered tables.

## Setup

```bash
uv sync
```

or install the dependencies from `pyproject.toml` with your preferred Python
environment manager.
