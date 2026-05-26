# OPAL Active Labeling Inference

Reproducible code examples for the OPAL active-labeling inference paper.

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

Run notebooks from the `BRCA/` directory so relative paths resolve.

## Setup

```bash
uv sync
```

or install the dependencies from `pyproject.toml` with your preferred Python
environment manager.
