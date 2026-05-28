# Simulation Examples

This directory contains cleaned scripts for the simulation appendix.

## Kendall's tau

Run the full paper-scale simulation:

```bash
python Simulation/kendall_tau_simulation.py --trials 500
python Simulation/plot_simulation_results.py --skip-odds
```

Outputs are written to `Simulation/results/` and `Simulation/plots/`. The
implementation writeup is in `Simulation/appendix_kendall_tau.md`.

The original archived Kendall tau simulation had no separate classical
label-only curve. The cleaned script adds one by default: it uses the same
uniform human-label sample as the prediction-assisted uniform curve, but
estimates Kendall's tau-a directly from the labeled units.
ESS is reported with respect to the full-population classical Kendall tau
variance, so the classical curve is exactly the diagonal `n_effective =
n_human` reference.

To reproduce the archived Kendall tau figure exactly, use the original
workspace convention where the labels used to fit the seed nuisance model also
remain in the Monte Carlo inference population:

```bash
python Simulation/kendall_tau_simulation.py --trials 500 --include-seed-in-inference
python Simulation/plot_simulation_results.py --skip-odds
```

## Odds Ratio

Run the four default odds-ratio cases:

```bash
python Simulation/odds_ratio_simulation.py --trials 500
python Simulation/plot_simulation_results.py --skip-kendall
```

The four cases cross group composition with the source of policy probabilities:

- `odds_ratio_20_80_oracle`: 20/80 hard/easy split, oracle probabilities.
- `odds_ratio_20_80_estimated`: 20/80 hard/easy split, estimated probabilities.
- `odds_ratio_balanced_oracle`: 50/50 hard/easy split, oracle probabilities.
- `odds_ratio_balanced_estimated`: 50/50 hard/easy split, estimated probabilities.

For a fast smoke test, pass `--quick` to either simulation script. To regenerate
the cleaned four-panel odds-ratio figure from the old 500-trial CSVs instead of
rerunning, use:

```bash
python Simulation/plot_simulation_results.py --copy-legacy-odds-results --skip-kendall
```
