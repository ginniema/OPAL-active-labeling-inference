# Simulation Examples

This directory contains cleaned scripts for the simulation appendix.

## Kendall's tau

Run the full paper-scale simulation:

```bash
python Simulation/kendall_tau_simulation.py --trials 500
python Simulation/plot_simulation_results.py --skip-odds
```

Outputs are written to `Simulation/results/` and `Simulation/plots/`.

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

## Uniform Mixing

Tune the method-uniform mixing weight for the odds-ratio simulation:

```bash
python Simulation/uniform_mixing_simulation.py --trials 500 --n-train 500
python Simulation/plot_simulation_results.py --skip-odds --skip-kendall --plot-uniform-mixing
```

The selected weights are written to `Simulation/results/uniform_mixing_lambda.csv`;
the full lambda grid is written to `Simulation/results/uniform_mixing_lambda_grid.csv`.
The convention is `pi_lambda = lambda * pi_method + (1 - lambda) * pi_uniform`,
so `lambda = 1` is the unmixed method.
