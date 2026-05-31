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
