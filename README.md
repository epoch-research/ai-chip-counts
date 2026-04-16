# ai-chip-counts
Code for estimating total quantities of AI chips/compute

See relevant data hub(s) on Epoch.ai for documentation

Nvidia annual prices and quarterly revenue data are stored in CSV here for reference/backup, but script currently looks at a Google sheet version

Run nvidia_estimates.ipynb

## How "Other" Nvidia ownership is computed

Ownership of Nvidia chips is split across hyperscalers (Microsoft, Meta, Amazon, Google, Oracle), China, and "Other" (everyone else). The Other category is built in two stages so we can combine a revenue-based baseline with independent estimates for specific companies.

1. **Baseline Other from the revenue model.** `nvidia_owners.ipynb` takes the total Nvidia chip estimates from the revenue model and subtracts hyperscaler and China allocations with correlated Monte Carlo samples. The per-sample residual is exported as `data_inputs/nvidia_other_cumulative_totals.csv`, `data_inputs/nvidia_other_cumulative_by_chip.csv`, and `data_inputs/nvidia_other_quarters_by_chip.csv`.

2. **Subtract named other owners.** `nvidia_owners_other.ipynb` reads that baseline and subtracts independently curated estimates for specific companies (e.g. xAI, CoreWeave, Tesla) from `data_inputs/named_other_owners_totals.csv` (cumulative H100e) and `data_inputs/named_other_owners_by_chip.csv` (cumulative units by chip type). These two inputs feed separate subtraction paths — totals input drives the totals output, by-chip input drives the by-chip output — so they don't need to be internally consistent. The leftover "Remainder" is exported to `owners_csv_export/`.

The point of this setup is to anchor on a revenue-based model for the overall envelope while letting hand-curated, non-revenue-based estimates pin down individual companies that show up in that envelope.
