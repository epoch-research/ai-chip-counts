# ai-chip-counts
Code for estimating total quantities of AI chips/compute

See relevant data hub(s) on Epoch.ai for documentation

Nvidia annual prices and quarterly revenue data are stored in CSV here for reference/backup, but script currently looks at a Google sheet version

Run nvidia_estimates.ipynb

## How "Other" Nvidia ownership is computed

Ownership of Nvidia chips is split across hyperscalers (Microsoft, Meta, Amazon, Google, Oracle), China, and "Other" (everyone else). The Other category is built in two stages so we can combine a revenue-based baseline with independent estimates for specific companies.

1. **Baseline Other from the revenue model.** `nvidia_owners.ipynb` takes the total Nvidia chip estimates from the revenue model and subtracts hyperscaler and China allocations with correlated Monte Carlo samples. The per-sample residual is exported as `data_inputs/nvidia_other_cumulative_totals.csv`, `data_inputs/nvidia_other_cumulative_by_chip.csv`, and `data_inputs/nvidia_other_quarters_by_chip.csv`.

2. **Subtract named other owners.** `nvidia_owners_other.ipynb` reads the baseline and subtracts independently curated estimates for specific companies (e.g. xAI, CoreWeave, Tesla) from two input sources combined at load time:
   - **Hand-curated rows** in `data_inputs/named_other_owners_cumulative_totals.csv` (cumulative H100e) and `data_inputs/named_other_owners_cumulative_by_chip.csv` (cumulative units by chip type), maintained manually in a Google Sheet for owners that don't have their own model.
   - **Generated rows** from standalone owner notebooks. Any subdirectory under `owners_csv_export/` (e.g. `owners_csv_export/coreweave/`) is auto-scanned for files ending in `cumulative_totals.csv` and `cumulative_by_chip.csv`, and those rows are pulled in alongside the hand-curated ones. See `coreweave_estimate.ipynb` for the first example — it estimates CoreWeave GPUs from power capacity and disclosed counts and writes to `owners_csv_export/coreweave/`.

   The two input types feed separate subtraction paths — totals drives the totals output, by-chip drives the by-chip output — so they don't need to be internally consistent. The leftover "Remainder" is exported to `owners_csv_export/`.

The point of this setup is to anchor on a revenue-based model for the overall envelope while letting hand-curated or notebook-generated estimates pin down individual companies that show up in that envelope. Adding a new named owner is just: write a notebook that exports to `owners_csv_export/<owner>/`, then add the owner's name to `OTHER_OWNERS_TO_EXCLUDE` in `nvidia_owners_other.ipynb`.
