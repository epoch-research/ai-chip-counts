use squiggle for probabilistic estimates, not numpy

example usage:

import squigglepy as sq
import numpy as np
import matplotlib.pyplot as plt
from squigglepy.numbers import K, M
from pprint import pprint

pop_of_ny_2022 = sq.to(8.1*M, 8.4*M)  # This means that you're 90% confident the value is between 8.1 and 8.4 Million.
pct_of_pop_w_pianos = sq.to(0.2, 1) * 0.01  # We assume there are almost no people with multiple pianos
pianos_per_piano_tuner = sq.to(2*K, 50*K)
piano_tuners_per_piano = 1 / pianos_per_piano_tuner
total_tuners_in_2022 = pop_of_ny_2022 * pct_of_pop_w_pianos * piano_tuners_per_piano
samples = total_tuners_in_2022 @ 1000  # Note: `@ 1000` is shorthand to get 1000 samples

# Get mean and SD
print('Mean: {}, SD: {}'.format(round(np.mean(samples), 2),
                                round(np.std(samples), 2)))

# Get percentiles
pprint(sq.get_percentiles(samples, digits=0))


import squigglepy as sq

# You can add and subtract distributions
(sq.norm(1,3) + sq.norm(4,5)) @ 100
(sq.norm(1,3) - sq.norm(4,5)) @ 100
(sq.norm(1,3) * sq.norm(4,5)) @ 100
(sq.norm(1,3) / sq.norm(4,5)) @ 100

# You can also do math with numbers
~((sq.norm(sd=5) + 2) * 2)
~(-sq.lognorm(0.1, 1) * sq.pareto(1) / 10)

# You can change the CI from 90% (default) to 80%
sq.norm(1, 3, credibility=80)

# You can clip
sq.norm(0, 3, lclip=0, rclip=5) # Sample norm with a 90% CI from 0-3, but anything lower than 0 gets clipped to 0 and anything higher than 5 gets clipped to 5.

# You can also clip with a function, and use pipes
sq.norm(0, 3) >> sq.clip(0, 5)

# You can correlate continuous distributions
a, b = sq.uniform(-1, 1), sq.to(0, 3)
a, b = sq.correlate((a, b), 0.5)  # Correlate a and b with a correlation of 0.5
# You can even pass your own correlation matrix!
a, b = sq.correlate((a, b), [[1, 0.5], [0.5, 1]])

## Style notes

- **Comments**: Write comments in plain English. Don't use math variables (`c_q`, `f_q`, `s_h`) in code comments. Explain what's happening intuitively instead.
- **Markdown cells**: Explain math intuitively rather than with LaTeX formulae. Use markdown cells for conceptual explanations, and trim redundant code comments accordingly.
- **Complex expressions**: Split multi-factor expressions into separate lines, each with a brief comment explaining that step.
- **Naming**: Function names should be specific/descriptive when the function is domain-specific (e.g. `compute_nvidia_hyperscaler_chip_shares`, not `compute_owner_chip_shares`). Avoid unnecessary indirection like name-mapping dicts when you can use the data directly.

## CSV export conventions

All notebooks export estimates to `csv_export/` as CSVs. Common column definitions:

- **Name**: human-readable row identifier (e.g. `"Q1 2024 - Trainium2"`, `"Microsoft Q3 2025"`)
- **Chip manufacturer**: `"Nvidia"`, `"AMD"`, `"Google"`, `"Amazon"`, etc.
- **Start date / End date**: `M/D/YYYY` format (not zero-padded), e.g. `1/1/2024`, `12/31/2025`
- **Compute estimate in H100e (median)**: median H100-equivalent compute. Computed as `units * (chip_TOPS / 1979)` where 1979 = H100 8-bit TOPS. Left blank if chip specs aren't available.
- **H100e (5th percentile) / H100e (95th percentile)**: bounds of 90% credible interval for H100e
- **Number of Units**: median chip count. **Number of Units (5th/95th percentile)**: bounds of 90% credible interval
- **Notes**: includes a generated timestamp, e.g. `"Estimates generated on: 01-28-2026 12:48"`
- **Source / Link, Last Modified By, Last Modified**: typically left blank (populated downstream)

## Notebook patterns

- **Common flow**: load data from Google Sheets → define squigglepy distributions → run Monte Carlo (N_SAMPLES=5000) → interpolate fiscal→calendar quarters → compute running totals → compute H100e → export CSVs
- **Key utility functions** in `chip_estimates_utils.py`: `estimate_cumulative_chip_sales`, `interpolate_samples_to_calendar_quarters`, `compute_running_totals`, `compute_h100e_samples`, `export_nvidia_owners_csvs`, `make_incomplete_note_fn`
- **Sample arrays**: results are stored as `{quarter: {chip_type: np.array(N_SAMPLES)}}` dicts, preserving per-sample correlations across chips/quarters
- **Owner notebooks** (`*_owners.ipynb`): extend the base chip estimation by allocating chips to owners (hyperscalers, China, Other). Export to `owners_csv_export/` with an extra `Owner` column
- **Data inputs**: `data_inputs/` has local CSVs; revenue/price/ownership data is loaded from Google Sheets URLs
- **Chip types**: Nvidia uses `['A100', 'A800', 'H100/H200', 'H800', 'H20', 'B200', 'B300']`. China-spec chips are A800, H800, H20
- **H100e conversion**: `units * (chip_TOPS / 1979)` where CHIP_SPECS dict maps chip name → {TOPS, TDP}