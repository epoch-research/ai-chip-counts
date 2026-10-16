"""Amazon Trainium chip-sales model.

Estimates quarterly and cumulative Trainium production (Trainium1, Trainium2, Trainium3)
from Q1 2024 through Q2 2026. Unlike the Nvidia/AMD/TPU models there is no revenue
line to divide by a chip price; instead the model is built from a handful of anchors:

- Trainium2, 2024-2025: Amazon's "1.4 million chips landed" (Q4 2025 earnings), spread
  over quarters using SemiAnalysis's production-share estimates.
- Trainium2, H1 2026: Morgan Stanley (500k in H1 2026, production ending mid-2026) and
  JPMorgan (600k Trainium2 for all of 2026) supply-chain forecasts.
- Trainium3, H1 2026: a direct range on units, informed by Alchip's Q2 2026 revenue bump
  from its first 3nm accelerator shipments and JPMorgan's 1.5M Trainium3 for 2026.
- Trainium1: a rough 2024 total with no production afterwards.

Methodology write-up (working draft tab):
https://docs.google.com/document/d/16j2nTJls-jIQ2gRBAo7Wq-fZUwV0UgRZP8sDuBL0XZ0/edit?tab=t.yjy8uqrdkmkf

This module holds the model itself; trainium_estimates.ipynb wraps it for tables, charts,
and sanity checks, and run_chip_model.py uses it for the run-and-validate workflow. The
three csv_export outputs feed the published dataset at https://epoch.ai/data/ai-chip-sales
(see validate_chip_sales.py).
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import squigglepy as sq
from squigglepy.numbers import K, M

# Identity used by run_chip_model.py: FAMILY names the csv_export file prefix, and
# DESIGNER is the key into validate_chip_sales.DESIGNER_CONFIGS.
FAMILY = "trainium"
DESIGNER = "amazon"

N_SAMPLES = 10_000
RANDOM_SEED = 42

QUARTERS_2024 = ['Q1 2024', 'Q2 2024', 'Q3 2024', 'Q4 2024']
QUARTERS_2025 = ['Q1 2025', 'Q2 2025', 'Q3 2025', 'Q4 2025']
QUARTERS_2026 = ['Q1 2026', 'Q2 2026']
QUARTERS = QUARTERS_2024 + QUARTERS_2025 + QUARTERS_2026

CHIPS = ['Trainium1', 'Trainium2', 'Trainium3']

CHIP_SPECS_ZIP_URL = "https://epoch.ai/data/ai_chip_sales.zip"

# Fallback specs if the epoch.ai download fails (TOPS from 8-bit OP/s, TDP in watts)
FALLBACK_SPECS = {
    'Trainium1': {'tops': 420,  'tdp': 300},
    'Trainium2': {'tops': 1299, 'tdp': 500},
    'Trainium3': {'tops': 2517, 'tdp': 700},
}

H100_TOPS = 1979  # reference for H100-equivalent calculation


# ---------------------------------------------------------------------------
# Model parameters (all 90% credible intervals)
# ---------------------------------------------------------------------------
# run_model accepts a `params` dict that overrides any of these by key, which is how
# the notebook runs sensitivity checks.
DEFAULT_PARAMS = {
    # --- Trainium2, Q1 2024 - Q4 2025 ---
    # Amazon: "Trainium2 is fully subscribed with 1.4 million chips landed" (Q4 2025
    # earnings). "Landed" likely means operational, so allow ~10% either way for lags.
    'trainium2_stock_end_2025': sq.norm(1.26 * M, 1.54 * M),
    # SemiAnalysis: 2025 production was roughly 4x 2024's, so 2024 is ~20% of the total.
    'trainium2_2024_share_of_2024_2025': sq.norm(0.15, 0.30, lclip=0.01, rclip=0.99),
    # SemiAnalysis: H1 was 40-45% of 2025 production (moderate growth within the year).
    'trainium2_2025_h1_share': sq.norm(0.40, 0.45, lclip=0.01, rclip=0.99),
    # SemiAnalysis chart implies ~2.5x quarter-over-quarter growth through 2024.
    'trainium2_2024_quarterly_growth': sq.to(2.0, 3.0),

    # --- Trainium2, H1 2026 ---
    # Morgan Stanley forecast 500k Trainium2 in H1 2026, after which production ends
    # (1.9M lifetime). JPMorgan forecast 600k Trainium2 for all of 2026, consistent with
    # front-loaded shipments. Lognormal 400k-900k (median ~600k): wide, because both are
    # third-party forecasts from late 2025 / early 2026.
    'trainium2_2026_h1_units': sq.to(400 * K, 900 * K),
    # Both forecasts imply production was ramping down (H1 2026 is below a flat
    # extrapolation of Q4 2025's ~335k/quarter), so Q1 gets somewhat more than half.
    'trainium2_2026_q1_share_of_h1': sq.norm(0.50, 0.65, lclip=0.3, rclip=0.8),

    # --- Trainium3, H1 2026 ---
    # Shipments started at the start of 2026 but ramp into a big H2. Alchip's Q2 2026
    # revenue bump from Trainium3 ("over $100 million", from May) suggests roughly
    # 10k-100k of its units, and Amazon's own "Mariana" tapeout carries the majority of
    # volume. Lognormal 30k (conservative extrapolation from Alchip revenue) to 300k
    # (~20% of JPMorgan's 1.5M for 2026), median just under 100k. All of it is assigned
    # to Q2; Q1 shipments were positive but negligible and are rounded down to zero.
    'trainium3_2026_h1_units': sq.to(30 * K, 300 * K),

    # --- Trainium1 ---
    # Very rough: 200k-1M units over 2024, split evenly by quarter, none afterwards.
    'trainium1_2024_units': sq.to(200 * K, 1 * M),
}


# ---------------------------------------------------------------------------
# Loading inputs
# ---------------------------------------------------------------------------
def _load_trainium_specs(chip_types_df, fallback_specs):
    """Extract TOPS and TDP per Trainium generation from epoch.ai's chip table, with fallbacks."""
    specs = {}
    for chip in CHIPS:
        row = chip_types_df[chip_types_df['Name'] == chip]
        if len(row) != 1:
            specs[chip] = fallback_specs[chip].copy()
            print(f"Warning: using fallback specs for {chip} (not found in chip_types.csv)")
            continue
        tops_raw = row['8-bit OP/s'].values[0]
        tops = float(tops_raw) / 1e12 if pd.notna(tops_raw) else fallback_specs[chip]['tops']
        # Trainium1 has no official TDP; the table carries an unofficial estimate instead
        tdp_raw = row['TDP (W) (from ML Hardware (linked))'].values[0]
        if pd.isna(tdp_raw) and 'Unofficial estimated TDP (W)' in row.columns:
            tdp_raw = row['Unofficial estimated TDP (W)'].values[0]
        tdp = float(tdp_raw) if pd.notna(tdp_raw) else fallback_specs[chip]['tdp']
        specs[chip] = {'tops': tops, 'tdp': tdp}
    return specs


def load_inputs(verbose=True):
    """Fetch chip specs from epoch.ai (falling back to local values).

    The model's quantitative inputs are the hardcoded DEFAULT_PARAMS; specs are the only
    thing fetched. Load once and pass to repeated run_model calls to avoid re-fetching.
    """
    try:
        response = requests.get(CHIP_SPECS_ZIP_URL, timeout=10)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            with zf.open("chip_types.csv") as fh:
                chip_types_df = pd.read_csv(fh)
        specs = _load_trainium_specs(chip_types_df[chip_types_df['Designer'] == 'Amazon'], FALLBACK_SPECS)
        if verbose:
            print("Loaded Trainium specs from epoch.ai chip_types.csv")
    except Exception as exc:
        print(f"Warning: could not download chip specs from epoch.ai ({exc}); using fallback specs")
        specs = {chip: spec.copy() for chip, spec in FALLBACK_SPECS.items()}
    return {"specs": specs}


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------
def _exponential_quarterly_split(year_total, growth):
    """Split a yearly total across four quarters that grow by `growth` each quarter."""
    # First quarter's share of a 4-term geometric series; flat split when growth is 1
    flat = np.isclose(growth, 1.0)
    safe_growth = np.where(flat, 2.0, growth)
    first_quarter = np.where(flat, year_total / 4,
                             year_total * (safe_growth - 1) / (safe_growth ** 4 - 1))
    return [first_quarter * growth ** i for i in range(4)]


def _simulate_trainium2(p, n):
    """Trainium2 units per quarter, Q1 2024 - Q2 2026."""
    stock_end_2025 = p['trainium2_stock_end_2025'] @ n
    share_2024 = p['trainium2_2024_share_of_2024_2025'] @ n
    production_2024 = stock_end_2025 * share_2024
    production_2025 = stock_end_2025 - production_2024

    # 2024 growth comes from the SemiAnalysis prior directly
    growth_2024 = p['trainium2_2024_quarterly_growth'] @ n

    # 2025 growth is backed out of the H1 share: for four quarters growing at a steady
    # rate, H1's share is 1 / (1 + growth squared)
    h1_share_2025 = p['trainium2_2025_h1_share'] @ n
    growth_2025 = np.sqrt(1.0 / h1_share_2025 - 1.0)

    quarterly = {}
    for q, units in zip(QUARTERS_2024, _exponential_quarterly_split(production_2024, growth_2024)):
        quarterly[q] = units
    for q, units in zip(QUARTERS_2025, _exponential_quarterly_split(production_2025, growth_2025)):
        quarterly[q] = units

    # H1 2026 is a separate forecast, split toward Q1 as production winds down
    h1_2026 = p['trainium2_2026_h1_units'] @ n
    q1_share = p['trainium2_2026_q1_share_of_h1'] @ n
    quarterly['Q1 2026'] = h1_2026 * q1_share
    quarterly['Q2 2026'] = h1_2026 * (1 - q1_share)
    return quarterly


def _simulate_trainium3(p, n):
    """Trainium3 units per quarter: zero before Q2 2026, then all of H1 2026 in Q2."""
    quarterly = {q: np.zeros(n) for q in QUARTERS}
    quarterly['Q2 2026'] = p['trainium3_2026_h1_units'] @ n
    return quarterly


def _simulate_trainium1(p, n):
    """Trainium1 units per quarter: an even split of the 2024 total, zero afterwards."""
    per_quarter = (p['trainium1_2024_units'] @ n) / 4
    return {q: (per_quarter.copy() if q in QUARTERS_2024 else np.zeros(n)) for q in QUARTERS}


def run_model(n_samples=N_SAMPLES, params=None, inputs=None, seed=RANDOM_SEED):
    """Run the Monte Carlo simulation.

    `params` overrides entries of DEFAULT_PARAMS by key (e.g. for sensitivity checks).
    Chips are sampled independently of each other; within a chip, every quarter of a
    given sample comes from the same draw, so cumulative totals carry the full
    uncertainty rather than averaging out.

    Returns a dict with quarterly samples {quarter: {chip: array}}, cumulative samples
    in the same shape, specs, and the parameters used.
    """
    if inputs is None:
        inputs = load_inputs()
    p = {**DEFAULT_PARAMS, **(params or {})}

    sq.set_seed(seed)
    np.random.seed(seed)

    by_chip = {
        'Trainium1': _simulate_trainium1(p, n_samples),
        'Trainium2': _simulate_trainium2(p, n_samples),
        'Trainium3': _simulate_trainium3(p, n_samples),
    }

    quarterly_samples = {q: {chip: by_chip[chip][q] for chip in CHIPS} for q in QUARTERS}

    cumulative_samples = {}
    running = {chip: np.zeros(n_samples) for chip in CHIPS}
    for q in QUARTERS:
        for chip in CHIPS:
            running[chip] = running[chip] + quarterly_samples[q][chip]
        cumulative_samples[q] = {chip: running[chip].copy() for chip in CHIPS}

    print("Simulation complete.")
    return {
        "inputs": inputs,
        "params": p,
        "n_samples": n_samples,
        "quarters": QUARTERS,
        "chips": CHIPS,
        "specs": inputs["specs"],
        "quarterly_samples": quarterly_samples,
        "cumulative_samples": cumulative_samples,
    }


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------
def get_quarter_dates(quarter):
    """Return (start_date, end_date) M/D/YYYY strings for a label like 'Q1 2024'."""
    q_num = int(quarter[1])
    year = int(quarter.split()[1])
    return {
        1: (f"1/1/{year}", f"3/31/{year}"),
        2: (f"4/1/{year}", f"6/30/{year}"),
        3: (f"7/1/{year}", f"9/30/{year}"),
        4: (f"10/1/{year}", f"12/31/{year}"),
    }[q_num]


def round_to_5k(x):
    """Round a unit count to the nearest 5,000 (per-chip exports avoid false precision)."""
    return int(round(x / 5000) * 5000)


# Every cumulative window starts at the first modeled quarter
CUMULATIVE_START_LABEL = QUARTERS[0]
CUMULATIVE_START_DATE = get_quarter_dates(QUARTERS[0])[0]


def export_canonical(results):
    """Write the canonical sales and owners tables (docs/schema.md) to canonical_export/trainium/.

    The schema library takes the per-quarter unit samples, one array per chip, and works out
    H100e, power and cost itself from the shared chip-spec table. Amazon owns every Trainium
    it makes, so the owners tables reuse the sales samples. The quarters are calendar quarters
    modelled directly, so none is flagged incomplete. Chip-quarters with no production, such as
    Trainium1 after 2024 or Trainium3 before Q2 2026, are dropped by the library.
    """
    from pipeline import chip_schema as cs

    quarterly = results["quarterly_samples"]
    cumulative = results["cumulative_samples"]
    tables = cs.sales_tables("Amazon", quarterly=quarterly, cumulative=cumulative)
    tables.update(cs.owners_tables("Amazon", {"Amazon": quarterly}, {"Amazon": cumulative}))
    return cs.write_tables("trainium", tables)


def export_csvs(results):
    """Write all Trainium estimate CSVs (csv_export/ and owners_csv_export/); return paths."""
    specs = results["specs"]
    n_samples = results["n_samples"]
    quarterly_samples = results["quarterly_samples"]
    cumulative_samples = results["cumulative_samples"]

    generated_note = f"Estimates generated on: {datetime.now().strftime('%m-%d-%Y %H:%M')}"

    written = []

    def write(rows, path):
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        print(f"Exported {len(df)} rows to {path}")
        written.append(path)

    def h100e_ratio(chip):
        return specs[chip]['tops'] / H100_TOPS

    def nonzero_chip_quarters(samples):
        """(quarter, chip, samples) for every chip-quarter with production."""
        for q in QUARTERS:
            for chip in CHIPS:
                arr = samples[q][chip]
                if np.median(arr) > 0:
                    yield q, chip, arr

    def all_chip_totals(q):
        """Cumulative units, H100e, and power (W) summed over chips, per sample."""
        units = np.zeros(n_samples)
        h100e = np.zeros(n_samples)
        power_w = np.zeros(n_samples)
        for chip in CHIPS:
            arr = cumulative_samples[q][chip]
            units += arr
            h100e += arr * h100e_ratio(chip)
            power_w += arr * specs[chip]['tdp']
        return units, h100e, power_w

    def owner_unit_columns(units, h100e, power_w):
        """The unit/H100e/TDP/power percentile columns shared by the owners exports."""
        power_mw = power_w / 1e6
        return {
            'Compute estimate in H100e (median)': int(np.percentile(h100e, 50)),
            'H100e (5th percentile)': int(np.percentile(h100e, 5)),
            'H100e (95th percentile)': int(np.percentile(h100e, 95)),
            'Number of Units': int(np.percentile(units, 50)),
            'Number of Units (5th percentile)': int(np.percentile(units, 5)),
            'Number of Units (95th percentile)': int(np.percentile(units, 95)),
            'Total TDP (W)': int(np.percentile(power_w, 50)),
            'Total TDP (W) (5th percentile)': int(np.percentile(power_w, 5)),
            'Total TDP (W) (95th percentile)': int(np.percentile(power_w, 95)),
            'Power in MW (median)': round(np.percentile(power_mw, 50), 1),
            'Power in MW (5th percentile)': round(np.percentile(power_mw, 5), 1),
            'Power in MW (95th percentile)': round(np.percentile(power_mw, 95), 1),
        }

    # --- Quarterly timelines by chip (csv_export) ---
    rows = []
    for q, chip, arr in nonzero_chip_quarters(quarterly_samples):
        start_date, end_date = get_quarter_dates(q)
        units = {pct: round_to_5k(np.percentile(arr, pct)) for pct in (5, 50, 95)}
        power_mw = arr * specs[chip]['tdp'] / 1e6
        rows.append({
            'Name': f"{q} - {chip}",
            'Chip manufacturer': 'Amazon',
            'Start date': start_date,
            'End date': end_date,
            'Compute estimate in H100e (median)': int(units[50] * h100e_ratio(chip)),
            'H100e (5th percentile)': int(units[5] * h100e_ratio(chip)),
            'H100e (95th percentile)': int(units[95] * h100e_ratio(chip)),
            'Number of Units': units[50],
            'Number of Units (5th percentile)': units[5],
            'Number of Units (95th percentile)': units[95],
            'Power in MW (median)': round(np.percentile(power_mw, 50), 1),
            'Power in MW (5th percentile)': round(np.percentile(power_mw, 5), 1),
            'Power in MW (95th percentile)': round(np.percentile(power_mw, 95), 1),
            'Source / Link': '',
            'Notes': generated_note,
            'Chip type': chip,
        })
    write(rows, 'csv_export/trainium_chip_timelines.csv')

    # --- Cumulative by chip (csv_export) ---
    rows = []
    for q, chip, arr in nonzero_chip_quarters(cumulative_samples):
        _, end_date = get_quarter_dates(q)
        units = {pct: round_to_5k(np.percentile(arr, pct)) for pct in (5, 50, 95)}
        rows.append({
            'Name': f"{chip} {CUMULATIVE_START_LABEL} to {q}",
            'Chip manufacturer': 'Amazon',
            'Start date': CUMULATIVE_START_DATE,
            'End date': end_date,
            'Chip type': chip,
            'Number of units (5th percentile)': units[5],
            'Number of units (median)': units[50],
            'Number of units (95th percentile)': units[95],
            'Compute estimate in H100e (5th percentile)': int(units[5] * h100e_ratio(chip)),
            'Compute estimate in H100e (median)': int(units[50] * h100e_ratio(chip)),
            'Compute estimate in H100e (95th percentile)': int(units[95] * h100e_ratio(chip)),
            'Incomplete': '',
            'Notes': generated_note,
        })
    write(rows, 'csv_export/trainium_cumulative_by_chip.csv')

    # --- Cumulative totals across all chips (csv_export) ---
    rows = []
    for q in QUARTERS:
        _, end_date = get_quarter_dates(q)
        units, h100e, power_w = all_chip_totals(q)
        rows.append({
            'Name': f"Amazon cumulative Trainium through {q}",
            'Chip manufacturer': 'Amazon',
            'Start date': CUMULATIVE_START_DATE,
            'End date': end_date,
            'Number of units (5th percentile)': int(np.percentile(units, 5)),
            'Number of units (median)': int(np.percentile(units, 50)),
            'Number of units (95th percentile)': int(np.percentile(units, 95)),
            'Compute estimate in H100e (5th percentile)': int(np.percentile(h100e, 5)),
            'Compute estimate in H100e (median)': int(np.percentile(h100e, 50)),
            'Compute estimate in H100e (95th percentile)': int(np.percentile(h100e, 95)),
            'Power in MW (5th percentile)': round(np.percentile(power_w / 1e6, 5), 1),
            'Power in MW (median)': round(np.percentile(power_w / 1e6, 50), 1),
            'Power in MW (95th percentile)': round(np.percentile(power_w / 1e6, 95), 1),
            'Incomplete': '',
            'Notes': generated_note,
        })
    write(rows, 'csv_export/trainium_cumulative_totals.csv')

    # --- Ownership format (owners_csv_export) ---
    # Amazon owns all Trainium chips (analogous to Google owning all TPUs).
    rows = []
    for q, chip, arr in nonzero_chip_quarters(cumulative_samples):
        _, end_date = get_quarter_dates(q)
        rows.append({
            'Name': f"Amazon {chip} cumulative through {q}",
            'Chip manufacturer': 'Amazon',
            'Owner': 'Amazon',
            'Start date': CUMULATIVE_START_DATE,
            'End date': end_date,
            **owner_unit_columns(arr, arr * h100e_ratio(chip), arr * specs[chip]['tdp']),
            'Chip type': chip,
            'Source / Link': '',
            'Notes': generated_note,
        })
    write(rows, 'owners_csv_export/trainium_owners_cumulative_by_chip.csv')

    rows = []
    for q, chip, arr in nonzero_chip_quarters(quarterly_samples):
        start_date, end_date = get_quarter_dates(q)
        rows.append({
            'Name': f"Amazon {chip} {q}",
            'Chip manufacturer': 'Amazon',
            'Owner': 'Amazon',
            'Start date': start_date,
            'End date': end_date,
            **owner_unit_columns(arr, arr * h100e_ratio(chip), arr * specs[chip]['tdp']),
            'Chip type': chip,
            'Source / Link': '',
            'Notes': generated_note,
        })
    write(rows, 'owners_csv_export/trainium_owners_quarters_by_chip.csv')

    rows = []
    for q in QUARTERS:
        _, end_date = get_quarter_dates(q)
        rows.append({
            'Name': f"Amazon cumulative Trainium through {q}",
            'Chip manufacturer': 'Amazon',
            'Owner': 'Amazon',
            'Start date': CUMULATIVE_START_DATE,
            'End date': end_date,
            **owner_unit_columns(*all_chip_totals(q)),
            'Source / Link': '',
            'Notes': generated_note,
        })
    write(rows, 'owners_csv_export/trainium_owners_cumulative_totals.csv')

    return written
