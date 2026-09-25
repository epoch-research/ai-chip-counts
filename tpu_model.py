"""Google TPU chip-sales model.

Estimates quarterly and cumulative TPU production by dividing Broadcom TPU revenue
across TPU versions (production mix) and dividing by per-chip price (manufacturing
cost marked up by Broadcom's margin). Monte Carlo throughout, with chip prices
presampled once per version and partially correlated across versions so that price
uncertainty propagates into cumulative totals instead of averaging out.

This module holds the model itself; tpu_estimates.ipynb wraps it for visualization
and sensitivity analysis, and run_chip_model.py uses it for the run-and-validate
workflow. The three csv_export outputs feed the published dataset at
https://epoch.ai/data/ai-chip-sales (see validate_chip_sales.py).

Data inputs come from a Google Sheet (quarterly TPU revenue on Broadcom's fiscal
calendar, production mix, costs):
https://docs.google.com/spreadsheets/d/1eGk2AAdewEO81vx-YBRTtdlhZvAMstY7vZuHrf3sgNI/edit
"""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import squigglepy as sq
from squigglepy.numbers import B

from chip_estimates_utils import (
    normalize_shares,
    estimate_cumulative_chip_sales,
    aggregate_by_chip_type,
    interpolate_samples_to_calendar_quarters,
    compute_running_totals,
    interpolate_to_calendar_quarters,
    make_incomplete_note_fn,
)

# Identity used by run_chip_model.py: FAMILY names the csv_export file prefix, and
# DESIGNER is the key into validate_chip_sales.DESIGNER_CONFIGS.
FAMILY = "tpu"
DESIGNER = "google"

N_SAMPLES = 5000
RANDOM_SEED = 42
PRICE_CORRELATION = 0.5  # correlation between chip prices across TPU versions

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------
SPREADSHEET_ID = "1eGk2AAdewEO81vx-YBRTtdlhZvAMstY7vZuHrf3sgNI"
REVENUE_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet=TPU_Revenue"
PROD_MIX_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet=Production_Mix"
PRICES_URL = f"https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/gviz/tq?tqx=out:csv&sheet=prices"

CHIP_SPECS_ZIP_URL = "https://epoch.ai/data/ai_chip_sales.zip"

# Map this model's TPU version names to the names in epoch.ai's chip_types.csv
TPU_NAME_MAP = {
    'v3': 'TPU v3',
    'v4i': 'TPU v4i',
    'v4': 'TPU v4',
    'v5e': 'TPU v5e',
    'v5p': 'TPU v5p',
    'v6e': 'TPU v6e',
    'v7': 'TPU v7',
}

# Fallback specs if the epoch.ai download fails (TOPS from 8-bit OP/s, TDP in watts)
FALLBACK_SPECS = {
    'v3':  {'tops': 123,  'tdp': 450},
    'v4i': {'tops': 138,  'tdp': 175},
    'v4':  {'tops': 275,  'tdp': 340},
    'v5e': {'tops': 393,  'tdp': 225},
    'v5p': {'tops': 918,  'tdp': 540},
    'v6e': {'tops': 1836, 'tdp': 380},
    'v7':  {'tops': 4614, 'tdp': 960},
}

H100_TOPS = 1979  # reference for H100-equivalent calculation


# ---------------------------------------------------------------------------
# Loading inputs
# ---------------------------------------------------------------------------
def _load_tpu_specs(chip_types_df, name_map, fallback_specs):
    """Extract TOPS and TDP per TPU version from epoch.ai's chip table, with fallbacks."""
    specs = {}
    for version, csv_name in name_map.items():
        row = chip_types_df[chip_types_df['Name'] == csv_name]
        if len(row) == 1:
            tops_raw = row['8-bit OP/s'].values[0]
            tops = float(tops_raw) / 1e12 if pd.notna(tops_raw) else fallback_specs[version]['tops']
            tdp_col = 'TDP (W) (from ML Hardware (linked))'
            tdp_raw = row[tdp_col].values[0] if tdp_col in row.columns else None
            tdp = float(tdp_raw) if pd.notna(tdp_raw) else fallback_specs[version]['tdp']
            specs[version] = {'tops': tops, 'tdp': tdp}
        else:
            specs[version] = fallback_specs[version].copy()
            print(f"Warning: using fallback specs for {version} (not found in chip_types.csv)")
    return specs


def load_inputs(verbose=True):
    """Fetch revenue/mix/price data from Google Sheets and chip specs from epoch.ai.

    Returns a dict of dataframes, squigglepy distributions, and specs that run_model
    consumes. Load once and pass to repeated run_model calls (e.g. for sensitivity
    analysis) to avoid re-fetching.
    """
    revenue_df = pd.read_csv(REVENUE_URL).dropna(axis=1, how="all")
    prod_mix_df = pd.read_csv(PROD_MIX_URL).dropna(axis=1, how="all")
    prices_df = pd.read_csv(PRICES_URL).dropna(axis=1, how="all")

    quarters = revenue_df['quarter'].tolist()
    versions = prod_mix_df['version'].dropna().unique().tolist()

    # Quarterly TPU revenue (billions), 90% CI per quarter
    tpu_revenue = {row['quarter']: sq.norm(row['revenue_p5'], row['revenue_p95'])
                   for _, row in revenue_df.iterrows()}

    # Production mix: share of revenue per TPU version per quarter
    prod_mix = {}
    for quarter in prod_mix_df['quarter'].unique():
        quarter_data = prod_mix_df[prod_mix_df['quarter'] == quarter]
        prod_mix[quarter] = {row['version']: sq.norm(row['share_p5'], row['share_p95'], lclip=0, rclip=1)
                             for _, row in quarter_data.iterrows()}

    # Manufacturing cost per version, and Broadcom margin per quarter
    tpu_cost = {row['version']: sq.to(row['cost_p5'], row['cost_p95']) for _, row in prices_df.iterrows()}
    margin_by_quarter = {row['quarter']: sq.to(row['broadcom_margin_p5'], row['broadcom_margin_p95'])
                         for _, row in revenue_df.iterrows()}

    # Fiscal quarter date ranges (raw M/D/YYYY strings from the sheet)
    quarter_dates = {q: (revenue_df.loc[revenue_df['quarter'] == q, 'start_date'].iloc[0],
                         revenue_df.loc[revenue_df['quarter'] == q, 'end_date'].iloc[0])
                     for q in quarters}

    # Chip specs (TOPS/TDP) from the published chip_types table, with local fallbacks
    try:
        response = requests.get(CHIP_SPECS_ZIP_URL, timeout=10)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            with zf.open("chip_types.csv") as fh:
                chip_types_df = pd.read_csv(fh)
        specs = _load_tpu_specs(chip_types_df[chip_types_df['Designer'] == 'Google'],
                                TPU_NAME_MAP, FALLBACK_SPECS)
        if verbose:
            print("Loaded TPU specs from epoch.ai chip_types.csv")
    except Exception as exc:
        print(f"Warning: could not download chip specs from epoch.ai ({exc}); using fallback specs")
        specs = {version: spec.copy() for version, spec in FALLBACK_SPECS.items()}

    if verbose:
        print(f"Loaded {len(revenue_df)} quarters of revenue data, {len(versions)} TPU versions")

    return {
        "revenue_df": revenue_df,
        "prod_mix_df": prod_mix_df,
        "prices_df": prices_df,
        "quarters": quarters,
        "versions": versions,
        "tpu_revenue": tpu_revenue,
        "prod_mix": prod_mix,
        "tpu_cost": tpu_cost,
        "margin_by_quarter": margin_by_quarter,
        "quarter_dates": quarter_dates,
        "specs": specs,
    }


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------
def _compute_expected_price_multiplier(margin_dist, n=10000):
    """Average price markup implied by a margin distribution, found by sampling.

    Price = cost / (1 - margin), and 1/(1-x) is convex, so the mean markup is not
    the markup of the mean margin - hence empirical sampling rather than closed form.
    """
    samples = margin_dist @ n
    return np.mean(1 / (1 - samples))


def run_model(n_samples=N_SAMPLES, price_correlation=PRICE_CORRELATION, inputs=None, seed=RANDOM_SEED):
    """Run the Monte Carlo simulation and all downstream transforms.

    Chip prices are presampled once per version (using the first quarter's margin)
    and reused across quarters, so price uncertainty stays correlated over time and
    accumulates into the cumulative totals. Re-running with a different
    price_correlation reuses the same seed, so runs differ only by the parameter.

    Returns a dict with the inputs and every samples structure the exports,
    visualizations, and sensitivity checks need.
    """
    if inputs is None:
        inputs = load_inputs()

    sq.set_seed(seed)
    np.random.seed(seed)
    np.seterr(invalid='raise')

    quarters = inputs["quarters"]
    versions = inputs["versions"]
    tpu_revenue = inputs["tpu_revenue"]
    prod_mix = inputs["prod_mix"]
    tpu_cost = inputs["tpu_cost"]
    margin_by_quarter = inputs["margin_by_quarter"]

    # Prices are anchored to the first quarter's margin; every other quarter gets a
    # deflation factor reflecting how much that quarter's expected price markup
    # differs from the base quarter's, using the sheet's per-quarter margins as the
    # source of truth. Computed empirically (see _compute_expected_price_multiplier)
    # and cached per distinct margin range so quarters with identical margin
    # assumptions get identical factors.
    base_quarter = quarters[0]
    base_prices = {version: tpu_cost[version] / (1 - margin_by_quarter[base_quarter])
                   for version in versions}

    multiplier_by_margin_range = {}
    for quarter in quarters:
        margin = margin_by_quarter[quarter]
        if (margin.x, margin.y) not in multiplier_by_margin_range:
            multiplier_by_margin_range[(margin.x, margin.y)] = _compute_expected_price_multiplier(margin)
    base_margin = margin_by_quarter[base_quarter]
    deflation_by_quarter = {
        quarter: (multiplier_by_margin_range[(margin_by_quarter[quarter].x, margin_by_quarter[quarter].y)]
                  / multiplier_by_margin_range[(base_margin.x, base_margin.y)])
        for quarter in quarters
    }

    def sample_revenue(quarter):
        return (tpu_revenue[quarter] @ 1) * B

    def sample_shares(quarter):
        raw_shares = {version: dist @ 1 for version, dist in prod_mix[quarter].items()}
        return normalize_shares(raw_shares)

    def sample_base_price(version):
        return (tpu_cost[version] / (1 - margin_by_quarter[base_quarter])) @ 1

    def get_margin_deflation_factor(quarter, version):
        return deflation_by_quarter[quarter]

    quarterly_samples = estimate_cumulative_chip_sales(
        quarters=quarters,
        chip_types=versions,
        sample_revenue=sample_revenue,
        sample_shares=sample_shares,
        sample_base_price=sample_base_price,
        get_deflation_factor=get_margin_deflation_factor,
        sample_revenue_uncertainty=None,
        price_correlation=price_correlation,
        base_price_distributions=base_prices,
        n_samples=n_samples,
    )

    # Downstream transforms: cumulative by version, running totals on the fiscal
    # calendar, sample-based interpolation onto calendar quarters, and a separate
    # summary-stat interpolation used by the calendar timelines export.
    cumulative_samples = aggregate_by_chip_type(quarterly_samples)
    running_totals_samples = compute_running_totals(quarterly_samples)
    calendar_quarterly_samples = interpolate_samples_to_calendar_quarters(quarterly_samples, inputs["quarter_dates"])
    calendar_running_totals_samples = compute_running_totals(calendar_quarterly_samples)
    calendar_results = interpolate_to_calendar_quarters(quarterly_samples, inputs["quarter_dates"], verbose=False)

    print("Simulation complete.")
    return {
        "inputs": inputs,
        "n_samples": n_samples,
        "price_correlation": price_correlation,
        "deflation_by_quarter": deflation_by_quarter,
        "quarters": quarters,
        "versions": versions,
        "specs": inputs["specs"],
        "quarterly_samples": quarterly_samples,
        "cumulative_samples": cumulative_samples,
        "running_totals_samples": running_totals_samples,
        "calendar_quarterly_samples": calendar_quarterly_samples,
        "calendar_running_totals_samples": calendar_running_totals_samples,
        "calendar_results": calendar_results,
    }


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------
def get_tpu_name(version):
    """Normalize a TPU version name to 'TPU vX' format."""
    version = version.strip()
    if version.startswith("TPU "):
        return version
    if version.startswith("TPU"):
        return f"TPU {version[3:]}"
    return f"TPU {version}"


def get_calendar_quarter_dates(cal_q):
    """Return (start_date, end_date) M/D/YYYY strings for a label like 'Q1 2024'."""
    parts = cal_q.split()
    q_num = int(parts[0][1])
    year = int(parts[1])
    if q_num == 1:
        return f"1/1/{year}", f"3/31/{year}"
    elif q_num == 2:
        return f"4/1/{year}", f"6/30/{year}"
    elif q_num == 3:
        return f"7/1/{year}", f"9/30/{year}"
    else:
        return f"10/1/{year}", f"12/31/{year}"


# By-chip cumulative exports start at 1/1/2023 because pre-2023 quarters have very
# incomplete coverage of TPU versions. This matches the website's cumulative-by-chip
# table.
CUMULATIVE_START_DATE = "1/1/2023"
CUMULATIVE_START_LABEL = "Q1 2023"
CUMULATIVE_CUTOFF = datetime(2023, 1, 1)

# The designer-totals export (tpu_cumulative_totals.csv) follows the website's
# designer-totals convention instead: the cumulative window starts at Q4 2022, the
# first calendar quarter touched by Broadcom fiscal data.
TOTALS_START_DATE = "10/1/2022"
TOTALS_START_LABEL = "Q4 2022"
TOTALS_CUTOFF = datetime(2022, 12, 31)


def tpu_prices(results):
    """The model's own price per TPU, in USD, for the canonical cost columns.

    The model prices each version at its manufacturing cost marked up by Broadcom's
    margin, adjusted each quarter as the margin changes. This averages that price over
    the quarters the version sold in, weighted by median units, so a chip sold mostly in
    lower-margin quarters gets the lower price. The chip_types table's list prices are
    not used for TPUs: TPU v4i has none, and the rest run 6-15% above the model's.
    """
    inputs = results["inputs"]
    base_quarter = results["quarters"][0]
    # Seeded so the prices, and so the cost columns, are identical run to run.
    sq.set_seed(RANDOM_SEED)
    prices = {}
    for version in results["versions"]:
        base = inputs["tpu_cost"][version] / (1 - inputs["margin_by_quarter"][base_quarter])
        base_median = float(np.median(base @ 20000))
        units = {q: float(np.median(results["quarterly_samples"][q][version])) for q in results["quarters"]}
        total = sum(units.values())
        weighted = (sum(base_median * results["deflation_by_quarter"][q] * u for q, u in units.items()) / total
                    if total else base_median)
        prices[get_tpu_name(version)] = round(weighted)
    return prices


def export_canonical(results):
    """Write the canonical sales and owners tables (docs/schema.md) to canonical_export/tpu/.

    Built from the sample-based calendar interpolation, so the quarterly numbers can
    differ slightly from tpu_calendar_quarter_chip_timelines.csv, which interpolates
    summary statistics instead. Google owns every TPU it ships.
    """
    from pipeline import chip_schema as cs

    revenue_df = results["inputs"]["revenue_df"]
    quarterly = {cq: {get_tpu_name(v): s for v, s in chips.items()}
                 for cq, chips in results["calendar_quarterly_samples"].items()}
    running = {cq: {get_tpu_name(v): s for v, s in chips.items()}
               for cq, chips in results["calendar_running_totals_samples"].items()}
    inc_q, inc_c = cs.incomplete_quarters(
        quarterly, pd.to_datetime(revenue_df['start_date'].iloc[0], format='%m/%d/%Y'),
        pd.to_datetime(revenue_df['end_date'].iloc[-1], format='%m/%d/%Y'))
    note = "Based on Broadcom fiscal quarters."
    prices = tpu_prices(results)
    tables = cs.sales_tables("Google", quarterly=quarterly, cumulative=running,
                             incomplete=inc_q, incomplete_cumulative=inc_c, note=note, prices=prices)
    tables.update(cs.owners_tables("Google", {"Google": quarterly}, {"Google": running},
                                   incomplete=inc_q, incomplete_cumulative=inc_c, note=note, prices=prices))
    return cs.write_tables("tpu", tables)


def export_csvs(results):
    """Write all TPU estimate CSVs (csv_export/ and owners_csv_export/); return paths."""
    inputs = results["inputs"]
    revenue_df = inputs["revenue_df"]
    quarters = results["quarters"]
    versions = results["versions"]
    specs = results["specs"]
    n_samples = results["n_samples"]

    timestamp = datetime.now().strftime("%m-%d-%Y %H:%M")
    generated_note = f"Estimates generated on: {timestamp}"

    # Two incomplete-data note helpers: the calendar timelines export quotes the
    # sheet's raw fiscal date strings; the cumulative exports use the shared helper.
    broadcom_first_start = revenue_df['start_date'].iloc[0]
    broadcom_last_end = revenue_df['end_date'].iloc[-1]
    broadcom_first_start_dt = pd.to_datetime(broadcom_first_start, format='%m/%d/%Y')
    broadcom_last_end_dt = pd.to_datetime(broadcom_last_end, format='%m/%d/%Y')

    def get_incomplete_note_raw(cal_q_start, cal_q_end):
        cal_start_dt = pd.to_datetime(cal_q_start, format='%m/%d/%Y')
        cal_end_dt = pd.to_datetime(cal_q_end, format='%m/%d/%Y')
        starts_before = cal_start_dt < broadcom_first_start_dt
        ends_after = cal_end_dt > broadcom_last_end_dt
        if starts_before and ends_after:
            return f"Incomplete: based on Broadcom fiscal quarters {broadcom_first_start} to {broadcom_last_end}"
        elif starts_before:
            return f"Incomplete: based on Broadcom fiscal quarters beginning {broadcom_first_start}"
        elif ends_after:
            return f"Incomplete: based on Broadcom fiscal quarters ending {broadcom_last_end}"
        return None

    get_incomplete_note = make_incomplete_note_fn(broadcom_first_start, broadcom_last_end,
                                                  source_label='Broadcom')

    def with_incomplete(incomplete_note):
        return f"{incomplete_note}. {generated_note}" if incomplete_note else generated_note

    written = []

    def write(df, path):
        df.to_csv(path, index=False)
        print(f"Exported {len(df)} rows to {path}")
        written.append(path)

    # --- Fiscal quarter timelines (csv_export) ---
    quarter_dates_lookup = {}
    for _, row in revenue_df.iterrows():
        start = row.get('start_date', '')
        end = row.get('end_date', '')
        if pd.isna(start):
            start = ''
        if pd.isna(end):
            end = ''
        quarter_dates_lookup[row['quarter']] = (start, end)

    rows = []
    for quarter in quarters:
        start_date, end_date = quarter_dates_lookup.get(quarter, ('', ''))
        for version in versions:
            arr = results["quarterly_samples"][quarter][version]
            if arr.sum() > 0:
                h100e_arr = arr * (specs[version]['tops'] / H100_TOPS)
                rows.append({
                    'Name': f"{quarter} - {get_tpu_name(version)}",
                    'Chip manufacturer': 'Google',
                    'Start date': start_date,
                    'End date': end_date,
                    'Compute estimate in H100e (median)': int(np.percentile(h100e_arr, 50)),
                    'H100e (5th percentile)': int(np.percentile(h100e_arr, 5)),
                    'H100e (95th percentile)': int(np.percentile(h100e_arr, 95)),
                    'Number of Units': int(np.percentile(arr, 50)),
                    'Number of Units (5th percentile)': int(np.percentile(arr, 5)),
                    'Number of Units (95th percentile)': int(np.percentile(arr, 95)),
                    'Chip type': get_tpu_name(version),
                    'Notes': generated_note,
                    'Source / Link': '',
                })
    write(pd.DataFrame(rows), 'csv_export/tpu_fiscal_quarter_chip_timelines.csv')

    # --- Calendar quarter timelines, from summary-stat interpolation (csv_export) ---
    rows = []
    for quarter in results["calendar_results"]:
        start_date, end_date = get_calendar_quarter_dates(quarter)
        incomplete_note = get_incomplete_note_raw(start_date, end_date)
        for version in versions:
            stats = results["calendar_results"][quarter][version]
            if stats['p50'] > 0:
                h100e_factor = specs[version]['tops'] / H100_TOPS
                rows.append({
                    'Name': f"{quarter} - {get_tpu_name(version)}",
                    'Chip manufacturer': 'Google',
                    'Start date': start_date,
                    'End date': end_date,
                    'Compute estimate in H100e (median)': int(stats['p50'] * h100e_factor),
                    'H100e (5th percentile)': int(stats['p5'] * h100e_factor),
                    'H100e (95th percentile)': int(stats['p95'] * h100e_factor),
                    'Number of Units': int(stats['p50']),
                    'Number of Units (5th percentile)': int(stats['p5']),
                    'Number of Units (95th percentile)': int(stats['p95']),
                    'Chip type': get_tpu_name(version),
                    'Incomplete': 'checked' if incomplete_note else '',
                    'Notes': with_incomplete(incomplete_note),
                    'Source / Link': '',
                })
    write(pd.DataFrame(rows), 'csv_export/tpu_calendar_quarter_chip_timelines.csv')

    # --- Cumulative running totals by version (csv_export) ---
    calendar_running = results["calendar_running_totals_samples"]
    rows = []
    for cq in calendar_running:
        _, end_date = get_calendar_quarter_dates(cq)
        if datetime.strptime(end_date, '%m/%d/%Y') < CUMULATIVE_CUTOFF:
            continue
        incomplete_note = get_incomplete_note(CUMULATIVE_START_DATE, end_date)
        for version in versions:
            arr = calendar_running[cq][version]
            if arr.sum() > 0:
                display_name = f"TPU {version}"
                h100e_arr = arr * (specs[version]['tops'] / H100_TOPS)
                rows.append({
                    'Name': f"{display_name} {CUMULATIVE_START_LABEL} to {cq}",
                    'Chip manufacturer': 'Google',
                    'Start date': CUMULATIVE_START_DATE,
                    'End date': end_date,
                    'Chip type': display_name,
                    'Number of units (5th percentile)': int(np.percentile(arr, 5)),
                    'Number of units (median)': int(np.percentile(arr, 50)),
                    'Number of units (95th percentile)': int(np.percentile(arr, 95)),
                    'Compute estimate in H100e (5th percentile)': int(np.percentile(h100e_arr, 5)),
                    'Compute estimate in H100e (median)': int(np.percentile(h100e_arr, 50)),
                    'Compute estimate in H100e (95th percentile)': int(np.percentile(h100e_arr, 95)),
                    'Incomplete': 'checked' if incomplete_note else '',
                    'Notes': with_incomplete(incomplete_note),
                })
    write(pd.DataFrame(rows), 'csv_export/tpu_cumulative_by_chip.csv')

    # --- Ownership format: cumulative by version (owners_csv_export) ---
    rows = []
    for cq in calendar_running:
        _, end_date = get_calendar_quarter_dates(cq)
        if datetime.strptime(end_date, '%m/%d/%Y') < CUMULATIVE_CUTOFF:
            continue
        incomplete_note = get_incomplete_note(CUMULATIVE_START_DATE, end_date)
        for version in versions:
            arr = calendar_running[cq][version]
            if arr.sum() > 0:
                display_name = f"TPU {version}"
                h100e_samples = arr * (specs[version]['tops'] / H100_TOPS)
                tdp_w = arr * specs[version]['tdp']
                rows.append({
                    'Name': f"Google {display_name} cumulative through {cq}",
                    'Chip manufacturer': 'Google',
                    'Owner': 'Google',
                    'Start date': CUMULATIVE_START_DATE,
                    'End date': end_date,
                    'Compute estimate in H100e (median)': int(np.percentile(h100e_samples, 50)),
                    'H100e (5th percentile)': int(np.percentile(h100e_samples, 5)),
                    'H100e (95th percentile)': int(np.percentile(h100e_samples, 95)),
                    'Number of Units': int(np.percentile(arr, 50)),
                    'Number of Units (5th percentile)': int(np.percentile(arr, 5)),
                    'Number of Units (95th percentile)': int(np.percentile(arr, 95)),
                    'Total TDP (W)': int(np.percentile(tdp_w, 50)),
                    'Total TDP (W) (5th percentile)': int(np.percentile(tdp_w, 5)),
                    'Total TDP (W) (95th percentile)': int(np.percentile(tdp_w, 95)),
                    'Chip type': display_name,
                    'Incomplete': 'checked' if incomplete_note else '',
                    'Source / Link': '',
                    'Notes': with_incomplete(incomplete_note),
                })
    write(pd.DataFrame(rows), 'owners_csv_export/tpu_owners_cumulative_by_chip.csv')

    # --- Ownership format: per-quarter additions by version (owners_csv_export) ---
    rows = []
    for cq in results["calendar_quarterly_samples"]:
        start_date, end_date = get_calendar_quarter_dates(cq)
        incomplete_note = get_incomplete_note(start_date, end_date)
        for version in versions:
            arr = results["calendar_quarterly_samples"][cq][version]
            if arr.sum() > 0:
                display_name = f"TPU {version}"
                h100e_samples = arr * (specs[version]['tops'] / H100_TOPS)
                tdp_w = arr * specs[version]['tdp']
                rows.append({
                    'Name': f"Google {display_name} {cq}",
                    'Chip manufacturer': 'Google',
                    'Owner': 'Google',
                    'Start date': start_date,
                    'End date': end_date,
                    'Compute estimate in H100e (median)': int(np.percentile(h100e_samples, 50)),
                    'H100e (5th percentile)': int(np.percentile(h100e_samples, 5)),
                    'H100e (95th percentile)': int(np.percentile(h100e_samples, 95)),
                    'Number of Units': int(np.percentile(arr, 50)),
                    'Number of Units (5th percentile)': int(np.percentile(arr, 5)),
                    'Number of Units (95th percentile)': int(np.percentile(arr, 95)),
                    'Total TDP (W)': int(np.percentile(tdp_w, 50)),
                    'Total TDP (W) (5th percentile)': int(np.percentile(tdp_w, 5)),
                    'Total TDP (W) (95th percentile)': int(np.percentile(tdp_w, 95)),
                    'Chip type': display_name,
                    'Incomplete': 'checked' if incomplete_note else '',
                    'Source / Link': '',
                    'Notes': with_incomplete(incomplete_note),
                })
    write(pd.DataFrame(rows), 'owners_csv_export/tpu_owners_quarters_by_chip.csv')

    # --- Ownership format: cumulative totals across all versions (owners_csv_export) ---
    rows = []
    for cq in calendar_running:
        _, end_date = get_calendar_quarter_dates(cq)
        if datetime.strptime(end_date, '%m/%d/%Y') < CUMULATIVE_CUTOFF:
            continue
        incomplete_note = get_incomplete_note(CUMULATIVE_START_DATE, end_date)

        units_total = np.zeros(n_samples)
        h100e_total = np.zeros(n_samples)
        power_total_w = np.zeros(n_samples)
        for version in versions:
            arr = calendar_running[cq][version]
            units_total += arr
            if version in specs:
                h100e_total += arr * (specs[version]['tops'] / H100_TOPS)
                power_total_w += arr * specs[version]['tdp']
        power_mw = power_total_w / 1e6

        rows.append({
            'Name': f"Google cumulative TPU through {cq}",
            'Chip manufacturer': 'Google',
            'Owner': 'Google',
            'Start date': CUMULATIVE_START_DATE,
            'End date': end_date,
            'Compute estimate in H100e (median)': int(np.percentile(h100e_total, 50)),
            'H100e (5th percentile)': int(np.percentile(h100e_total, 5)),
            'H100e (95th percentile)': int(np.percentile(h100e_total, 95)),
            'Number of Units': int(np.percentile(units_total, 50)),
            'Number of Units (5th percentile)': int(np.percentile(units_total, 5)),
            'Number of Units (95th percentile)': int(np.percentile(units_total, 95)),
            'Total TDP (W)': int(np.percentile(power_total_w, 50)),
            'Total TDP (W) (5th percentile)': int(np.percentile(power_total_w, 5)),
            'Total TDP (W) (95th percentile)': int(np.percentile(power_total_w, 95)),
            'Power in MW (median)': int(np.percentile(power_mw, 50)),
            'Power in MW (5th percentile)': int(np.percentile(power_mw, 5)),
            'Power in MW (95th percentile)': int(np.percentile(power_mw, 95)),
            'Incomplete': 'checked' if incomplete_note else '',
            'Source / Link': '',
            'Notes': with_incomplete(incomplete_note),
        })
    write(pd.DataFrame(rows), 'owners_csv_export/tpu_owners_cumulative_totals.csv')

    # --- Cumulative totals across all versions (csv_export) ---
    # The Q4 2022 window start predates fiscal coverage by convention (mirroring the
    # published designer-totals table), so rows are only flagged incomplete when their
    # end extends past the available fiscal data - not at the head of the window.
    def get_tail_incomplete_note(end_date):
        if pd.to_datetime(end_date, format='%m/%d/%Y') > broadcom_last_end_dt:
            last_str = f"{broadcom_last_end_dt.month}/{broadcom_last_end_dt.day}/{broadcom_last_end_dt.year}"
            return f"Incomplete: based on Broadcom fiscal quarters ending {last_str}"
        return None

    rows = []
    for cq in calendar_running:
        _, end_date = get_calendar_quarter_dates(cq)
        if datetime.strptime(end_date, '%m/%d/%Y') < TOTALS_CUTOFF:
            continue
        incomplete_note = get_tail_incomplete_note(end_date)

        units_total = np.zeros(n_samples)
        h100e_total = np.zeros(n_samples)
        power_total_w = np.zeros(n_samples)
        for version in versions:
            arr = calendar_running[cq][version]
            units_total += arr
            if version in specs:
                h100e_total += arr * (specs[version]['tops'] / H100_TOPS)
                power_total_w += arr * specs[version]['tdp']

        rows.append({
            'Name': f"Google TPU total {TOTALS_START_LABEL} to {cq}",
            'Chip manufacturer': 'Google',
            'Start date': TOTALS_START_DATE,
            'End date': end_date,
            'Number of units (5th percentile)': int(np.percentile(units_total, 5)),
            'Number of units (median)': int(np.percentile(units_total, 50)),
            'Number of units (95th percentile)': int(np.percentile(units_total, 95)),
            'Compute estimate in H100e (5th percentile)': int(np.percentile(h100e_total, 5)),
            'Compute estimate in H100e (median)': int(np.percentile(h100e_total, 50)),
            'Compute estimate in H100e (95th percentile)': int(np.percentile(h100e_total, 95)),
            'Power in MW (5th percentile)': int(np.percentile(power_total_w / 1e6, 5)),
            'Power in MW (median)': int(np.percentile(power_total_w / 1e6, 50)),
            'Power in MW (95th percentile)': int(np.percentile(power_total_w / 1e6, 95)),
            'Incomplete': 'checked' if incomplete_note else '',
            'Notes': with_incomplete(incomplete_note),
        })
    write(pd.DataFrame(rows), 'csv_export/tpu_cumulative_totals.csv')

    return written
