"""Shared utilities for chip volume estimation (NVIDIA and TPU)."""

import numpy as np
import pandas as pd
from datetime import datetime


# ===============================
# Core quarterly simulation logic
# ===============================
def estimate_chip_sales(quarters, versions, sample_revenue, sample_shares, sample_price, n_samples=5000):
    """
    Run Monte Carlo simulation to estimate chip volumes.

    Args:
        quarters: list of quarter identifiers (e.g., ['Q1_FY23', 'Q2_FY23', ...])
        versions: list of chip types (e.g., ['v3', 'v4', 'v5e', ...])
        sample_revenue: fn(quarter) -> float, samples or looks up total chip revenue in dollars for a quarter
        sample_shares: fn(quarter) -> dict, samples {version: share} for a quarter (should sum to 1)
        sample_price: fn(quarter, version) -> float, samples or looks up price for a chip type in a quarter
        n_samples: number of Monte Carlo samples

    Returns:
        Dictionary of {quarter: {version: [array of samples of chip unit counts]}}
        To find median, confidence intervals, etc you will need to take the percentiles of the result

    Note on cross-quarter correlations:
        The sampling functions are called independently for each quarter within each iteration.
        This means any parameters you want correlated across quarters (e.g., a single margin
        value affecting all quarters) will NOT be correlated by default. To preserve cross-quarter
        correlations, pre-sample those parameters outside this function and have your sampling
        functions reference them.

        
    """
    results = {quarter: {version: [] for version in versions} for quarter in quarters}

    for _ in range(n_samples):
        for quarter in quarters:
            revenue = sample_revenue(quarter)
            shares = sample_shares(quarter)

            for version in versions:
                share = shares.get(version, 0)
                if share > 0:
                    price = sample_price(quarter, version)
                    chips = (revenue * share) / price
                else:
                    chips = 0
                results[quarter][version].append(chips)

    return results


def estimate_cumulative_chip_sales(
    quarters,
    chip_types,
    sample_revenue,
    sample_shares,
    sample_base_price,
    get_deflation_factor=None,
    revenue_bias_dist=None,
    n_samples=5000,
):
    """
    Run Monte Carlo simulation to estimate cumulative chip volumes with correlated parameters.

    Similar to estimate_chip_sales, but presamples certain parameters to correlate them
    across quarters. Use this when estimating cumulative totals where you want price
    uncertainty (and optionally revenue bias) to compound rather than average out.

    Args:
        quarters: list of quarter identifiers (e.g., ['Q1_2023', 'Q2_2023', ...])
        chip_types: list of chip types (e.g., ['alpha', 'beta', 'gamma', ...])
        sample_revenue: fn(quarter) -> float, samples total chip revenue in dollars for a quarter
        sample_shares: fn(quarter) -> dict, samples {chip: share} for a quarter (should sum to 1)
        sample_base_price: fn(chip) -> float, samples the BASE price for a chip type
            (i.e., the price when the chip was first introduced). Called once per chip;
            subsequent quarters use this base price scaled by get_deflation_factor.
        get_deflation_factor: fn(quarter, chip) -> float, returns price multiplier for a
            quarter relative to the base price. Should return 1.0 for the chip's first
            quarter and <1.0 for later quarters as prices decline. If None, no deflation.
        revenue_bias_dist: squigglepy distribution for systematic revenue estimation error.
            Sampled once and applied as a multiplier to all quarters. If None, no bias.
        n_samples: number of Monte Carlo samples

    Returns:
        dict of {chip: np.array of cumulative chip counts across all quarters}
        Each array has n_samples elements representing the distribution of total chips.

    Note on correlations:
        - Prices are sampled once per chip type and reused (with deflation) across all quarters.
          This means if we sample a "high price world", that persists for the entire simulation.
        - Revenue bias (if provided) is sampled once and applied to all quarters.
        - Revenue and production mix shares are sampled independently each quarter.
    """
    # === PRESAMPLE CORRELATED PARAMS ===

    # Prices: sample once per chip
    base_price_samples = {
        chip: np.array([sample_base_price(chip) for _ in range(n_samples)])
        for chip in chip_types
    }

    # Revenue bias (if provided)
    rev_bias = revenue_bias_dist @ n_samples if revenue_bias_dist else np.ones(n_samples)

    # === MAIN LOOP ===
    results = {chip: np.zeros(n_samples) for chip in chip_types}

    for quarter in quarters:
        # Sample revenue (uncorrelated) with bias (correlated)
        revenue = np.array([sample_revenue(quarter) for _ in range(n_samples)]) * rev_bias

        # Sample shares (uncorrelated)
        shares_list = [sample_shares(quarter) for _ in range(n_samples)]

        for chip in chip_types:
            shares = np.array([s.get(chip, 0) for s in shares_list])
            deflation = get_deflation_factor(quarter, chip) if get_deflation_factor else 1.0
            price = base_price_samples[chip] * deflation
            results[chip] += (revenue * shares) / price

    return results


def normalize_shares(raw_shares):
    """Normalize share values to sum to 1."""
    total = sum(raw_shares.values())
    return {k: v / total for k, v in raw_shares.items()}


def get_percentiles(samples, percentiles=[5, 50, 95]):
    """Get percentile values from samples array."""
    return {p: np.percentile(samples, p) for p in percentiles}


def compute_h100_equivalents(chip_counts, chip_specs, h100_tops=1979):
    """
    Convert chip counts to H100 equivalents based on 8-bit TOPS.

    Args:
        chip_counts: dict of {version: count} or {version: array of samples}
        chip_specs: dict with 'tops' key for each version
        h100_tops: H100 8-bit TOPS (default 1979)

    Returns:
        dict of {version: h100_equivalent_count}
    """
    return {
        version: counts * (chip_specs[version]['tops'] / h100_tops)
        for version, counts in chip_counts.items()
    }


def samples_to_percentile_dict(samples, percentiles=[5, 50, 95]):
    """Convert samples array to dict with percentile keys."""
    return {f'p{p}': int(np.percentile(samples, p)) for p in percentiles}


def export_quarterly_by_version(results, chip_specs, output_path, n_samples, h100_tops=1979):
    """
    Export quarterly chip volumes by version to CSV.

    Args:
        results: dict of {quarter: {version: list of samples}}
        chip_specs: dict with specs for each version
        output_path: path for CSV output
        n_samples: number of samples per distribution
        h100_tops: H100 8-bit TOPS for equivalence calculation

    Returns:
        DataFrame with exported data
    """
    rows = []
    for quarter in results:
        for version in chip_specs:
            arr = np.array(results[quarter][version])
            if arr.sum() > 0:
                h100e_arr = arr * (chip_specs[version]['tops'] / h100_tops)
                rows.append({
                    'quarter': quarter,
                    'version': version,
                    'chips_p5': int(np.percentile(arr, 5)),
                    'chips_p50': int(np.percentile(arr, 50)),
                    'chips_p95': int(np.percentile(arr, 95)),
                    'h100e_p5': int(np.percentile(h100e_arr, 5)),
                    'h100e_p50': int(np.percentile(h100e_arr, 50)),
                    'h100e_p95': int(np.percentile(h100e_arr, 95)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def print_cumulative_summary(cumulative, chip_specs, title="Cumulative Production"):
    """Print formatted summary of cumulative chip counts with percentiles."""
    print(f"\n{title}")
    print(f"{'Version':<6} {'p5':>12} {'p50':>12} {'p95':>12}")
    print("-" * 45)

    grand_total = None
    for version in chip_specs:
        arr = cumulative[version]
        if arr.sum() > 0:
            if grand_total is None:
                grand_total = np.zeros_like(arr)
            grand_total += arr
            print(f"{version:<6} {int(np.percentile(arr, 5)):>12,} {int(np.percentile(arr, 50)):>12,} {int(np.percentile(arr, 95)):>12,}")

    if grand_total is not None:
        print("-" * 45)
        print(f"{'TOTAL':<6} {int(np.percentile(grand_total, 5)):>12,} {int(np.percentile(grand_total, 50)):>12,} {int(np.percentile(grand_total, 95)):>12,}")


def format_thousands(n):
    """Format number as Xk (rounded to nearest thousand)."""
    return f"{round(n / 1000)}k"


def summarize_calendar_quarters(calendar_results, format_fn=format_thousands):
    """
    Create summary DataFrame from calendar quarter results (percentile dicts).

    Args:
        calendar_results: dict of {calendar_quarter: {version: {'p5': float, 'p50': float, 'p95': float}}}
                          Output from interpolate_to_calendar_quarters()
        format_fn: function to format numbers (default: format_thousands)

    Returns:
        DataFrame with Quarter, one column per chip type, and Total column.
        Each cell shows "median (p5 - p95)".
    """
    quarters = list(calendar_results.keys())
    versions = list(calendar_results[quarters[0]].keys())

    rows = []
    for quarter in quarters:
        row = {'Quarter': quarter}
        total_p5 = 0.0
        total_p50 = 0.0
        total_p95 = 0.0

        for version in versions:
            stats = calendar_results[quarter][version]
            total_p5 += stats['p5']
            total_p50 += stats['p50']
            total_p95 += stats['p95']

            if stats['p50'] > 0:
                p5 = format_fn(stats['p5'])
                p50 = format_fn(stats['p50'])
                p95 = format_fn(stats['p95'])
                row[version] = f"{p50} ({p5}-{p95})"
            else:
                row[version] = "-"

        # Add total column
        p5 = format_fn(total_p5)
        p50 = format_fn(total_p50)
        p95 = format_fn(total_p95)
        row['Total'] = f"{p50} ({p5}-{p95})"

        rows.append(row)

    # Order columns: Quarter, versions (in order), Total
    cols = ['Quarter'] + versions + ['Total']
    return pd.DataFrame(rows)[cols]


def summarize_quarterly_by_chip(results, format_fn=format_thousands):
    """
    Create summary DataFrame with each chip type as separate columns with 90% CI.

    Args:
        results: dict of {quarter: {chip_type: list of samples}}
                 Output from estimate_chip_sales()
        format_fn: function to format numbers (default: format_thousands)

    Returns:
        DataFrame with Quarter, one column per chip type, and Total column.
        Each cell shows "median (p5 - p95)".
    """
    # Infer quarters and chip types from results
    quarters = list(results.keys())
    chip_types = list(results[quarters[0]].keys())
    n_samples = len(results[quarters[0]][chip_types[0]])

    rows = []
    for quarter in quarters:
        row = {'Quarter': quarter}
        total = np.zeros(n_samples)

        for chip_type in chip_types:
            arr = np.array(results[quarter][chip_type])
            total += arr
            if arr.sum() > 0:
                p5 = format_fn(np.percentile(arr, 5))
                p50 = format_fn(np.percentile(arr, 50))
                p95 = format_fn(np.percentile(arr, 95))
                row[chip_type] = f"{p50} ({p5}-{p95})"
            else:
                row[chip_type] = "-"

        # Add total column
        p5 = format_fn(np.percentile(total, 5))
        p50 = format_fn(np.percentile(total, 50))
        p95 = format_fn(np.percentile(total, 95))
        row['Total'] = f"{p50} ({p5}-{p95})"

        rows.append(row)

    # Order columns: Quarter, chip types (in order), Total
    cols = ['Quarter'] + chip_types + ['Total']
    return pd.DataFrame(rows)[cols]


# ===============================
# Calendar quarter interpolation
# ===============================

def _get_calendar_quarter(date):
    """Return calendar quarter string like 'Q1 2024' for a given date."""
    month = date.month
    year = date.year
    if month <= 3:
        return f"Q1 {year}"
    elif month <= 6:
        return f"Q2 {year}"
    elif month <= 9:
        return f"Q3 {year}"
    else:
        return f"Q4 {year}"


def _get_calendar_quarter_bounds(cal_q):
    """Return (start_date, end_date) for a calendar quarter like 'Q1 2024'."""
    parts = cal_q.split()
    q_num = int(parts[0][1])
    year = int(parts[1])
    if q_num == 1:
        return datetime(year, 1, 1), datetime(year, 3, 31)
    elif q_num == 2:
        return datetime(year, 4, 1), datetime(year, 6, 30)
    elif q_num == 3:
        return datetime(year, 7, 1), datetime(year, 9, 30)
    else:
        return datetime(year, 10, 1), datetime(year, 12, 31)


def _days_overlap(start1, end1, start2, end2):
    """Calculate the number of overlapping days between two date ranges."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    if overlap_start <= overlap_end:
        return (overlap_end - overlap_start).days + 1
    return 0


def summarize_sim_results(sim_results):
    """
    Compute median and 90% CI for each chip and each quarter from simulation results.

    Args:
        sim_results: dict of {quarter: {version: array of samples}}
                     Output from estimate_chip_sales()

    Returns:
        dict of {quarter: {version: {'p5': float, 'p50': float, 'p95': float}}}
        Values are not rounded.
    """
    quarters = list(sim_results.keys())
    versions = list(sim_results[quarters[0]].keys())

    summary = {}
    for quarter in quarters:
        summary[quarter] = {}
        for version in versions:
            arr = np.array(sim_results[quarter][version])
            summary[quarter][version] = {
                'p5': np.percentile(arr, 5),
                'p50': np.percentile(arr, 50),
                'p95': np.percentile(arr, 95),
            }
    return summary


def interpolate_to_calendar_quarters(sim_results, quarter_dates, verbose=True):
    """
    Interpolate fiscal quarter chip estimates to calendar quarters.

    First computes median and 90% CI for each chip/quarter, then interpolates
    to calendar quarters by taking weighted averages of these summary statistics
    based on the day overlap between fiscal and calendar quarters.

    Args:
        sim_results: dict of {quarter: {version: array of samples}}
                     Output from estimate_chip_sales()
        quarter_dates: dict of {quarter: (start_date, end_date)} where dates are
                       datetime objects or strings parseable by pd.to_datetime
        verbose: if True, print progress info

    Returns:
        dict of {calendar_quarter: {version: {'p5': float, 'p50': float, 'p95': float}}}
        Calendar quarters are named like 'Q1 2024', 'Q2 2024', etc.
    """
    # First summarize sim_results to get percentiles for each fiscal quarter
    fiscal_summary = summarize_sim_results(sim_results)

    # Parse quarter dates
    fiscal_quarters = []
    for quarter in sim_results.keys():
        start, end = quarter_dates[quarter]
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        fiscal_quarters.append({
            'quarter': quarter,
            'start': start,
            'end': end,
            'days': (end - start).days + 1
        })

    # Get versions from first quarter
    versions = list(sim_results[fiscal_quarters[0]['quarter']].keys())

    # Determine the range of calendar quarters we need
    all_dates = []
    for fq in fiscal_quarters:
        all_dates.extend([fq['start'], fq['end']])
    min_date, max_date = min(all_dates), max(all_dates)

    # Generate all calendar quarters in the range
    calendar_quarters = []
    current = datetime(min_date.year, ((min_date.month - 1) // 3) * 3 + 1, 1)
    while current <= max_date:
        cal_q = _get_calendar_quarter(current)
        if cal_q not in calendar_quarters:
            calendar_quarters.append(cal_q)
        # Move to next quarter
        if current.month >= 10:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 3, 1)

    # Build intermediate mapping: calendar_quarter -> list of overlapping fiscal quarters
    # Each entry contains: fiscal_quarter name, days_overlap, pct_of_fiscal_quarter
    calendar_map = {}
    for cq in calendar_quarters:
        cq_start, cq_end = _get_calendar_quarter_bounds(cq)
        overlaps = []
        for fq in fiscal_quarters:
            days_overlap = _days_overlap(fq['start'], fq['end'], cq_start, cq_end)
            if days_overlap > 0:
                pct_of_fq = days_overlap / fq['days']
                overlaps.append({
                    'fiscal_quarter': fq['quarter'],
                    'days_overlap': days_overlap,
                    'pct_of_fiscal_quarter': pct_of_fq,
                })
        calendar_map[cq] = overlaps

    # Initialize results with percentile dicts
    calendar_results = {
        cq: {version: {'p5': 0.0, 'p50': 0.0, 'p95': 0.0} for version in versions}
        for cq in calendar_quarters
    }

    # Use calendar_map to compute weighted average of percentiles
    for cq, overlaps in calendar_map.items():
        for overlap in overlaps:
            fq_name = overlap['fiscal_quarter']
            fraction = overlap['pct_of_fiscal_quarter']
            for version in versions:
                fq_stats = fiscal_summary[fq_name][version]
                calendar_results[cq][version]['p5'] += fq_stats['p5'] * fraction
                calendar_results[cq][version]['p50'] += fq_stats['p50'] * fraction
                calendar_results[cq][version]['p95'] += fq_stats['p95'] * fraction

    return calendar_results


def verify_calendar_quarter_interpolation(sim_results, calendar_results, quarter_dates, verbose=True):
    """
    Run sanity checks on calendar quarter interpolation.

    Args:
        sim_results: original fiscal quarter results (dict of {quarter: {version: array of samples}})
        calendar_results: interpolated calendar quarter results
                          (dict of {calendar_quarter: {version: {'p5': float, 'p50': float, 'p95': float}}})
        quarter_dates: dict of {quarter: (start_date, end_date)}
        verbose: if True, print detailed output

    Returns:
        True if all checks pass, False otherwise
    """
    all_passed = True
    versions = list(sim_results[list(sim_results.keys())[0]].keys())

    # Get fiscal summary for comparison
    fiscal_summary = summarize_sim_results(sim_results)

    # Parse fiscal quarters
    fiscal_quarters = []
    for quarter in sim_results.keys():
        start, end = quarter_dates[quarter]
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        fiscal_quarters.append({
            'quarter': quarter,
            'start': start,
            'end': end,
            'days': (end - start).days + 1
        })

    # Check 1: Total median chips should approximately match
    if verbose:
        print("=== Check 1: Total median chips should approximately match ===")

    fiscal_total_p50 = {v: 0.0 for v in versions}
    calendar_total_p50 = {v: 0.0 for v in versions}

    for fq in fiscal_summary:
        for v in versions:
            fiscal_total_p50[v] += fiscal_summary[fq][v]['p50']

    for cq in calendar_results:
        for v in versions:
            calendar_total_p50[v] += calendar_results[cq][v]['p50']

    if verbose:
        print(f"{'Version':<6} {'Fiscal p50':>14} {'Calendar p50':>14} {'Diff':>12}")
        print("-" * 50)
    for v in versions:
        diff = abs(fiscal_total_p50[v] - calendar_total_p50[v])
        # Allow small relative difference due to weighted averaging of percentiles
        rel_diff = diff / max(fiscal_total_p50[v], 1) * 100
        passed = rel_diff < 5  # Allow up to 5% difference
        if not passed:
            all_passed = False
        if verbose:
            status = "✓" if passed else "✗"
            print(f"{v:<6} {fiscal_total_p50[v]:>14,.0f} {calendar_total_p50[v]:>14,.0f} {rel_diff:>11.1f}% {status}")

    # Check 2: Spot-check first fiscal quarter split
    if verbose:
        print("\n=== Check 2: Spot-check first fiscal quarter date split ===")
    fq = fiscal_quarters[0]
    if verbose:
        print(f"{fq['quarter']}: {fq['start'].date()} to {fq['end'].date()} ({fq['days']} days)")

    cq_first = _get_calendar_quarter(fq['start'])
    cq_second = _get_calendar_quarter(fq['end'])

    cq_first_start, cq_first_end = _get_calendar_quarter_bounds(cq_first)
    cq_second_start, cq_second_end = _get_calendar_quarter_bounds(cq_second)

    overlap_first = _days_overlap(fq['start'], fq['end'], cq_first_start, cq_first_end)
    overlap_second = _days_overlap(fq['start'], fq['end'], cq_second_start, cq_second_end)

    if verbose:
        print(f"Overlap with {cq_first}: {overlap_first} days ({overlap_first/fq['days']*100:.1f}%)")
        if cq_first != cq_second:
            print(f"Overlap with {cq_second}: {overlap_second} days ({overlap_second/fq['days']*100:.1f}%)")
        print(f"Total accounted: {overlap_first + overlap_second} days (should equal {fq['days']})")

    if overlap_first + overlap_second != fq['days']:
        all_passed = False

    # Check 3: Verify "pure" calendar quarters (first and last) - check that weighted avg matches
    if verbose:
        print("\n=== Check 3: Verify 'pure' calendar quarters (single fiscal quarter source) ===")

    # First calendar quarter: only from first fiscal quarter
    fq_first = fiscal_quarters[0]
    cq_first_name = _get_calendar_quarter(fq_first['start'])
    cq_first_start, cq_first_end = _get_calendar_quarter_bounds(cq_first_name)
    overlap_first = _days_overlap(fq_first['start'], fq_first['end'], cq_first_start, cq_first_end)
    fraction_first = overlap_first / fq_first['days']

    if verbose:
        print(f"{cq_first_name} receives {fraction_first*100:.1f}% of {fq_first['quarter']}")
        print(f"  {fq_first['quarter']} runs {fq_first['start'].date()} to {fq_first['end'].date()} ({fq_first['days']} days)")
        print(f"  {cq_first_name} runs {cq_first_start.date()} to {cq_first_end.date()}")
        print(f"  Overlap: {fq_first['start'].date()} to {cq_first_end.date()} = {overlap_first} days")
        print(f"\n{'Version':<6} {'FQ p50':>12} {'x':>3} {'frac':>6} {'=':>3} {'Expected':>12} {'Actual':>12} {'Match':>6}")
        print("-" * 65)

    for v in versions:
        fq_p50 = fiscal_summary[fq_first['quarter']][v]['p50']
        expected_p50 = fq_p50 * fraction_first
        actual_p50 = calendar_results[cq_first_name][v]['p50']
        match = abs(expected_p50 - actual_p50) < 0.01
        if not match:
            all_passed = False
        if verbose and fq_p50 > 0:
            print(f"{v:<6} {fq_p50:>12,.0f} {'x':>3} {fraction_first:>6.1%} {'=':>3} {expected_p50:>12,.0f} {actual_p50:>12,.0f} {'✓' if match else '✗':>6}")

    # Last calendar quarter: only from last fiscal quarter
    fq_last = fiscal_quarters[-1]
    cq_last_name = _get_calendar_quarter(fq_last['end'])
    cq_last_start, cq_last_end = _get_calendar_quarter_bounds(cq_last_name)
    overlap_last = _days_overlap(fq_last['start'], fq_last['end'], cq_last_start, cq_last_end)
    fraction_last = overlap_last / fq_last['days']

    if verbose:
        print(f"\n{cq_last_name} receives {fraction_last*100:.1f}% of {fq_last['quarter']}")
        print(f"  {fq_last['quarter']} runs {fq_last['start'].date()} to {fq_last['end'].date()} ({fq_last['days']} days)")
        print(f"  {cq_last_name} runs {cq_last_start.date()} to {cq_last_end.date()}")
        print(f"  Overlap: {cq_last_start.date()} to {fq_last['end'].date()} = {overlap_last} days")
        print(f"\n{'Version':<6} {'FQ p50':>12} {'x':>3} {'frac':>6} {'=':>3} {'Expected':>12} {'Actual':>12} {'Match':>6}")
        print("-" * 65)

    for v in versions:
        fq_p50 = fiscal_summary[fq_last['quarter']][v]['p50']
        expected_p50 = fq_p50 * fraction_last
        actual_p50 = calendar_results[cq_last_name][v]['p50']
        match = abs(expected_p50 - actual_p50) < 0.01
        if not match:
            all_passed = False
        if verbose and fq_p50 > 0:
            print(f"{v:<6} {fq_p50:>12,.0f} {'x':>3} {fraction_last:>6.1%} {'=':>3} {expected_p50:>12,.0f} {actual_p50:>12,.0f} {'✓' if match else '✗':>6}")

    # Check 4: Verify a middle calendar quarter is the correct blend of fiscal quarters
    if verbose:
        print("\n=== Check 4: Verify blended calendar quarter (random middle quarter) ===")

    # Pick a calendar quarter in the middle (not first or last)
    cal_quarter_list = list(calendar_results.keys())
    if len(cal_quarter_list) > 2:
        import random
        random.seed(42)
        middle_cq = random.choice(cal_quarter_list[1:-1])
    else:
        middle_cq = cal_quarter_list[0]

    cq_start, cq_end = _get_calendar_quarter_bounds(middle_cq)

    if verbose:
        print(f"Selected calendar quarter: {middle_cq}")
        print(f"  {middle_cq} runs {cq_start.date()} to {cq_end.date()}")
        print(f"\nContributing fiscal quarters:")

    # Find all fiscal quarters that contribute to this calendar quarter
    contributions = []
    for fq in fiscal_quarters:
        overlap = _days_overlap(fq['start'], fq['end'], cq_start, cq_end)
        if overlap > 0:
            fraction = overlap / fq['days']
            contributions.append({
                'fq': fq,
                'overlap': overlap,
                'fraction': fraction
            })
            if verbose:
                print(f"  {fq['quarter']}: {fq['start'].date()} to {fq['end'].date()} ({fq['days']} days)")
                print(f"    Overlap with {middle_cq}: {overlap} days")
                print(f"    Contribution: {overlap}/{fq['days']} = {fraction*100:.1f}% of {fq['quarter']}'s chips")

    # Compute expected values by summing contributions from each fiscal quarter
    if verbose:
        print(f"\nExpected {middle_cq} p50 = ", end="")
        contrib_strs = [f"{c['fraction']*100:.1f}% of {c['fq']['quarter']}" for c in contributions]
        print(" + ".join(contrib_strs))

        # Build dynamic header showing: Version | FQ1 p50 | x frac1 | + | FQ2 p50 | x frac2 | = | Sum | Actual | Match
        print(f"\n{'Version':<6}", end="")
        for i, c in enumerate(contributions):
            fq_name = c['fq']['quarter']
            if i > 0:
                print(f" {'':>3}", end="")  # spacing for '+'
            print(f" {fq_name + ' p50':>12} {'x':>3} {'frac':>6}", end="")
        print(f" {'=':>3} {'Sum':>10} {'Actual':>10} {'Match':>6}")
        print("-" * (6 + len(contributions) * 28 + 35))

    for v in versions:
        expected_p50 = 0.0
        row_data = []

        for c in contributions:
            fq_p50 = fiscal_summary[c['fq']['quarter']][v]['p50']
            contrib = fq_p50 * c['fraction']
            expected_p50 += contrib
            row_data.append({
                'fq_p50': fq_p50,
                'fraction': c['fraction'],
                'contrib': contrib,
            })

        actual_p50 = calendar_results[middle_cq][v]['p50']
        match = abs(expected_p50 - actual_p50) < 0.01
        if not match:
            all_passed = False

        # Only print if there's data for this version
        if verbose and (expected_p50 > 0 or actual_p50 > 0):
            row = f"{v:<6}"
            for i, rd in enumerate(row_data):
                if i > 0:
                    row += f" {'+':>3}"
                row += f" {rd['fq_p50']:>12,.0f} {'x':>3} {rd['fraction']:>6.1%}"
            status = '✓' if match else '✗'
            row += f" {'=':>3} {int(expected_p50):>10,} {int(actual_p50):>10,} {status:>6}"
            print(row)

    if verbose:
        print(f"\n{'All checks passed!' if all_passed else 'Some checks FAILED!'}")

    return all_passed
