"""
Aggregate xAI data center data into cumulative chip counts and compute power over time.

Inputs:
  - data_inputs/xai_chip_quantities.csv  (chip-level snapshots per data center)
  - data_inputs/xai_datacenter_timelines.csv  (running totals of H100e / power per data center)

Outputs (printed as copy-able CSVs):
  1. Cumulative by chip type per quarter
  2. Cumulative totals per quarter
  3. Quarterly additions by chip type

Logic:
  - Build chip inventory snapshots from chip_quantities (primary) and timeline descriptions (fallback for early dates).
  - For each calendar quarter, take the latest snapshot on or before quarter end.
  - Sum across all data centers.
  - If timeline H100e exceeds chip-derived H100e for a quarter, we still use chip quantities (chip data is authoritative).
  - Timeline power (MW) is used directly since chip data doesn't provide power info.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
import re
import os
import io

# ---------------------------------------------------------------------------
# Chip specs (matching nvidia_estimates.ipynb FALLBACK_SPECS)
# ---------------------------------------------------------------------------
CHIP_SPECS = {
    'H100':  {'tops': 1979, 'tdp': 700},
    'H200':  {'tops': 1979, 'tdp': 700},
    'B200':  {'tops': 5000, 'tdp': 1200},
    'B300':  {'tops': 5000, 'tdp': 1400},
}
H100_TOPS = 1979

# Use H100/H200 as the combined chip name in output (matching existing convention)
# but input data has separate H100 and H200 entries
CHIP_DISPLAY_MAP = {
    'H100': 'H100/H200',
    'H200': 'H100/H200',
    'B200': 'B200',
    'B300': 'B300',
}

# ---------------------------------------------------------------------------
# Quarter utilities
# ---------------------------------------------------------------------------
def date_to_quarter(d):
    """Return calendar quarter string like 'Q3 2024' for a date."""
    q = (d.month - 1) // 3 + 1
    return f"Q{q} {d.year}"

def quarter_end(q_str):
    """Return the last date of a calendar quarter like 'Q3 2024'."""
    q, year = int(q_str[1]), int(q_str[3:])
    month = q * 3
    if month == 12:
        return date(year, 12, 31)
    else:
        return date(year, month + 1, 1) - pd.Timedelta(days=1)

def quarter_start(q_str):
    """Return the first date of a calendar quarter."""
    q, year = int(q_str[1]), int(q_str[3:])
    month = (q - 1) * 3 + 1
    return date(year, month, 1)

def format_date(d):
    """Format date as M/D/YYYY (no zero-padding)."""
    return f"{d.month}/{d.day}/{d.year}"

# ---------------------------------------------------------------------------
# Load chip quantities (authoritative chip-level data)
# ---------------------------------------------------------------------------
def load_chip_quantities(path):
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        chip = row['Chip type'].strip()
        records.append({
            'data_center': row['Data center'].strip(),
            'date': pd.to_datetime(row['Date']).date(),
            'chip_type': chip,
            'units': int(row['Number of Units']),
        })
    return records

# ---------------------------------------------------------------------------
# Load data center timelines (running totals with H100e and power)
# ---------------------------------------------------------------------------
def load_timelines(path):
    df = pd.read_csv(path)
    records = []
    for _, row in df.iterrows():
        h100e = row.get('H100 equivalents', 0)
        if pd.isna(h100e) or h100e == '':
            h100e = 0
        h100e = int(float(h100e))

        power = row.get('Power (MW)', 0)
        if pd.isna(power):
            power = 0
        power = float(power)

        status = str(row.get('Construction status', ''))

        records.append({
            'data_center': row['Data center'].strip(),
            'date': pd.to_datetime(row['Date']).date(),
            'h100e': h100e,
            'power_mw': power,
            'status': status,
        })
    return records

# ---------------------------------------------------------------------------
# Parse chip breakdown from timeline descriptions (for early dates)
# ---------------------------------------------------------------------------
def parse_chips_from_status(status, h100e_total):
    """
    Try to extract chip counts from timeline status descriptions.
    Returns dict of {chip_type: units} or None if can't parse.
    """
    chips = {}

    # Pattern: "Nk chiptype" e.g. "25k H100s", "150k H100s and 50k H200s"
    matches = re.findall(r'(\d+)k\s+(H100|H200|B200|B300)s?', status, re.IGNORECASE)
    if matches:
        for count_str, chip in matches:
            count = int(count_str) * 1000
            chips[chip] = chips.get(chip, 0) + count
        return chips

    # Pattern: bare number like "110000 NVIDIA GB200"
    match = re.search(r'(\d{4,})\s+NVIDIA\s+GB200', status)
    if match:
        chips['B200'] = int(match.group(1))
        return chips

    return None

# ---------------------------------------------------------------------------
# Build per-data-center chip inventory over time
# ---------------------------------------------------------------------------
def build_chip_inventory(chip_records, timeline_records):
    """
    Build a time series of chip inventory per data center.

    Returns dict: {data_center: [(date, {chip_type: units})]}
    sorted by date.

    Chip quantities are authoritative. Timeline records fill in earlier dates.
    """
    # Group chip quantities by (data_center, date)
    chip_snapshots = {}
    for r in chip_records:
        key = (r['data_center'], r['date'])
        if key not in chip_snapshots:
            chip_snapshots[key] = {}
        chip_snapshots[key][r['chip_type']] = r['units']

    # Group timelines by data_center
    timeline_by_dc = {}
    for r in timeline_records:
        dc = r['data_center']
        if dc not in timeline_by_dc:
            timeline_by_dc[dc] = []
        timeline_by_dc[dc].append(r)
    for dc in timeline_by_dc:
        timeline_by_dc[dc].sort(key=lambda x: x['date'])

    inventory = {}
    all_dcs = set([r['data_center'] for r in chip_records] +
                  [r['data_center'] for r in timeline_records])

    for dc in sorted(all_dcs):
        snapshots = []
        seen_dates = set()

        # Chip quantity snapshots (authoritative)
        for (d, dt), chips in sorted(chip_snapshots.items()):
            if d == dc:
                snapshots.append((dt, dict(chips)))
                seen_dates.add(dt)

        # Fill in from timelines for dates not covered by chip quantities.
        for tl in timeline_by_dc.get(dc, []):
            if tl['date'] in seen_dates:
                continue
            if tl['h100e'] == 0:
                # Could be either "no chips yet" or "blank field, chips unchanged".
                # Mark for resolution in second pass based on context.
                snapshots.append((tl['date'], {'_h100e_only': 0}))
                continue

            parsed = parse_chips_from_status(tl['status'], tl['h100e'])
            if parsed:
                snapshots.append((tl['date'], parsed))
            else:
                # No chip breakdown available from this timeline entry.
                # We'll resolve this in a second pass below.
                snapshots.append((tl['date'], {'_h100e_only': tl['h100e']}))

        snapshots.sort(key=lambda x: x[0])

        # Second pass: resolve _h100e_only entries.
        # If we've seen real chip data before, carry forward that breakdown.
        # If not, treat as genuinely empty (pre-deployment).
        last_chip_breakdown = None
        resolved = []
        for snap_date, chips in snapshots:
            if '_h100e_only' in chips:
                if last_chip_breakdown is not None:
                    # Data center had chips before; this is a "no change" observation
                    resolved.append((snap_date, dict(last_chip_breakdown)))
                else:
                    # No prior chips; this is genuinely empty
                    resolved.append((snap_date, {}))
            elif chips:
                last_chip_breakdown = dict(chips)
                resolved.append((snap_date, chips))
            else:
                resolved.append((snap_date, {}))

        inventory[dc] = resolved

    return inventory

# ---------------------------------------------------------------------------
# Resolve chip inventory for a given data center at a given date
# ---------------------------------------------------------------------------
def get_inventory_at_date(dc_snapshots, target_date):
    result = {}
    for snap_date, chips in dc_snapshots:
        if snap_date <= target_date:
            result = chips
        else:
            break
    return result

# ---------------------------------------------------------------------------
# Compute H100e from chip counts
# ---------------------------------------------------------------------------
def compute_h100e_from_chips(chip_dict):
    if '_h100e_only' in chip_dict:
        return chip_dict['_h100e_only']
    total = 0
    for chip, units in chip_dict.items():
        if chip in CHIP_SPECS:
            total += units * CHIP_SPECS[chip]['tops'] / H100_TOPS
    return total

# ---------------------------------------------------------------------------
# Get power (MW) from timeline for a data center at a date
# ---------------------------------------------------------------------------
def get_power_at_date(timeline_records, dc, target_date):
    power = 0
    for r in sorted(timeline_records, key=lambda x: x['date']):
        if r['data_center'] == dc and r['date'] <= target_date:
            power = r['power_mw']
    return power

# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------
def aggregate_xai_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'data_inputs')

    chip_records = load_chip_quantities(os.path.join(input_dir, 'xai_chip_quantities.csv'))
    timeline_records = load_timelines(os.path.join(input_dir, 'xai_datacenter_timelines.csv'))

    inventory = build_chip_inventory(chip_records, timeline_records)

    # Print parsed inventory for debugging
    print("=== Parsed Inventory ===")
    for dc, snaps in inventory.items():
        print(f"\n  {dc}:")
        for d, chips in snaps:
            if '_h100e_only' in chips:
                print(f"    {d}: {chips['_h100e_only']:,} H100e (no chip breakdown)")
            elif chips:
                parts = [f"{u:,} {c}" for c, u in sorted(chips.items())]
                print(f"    {d}: {', '.join(parts)}")
            else:
                print(f"    {d}: (no chips)")

    # Determine quarter range
    all_dates = [d for dc, snaps in inventory.items() for d, _ in snaps]
    min_date, max_date = min(all_dates), max(all_dates)

    quarters = []
    q = date_to_quarter(min_date)
    while quarter_start(q) <= max_date:
        quarters.append(q)
        s = quarter_start(q)
        next_month = s.month + 3
        next_year = s.year
        if next_month > 12:
            next_month -= 12
            next_year += 1
        q = date_to_quarter(date(next_year, next_month, 1))

    timestamp = datetime.now().strftime("%m-%d-%Y %H:%M")

    # -----------------------------------------------------------------------
    # Build per-quarter aggregated data
    # -----------------------------------------------------------------------
    all_chip_types_seen = set()
    quarter_data = {}   # {quarter: {chip_display_name: total_units}}
    quarter_h100e = {}
    quarter_power = {}

    for q in quarters:
        qend = quarter_end(q)
        combined_chips = {}
        total_h100e = 0
        total_power = 0

        for dc, snaps in inventory.items():
            inv = get_inventory_at_date(snaps, qend)
            if not inv:
                continue

            total_power += get_power_at_date(timeline_records, dc, qend)

            if '_h100e_only' in inv:
                total_h100e += inv['_h100e_only']
                continue

            for chip, units in inv.items():
                display = CHIP_DISPLAY_MAP.get(chip, chip)
                combined_chips[display] = combined_chips.get(display, 0) + units
                all_chip_types_seen.add(display)

            total_h100e += compute_h100e_from_chips(inv)

        quarter_data[q] = combined_chips
        quarter_h100e[q] = total_h100e
        quarter_power[q] = total_power

    chip_types_ordered = sorted(all_chip_types_seen)

    # -----------------------------------------------------------------------
    # Helper to get chip spec for a display name
    # -----------------------------------------------------------------------
    def chip_spec(chip_display):
        if chip_display == 'H100/H200':
            return CHIP_SPECS['H100']
        return CHIP_SPECS[chip_display]

    # -----------------------------------------------------------------------
    # Output 1: Cumulative by chip type (CSV)
    # -----------------------------------------------------------------------
    by_chip_rows = []
    for q in quarters:
        chips = quarter_data[q]
        for chip_display in chip_types_ordered:
            units = chips.get(chip_display, 0)
            if units == 0:
                continue
            spec = chip_spec(chip_display)
            h100e = int(units * spec['tops'] / H100_TOPS)
            total_tdp = int(units * spec['tdp'])

            by_chip_rows.append({
                'Name': f"xAI {chip_display} cumulative through {q}",
                'Chip manufacturer': 'Nvidia',
                'Owner': 'xAI',
                'Chip type': chip_display,
                'Start date': '1/1/2024',
                'End date': format_date(quarter_end(q)),
                'Compute estimate in H100e (median)': h100e,
                'H100e (5th percentile)': '',
                'H100e (95th percentile)': '',
                'Number of Units (median)': units,
                'Number of Units (5th percentile)': '',
                'Number of Units (95th percentile)': '',
                'Source / Link': '',
                'Notes': f"Estimates generated on: {timestamp}",
                'Last Modified By': '',
                'Last Modified': '',
                'Incomplete': '',
                'Created': timestamp,
                'Total TDP (W)': total_tdp,
                'Total TDP (W) (5th percentile)': '',
                'Total TDP (W) (95th percentile)': '',
            })

    by_chip_df = pd.DataFrame(by_chip_rows)

    # -----------------------------------------------------------------------
    # Output 2: Cumulative totals (CSV)
    # -----------------------------------------------------------------------
    totals_rows = []
    for q in quarters:
        h100e = int(quarter_h100e[q])
        power = quarter_power[q]
        chips = quarter_data[q]
        total_units = sum(chips.values())

        total_tdp = 0
        has_chip_breakdown = True
        for chip_display, units in chips.items():
            spec = chip_spec(chip_display)
            total_tdp += units * spec['tdp']

        # Notes with per-DC breakdown
        dc_notes = []
        qend = quarter_end(q)
        for dc in sorted(inventory.keys()):
            inv = get_inventory_at_date(inventory[dc], qend)
            if inv and inv != {}:
                if '_h100e_only' in inv:
                    dc_notes.append(f"{dc}: {inv['_h100e_only']:,} H100e (no chip breakdown)")
                else:
                    parts = [f"{u:,} {c}" for c, u in sorted(inv.items())]
                    dc_notes.append(f"{dc}: {', '.join(parts)}")

        notes_str = "; ".join(dc_notes)
        notes_str += f". Estimates generated on: {timestamp}"

        totals_rows.append({
            'Name': f"xAI cumulative Nvidia through {q}",
            'Chip manufacturer': 'Nvidia',
            'Owner': 'xAI',
            'Start date': '1/1/2024',
            'End date': format_date(quarter_end(q)),
            'Compute estimate in H100e (median)': h100e,
            'H100e (5th percentile)': '',
            'H100e (95th percentile)': '',
            'Number of Units (median)': total_units if total_units > 0 else '',
            'Number of Units (5th percentile)': '',
            'Number of Units (95th percentile)': '',
            'Total TDP (W)': int(total_tdp) if has_chip_breakdown and total_tdp > 0 else '',
            'Total TDP (W) (5th percentile)': '',
            'Total TDP (W) (95th percentile)': '',
            'Power in MW (median)': round(power, 2) if power > 0 else '',
            'Power in MW (5th percentile)': '',
            'Power in MW (95th percentile)': '',
            'Incomplete': '',
            'Source / Link': '',
            'Notes': notes_str,
        })

    totals_df = pd.DataFrame(totals_rows)

    # -----------------------------------------------------------------------
    # Output 3: Quarterly additions by chip type
    # -----------------------------------------------------------------------
    quarterly_add_rows = []
    prev_chips = {}
    # Track the first quarter each chip type appears, so we can emit 0-rows after that
    chip_first_quarter = {}
    for q in quarters:
        chips = quarter_data[q]
        for chip_display in chip_types_ordered:
            if chip_display not in chip_first_quarter and chips.get(chip_display, 0) > 0:
                chip_first_quarter[chip_display] = q

    prev_chips = {}
    for q in quarters:
        chips = quarter_data[q]
        for chip_display in chip_types_ordered:
            # Only emit rows from the first quarter the chip type appears onward
            if chip_display not in chip_first_quarter:
                continue
            if quarters.index(q) < quarters.index(chip_first_quarter[chip_display]):
                continue

            curr = chips.get(chip_display, 0)
            prev = prev_chips.get(chip_display, 0)
            added = curr - prev

            spec = chip_spec(chip_display)
            h100e_added = int(added * spec['tops'] / H100_TOPS)
            tdp_added = int(added * spec['tdp'])

            quarterly_add_rows.append({
                'Name': f"xAI {chip_display} added in {q}",
                'Chip manufacturer': 'Nvidia',
                'Owner': 'xAI',
                'Chip type': chip_display,
                'Start date': format_date(quarter_start(q)),
                'End date': format_date(quarter_end(q)),
                'Compute estimate in H100e (median)': h100e_added,
                'H100e (5th percentile)': '',
                'H100e (95th percentile)': '',
                'Number of Units (median)': added,
                'Number of Units (5th percentile)': '',
                'Number of Units (95th percentile)': '',
                'Source / Link': '',
                'Notes': f"Estimates generated on: {timestamp}",
                'Last Modified By': '',
                'Last Modified': '',
                'Incomplete': '',
                'Total TDP (W)': tdp_added,
                'Total TDP (W) (5th percentile)': '',
                'Total TDP (W) (95th percentile)': '',
            })
        prev_chips = dict(chips)

    quarterly_add_df = pd.DataFrame(quarterly_add_rows)

    # -----------------------------------------------------------------------
    # Print everything
    # -----------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("=== CUMULATIVE BY CHIP TYPE (CSV) ===")
    print("=" * 80)
    print(by_chip_df.to_csv(index=False))

    print("\n" + "=" * 80)
    print("=== CUMULATIVE TOTALS (CSV) ===")
    print("=" * 80)
    print(totals_df.to_csv(index=False))

    print("\n" + "=" * 80)
    print("=== QUARTERLY ADDITIONS BY CHIP TYPE (CSV) ===")
    print("=" * 80)
    print(quarterly_add_df.to_csv(index=False))

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== SUMMARY TABLE ===")
    print("=" * 80)
    print(f"{'Quarter':<12} {'H100e':>12} {'Units':>10} {'Power MW':>10}  Chip breakdown")
    print("-" * 80)
    for q in quarters:
        h100e = quarter_h100e[q]
        chips = quarter_data[q]
        total_units = sum(chips.values())
        power = quarter_power[q]
        chip_str = ", ".join(f"{u:,} {c}" for c, u in sorted(chips.items())) if chips else "(no breakdown)"
        print(f"  {q:<10} {int(h100e):>12,} {total_units:>10,} {power:>10.0f}  {chip_str}")

    # Quarterly additions summary
    print(f"\n{'Quarter':<12} {'H100e added':>12} {'Units added':>12}  Chip additions")
    print("-" * 80)
    prev_chips = {}
    prev_h100e = 0
    for q in quarters:
        h100e = quarter_h100e[q]
        chips = quarter_data[q]
        h100e_added = h100e - prev_h100e

        additions = []
        for chip_display in chip_types_ordered:
            curr = chips.get(chip_display, 0)
            prev = prev_chips.get(chip_display, 0)
            diff = curr - prev
            if diff != 0:
                additions.append(f"{diff:+,} {chip_display}")

        total_added = sum(chips.values()) - sum(prev_chips.values())
        add_str = ", ".join(additions) if additions else "(no change)"
        print(f"  {q:<10} {int(h100e_added):>+12,} {total_added:>+12,}  {add_str}")

        prev_chips = dict(chips)
        prev_h100e = h100e


if __name__ == '__main__':
    aggregate_xai_data()
