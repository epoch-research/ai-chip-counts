"""Generate the interactive chip ownership HTML visualization.

Reads CSV exports from owners_csv_export/ and produces index.html.
Can be run standalone: python visualizations/chip_ownership/generate.py
"""

import json
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CSV_DIR = os.path.join(PROJECT_ROOT, 'owners_csv_export')
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'template.html')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'index.html')

CHIP_COLORS = {
    # Nvidia
    'A100': 'lightcoral',
    'A800': 'sienna',
    'H100/H200': 'steelblue',
    'H800': 'firebrick',
    'H20': 'orange',
    'B200': 'mediumseagreen',
    'B300': 'seagreen',
    # Google TPU
    'TPU v3': '#7B1FA2',
    'TPU v4': '#9C27B0',
    'TPU v4i': '#BA68C8',
    'TPU v5e': '#6A1B9A',
    'TPU v5p': '#4A148C',
    'TPU v6e': '#8E24AA',
    'TPU v7': '#CE93D8',
}

# Display order for owners and chips
OWNER_ORDER = ['Microsoft', 'Meta', 'Amazon', 'Google', 'Other (ex-Big 4 hyperscalers)']
CHIP_ORDER = [
    'A100', 'A800', 'H100/H200', 'H800', 'H20', 'B200', 'B300',
    'TPU v3', 'TPU v4', 'TPU v4i', 'TPU v5e', 'TPU v5p', 'TPU v6e', 'TPU v7',
]

# CSV file pairs: (cumulative_file, timelines_file)
CSV_SOURCES = [
    ('nvidia_ownership_cumulative_by_chip.csv', 'nvidia_ownership_timelines_by_chip.csv'),
    ('tpu_ownership_cumulative_by_chip.csv', 'tpu_ownership_timelines_by_chip.csv'),
]


def _end_date_to_quarter(end_date_str):
    """Convert end date like '3/31/2022' to quarter string like 'Q1 2022'."""
    parts = end_date_str.split('/')
    month = int(parts[0])
    year = int(parts[2])
    quarter = (month - 1) // 3 + 1
    return f"Q{quarter} {year}"


def _quarter_sort_key(q):
    """Sort key for quarter strings like 'Q1 2022'."""
    parts = q.split()
    return (int(parts[1]), int(parts[0][1]))


def _load_csv_to_viz_data(csv_path):
    """Load a by-chip CSV and return {quarter: {owner: {chip: {units/h100e: [p5,p50,p95]}}}}."""
    df = pd.read_csv(csv_path)
    data = {}
    for _, row in df.iterrows():
        quarter = _end_date_to_quarter(row['End date'])
        owner = row['Owner']
        chip = row['Chip type']

        units_p50 = int(row['Number of Units'])
        if units_p50 == 0:
            continue

        data.setdefault(quarter, {}).setdefault(owner, {})[chip] = {
            'units': [
                int(row['Number of Units (5th percentile)']),
                units_p50,
                int(row['Number of Units (95th percentile)']),
            ],
            'h100e': [
                int(row['H100e (5th percentile)']),
                int(row['Compute estimate in H100e (median)']),
                int(row['H100e (95th percentile)']),
            ],
        }
    return data


def _merge_viz_data(base, addition):
    """Merge two viz data dicts (addition's entries are added into base)."""
    for quarter, owners in addition.items():
        for owner, chips in owners.items():
            for chip, values in chips.items():
                base.setdefault(quarter, {}).setdefault(owner, {})[chip] = values


def generate():
    """Read CSV exports and write index.html."""
    cumulative_data = {}
    flow_data = {}

    for cumulative_file, timelines_file in CSV_SOURCES:
        cumulative_path = os.path.join(CSV_DIR, cumulative_file)
        timelines_path = os.path.join(CSV_DIR, timelines_file)

        if os.path.exists(cumulative_path):
            _merge_viz_data(cumulative_data, _load_csv_to_viz_data(cumulative_path))
        if os.path.exists(timelines_path):
            _merge_viz_data(flow_data, _load_csv_to_viz_data(timelines_path))

    # For cumulative data, carry forward the last known values into later quarters
    # where a source is missing (e.g. TPU data ends at Q4 2025 but Nvidia goes to Q1 2026)
    all_quarters = sorted(
        set(cumulative_data.keys()) | set(flow_data.keys()),
        key=_quarter_sort_key,
    )
    prev = {}
    for q in all_quarters:
        if q in cumulative_data:
            for owner, chips in cumulative_data[q].items():
                prev.setdefault(owner, {}).update(chips)
        for owner, chips in prev.items():
            for chip, values in chips.items():
                cumulative_data.setdefault(q, {}).setdefault(owner, {}).setdefault(chip, values)

    viz_data = {'cumulative': cumulative_data, 'flow': flow_data}

    all_owners_set = set()
    all_chips_set = set()
    for view_data in [cumulative_data, flow_data]:
        for q_data in view_data.values():
            for owner, chips in q_data.items():
                all_owners_set.add(owner)
                all_chips_set.update(chips.keys())

    # Use predefined display order, filtering to only items present in data
    all_owners = [o for o in OWNER_ORDER if o in all_owners_set]
    all_chips = [c for c in CHIP_ORDER if c in all_chips_set]

    with open(TEMPLATE_PATH) as f:
        html = f.read()

    html = html.replace('__PLACEHOLDER_viz_json__', json.dumps(viz_data))
    html = html.replace('__PLACEHOLDER_colors_json__', json.dumps(CHIP_COLORS))
    html = html.replace('__PLACEHOLDER_owners_json__', json.dumps(all_owners))
    html = html.replace('__PLACEHOLDER_chips_json__', json.dumps(all_chips))
    html = html.replace('__PLACEHOLDER_quarters_json__', json.dumps(all_quarters))

    with open(OUTPUT_PATH, 'w') as f:
        f.write(html)

    print(f"Wrote {OUTPUT_PATH} ({len(all_quarters)} quarters, "
          f"{len(all_owners)} owners, {len(all_chips)} chip types)")


if __name__ == '__main__':
    generate()
