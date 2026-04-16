"""Point-estimate CoreWeave fleet buildout using IT power and Nvidia unit mix."""

from collections import OrderedDict
from pathlib import Path

import pandas as pd

from lab_compute_utils import convert_it_power_to_chips


SCRIPT_DIR = Path(__file__).resolve().parent
NVIDIA_TIMELINE_CSV = SCRIPT_DIR / 'nvidia_calendar_quarter_chip_timelines.csv'

# CoreWeave-specific assumptions from the notebook.
CHIP_SPECS = {
    'A100': {'IT_power': 926, 'H100e': 0.33},
    'H100/H200': {'IT_power': 1472, 'H100e': 1.0},
    'B200': {'IT_power': 2083, 'H100e': 2.5265},
    'B300': {'IT_power': 2222, 'H100e': 2.5265},
}

ALLOWED_NVIDIA_CHIPS = {'A100', 'H100', 'B200', 'B300'}
CSV_TO_HELPER_CHIP_NAME = {
    'A100': 'A100',
    'H100': 'H100/H200',
    'B200': 'B200',
    'B300': 'B300',
}
OUTPUT_CHIPS = ['H100/H200', 'B200', 'B300']

# Notebook assumption: 2023 and 2024 deployments were all H100.
ANNUAL_POWER_ADDITIONS_MW = OrderedDict([
    ('2023', 44_158 * CHIP_SPECS['H100/H200']['IT_power'] / 1e6),
    ('2024', 197_011 * CHIP_SPECS['H100/H200']['IT_power'] / 1e6),
])
ANNUAL_CHIP_COMPOSITION = {
    '2023': {'H100/H200': 1.0},
    '2024': {'H100/H200': 1.0},
}

QUARTERLY_POWER_ADDITIONS_MW = OrderedDict([
    ('Q1 2025', 60.0),
    ('Q2 2025', 50.0),
    ('Q3 2025', 120.0),
    ('Q4 2025', 260.0),
])

# Deterministic lag assumptions from the notebook's point-estimate version.
# Each entry is:
#   CoreWeave deployment quarter -> (anchor Nvidia quarter, lag in quarters)
#
# The lag says how far back to look for the Nvidia sales mix that best matches
# the chips being deployed by CoreWeave in that quarter.
# Examples:
# - 1.0 means use the anchor quarter directly.
# - 0.8 means use a mix that is 80% of a quarter earlier than the anchor, so
#   the script interpolates between the anchor quarter and the next quarter.
# - 0.4 means use a mix that is 40% of a quarter earlier than the anchor, which
#   puts even more weight on the next quarter's mix.
#
# Concretely:
# - Q1 2025 uses Q4 2024 with a 1.0-quarter lag, so it takes the Q4 2024 mix.
# - Q2 2025 uses Q1 2025 with a 1.0-quarter lag, so it takes the Q1 2025 mix.
# - Q3 2025 uses Q2 2025 with a 0.8-quarter lag, so it blends Q2 2025 and Q3 2025.
# - Q4 2025 uses Q3 2025 with a 0.4-quarter lag, so it blends Q3 2025 and Q4 2025.
LAG_ASSUMPTIONS = {
    'Q1 2025': ('Q4 2024', 1.0),
    'Q2 2025': ('Q1 2025', 1.0),
    'Q3 2025': ('Q2 2025', 0.8),
    'Q4 2025': ('Q3 2025', 0.4),
}

MW_2026_TOTAL = 850.0
Q1_2026_SHARE_LOW = 6 / 35
Q1_2026_SHARE_HIGH = 7 / 30
Q1_2026_LAG = 0.4


def load_nvidia_unit_counts():
    """Load Nvidia quarterly unit counts for the allowed chip set."""
    df = pd.read_csv(NVIDIA_TIMELINE_CSV)
    df['Start date'] = pd.to_datetime(df['Start date'])
    df['Quarter'] = df['Start date'].dt.to_period('Q').dt.strftime('Q%q %Y')

    df = df[df['Chip manufacturer'] == 'Nvidia'].copy()
    df = df[df['Chip type'].isin(ALLOWED_NVIDIA_CHIPS)].copy()
    df['Chip type'] = df['Chip type'].map(CSV_TO_HELPER_CHIP_NAME)

    grouped = (
        df.groupby(['Quarter', 'Chip type'], as_index=False)
        .agg(number_of_units=('Number of Units', 'sum'))
    )

    lookup = {}
    for quarter, quarter_df in grouped.groupby('Quarter'):
        lookup[quarter] = {
            row['Chip type']: float(row['number_of_units'])
            for _, row in quarter_df.iterrows()
        }
    return lookup


def interpolate_unit_mix(unit_lookup, quarter_low, quarter_high, weight_high):
    """Interpolate chip unit counts between two quarters, then normalize."""
    interpolated_units = {}
    for chip in CHIP_SPECS:
        units_low = unit_lookup.get(quarter_low, {}).get(chip, 0.0)
        units_high = unit_lookup.get(quarter_high, {}).get(chip, 0.0)
        interpolated_units[chip] = (
            (1.0 - weight_high) * units_low + weight_high * units_high
        )

    # CoreWeave 2025 mix should only use Hopper and Blackwell chips from the CSV.
    included_units = {
        chip: interpolated_units.get(chip, 0.0)
        for chip in OUTPUT_CHIPS
        if interpolated_units.get(chip, 0.0) > 0
    }
    total_units = sum(included_units.values())
    if total_units <= 0:
        raise ValueError(
            f'No allowed Nvidia unit counts found for {quarter_low} -> {quarter_high}'
        )

    return {
        chip: units / total_units
        for chip, units in included_units.items()
    }


def build_2025_compositions(unit_lookup):
    """Build the 2025 CoreWeave unit compositions from Nvidia quarterly data."""
    quarter_order = ['Q4 2024', 'Q1 2025', 'Q2 2025', 'Q3 2025', 'Q4 2025']
    quarter_index = {quarter: i for i, quarter in enumerate(quarter_order)}

    compositions = {}
    for coreweave_quarter, (anchor_quarter, lag_quarters) in LAG_ASSUMPTIONS.items():
        anchor_idx = quarter_index[anchor_quarter]
        float_idx = max(
            0.0,
            min(anchor_idx - (lag_quarters - 1.0), len(quarter_order) - 1),
        )
        idx_low = int(float_idx)
        idx_high = min(idx_low + 1, len(quarter_order) - 1)
        weight_high = float_idx - idx_low

        compositions[coreweave_quarter] = interpolate_unit_mix(
            unit_lookup=unit_lookup,
            quarter_low=quarter_order[idx_low],
            quarter_high=quarter_order[idx_high],
            weight_high=weight_high,
        )

    return compositions


def summarize_rows(df):
    """Summarize helper output rows into chip counts, total chips, and H100e."""
    chips_added = df.groupby('Chip Type')['Chips Added'].sum()
    h100e_added = df['H100e Added'].sum()

    metrics = OrderedDict()
    for chip in OUTPUT_CHIPS:
        metrics[chip] = float(chips_added.get(chip, 0.0))
    metrics['Total'] = sum(metrics[chip] for chip in OUTPUT_CHIPS)
    metrics['H100e'] = float(h100e_added)

    return metrics


def combine_metric_dicts(*metric_dicts):
    """Add metric dictionaries together by key."""
    combined = OrderedDict((metric, 0.0) for metric in ['H100/H200', 'B200', 'B300', 'Total', 'H100e'])
    for metric_dict in metric_dicts:
        for metric in combined:
            combined[metric] += metric_dict.get(metric, 0.0)
    return combined


def metric_dict_to_frame(metric_dict):
    return pd.DataFrame({
        'Metric': list(metric_dict.keys()),
        'Point estimate': list(metric_dict.values()),
    })


def format_metric_frame(metric_frame):
    display = metric_frame.copy()
    display['Point estimate'] = display['Point estimate'].map(lambda x: f'{x:,.0f}')
    return display.to_string(index=False)


def print_section(title, metric_dict):
    print('\n' + '=' * 80)
    print(f'  {title}')
    print('=' * 80)
    print(format_metric_frame(metric_dict_to_frame(metric_dict)))


def main():
    unit_lookup = load_nvidia_unit_counts()
    quarterly_compositions = build_2025_compositions(unit_lookup)

    annual_df = convert_it_power_to_chips(
        power_added_mw_by_period=ANNUAL_POWER_ADDITIONS_MW,
        chip_specs=CHIP_SPECS,
        chip_composition_by_period=ANNUAL_CHIP_COMPOSITION,
    )
    annual_2023_metrics = summarize_rows(annual_df[annual_df['Period'] == '2023'])
    annual_2024_metrics = summarize_rows(annual_df[annual_df['Period'] == '2024'])

    print_section('2023 DEPLOYMENTS (point estimate)', annual_2023_metrics)
    print_section('2024 DEPLOYMENTS (point estimate)', annual_2024_metrics)

    quarterly_df = convert_it_power_to_chips(
        power_added_mw_by_period=QUARTERLY_POWER_ADDITIONS_MW,
        chip_specs=CHIP_SPECS,
        chip_composition_by_period=quarterly_compositions,
    )

    print('\n' + '=' * 80)
    print('  2025 QUARTERLY DEPLOYMENTS')
    print('=' * 80)
    quarterly_metrics = {}
    for quarter in QUARTERLY_POWER_ADDITIONS_MW:
        quarter_metrics = summarize_rows(quarterly_df[quarterly_df['Period'] == quarter])
        quarterly_metrics[quarter] = quarter_metrics
        print(f'\n{quarter}')
        print(format_metric_frame(metric_dict_to_frame(quarter_metrics)))

    q1_2026_share_midpoint = (Q1_2026_SHARE_LOW + Q1_2026_SHARE_HIGH) / 2
    q1_2026_power_mw = MW_2026_TOTAL * q1_2026_share_midpoint * Q1_2026_LAG
    q1_2026_composition = interpolate_unit_mix(
        unit_lookup=unit_lookup,
        quarter_low='Q4 2025',
        quarter_high='Q4 2025',
        weight_high=0.0,
    )
    q1_2026_df = convert_it_power_to_chips(
        power_added_mw_by_period=OrderedDict([('Q1 2026 (pre-purchased)', q1_2026_power_mw)]),
        chip_specs=CHIP_SPECS,
        chip_composition_by_period=OrderedDict([('Q1 2026 (pre-purchased)', q1_2026_composition)]),
    )
    q1_2026_metrics = summarize_rows(q1_2026_df)

    print_section('Q1 2026 PRE-PURCHASED INVENTORY (held at end of 2025)', q1_2026_metrics)

    deployed_2025_total = combine_metric_dicts(*quarterly_metrics.values())
    print_section('2025 DEPLOYED TOTAL', deployed_2025_total)

    end_of_2025_inventory = combine_metric_dicts(deployed_2025_total, q1_2026_metrics)
    print_section('END-OF-2025 INVENTORY (deployed + pre-purchased)', end_of_2025_inventory)

    grand_total = combine_metric_dicts(
        annual_2023_metrics,
        annual_2024_metrics,
        deployed_2025_total,
        q1_2026_metrics,
    )
    print_section('GRAND TOTAL incl. 2023/2024 (deployed + pre-purchased)', grand_total)


if __name__ == '__main__':
    main()
