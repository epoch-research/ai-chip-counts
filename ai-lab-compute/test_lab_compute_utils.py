"""Sanity-check script for lab_compute_utils.convert_it_power_to_chips."""

import pandas as pd

from lab_compute_utils import convert_it_power_to_chips


CHIP_SPECS = {
    'A100':      {'IT_power':   926, 'H100e': 0.33},
    'H100/H200': {'IT_power':  1389, 'H100e': 1.10},
    'B200':      {'IT_power':  2083, 'H100e': 2.50},
    'B300':      {'IT_power':  2222, 'H100e': 2.50},
}

# Three year-end periods, mimicking the Microsoft/OpenAI-style buildout.
POWER_ADDED_MW = {
    '2023-12-31':  200.0,   # first period — 200 MW deployed
    '2024-12-31':  400.0,   # +400 MW in 2024
    '2025-12-31': 1300.0,   # +1300 MW in 2025
}

# Composition of chips added IN EACH PERIOD (shares of new units that period).
COMPOSITION = {
    '2023-12-31': {'A100': 0.60, 'H100/H200': 0.40},
    '2024-12-31': {'A100': 0.05, 'H100/H200': 0.85, 'B200': 0.10},
    '2025-12-31': {'H100/H200': 0.30, 'B200': 0.50, 'B300': 0.20},
}


def main():
    df = convert_it_power_to_chips(
        power_added_mw_by_period=POWER_ADDED_MW,
        chip_specs=CHIP_SPECS,
        chip_composition_by_period=COMPOSITION,
    )

    print('=== Full per-period × per-chip results ===')
    pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
    print(df.to_string(index=False))

    print('\n=== Cumulative H100e by period (totals across chip types) ===')
    totals = df.groupby('Period').agg(
        total_power_added_mw=('Power Added (MW)', 'sum'),
        total_chips_cumulative=('Cumulative Chips', 'sum'),
        total_h100e_cumulative=('Cumulative H100e', 'sum'),
    )
    print(totals.to_string())

    # Sanity checks
    for period, expected_mw in POWER_ADDED_MW.items():
        got = df.loc[df['Period'] == period, 'Power Added (MW)'].sum()
        assert abs(got - expected_mw) < 1e-6, (
            f"{period}: allocated power {got} != input {expected_mw}"
        )
    print('\nOK: per-period allocated power matches inputs.')

    for period, expected_comp in COMPOSITION.items():
        period_df = df.loc[df['Period'] == period].set_index('Chip Type')
        for chip, expected_share in expected_comp.items():
            got = period_df.loc[chip, 'Share of Units Added']
            assert abs(got - expected_share) < 1e-6, (
                f"{period} {chip}: unit share {got} != input {expected_share}"
            )
    print('OK: per-period unit shares match inputs.')

    # The function should derive power shares from unit shares plus IT power.
    expected_power_shares_2023 = {
        'A100': (0.60 * 926) / (0.60 * 926 + 0.40 * 1389),
        'H100/H200': (0.40 * 1389) / (0.60 * 926 + 0.40 * 1389),
    }
    period_df = df.loc[df['Period'] == '2023-12-31'].set_index('Chip Type')
    for chip, expected_share in expected_power_shares_2023.items():
        got = period_df.loc[chip, 'Share of Power Added']
        assert abs(got - expected_share) < 1e-6, (
            f"2023-12-31 {chip}: power share {got} != expected {expected_share}"
        )
    print('OK: power shares are derived from unit shares and IT power.')

    # Schema enforcement demo. This should stop execution when the input is invalid.
    print('\n=== Schema enforcement (expected to raise and stop here) ===')
    bad_composition = {
        '2023-12-31': {'A100': 0.60, 'H100/H200': 0.30},  # sums to 0.9
        '2024-12-31': {'A100': 0.05, 'H100/H200': 0.85, 'B200': 0.10},
        '2025-12-31': {'H100/H200': 0.30, 'B200': 0.50, 'B300': 0.20},
    }
    convert_it_power_to_chips(POWER_ADDED_MW, CHIP_SPECS, bad_composition)


if __name__ == '__main__':
    main()
