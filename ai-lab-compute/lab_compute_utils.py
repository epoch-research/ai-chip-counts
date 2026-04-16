"""Helpers for modeling AI lab compute fleet buildout."""

import numpy as np
import pandas as pd


def convert_it_power_to_chips(
    power_added_mw_by_period,
    chip_specs,
    chip_composition_by_period,
    tolerance=1e-6,
):
    """Run an incremental fleet buildout model.

    In each period, some new IT power (MW) is deployed and allocated across chip
    types by a composition of new units (shares summing to 1). The function
    converts those unit shares into power shares using each chip's IT power,
    then converts deployed power into chip counts and H100-equivalents.
    Cumulative totals carry across periods; chips deployed in an earlier period
    are never retroactively reallocated.

    Args:
        power_added_mw_by_period: mapping of period key → new IT power (MW) in
            that period. Period keys can be any sortable/hashable type (dates,
            quarter labels, ints).
        chip_specs: mapping of chip name → {"IT_power": float, "H100e": float}.
        composition_by_period: mapping of period key → {chip name: share of
            new units added in that period}. Shares must sum to 1 (or to 0 if
            no additions that period). Chips omitted from a period's
            composition are treated as 0.
        tolerance: numerical tolerance for the composition-sum check.

    Returns:
        DataFrame with one row per (period, chip), ordered by period then by
        the order of chips in `chip_specs`.
    """
    periods = sorted(power_added_mw_by_period.keys())
    chips = list(chip_specs.keys())

    for chip, spec in chip_specs.items():
        missing = {'IT_power', 'H100e'} - set(spec.keys())
        if missing:
            raise ValueError(f"chip_specs[{chip!r}] missing keys: {missing}")
        if spec['IT_power'] <= 0:
            raise ValueError(f"chip_specs[{chip!r}].IT_power must be > 0")

    if set(chip_composition_by_period.keys()) != set(periods):
        raise ValueError(
            "composition_by_period keys must exactly match "
            "power_added_mw_by_period keys"
        )

    for period, comp in chip_composition_by_period.items():
        unknown = set(comp.keys()) - set(chips)
        if unknown:
            raise ValueError(
                f"period {period!r}: composition references unknown chips {unknown}"
            )
        total = sum(comp.values())
        sums_to_one = np.isclose(total, 1.0, atol=tolerance)
        sums_to_zero = np.isclose(total, 0.0, atol=tolerance)
        if not (sums_to_one or sums_to_zero):
            raise ValueError(
                f"period {period!r}: composition shares sum to {total:.6f}, "
                "expected 1.0 (or 0.0 if no additions)"
            )

    cumulative_chips = {chip: 0.0 for chip in chips}
    cumulative_h100e = {chip: 0.0 for chip in chips}

    rows = []
    for period in periods:
        new_power_mw = power_added_mw_by_period[period]
        comp = chip_composition_by_period[period]
        weighted_average_it_power = sum(
            comp.get(chip, 0.0) * chip_specs[chip]['IT_power']
            for chip in chips
        )

        for chip in chips:
            share_of_new_units = comp.get(chip, 0.0)
            it_power = chip_specs[chip]['IT_power']
            h100e_per_gpu = chip_specs[chip]['H100e']

            if np.isclose(weighted_average_it_power, 0.0, atol=tolerance):
                share_of_power_added = 0.0
            else:
                # More power-hungry chips get a larger share of the same unit mix.
                share_of_power_added = (
                    share_of_new_units * it_power / weighted_average_it_power
                )

            chip_power_mw = new_power_mw * share_of_power_added
            new_chips = chip_power_mw * 1e6 / it_power
            new_h100e = new_chips * h100e_per_gpu

            cumulative_chips[chip] += new_chips
            cumulative_h100e[chip] += new_h100e

            rows.append({
                'Period': period,
                'Chip Type': chip,
                'Share of Units Added': share_of_new_units,
                'Share of Power Added': share_of_power_added,
                'Power Added (MW)': chip_power_mw,
                'Chips Added': new_chips,
                'H100e Added': new_h100e,
                'Cumulative Chips': cumulative_chips[chip],
                'Cumulative Power (MW)': cumulative_chips[chip] * it_power / 1e6,
                'Cumulative H100e': cumulative_h100e[chip],
            })

    return pd.DataFrame(rows)
