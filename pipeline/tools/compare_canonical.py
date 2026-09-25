#!/usr/bin/env python3.11
"""Check a family's canonical export against the legacy CSVs it replaces.

    python3.11 -m pipeline.tools.compare_canonical tpu [--tol 1.0]

Rows are matched on designer, quarter, chip type and owner, never on Name, since the
two sides use different Name templates. For each metric the legacy files carry, prints
how many matched rows agree within the tolerance (percent), then the worst mismatches
and any rows present on only one side.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import chip_schema as cs  # noqa: E402

# family -> canonical table -> legacy files that hold the same rows.
LEGACY = {
    "nvidia": {
        "sales_quarterly_by_chip": ["csv_export/nvidia_calendar_quarter_chip_timelines.csv"],
        "sales_cumulative_by_chip": ["csv_export/nvidia_cumulative_by_chip.csv"],
        "sales_cumulative_by_designer": ["csv_export/nvidia_cumulative_totals.csv"],
    },
    "nvidia_owners": {
        "owners_quarterly_by_chip": ["owners_csv_export/nvidia_owners_quarters_by_chip.csv"],
        "owners_cumulative_by_chip": ["owners_csv_export/nvidia_owners_cumulative_by_chip.csv"],
        "owners_cumulative_by_designer": ["owners_csv_export/nvidia_owners_cumulative_totals.csv"],
    },
    "amd": {
        "sales_quarterly_by_chip": ["csv_export/amd_chip_timelines.csv"],
        "sales_cumulative_by_chip": ["csv_export/amd_cumulative_by_chip.csv"],
        "sales_cumulative_by_designer": ["csv_export/amd_cumulative_totals.csv"],
        "owners_quarterly_by_chip": ["owners_csv_export/amd_owners_quarters_by_chip.csv"],
        "owners_cumulative_by_chip": ["owners_csv_export/amd_owners_cumulative_by_chip.csv"],
        "owners_cumulative_by_designer": ["owners_csv_export/amd_owners_cumulative_totals.csv"],
    },
    "tpu": {
        "sales_quarterly_by_chip": ["csv_export/tpu_calendar_quarter_chip_timelines.csv"],
        "sales_cumulative_by_chip": ["csv_export/tpu_cumulative_by_chip.csv"],
        "sales_cumulative_by_designer": ["csv_export/tpu_cumulative_totals.csv"],
        "owners_quarterly_by_chip": ["owners_csv_export/tpu_owners_quarters_by_chip.csv"],
        "owners_cumulative_by_chip": ["owners_csv_export/tpu_owners_cumulative_by_chip.csv"],
        "owners_cumulative_by_designer": ["owners_csv_export/tpu_owners_cumulative_totals.csv"],
    },
    "trainium": {
        "sales_quarterly_by_chip": ["csv_export/trainium_chip_timelines.csv"],
        "sales_cumulative_by_chip": ["csv_export/trainium_cumulative_by_chip.csv"],
        "sales_cumulative_by_designer": ["csv_export/trainium_cumulative_totals.csv"],
        "owners_quarterly_by_chip": ["owners_csv_export/trainium_owners_quarters_by_chip.csv"],
        "owners_cumulative_by_chip": ["owners_csv_export/trainium_owners_cumulative_by_chip.csv"],
        "owners_cumulative_by_designer": ["owners_csv_export/trainium_owners_cumulative_totals.csv"],
    },
}


def _first(df, *names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def legacy_frame(paths):
    """Read legacy CSVs into canonical keys plus the median of each metric they carry."""
    frames = []
    for p in paths:
        df = pd.read_csv(p, dtype=str)
        out = pd.DataFrame({
            "designer": df["Chip manufacturer"],
            "quarter": pd.to_datetime(df["End date"], format="mixed").map(cs.quarter_of),
            "chip_type": df["Chip type"].map(cs.canonical_chip) if "Chip type" in df else "",
            "owner": df["Owner"].map(cs.canonical_owner) if "Owner" in df else "",
        })
        out["units_med"] = _first(df, "Number of units (median)", "Number of Units (median)", "Number of Units")
        out["h100e_med"] = _first(df, "Compute estimate in H100e (median)", "H100e compute power (median)")
        power = _first(df, "Power in MW (median)")
        out["power_mw_med"] = power.fillna(_first(df, "Total TDP (W)", "Total TDP (W) (median)") / 1e6)
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def compare(family, tol):
    ok = True
    for table, paths in LEGACY[family].items():
        path = cs.CANONICAL_EXPORT_DIR / family / f"{table}.csv"
        if not path.exists():
            print(f"\n{table}: no canonical file at {path}")
            ok = False
            continue
        new = pd.read_csv(path, dtype={"chip_type": str, "owner": str})
        new["quarter"] = new["end_date"].map(cs.quarter_of)
        for col in ("chip_type", "owner"):
            if col not in new:
                new[col] = ""
        new[["chip_type", "owner"]] = new[["chip_type", "owner"]].fillna("")
        old = legacy_frame(paths)
        if "_by_chip" not in table:
            old["chip_type"] = ""
        keys = ["designer", "quarter", "chip_type", "owner"]
        m = new.merge(old, on=keys, how="outer", suffixes=("", "_legacy"), indicator=True)
        both = m[m["_merge"] == "both"]
        print(f"\n{table}: {len(both)} matched, {int((m['_merge'] == 'left_only').sum())} canonical-only, "
              f"{int((m['_merge'] == 'right_only').sum())} legacy-only")
        for metric in ("units_med", "h100e_med", "power_mw_med"):
            legacy = both[f"{metric}_legacy"]
            have = legacy.notna()
            if not have.any():
                continue
            diff = (both.loc[have, metric] - legacy[have]).abs()
            pct = diff / legacy[have].abs().clip(lower=1) * 100
            # Legacy exports truncated to whole numbers, so a gap under 1 is rounding.
            n_ok = int(((pct <= tol) | (diff < 1)).sum())
            print(f"  {metric:<13} {n_ok}/{int(have.sum())} within {tol}%  (max diff {pct.max():.2f}%)")
            if n_ok < have.sum():
                ok = False
                worst = both.loc[have].assign(pct=pct.where(diff >= 1, 0)).nlargest(3, "pct")
                for _, r in worst.iterrows():
                    print(f"      {r['quarter']} {r['chip_type'] or '-'} {r['owner'] or ''}: "
                          f"{r[metric]:,.2f} vs legacy {r[f'{metric}_legacy']:,.2f} ({r['pct']:.1f}%)")
        for side, label in (("left_only", "canonical-only"), ("right_only", "legacy-only")):
            extra = m[m["_merge"] == side]
            if len(extra):
                ok = False if side == "right_only" else ok
                sample = extra[keys].head(4).apply(lambda r: " ".join(filter(None, r.astype(str))), axis=1).tolist()
                print(f"  {label}: {sample}{' ...' if len(extra) > 4 else ''}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("family", choices=sorted(LEGACY))
    ap.add_argument("--tol", type=float, default=1.0)
    args = ap.parse_args()
    sys.exit(0 if compare(args.family, args.tol) else 1)
