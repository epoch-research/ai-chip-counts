#!/usr/bin/env python3.11
"""Copy the rows no model produces out of the published data into staging/curated/.

    python3.11 -m pipeline.tools.snapshot_curated_rows [--mirror PATH]

Huawei and Cambricon (every table), xAI and smuggled China (owners tables) exist only in
Airtable today, typed in by hand. This reads them from the website's hourly mirror of the
Airtable views, converts them to the canonical schema, and writes them in display form to
staging/curated/<table>.csv, where they are maintained by hand from then on. Run it once;
rerunning overwrites any hand edits.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import chip_schema as cs  # noqa: E402
from pipeline.build_staging import CURATED, canonical_from_legacy, fill_designer_cost  # noqa: E402

MIRROR = Path.home() / "Desktop/Scripts/epoch-website-astro/src/public/data/generated"
PUBLISHED = {
    "sales_quarterly_by_chip": "ai_chip_sales_timelines_by_chip.csv",
    "sales_cumulative_by_chip": "ai_chip_sales_cumulative_timelines.csv",
    "sales_cumulative_by_designer": "ai_chip_sales_cumulative_timelines_by_designer.csv",
    "owners_quarterly_by_chip": "ai_chip_owners_quarters_by_chip_type.csv",
    "owners_cumulative_by_chip": "ai_chip_owners_cumulative_by_chip_type.csv",
    "owners_cumulative_by_designer": "ai_chip_owners_cumulative_by_designer.csv",
}
CURATED_DESIGNERS = {"Huawei", "Cambricon"}
CURATED_OWNERS = {"xAI", "China (smuggled)"}

# Typos found in the published data, corrected on the way in and noted on the row.
# (table, Name as published, column) -> (published value, corrected value, reason)
FIXES = {
    ("owners_quarterly_by_chip", "China Cambricon Siyuan 590 Q2 2024", "H100e (95th percentile)"): (
        "23.0", "233",
        "H100e 95th percentile corrected from 23 to 233: below its own median, and the sales row for the "
        "same chip and quarter (China is Cambricon's only owner) says 233."),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mirror", type=Path, default=MIRROR)
    args = ap.parse_args()
    CURATED.mkdir(parents=True, exist_ok=True)
    tables = {}
    for table, fname in PUBLISHED.items():
        df = pd.read_csv(args.mirror / fname, dtype=str)
        keep = df["Chip manufacturer"].isin(CURATED_DESIGNERS)
        if "Owner" in df:
            keep |= df["Owner"].isin(CURATED_OWNERS)
        df = df[keep].reset_index(drop=True)
        fixed = {}
        for (t, name, col), (was, now, why) in FIXES.items():
            hit = (df["Name"] == name) if t == table else pd.Series(False, index=df.index)
            if hit.any():
                assert df.loc[hit, col].iloc[0] == was, f"{name}: expected {was}, found {df.loc[hit, col].iloc[0]}"
                df.loc[hit, col] = now
                fixed[hit.idxmax()] = why
        # Curated rows keep their own Notes: they are the rows' documentation in Airtable.
        tables[table] = canonical_from_legacy(df, table)
        for i, why in fixed.items():
            tables[table].loc[i, "notes"] += f" {why}"
    for table, df in fill_designer_cost(tables).items():
        cs.validate(table, df)
        out = cs.to_display(df)
        out["Incomplete"] = out["Incomplete"].map({True: "true", False: ""})
        out.to_csv(CURATED / f"{table}.csv", index=False)
        print(f"{table:<32} {len(df):>3} curated rows: {sorted(df['designer'].unique())}"
              + (f" owners {sorted(df['owner'].unique())}" if "owner" in df else ""))
