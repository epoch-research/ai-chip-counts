#!/usr/bin/env python3.11
"""Assemble staging/, the stand-in for the Airtable base, from every canonical source.

    python3.11 -m pipeline.build_staging

staging/ holds the six results tables in their public form (display column names, one
Name template per table), plus the chip_types and organizations reference tables. It is
what the Airtable base would hold after a push, and what the HTML hub in hub/ reads.

Rows come from three kinds of source:

1. canonical_export/<family>/, written by the models from their Monte Carlo samples
   (see chip_schema.py). Most rows come from here.
2. Legacy exports with no samples behind them, converted here on every run:
   the Nvidia "Other" remainder (nvidia_owners_other.ipynb refits distributions to
   percentiles rather than carrying samples) and CoreWeave (point estimates).
3. staging/curated/, rows no model produces and that are maintained by hand, as they
   are in Airtable today: Huawei, Cambricon, xAI and smuggled China. They were
   snapshotted from the published data by pipeline/tools/snapshot_curated_rows.py.

The build refuses to write if two sources claim the same row, a chip type is unknown,
or a table breaks the schema. Legacy exports and the current Airtable workflow are
untouched by all of this.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path

# Runnable as a script or with -m from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pipeline import chip_schema as cs  # noqa: E402

STAGING = cs.REPO / "staging"
CURATED = STAGING / "curated"

# Legacy sources converted on every build: family -> {canonical table: legacy CSV}.
LEGACY_ONLY = {
    "nvidia_other": {
        "owners_quarterly_by_chip": "owners_csv_export/nvidia_owners_OTHER_quarters_by_chip.csv",
        "owners_cumulative_by_chip": "owners_csv_export/nvidia_owners_OTHER_cumulative_by_chip.csv",
        "owners_cumulative_by_designer": "owners_csv_export/nvidia_owners_OTHER_cumulative_totals.csv",
    },
    "coreweave": {
        "owners_quarterly_by_chip": "owners_csv_export/coreweave/coreweave_owners_quarters_by_chip.csv",
        "owners_cumulative_by_chip": "owners_csv_export/coreweave/coreweave_owners_cumulative_by_chip.csv",
        "owners_cumulative_by_designer": "owners_csv_export/coreweave/coreweave_owners_cumulative_totals.csv",
    },
}

# Every spelling the legacy files and the published tables use for each canonical
# column, first match wins.
LEGACY_COLUMNS = {
    "units_med": ["Number of units (median)", "Number of Units (median)", "Number of Units"],
    "units_p5": ["Number of units (5th percentile)", "Number of Units (5th percentile)"],
    "units_p95": ["Number of units (95th percentile)", "Number of Units (95th percentile)"],
    "h100e_med": ["Compute estimate in H100e (median)", "H100e compute power (median)"],
    "h100e_p5": ["Compute estimate in H100e (5th percentile)", "H100e (5th percentile)",
                 "H100e compute power (5th percentile)"],
    "h100e_p95": ["Compute estimate in H100e (95th percentile)", "H100e (95th percentile)",
                  "H100e compute power (95th percentile)"],
    "power_mw_med": ["Power in MW (median)"],
    "power_mw_p5": ["Power in MW (5th percentile)"],
    "power_mw_p95": ["Power in MW (95th percentile)"],
}
LEGACY_TDP_W = {"med": ["Total TDP (W) (median)", "Total TDP (W)"], "p5": ["Total TDP (W) (5th percentile)"],
                "p95": ["Total TDP (W) (95th percentile)"]}


def _col(df, names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce")
    return pd.Series(np.nan, index=df.index)


# "Estimates generated on: 04-22-2026 11:57", the legacy exporters' stamp.
LEGACY_STAMP = re.compile(r"Estimates generated on:?\s*(\d{1,2})-(\d{1,2})-(\d{4}) (\d{1,2}):(\d{2})")


def source_notes(text, note=None):
    """Keep a source row's own Notes, with its generation stamp in the canonical format.

    Converted rows are re-read on every build, so stamping them with the build time would
    mark them changed each run and misstate when the numbers were produced. The source's
    stamp is the true one; a row without one gets none. `note` is appended, for provenance.
    """
    text = "" if pd.isna(text) else str(text).strip()
    text = LEGACY_STAMP.sub(lambda m: f"Generated on {m[3]}-{int(m[1]):02d}-{int(m[2]):02d} {int(m[4]):02d}:{m[5]}", text)
    return " ".join(x for x in (text, note) if x)


def canonical_from_legacy(df, table, note=None):
    """Convert legacy or published rows (percentiles only) into a canonical table.

    Units and H100e are taken as given. Power comes from the row, or from its total TDP.
    Cost is units times the chip's price, which is exact per chip because a percentile
    scales with a constant. Designer-level cost has no per-chip breakdown, so it is the
    sum of the matching per-chip rows; that sum is exact for the median only when the
    chips move together, and the note on the row says so.
    """
    if df.empty:
        return pd.DataFrame(columns=cs.table_columns(table))
    specs = cs.chip_specs()
    out = pd.DataFrame(index=df.index)
    out["designer"] = df.get("Designer", df.get("Chip manufacturer"))
    if table.startswith("owners_"):
        out["owner"] = df["Owner"].map(cs.canonical_owner)
    if table.endswith("_by_chip"):
        out["chip_type"] = df["Chip type"].map(cs.canonical_chip)
    end = pd.to_datetime(df["End date"], format="mixed")
    quarter = end.map(cs.quarter_of)
    if "_quarterly_" in table:
        out["start_date"] = quarter.map(lambda q: cs.quarter_bounds(q)[0].isoformat())
    else:
        start = df["Series start date"] if "Series start date" in df else df["Start date"]
        out["series_start"] = pd.to_datetime(start, format="mixed").dt.strftime("%Y-%m-%d")
    out["end_date"] = quarter.map(lambda q: cs.quarter_bounds(q)[1].isoformat())

    for c, names in LEGACY_COLUMNS.items():
        out[c] = _col(df, [cs.DISPLAY_NAMES[c], *names])
    for s, names in LEGACY_TDP_W.items():
        out[f"power_mw_{s}"] = out[f"power_mw_{s}"].fillna(_col(df, names) / 1e6)
    # A median with no interval stays that way: filling p5 and p95 with the median would
    # present a point estimate as if it had no uncertainty.

    for s in cs.STATS:
        if "chip_type" in out:
            price = out["chip_type"].map(lambda c: specs[c]["price_usd"])
            out[f"cost_usd_{s}"] = (out[f"units_{s}"] * price).round()
        else:
            out[f"cost_usd_{s}"] = np.nan
    inc = df.get("Incomplete", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    out["incomplete"] = inc.isin(["true", "checked", "1", "yes"])
    out["source"] = df.get("Source", df.get("Source / Link", pd.Series("", index=df.index))).fillna("")
    out["notes"] = [source_notes(n, note) for n in df.get("Notes", pd.Series("", index=df.index))]

    owner = out["owner"] if "owner" in out else None
    chip = out["chip_type"] if "chip_type" in out else None
    first = out["series_start"].map(cs.quarter_of) if "series_start" in out else pd.Series(None, index=out.index)
    out["name"] = [cs.row_name(table, out.at[i, "designer"], quarter[i], None if chip is None else chip[i],
                               None if owner is None else owner[i], first[i]) for i in out.index]
    return out[cs.table_columns(table)].reset_index(drop=True)


def fill_designer_cost(tables):
    """Designer-level cost for converted rows: the sum of their per-chip cost rows."""
    for prefix in ("sales", "owners"):
        des, chip = tables.get(f"{prefix}_cumulative_by_designer"), tables.get(f"{prefix}_cumulative_by_chip")
        if des is None or chip is None or des.empty or chip.empty:
            continue
        keys = ["designer", "end_date"] + (["owner"] if prefix == "owners" else [])
        sums = chip.groupby(keys)[[f"cost_usd_{s}" for s in cs.STATS]].sum(min_count=1).reset_index()
        merged = des.drop(columns=[f"cost_usd_{s}" for s in cs.STATS]).merge(sums, on=keys, how="left")
        merged["notes"] = np.where(merged["cost_usd_med"].notna(),
                                   merged["notes"] + " Cost interval summed from per-chip intervals.", merged["notes"])
        tables[f"{prefix}_cumulative_by_designer"] = merged[cs.table_columns(f"{prefix}_cumulative_by_designer")]
    return tables


def convert_legacy_only():
    for family, files in LEGACY_ONLY.items():
        note = f"Converted from the percentile-only CSVs in {Path(next(iter(files.values()))).parent}/."
        tables = {t: canonical_from_legacy(pd.read_csv(cs.REPO / f, dtype=str), t, note) for t, f in files.items()}
        cs.write_tables(family, fill_designer_cost(tables))


def load_sources():
    """{table: [(source label, canonical frame), ...]} across canonical_export and curated rows."""
    sources = {t: [] for t in cs.TABLES}
    for folder in sorted(p for p in cs.CANONICAL_EXPORT_DIR.iterdir() if p.is_dir()):
        for t in cs.TABLES:
            f = folder / f"{t}.csv"
            if f.exists():
                sources[t].append((f"canonical_export/{folder.name}", pd.read_csv(f, dtype=str, keep_default_na=False)))
    for t in cs.TABLES:
        f = CURATED / f"{t}.csv"
        if f.exists():
            sources[t].append(("staging/curated", cs.from_display(pd.read_csv(f, dtype=str, keep_default_na=False))))
    return sources


def merge(table, parts):
    frames = [df.assign(_source=label) for label, df in parts if len(df)]
    if not frames:
        return pd.DataFrame(columns=cs.table_columns(table)), {}
    df = pd.concat(frames, ignore_index=True)
    keys = [k for k in ("designer", "owner", "chip_type", "end_date") if k in df]
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        clash = df[dup].sort_values(keys)[keys + ["_source"]].head(10).to_string(index=False)
        raise SystemExit(f"{table}: two sources claim the same rows:\n{clash}")
    counts = df["_source"].value_counts().to_dict()
    sort = ["designer", "end_date"] + [k for k in ("owner", "chip_type") if k in df]
    df = df.sort_values(sort).drop(columns=["_source"])[cs.table_columns(table)]
    for c in cs.metric_columns():
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["incomplete"] = df["incomplete"].astype(str).str.lower().isin(["true", "1"])
    cs.validate(table, df.reset_index(drop=True))
    return df, counts


def main():
    convert_legacy_only()
    sources = load_sources()
    manifest = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "tables": {}}
    for t in cs.TABLES:
        df, counts = merge(t, sources[t])
        out = cs.to_display(df)
        out["Incomplete"] = out["Incomplete"].map({True: "true", False: ""})
        out.to_csv(STAGING / f"{t}.csv", index=False)
        manifest["tables"][t] = {"rows": len(df), "sources": counts}
        print(f"{t:<32} {len(df):>4} rows  " + ", ".join(f"{k.split('/')[-1]} {v}" for k, v in counts.items()))
    (STAGING / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # A single-file copy of everything above, for checking by eye without a server.
    from pipeline.tools.build_viewer import build as build_viewer
    build_viewer()
    return 0


if __name__ == "__main__":
    sys.exit(main())
