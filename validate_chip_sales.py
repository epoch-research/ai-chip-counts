#!/usr/bin/env python3
"""Validate local AI chip-sales model outputs against Epoch's published website data.

The notebooks in this repo (e.g. nvidia_chip_estimates.ipynb) run a Monte Carlo model and
write per-chip and cumulative estimates to csv_export/. Those CSVs are what ultimately feed
the public dataset at https://epoch.ai/data/ai-chip-sales. This module downloads the
currently published dataset (in memory, nothing is written to disk) and compares it, cell
by cell, against the local CSVs so we can catch drift, stale uploads, or accidental changes
before/after publishing.

For each designer it lines up three published tables with their local counterparts:

    Published (in the zip)                local CSV (csv_export/)
    ------------------------------------  --------------------------------------------
    timelines_by_chip                     <designer>_calendar_quarter_chip_timelines.csv
    cumulative_timelines                  <designer>_cumulative_by_chip.csv
    cumulative_timelines_by_designer      <designer>_cumulative_totals.csv

What it checks per table:
  1. Row coverage  - rows present locally but not on the website, and vice versa.
  2. Per-cell diff - percent difference for every numeric metric (units, H100e, power),
                     with each cell classified ok / minor / major against tolerances.
  3. Timestamp     - the "Estimates generated on" stamp in each file, to explain diffs
                     (e.g. the website table was generated from an older model run).

Nvidia, Google (TPU) and AMD are wired up today. To extend to Amazon (Trainium) etc.,
add an entry to DESIGNER_CONFIGS: the three published tables share the same layout across
designers, so only the local file paths, the website "Chip manufacturer" label, and any
chip-name aliases need to be supplied.

Usage:
    python validate_chip_sales.py                 # validate Nvidia, print a text report
    python validate_chip_sales.py --designer all  # validate every configured designer

The companion notebook validate_chip_sales.ipynb wraps the same functions and renders
the report inline: coverage/timestamp and per-metric summary tables, 2D percent-diff
tables (chip x quarter), and a cumulative time-series figure.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
import zipfile
import urllib.request
import urllib.error
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


WEBSITE_ZIP_URL = "https://epoch.ai/data/ai_chip_sales.zip"

# Cells whose percent difference is within TOL_PCT are considered a match; between TOL_PCT
# and WARN_PCT is "minor" drift; above WARN_PCT is "major" drift worth investigating.
DEFAULT_TOL_PCT = 1.0
DEFAULT_WARN_PCT = 10.0


# ---------------------------------------------------------------------------
# Table layouts (shared across all designers)
# ---------------------------------------------------------------------------
# Each entry describes one published table, whether rows are keyed by chip type, and the
# list of metrics to compare. Each metric is a triple of (local column name, website
# column name, short label). The column names differ between the two sides in places -
# most notably the cumulative-by-chip H100e columns - so they are spelled out explicitly
# rather than assumed equal.
TABLE_SPECS = {
    "timelines_by_chip": {
        "title": "Quarterly timelines by chip",
        "web_file": "timelines_by_chip.csv",
        "has_chip": True,
        "metrics": [
            ("Number of Units", "Number of Units", "Units median"),
            ("Number of Units (5th percentile)", "Number of Units (5th percentile)", "Units p5"),
            ("Number of Units (95th percentile)", "Number of Units (95th percentile)", "Units p95"),
            ("Compute estimate in H100e (median)", "Compute estimate in H100e (median)", "H100e median"),
            ("H100e (5th percentile)", "H100e (5th percentile)", "H100e p5"),
            ("H100e (95th percentile)", "H100e (95th percentile)", "H100e p95"),
        ],
    },
    "cumulative_by_chip": {
        "title": "Cumulative timelines by chip",
        "web_file": "cumulative_timelines.csv",
        "has_chip": True,
        "metrics": [
            ("Number of units (median)", "Number of units (median)", "Units median"),
            ("Number of units (5th percentile)", "Number of units (5th percentile)", "Units p5"),
            ("Number of units (95th percentile)", "Number of units (95th percentile)", "Units p95"),
            ("Compute estimate in H100e (median)", "H100e compute power (median)", "H100e median"),
            ("Compute estimate in H100e (5th percentile)", "H100e compute power (5th percentile)", "H100e p5"),
            ("Compute estimate in H100e (95th percentile)", "H100e compute power (95th percentile)", "H100e p95"),
        ],
    },
    "cumulative_by_designer": {
        "title": "Cumulative totals by designer",
        "web_file": "cumulative_timelines_by_designer.csv",
        "has_chip": False,
        "metrics": [
            ("Number of units (median)", "Number of units (median)", "Units median"),
            ("Number of units (5th percentile)", "Number of units (5th percentile)", "Units p5"),
            ("Number of units (95th percentile)", "Number of units (95th percentile)", "Units p95"),
            ("Compute estimate in H100e (median)", "Compute estimate in H100e (median)", "H100e median"),
            ("Compute estimate in H100e (5th percentile)", "Compute estimate in H100e (5th percentile)", "H100e p5"),
            ("Compute estimate in H100e (95th percentile)", "Compute estimate in H100e (95th percentile)", "H100e p95"),
            ("Power in MW (median)", "Power in MW (median)", "Power MW median"),
            ("Power in MW (5th percentile)", "Power in MW (5th percentile)", "Power MW p5"),
            ("Power in MW (95th percentile)", "Power in MW (95th percentile)", "Power MW p95"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Per-designer wiring
# ---------------------------------------------------------------------------
# To add a designer, copy the Nvidia block and fill in:
#   web_name      - the label used in the website's "Chip manufacturer" column
#   chip_aliases  - any chip names that differ between website and local exports
#   local_files   - the three local CSVs (see csv_export/ for the naming conventions)
DESIGNER_CONFIGS = {
    "nvidia": {
        "web_name": "Nvidia",
        # Chip naming isn't yet consistent across published files: the per-quarter table and
        # the local exports use "H100/H200", but the published cumulative_timelines.csv still
        # says "H100". Normalize both sides so rows line up regardless.
        "chip_aliases": {"H100/H200": "H100"},
        "local_files": {
            "timelines_by_chip": "csv_export/nvidia_calendar_quarter_chip_timelines.csv",
            "cumulative_by_chip": "csv_export/nvidia_cumulative_by_chip.csv",
            "cumulative_by_designer": "csv_export/nvidia_cumulative_totals.csv",
        },
    },
    "google": {
        "web_name": "Google",
        # TPU chip names match between the website and the local exports.
        "chip_aliases": {},
        "local_files": {
            "timelines_by_chip": "csv_export/tpu_calendar_quarter_chip_timelines.csv",
            "cumulative_by_chip": "csv_export/tpu_cumulative_by_chip.csv",
            "cumulative_by_designer": "csv_export/tpu_cumulative_totals.csv",
        },
    },
    "amd": {
        "web_name": "AMD",
        # The published AMD cumulative table is the one place on the site that spells chip
        # types with an "Instinct " prefix - its own Name column and every other designer
        # and table use the bare name. Drop the prefix so those rows still line up.
        "chip_aliases": {f"Instinct {c}": c for c in
                         ("MI250X", "MI300A", "MI300X", "MI308X", "MI325X", "MI350X", "MI355X")},
        # AMD reports on calendar quarters, so there is a single quarterly timelines file
        # rather than the separate fiscal/calendar pair Nvidia needs.
        "local_files": {
            "timelines_by_chip": "csv_export/amd_chip_timelines.csv",
            "cumulative_by_chip": "csv_export/amd_cumulative_by_chip.csv",
            "cumulative_by_designer": "csv_export/amd_cumulative_totals.csv",
        },
    },
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_TS_RE = re.compile(r"Estimates generated on:\s*([0-9]{1,2}-[0-9]{1,2}-[0-9]{4} [0-9]{1,2}:[0-9]{2})")


def _to_float(value):
    """Coerce a cell to float, treating blanks/None/non-numeric as NaN."""
    if value is None:
        return np.nan
    if isinstance(value, str) and value.strip() == "":
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _iso_dates(series):
    """Parse a date column (either M/D/YYYY or YYYY-MM-DD) to canonical ISO strings."""
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


def _quarter_label(iso_date):
    """Turn an ISO start date into a calendar-quarter label like 'Q1 2024'."""
    dt = pd.to_datetime(iso_date)
    return f"Q{(dt.month - 1) // 3 + 1} {dt.year}"


def extract_timestamps(notes_series):
    """Return a Counter of the 'Estimates generated on' stamps found in a Notes column."""
    stamps = []
    for note in notes_series.dropna():
        match = _TS_RE.search(str(note))
        if match:
            stamps.append(match.group(1))
    return Counter(stamps)


def _format_timestamps(counter):
    """Render a timestamp Counter as a compact human-readable string."""
    if not counter:
        return "(no timestamp found)"
    parts = [f"{ts} (x{n})" for ts, n in counter.most_common()]
    return ", ".join(parts)


def _dominant_timestamp(counter):
    """The most common run timestamp in a file, or None if there is none."""
    return counter.most_common(1)[0][0] if counter else None


def percent_diff(local_v, web_v):
    """Percent difference of local relative to the published website value.

    Returns NaN if either side is missing, 0 if both are exactly zero, and +/-inf if the
    website value is zero but the local value is not (an unbounded relative change).
    """
    if np.isnan(local_v) or np.isnan(web_v):
        return np.nan
    if web_v == 0:
        return 0.0 if local_v == 0 else np.inf
    return (local_v - web_v) / web_v * 100.0


def classify(pct, tol_pct, warn_pct):
    """Bucket a percent difference into ok / minor / major / nodata."""
    if np.isnan(pct):
        return "nodata"
    ap = abs(pct)
    if np.isinf(ap):
        return "major"
    if ap <= tol_pct:
        return "ok"
    if ap <= warn_pct:
        return "minor"
    return "major"


# ---------------------------------------------------------------------------
# Fetching the published data
# ---------------------------------------------------------------------------
def fetch_website_tables(url=WEBSITE_ZIP_URL):
    """Download the published zip and return {table_key: DataFrame}, all in memory."""
    print(f"Downloading published dataset from {url} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "epoch-chip-validation/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not download {url}: {exc}") from exc

    tables = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        # Index zip members by basename so a wrapping folder inside the zip doesn't matter.
        members = {os.path.basename(name): name for name in zf.namelist()}
        for table_key, spec in TABLE_SPECS.items():
            if spec["web_file"] not in members:
                raise FileNotFoundError(f"{spec['web_file']} not found in the published zip")
            with zf.open(members[spec["web_file"]]) as fh:
                tables[table_key] = pd.read_csv(fh)
    print(f"Loaded {len(tables)} published tables ({len(payload):,} bytes)")
    return tables


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------
def compare_table(table_key, local_df, web_df, spec, web_name, chip_aliases, tol_pct, warn_pct):
    """Compare one local table against its published counterpart.

    Returns a dict with the long-format per-cell diffs, the unmatched row keys on each
    side, the run timestamps, and any metrics that had to be skipped.
    """
    local = local_df.copy()
    web = web_df[web_df["Chip manufacturer"] == web_name].copy()

    # Canonicalize dates and chip names so the two sides line up on a common row key.
    for df in (local, web):
        df["start"] = _iso_dates(df["Start date"])
        df["end"] = _iso_dates(df["End date"])
        if spec["has_chip"]:
            df["chip"] = df["Chip type"].map(lambda c: chip_aliases.get(str(c), str(c)))
        else:
            df["chip"] = "(total)"
    key_cols = ["chip", "start", "end"]

    # Guard against accidental duplicate keys so the merge below stays one-to-one.
    for side, df in (("local", local), ("website", web)):
        n_dup = int(df.duplicated(key_cols).sum())
        if n_dup:
            print(f"  warning: {n_dup} duplicate row key(s) on {side} side of {table_key}; keeping first")
    local = local.drop_duplicates(key_cols)
    web = web.drop_duplicates(key_cols)

    local_ts = extract_timestamps(local["Notes"]) if "Notes" in local.columns else Counter()
    web_ts = extract_timestamps(web["Notes"]) if "Notes" in web.columns else Counter()

    # Skip any metric whose column is absent on either side (keeps the comparison robust
    # as the published schema evolves), and note what was skipped.
    usable_metrics, skipped_metrics = [], []
    for local_col, web_col, metric_label in spec["metrics"]:
        if local_col in local.columns and web_col in web.columns:
            usable_metrics.append((local_col, web_col, metric_label))
        else:
            skipped_metrics.append(metric_label)

    # Narrow each side down to the row key plus one numeric column per metric, then line
    # the sides up with an outer merge; the merge indicator flags one-sided rows.
    def keyed_metric_values(df, col_index):
        out = df[key_cols].copy()
        for metric_cols in usable_metrics:
            out[metric_cols[2]] = df[metric_cols[col_index]].map(_to_float)
        return out

    merged = keyed_metric_values(local, 0).merge(
        keyed_metric_values(web, 1),
        on=key_cols, how="outer", suffixes=(" local", " web"), indicator=True,
    )
    local_only = sorted(map(tuple, merged.loc[merged["_merge"] == "left_only", key_cols].values))
    web_only = sorted(map(tuple, merged.loc[merged["_merge"] == "right_only", key_cols].values))
    both = merged[merged["_merge"] == "both"]

    # Reshape to long format: one row per (chip, quarter, metric) cell with the local and
    # website values side by side, classified against the tolerances.
    cells = []
    for _, _, metric_label in usable_metrics:
        cell = both[key_cols].copy()
        cell["metric"] = metric_label
        cell["local"] = both[f"{metric_label} local"].values
        cell["website"] = both[f"{metric_label} web"].values
        cells.append(cell)
    if cells:
        diffs = pd.concat(cells, ignore_index=True)
        # Label each row with the quarter of its END date: for quarterly rows start and
        # end fall in the same quarter, and for cumulative rows (which all share one
        # start date) the end date is what distinguishes them.
        diffs["quarter"] = diffs["end"].map(_quarter_label)
        diffs["pct_diff"] = [percent_diff(lv, wv) for lv, wv in zip(diffs["local"], diffs["website"])]
        diffs["status"] = [classify(p, tol_pct, warn_pct) for p in diffs["pct_diff"]]
    else:
        diffs = pd.DataFrame(columns=key_cols + ["metric", "local", "website", "quarter", "pct_diff", "status"])

    return {
        "table": table_key,
        "title": spec["title"],
        "diffs": diffs,
        "n_matched": int(len(both)),
        "local_only": local_only,
        "web_only": web_only,
        "local_ts": local_ts,
        "web_ts": web_ts,
        "skipped_metrics": skipped_metrics,
        "tol_pct": tol_pct,
    }


def validate_designer(designer, web_tables, tol_pct=DEFAULT_TOL_PCT, warn_pct=DEFAULT_WARN_PCT):
    """Compare every configured table for one designer; returns a list of result dicts."""
    config = DESIGNER_CONFIGS[designer]
    results = []
    for table_key, spec in TABLE_SPECS.items():
        local_path = config["local_files"][table_key]
        if not os.path.exists(local_path):
            print(f"  skipping {table_key}: local file not found ({local_path})")
            continue
        results.append(compare_table(
            table_key, pd.read_csv(local_path), web_tables[table_key], spec,
            config["web_name"], config.get("chip_aliases", {}), tol_pct, warn_pct,
        ))
    return results


def verdict(results):
    """Aggregate pass/fail: any major cell or any unmatched row is worth a human look."""
    major_cells = sum(int((r["diffs"]["status"] == "major").sum()) for r in results if not r["diffs"].empty)
    unmatched_rows = sum(len(r["local_only"]) + len(r["web_only"]) for r in results)
    return {
        "major_cells": major_cells,
        "unmatched_rows": unmatched_rows,
        "ok": major_cells == 0 and unmatched_rows == 0,
    }


# ---------------------------------------------------------------------------
# Summary tables (used by both the console report and the notebook)
# ---------------------------------------------------------------------------
def summarize_metrics(diffs):
    """Per-metric summary: counts by status plus bias, typical magnitude, and worst cell."""
    if diffs.empty:
        return pd.DataFrame()
    out = []
    for metric, grp in diffs.groupby("metric", sort=False):
        finite = grp["pct_diff"].replace([np.inf, -np.inf], np.nan).dropna()
        out.append({
            "metric": metric,
            "n": len(grp),
            "ok": int((grp["status"] == "ok").sum()),
            "minor": int((grp["status"] == "minor").sum()),
            "major": int((grp["status"] == "major").sum()),
            "mean_signed_pct": finite.mean() if len(finite) else np.nan,
            "mean_abs_pct": finite.abs().mean() if len(finite) else np.nan,
            "max_abs_pct": finite.abs().max() if len(finite) else np.nan,
        })
    return pd.DataFrame(out)


def metric_summary(results):
    """One row per (table, metric) across all comparisons - the notebook's main table."""
    frames = []
    for res in results:
        summary = summarize_metrics(res["diffs"])
        if summary.empty:
            continue
        summary.insert(0, "table", res["title"])
        frames.append(summary)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def coverage_summary(results):
    """One row per table: row-coverage counts and the run timestamp on each side.

    'same run' is False when the dominant timestamps differ, i.e. the published table
    came from a different model run than the local CSV - in that case diffs are expected.
    """
    rows = []
    for res in results:
        rows.append({
            "table": res["title"],
            "rows matched": res["n_matched"],
            "local only": len(res["local_only"]),
            "website only": len(res["web_only"]),
            "local run": _format_timestamps(res["local_ts"]),
            "website run": _format_timestamps(res["web_ts"]),
            "same run": _dominant_timestamp(res["local_ts"]) == _dominant_timestamp(res["web_ts"]),
        })
    return pd.DataFrame(rows)


def pct_diff_table(results, table, metric="Units median"):
    """Readable 2D table of percent differences for one comparison: rows x quarters.

    For the chip-level tables, rows are chip types and the table shows one metric
    (median units by default). For the designer-totals table, which has no chip
    breakdown, pass metric=None to get all metrics as rows instead.

    Cell legend: a number is the percent difference (local vs website), '-' means
    neither dataset has that cell (e.g. before a chip launched), 'local only' /
    'website only' flag rows that exist on just one side, and 'no data' means the
    row matched but the value is blank on at least one side. When every cell is
    within tolerance and no row is one-sided, a confirmation line is printed too.
    """
    res = next((r for r in results if r["table"] == table), None)
    if res is None:
        return pd.DataFrame()
    d = res["diffs"]
    if metric is not None:
        d = d[d["metric"] == metric]
        row_field = "chip"
    else:
        row_field = "metric"

    # Collect the percent diff of every matched cell, remembering each quarter's date
    # so the columns can be ordered chronologically.
    cell_pct = {}
    quarter_dates = {}
    for _, row in d.iterrows():
        cell_pct[(row[row_field], row["quarter"])] = row["pct_diff"]
        quarter_dates.setdefault(row["quarter"], pd.to_datetime(row["end"]))

    # Rows that exist on one side only become text flags. A one-sided row is missing
    # all of its metrics, so in by-metric mode the flag spans every metric row.
    metric_rows = list(dict.fromkeys(d["metric"])) if metric is None else []
    flagged = {}
    for keys, flag in ((res["local_only"], "local only"), (res["web_only"], "website only")):
        for chip, start, end in keys:
            quarter = _quarter_label(end)
            quarter_dates.setdefault(quarter, pd.to_datetime(end))
            for row_label in ([chip] if metric is not None else metric_rows or ["(all metrics)"]):
                flagged[(row_label, quarter)] = flag

    if not cell_pct and not flagged:
        return pd.DataFrame()

    if metric is not None:
        row_labels = sorted({label for label, _ in cell_pct} | {label for label, _ in flagged})
    else:
        row_labels = metric_rows or ["(all metrics)"]
    quarters = sorted(quarter_dates, key=quarter_dates.get)

    def render_cell(row_label, quarter):
        if (row_label, quarter) in flagged:
            return flagged[(row_label, quarter)]
        if (row_label, quarter) not in cell_pct:
            return "-"
        pct = cell_pct[(row_label, quarter)]
        if np.isnan(pct):
            return "no data"
        if np.isinf(pct):
            return "inf"
        text = f"{pct:.1f}"
        return "0.0" if text == "-0.0" else text

    grid = pd.DataFrame([[render_cell(label, q) for q in quarters] for label in row_labels],
                        index=row_labels, columns=quarters)
    grid.index.name = row_field
    grid.columns.name = "quarter"

    # Say so explicitly when there is nothing to look at in this table.
    one_sided = res["local_only"] or res["web_only"]
    if not one_sided and len(d) and (d["status"] == "ok").all():
        scope = f"{metric} " if metric is not None else ""
        print(f"{res['title']}: all {len(d)} {scope}cells match within the {res['tol_pct']}% tolerance; "
              f"no one-sided rows")
    return grid


def worst_discrepancies(results, n=10):
    """The n cells with the largest percent difference (minor or worse) across all tables."""
    frames = [res["diffs"].assign(table=res["title"]) for res in results if not res["diffs"].empty]
    if not frames:
        return pd.DataFrame()
    cells = pd.concat(frames, ignore_index=True)
    cells = cells[cells["status"].isin(["minor", "major"])].copy()
    cells["abs_pct"] = cells["pct_diff"].abs()
    cells = cells.sort_values("abs_pct", ascending=False).drop(columns="abs_pct")
    columns = ["table", "chip", "quarter", "metric", "local", "website", "pct_diff", "status"]
    return cells.head(n)[columns].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figure (returned as a matplotlib figure; the notebook displays it inline)
# ---------------------------------------------------------------------------
def plot_cumulative_timeseries(results, designer="nvidia"):
    """Cumulative designer totals over time, local vs website, for units/H100e/power.

    Returns the figure, or None if the designer-totals table wasn't compared.
    """
    res = next((r for r in results if r["table"] == "cumulative_by_designer"), None)
    if res is None or res["diffs"].empty:
        return None
    d = res["diffs"].copy()
    d["end_dt"] = pd.to_datetime(d["end"])

    panels = [
        ("Units median", "Units p5", "Units p95", "cumulative units"),
        ("H100e median", "H100e p5", "H100e p95", "cumulative H100e"),
        ("Power MW median", "Power MW p5", "Power MW p95", "cumulative power (MW)"),
    ]
    panels = [p for p in panels if p[0] in set(d["metric"])]
    if not panels:
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
    axes = axes[0]
    for ax, (med, p5, p95, ylab) in zip(axes, panels):
        # Draw website as a wide line underneath and local as a thin line on top, so that
        # when they match the website color shows as a halo (rather than being hidden).
        for side, color, lw, ms, z in (("website", "tab:orange", 5, 0, 1), ("local", "tab:blue", 1.6, 3, 2)):
            sub_med = d[d["metric"] == med].sort_values("end_dt")
            ax.plot(sub_med["end_dt"], sub_med[side], marker="o", ms=ms, color=color,
                    lw=lw, alpha=0.5 if side == "website" else 1.0, label=side, zorder=z)
            sub5 = d[d["metric"] == p5].sort_values("end_dt")
            sub95 = d[d["metric"] == p95].sort_values("end_dt")
            if side == "local" and not sub5.empty and not sub95.empty:
                ax.fill_between(sub5["end_dt"], sub5[side], sub95[side], color=color, alpha=0.12)
        ax.set_title(ylab, fontsize=11)
        ax.set_xlabel("cumulative through")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"{designer}: cumulative totals over time (band = local p5-p95)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ---------------------------------------------------------------------------
# Console report (CLI)
# ---------------------------------------------------------------------------
def print_table_report(result, top_n=8):
    """Print a per-table console summary in the style of the repo's other check output."""
    print("\n" + "=" * 78)
    print(f"  {result['title']}  [{result['table']}]")
    print("=" * 78)
    print(f"  local run:   {_format_timestamps(result['local_ts'])}")
    print(f"  website run: {_format_timestamps(result['web_ts'])}")
    if _dominant_timestamp(result["local_ts"]) != _dominant_timestamp(result["web_ts"]):
        print("  ** timestamps differ - the published table is from a different model run **")

    print(f"\n  rows: {result['n_matched']} matched, "
          f"{len(result['local_only'])} local-only, {len(result['web_only'])} website-only")
    if result["skipped_metrics"]:
        print(f"  skipped metrics (column absent on one side): {', '.join(result['skipped_metrics'])}")

    _print_unmatched(result["local_only"], "local-only (in csv_export, not on website)")
    _print_unmatched(result["web_only"], "website-only (published, missing locally)")

    summary = summarize_metrics(result["diffs"])
    if summary.empty:
        print("  no comparable cells")
        return

    print()
    header = f"  {'metric':<16}{'n':>5}{'ok':>5}{'minor':>7}{'major':>7}{'bias%':>9}{'|mean|%':>9}{'max%':>9}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for _, r in summary.iterrows():
        flag = "  <-- check" if r["major"] > 0 else ""
        print(f"  {r['metric']:<16}{int(r['n']):>5}{int(r['ok']):>5}{int(r['minor']):>7}{int(r['major']):>7}"
              f"{_fmt_pct(r['mean_signed_pct']):>9}{_fmt_pct(r['mean_abs_pct']):>9}{_fmt_pct(r['max_abs_pct']):>9}{flag}")

    worst = worst_discrepancies([result], n=top_n)
    if not worst.empty:
        print(f"\n  largest discrepancies (top {len(worst)}):")
        for _, r in worst.iterrows():
            chip_label = "" if r["chip"] == "(total)" else f"{r['chip']} "
            print(f"    {chip_label}{r['quarter']:<8} {r['metric']:<14} "
                  f"local={_fmt_num(r['local'])}  web={_fmt_num(r['website'])}  ({_fmt_pct(r['pct_diff'])})")


def _print_unmatched(keys, label):
    if not keys:
        return
    shown = ", ".join("/".join(str(p) for p in k) for k in keys[:10])
    more = f" (+{len(keys) - 10} more)" if len(keys) > 10 else ""
    print(f"  {label}: {shown}{more}")


def _fmt_pct(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    if np.isinf(x):
        return "inf"
    return f"{x:+.1f}%" if x else "0.0%"


def _fmt_num(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--designer", default="nvidia",
                        help="designer to validate, or 'all' (default: nvidia). "
                             f"Configured: {', '.join(DESIGNER_CONFIGS)}")
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL_PCT,
                        help=f"percent tolerance for an exact match (default {DEFAULT_TOL_PCT})")
    parser.add_argument("--warn", type=float, default=DEFAULT_WARN_PCT,
                        help=f"percent threshold for major drift (default {DEFAULT_WARN_PCT})")
    args = parser.parse_args(argv)

    if args.designer == "all":
        designers = list(DESIGNER_CONFIGS)
    elif args.designer in DESIGNER_CONFIGS:
        designers = [args.designer]
    else:
        parser.error(f"unknown designer '{args.designer}'. Configured: {', '.join(DESIGNER_CONFIGS)} (or 'all')")

    web_tables = fetch_website_tables()

    all_ok = True
    for designer in designers:
        print("\n" + "#" * 78)
        print(f"#  Validating: {designer}")
        print("#" * 78)
        results = validate_designer(designer, web_tables, args.tol, args.warn)
        for result in results:
            print_table_report(result)
        v = verdict(results)
        print("\n" + "-" * 78)
        print(f"  {designer} summary: {v['major_cells']} major cell diff(s), {v['unmatched_rows']} unmatched row(s)")
        print(f"  verdict: {'PASS - matches published data within tolerance' if v['ok'] else 'REVIEW - drift detected (see above)'}")
        all_ok = all_ok and v["ok"]

    print("\n" + "=" * 78)
    print(f"  Overall: {'PASS' if all_ok else 'REVIEW - drift detected in at least one table'}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
