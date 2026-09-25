#!/usr/bin/env python3
"""Inventory the current chip-sales schema: local export CSVs vs the Airtable tables.

Reads every CSV in csv_export/ and owners_csv_export/ plus the website's hourly mirror of
the Airtable tables (epoch-website-astro/src/public/data/generated/ai_chip_*.csv), maps
each column to a normalized concept (units_med, h100e_p5, ...), and writes
docs/current_schema.md: one matrix per Airtable table showing the spelling each source
uses, plus Name templates, date formats, vocabularies and orphans.

    python3.11 -m pipeline.tools.schema_inventory [--website-generated PATH]
"""
import argparse
import glob
import os
import re
from collections import Counter, defaultdict

import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

WEBSITE_GENERATED = os.path.expanduser(
    "~/Desktop/Scripts/epoch-website-astro/src/public/data/generated")

# Airtable table (as published) -> local files that are hand-imported into it.
TARGETS = {
    "ai_chip_sales_timelines_by_chip": [
        "csv_export/nvidia_calendar_quarter_chip_timelines.csv",
        "csv_export/tpu_calendar_quarter_chip_timelines.csv",
        "csv_export/amd_chip_timelines.csv",
        "csv_export/trainium_chip_timelines.csv",
    ],
    "ai_chip_sales_cumulative_timelines": [
        "csv_export/nvidia_cumulative_by_chip.csv",
        "csv_export/tpu_cumulative_by_chip.csv",
        "csv_export/amd_cumulative_by_chip.csv",
        "csv_export/trainium_cumulative_by_chip.csv",
    ],
    "ai_chip_sales_cumulative_timelines_by_designer": [
        "csv_export/nvidia_cumulative_totals.csv",
        "csv_export/tpu_cumulative_totals.csv",
        "csv_export/amd_cumulative_totals.csv",
        "csv_export/trainium_cumulative_totals.csv",
    ],
    "ai_chip_owners_quarters_by_chip_type": [
        "owners_csv_export/nvidia_owners_quarters_by_chip.csv",
        "owners_csv_export/nvidia_owners_OTHER_quarters_by_chip.csv",
        "owners_csv_export/tpu_owners_quarters_by_chip.csv",
        "owners_csv_export/amd_owners_quarters_by_chip.csv",
        "owners_csv_export/trainium_owners_quarters_by_chip.csv",
        "owners_csv_export/coreweave/coreweave_owners_quarters_by_chip.csv",
    ],
    "ai_chip_owners_cumulative_by_chip_type": [
        "owners_csv_export/nvidia_owners_cumulative_by_chip.csv",
        "owners_csv_export/nvidia_owners_OTHER_cumulative_by_chip.csv",
        "owners_csv_export/tpu_owners_cumulative_by_chip.csv",
        "owners_csv_export/amd_owners_cumulative_by_chip.csv",
        "owners_csv_export/trainium_owners_cumulative_by_chip.csv",
        "owners_csv_export/coreweave/coreweave_owners_cumulative_by_chip.csv",
    ],
    "ai_chip_owners_cumulative_by_designer": [
        "owners_csv_export/nvidia_owners_cumulative_totals.csv",
        "owners_csv_export/nvidia_owners_OTHER_cumulative_totals.csv",
        "owners_csv_export/tpu_owners_cumulative_totals.csv",
        "owners_csv_export/amd_owners_cumulative_totals.csv",
        "owners_csv_export/trainium_owners_cumulative_totals.csv",
        "owners_csv_export/coreweave/coreweave_owners_cumulative_totals.csv",
    ],
}

PCT = {"median": "med", "5th percentile": "p5", "95th percentile": "p95"}
PCT_RE = r"(?: \((median|5th percentile|95th percentile)\))?"

# (regex, concept stem). Bare metric names (no suffix) are treated as the median.
METRIC_PATTERNS = [
    (re.compile(rf"^number of units{PCT_RE}$", re.I), "units"),
    (re.compile(rf"^(?:compute estimate in h100e|h100e compute power|h100e){PCT_RE}$", re.I), "h100e"),
    (re.compile(rf"^power in mw{PCT_RE}$", re.I), "power_mw"),
    (re.compile(rf"^total tdp \(w\){PCT_RE}$", re.I), "total_tdp_w"),
]
EXACT = {
    "Name": "name", "Chip manufacturer": "designer", "Designer": "designer", "Owner": "owner",
    "Chip type": "chip_type", "Start date": "start_date", "End date": "end_date",
    "Incomplete": "incomplete", "Notes": "notes", "Source / Link": "source",
    "Last Modified By": "meta: last modified by", "Last Modified": "meta: last modified",
    "Last Modified 2": "meta: last modified 2", "Created": "meta: created",
}
AIRTABLE_COMPUTED = {  # lookups/formulas that exist only in Airtable
    "Chip TDP (W)", "Chip H100e", "Chip type (linked)", "Chip Cost (Linked)", "Chip Cost (USD)",
    "Cost Estimate (USD)", "Unofficial estimated TDP (W)",
}

CONCEPT_ORDER = (
    ["name", "designer", "owner", "chip_type", "start_date", "end_date"]
    + [f"{m}_{s}" for m in ("units", "h100e", "power_mw", "total_tdp_w") for s in ("med", "p5", "p95")]
    + ["incomplete", "notes", "source", "meta: last modified by", "meta: last modified",
       "meta: last modified 2", "meta: created"]
)


def concept_of(col):
    if col in EXACT:
        return EXACT[col]
    if col in AIRTABLE_COMPUTED:
        return f"airtable-only: {col}"
    for rx, stem in METRIC_PATTERNS:
        m = rx.match(col)
        if m:
            return f"{stem}_{PCT.get(m.group(1), 'med') if m.group(1) else 'med'}"
    return f"?: {col}"


QUARTER_RES = [
    (re.compile(r"\bFY\d{2} ?Q\d\b"), "{FYqtr}"),
    (re.compile(r"\bQ\d_FY\d{2}\b"), "{FYqtr}"),
    (re.compile(r"\bQ\d 20\d{2}\b"), "{Qtr}"),
    (re.compile(r"\bFY\d{2}\b"), "{FY}"),
]
DESIGNERS = ["Nvidia", "Google TPU", "Google", "AMD", "Amazon", "Huawei", "Cambricon", "Trainium", "TPU"]


def name_template(df):
    chips = sorted({str(c) for c in df.get("Chip type", pd.Series(dtype=str)).dropna()}, key=len, reverse=True)
    owners = sorted({str(o) for o in df.get("Owner", pd.Series(dtype=str)).dropna()}, key=len, reverse=True)
    out = Counter()
    for raw in df["Name"].astype(str):
        s = raw
        for c in chips:
            s = s.replace(c, "{Chip}")
        for rx, tok in QUARTER_RES:
            s = rx.sub(tok, s)
        for o in owners:
            s = re.sub(rf"\b{re.escape(o)}\b", "{Owner}", s)
        for d in DESIGNERS:
            s = re.sub(rf"\b{re.escape(d)}\b", "{Designer}", s)
        s = re.sub(r"(\{Designer\} ?)+", "{Designer} ", s).strip()
        out[s] += 1
    return out


def date_format(series):
    v = str(series.dropna().astype(str).iloc[0]) if len(series.dropna()) else ""
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        return "YYYY-MM-DD"
    if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", v):
        return "M/D/YYYY"
    return v or "(empty)"


def load(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False).replace("", pd.NA)


def short(path):
    return os.path.basename(path).replace(".csv", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--website-generated", default=WEBSITE_GENERATED)
    args = ap.parse_args()

    tables = {}  # label -> (df, kind)
    for target, locals_ in TARGETS.items():
        tables[target] = (load(os.path.join(args.website_generated, target + ".csv")), "airtable")
        for f in locals_:
            tables[f] = (load(f), "local")
    all_local = sorted(glob.glob("csv_export/*.csv") + glob.glob("owners_csv_export/**/*.csv", recursive=True))
    mapped = {f for fs in TARGETS.values() for f in fs}
    orphans = [f for f in all_local if f not in mapped]
    for f in orphans:
        tables[f] = (load(f), "orphan")

    L = []
    L.append("# The current schema: local exports vs Airtable\n")
    L.append("Sections 1 to 5 are generated by `pipeline/tools/schema_inventory.py`; rerun it after any export "
             "change. The findings section above them is written by hand and survives reruns. "
             "Airtable's side comes from the website's hourly mirror of each published view "
             "(`epoch-website-astro/src/public/data/generated/`), so it shows visible fields only. "
             "Cells give the literal column name; a dash means the table lacks that quantity.\n")

    # --- Global dialect table -------------------------------------------------------
    L.append("## 1. Spellings per concept, across everything\n")
    L.append("How many names each quantity goes by, across all 40 tables. Any count above 1 is a translation someone makes by hand.\n")
    spell = defaultdict(Counter)
    for label, (df, kind) in tables.items():
        for col in df.columns:
            spell[concept_of(col)][col] += 1
    L.append("| Concept | Distinct spellings | Used as (count of tables) |")
    L.append("|---|---|---|")
    for c in CONCEPT_ORDER:
        if c in spell:
            names = "; ".join(f"`{n}` ({k})" for n, k in spell[c].most_common())
            L.append(f"| {c} | {len(spell[c])} | {names} |")
    others = [c for c in spell if c not in CONCEPT_ORDER]
    for c in sorted(others):
        names = "; ".join(f"`{n}` ({k})" for n, k in spell[c].most_common())
        L.append(f"| {c} | {len(spell[c])} | {names} |")
    L.append("")

    # --- Per-target matrices --------------------------------------------------------
    L.append("## 2. Per Airtable table: what each source calls each column\n")
    L.append("The first column is the Airtable table; the rest are the local CSVs imported into it. "
             "**Bold** marks a local name that differs from Airtable's, which means a re-mapping on "
             "every import.\n")
    for target, locals_ in TARGETS.items():
        cols = [target] + locals_
        L.append(f"### `{target}`\n")
        L.append("| Concept | " + " | ".join(f"`{short(c)}`" for c in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        concept_cols = {}
        for c in cols:
            df = tables[c][0]
            m = defaultdict(list)
            for col in df.columns:
                m[concept_of(col)].append(col)
            concept_cols[c] = m
        present = [k for k in CONCEPT_ORDER if any(k in concept_cols[c] for c in cols)]
        present += sorted(k for c in cols for k in concept_cols[c] if k not in CONCEPT_ORDER and k not in present)
        at = concept_cols[target]
        for k in present:
            cells = []
            for c in cols:
                names = concept_cols[c].get(k)
                if not names:
                    cells.append("—")
                    continue
                txt = ", ".join(f"`{n}`" for n in names)
                if c != target and at.get(k) and names != at[k]:
                    txt = f"**{txt}**"
                if c != target and not at.get(k):
                    txt = f"**{txt}** (no Airtable field)"
                cells.append(txt)
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        # row-level facts
        L.append("")
        L.append("| | " + " | ".join(f"`{short(c)}`" for c in cols) + " |")
        L.append("|---|" + "---|" * len(cols))
        L.append("| rows | " + " | ".join(str(len(tables[c][0])) for c in cols) + " |")
        L.append("| date format | " + " | ".join(date_format(tables[c][0]["Start date"]) for c in cols) + " |")
        des = []
        for c in cols:
            df = tables[c][0]
            col = "Chip manufacturer" if "Chip manufacturer" in df else "Designer"
            des.append(", ".join(sorted(df[col].dropna().unique())) if col in df else "—")
        L.append("| designers | " + " | ".join(des) + " |")
        if "Owner" in tables[target][0]:
            L.append("| owners | " + " | ".join(
                ", ".join(sorted(tables[c][0]["Owner"].dropna().unique())) if "Owner" in tables[c][0] else "—"
                for c in cols) + " |")
        L.append("")
        L.append("Name templates (most common first):\n")
        for c in cols:
            tpl = name_template(tables[c][0])
            L.append(f"- `{short(c)}`: " + "; ".join(f"`{t}` ×{n}" for t, n in tpl.most_common(3)))
        L.append("")

    # --- Chip type vocab --------------------------------------------------------------
    L.append("## 3. Chip type vocabulary\n")
    L.append("Chip names as spelled in each table. Aliases across tables (`H100/H200` vs `H100`, "
             "`Instinct MI300X` vs `MI300X`) are re-mapped by hand today.\n")
    vocab = defaultdict(set)
    for label, (df, kind) in tables.items():
        if "Chip type" in df:
            for v in df["Chip type"].dropna().unique():
                vocab[v].add(short(label))
    for v in sorted(vocab):
        L.append(f"- `{v}`: {', '.join(sorted(vocab[v]))}")
    L.append("")

    # --- Orphans and Airtable-only ------------------------------------------------------
    L.append("## 4. Local files with no Airtable target\n")
    for f in orphans:
        df = tables[f][0]
        L.append(f"- `{f}` ({len(df)} rows): {', '.join(df.columns[:6])}, …")
    L.append("")
    L.append("## 5. Rows that exist only in Airtable\n")
    L.append("Designers present in the Airtable table but produced by no local export (hand-curated):\n")
    for target, locals_ in TARGETS.items():
        adf = tables[target][0]
        local_des = set()
        for f in locals_:
            d = tables[f][0]
            local_des |= set(d["Chip manufacturer"].dropna().unique())
        only = adf[~adf["Chip manufacturer"].isin(local_des)].groupby("Chip manufacturer").size()
        L.append(f"- `{target}`: " + (", ".join(f"{k} ({v} rows)" for k, v in only.items()) or "none"))
        if "Owner" in adf:
            local_own = set()
            for f in locals_:
                d = tables[f][0]
                if "Owner" in d:
                    local_own |= set(d["Owner"].dropna().unique())
            oo = adf[~adf["Owner"].isin(local_own)].groupby("Owner").size()
            if len(oo):
                L.append(f"  - owners with no local export: " + ", ".join(f"{k} ({v} rows)" for k, v in oo.items()))
    L.append("")

    # Keep the hand-written findings: everything in the existing file between the title
    # block and the first generated section survives a rerun.
    out_path = "docs/current_schema.md"
    preamble = ""
    if os.path.exists(out_path):
        existing = open(out_path).read()
        start = existing.find("\n## ")
        end = existing.find("\n## 1. ")
        if 0 <= start < end:
            preamble = existing[start + 1:end + 1]
    first_section = next(i for i, line in enumerate(L) if line.startswith("## 1. "))
    L = L[:first_section] + ([preamble] if preamble else []) + L[first_section:]

    os.makedirs("docs", exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote docs/current_schema.md", len(L), "lines")


if __name__ == "__main__":
    main()
