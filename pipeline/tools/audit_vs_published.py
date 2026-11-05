#!/usr/bin/env python3.11
"""Compare staging/ with the published data, field by field.

    python3.11 -m pipeline.tools.audit_vs_published [--mirror PATH] [--tol 1.0] [--csv OUT]

The published side is the website's hourly mirror of the Airtable views
(epoch-website-astro/src/public/data/generated/). Rows are matched on designer, owner,
chip type and quarter, never on Name. For every field both sides carry (each metric at
each percentile, the dates, the Incomplete flag), it counts matched cells that agree,
differ, or are filled on one side only, and breaks disagreements down by designer and
chip. Numbers agree when within --tol percent or less than 1 apart, since the legacy
exports truncated to whole numbers.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline import chip_schema as cs  # noqa: E402

MIRROR = Path.home() / "Desktop/Scripts/epoch-website-astro/src/public/data/generated"
PUBLISHED = {
    "sales_quarterly_by_chip": "ai_chip_sales_timelines_by_chip.csv",
    "sales_cumulative_by_chip": "ai_chip_sales_cumulative_timelines.csv",
    "sales_cumulative_by_designer": "ai_chip_sales_cumulative_timelines_by_designer.csv",
    "owners_quarterly_by_chip": "ai_chip_owners_quarters_by_chip_type.csv",
    "owners_cumulative_by_chip": "ai_chip_owners_cumulative_by_chip_type.csv",
    "owners_cumulative_by_designer": "ai_chip_owners_cumulative_by_designer.csv",
}
STAT_WORD = {"p5": "5th percentile", "med": "median", "p95": "95th percentile"}

# Published spellings of each canonical metric, first match wins; scale converts to
# the canonical unit (power in MW).
PUBLISHED_METRICS = {
    "units": [("Number of units ({w})", 1), ("Number of Units ({w})", 1), ("Number of Units", 1, "med")],
    "h100e": [("Compute estimate in H100e ({w})", 1), ("H100e compute power ({w})", 1), ("H100e ({w})", 1)],
    "power_mw": [("Power in MW ({w})", 1), ("Total TDP (W) ({w})", 1e-6), ("Total TDP (W)", 1e-6, "med")],
    "cost_usd": [("Cost Estimate (USD)", 1, "med")],
}


def published_metric(df, metric, stat):
    for spec in PUBLISHED_METRICS[metric]:
        pattern, scale = spec[0], spec[1]
        only = spec[2] if len(spec) > 2 else None
        if only and only != stat:
            continue
        col = pattern.format(w=STAT_WORD[stat])
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce") * scale, col
    return None, None


def keys(df, designer_col):
    k = pd.DataFrame(index=df.index)
    k["designer"] = df[designer_col]
    k["owner"] = df["Owner"].map(cs.canonical_owner) if "Owner" in df else ""
    k["chip"] = df["Chip type"].map(cs.canonical_chip) if "Chip type" in df else ""
    k["quarter"] = pd.to_datetime(df["End date"], format="mixed").map(cs.quarter_of)
    return k


def flag(series):
    return series.fillna("").astype(str).str.lower().isin(["true", "checked", "1", "yes"])


def audit(mirror, tol):
    findings, summary = [], []
    for table, fname in PUBLISHED.items():
        pub = pd.read_csv(mirror / fname, dtype=str)
        stg = pd.read_csv(cs.REPO / "staging" / f"{table}.csv", dtype=str)
        kp, ks = keys(pub, "Chip manufacturer"), keys(stg, "Designer")
        on = ["designer", "owner", "chip", "quarter"]
        kp["_p"], ks["_s"] = kp.index, ks.index
        m = kp.merge(ks, on=on, how="outer", indicator=True)
        both = m[m["_merge"] == "both"]
        P = pub.loc[both["_p"].astype(int)].reset_index(drop=True)
        S = stg.loc[both["_s"].astype(int)].reset_index(drop=True)
        K = both[on].reset_index(drop=True)
        # kp (published) is the left side of the merge, ks (staging) the right.
        summary.append((table, "rows", len(both), int((m["_merge"] == "left_only").sum()),
                        int((m["_merge"] == "right_only").sum()), None))

        fields = []
        for metric in cs.METRICS:
            for stat in cs.STATS:
                pv, pcol = published_metric(P, metric, stat)
                if pv is None:
                    continue
                sv = pd.to_numeric(S[cs.DISPLAY_NAMES[f"{metric}_{stat}"]], errors="coerce")
                if metric == "cost_usd":
                    pv = pv.replace(0, np.nan)   # production writes 0 where a chip has no price
                fields.append((f"{metric}_{stat}", pv, sv, "num", pcol))
        start_col = "Start date" if "_quarterly_" in table else "Series start date"
        fields.append(("start date", pd.to_datetime(P["Start date"], format="mixed").dt.strftime("%Y-%m-%d"),
                       S[start_col], "text", "Start date"))
        if "Incomplete" in P:
            fields.append(("incomplete", flag(P["Incomplete"]), flag(S["Incomplete"]), "bool", "Incomplete"))

        for name, pv, sv, kind, pcol in fields:
            has_p, has_s = pv.notna() & (pv != ""), sv.notna() & (sv != "")
            if kind == "num":
                diff = (sv - pv).abs()
                agree = has_p & has_s & ((diff <= np.maximum(1, tol / 100 * pv.abs())))
            elif kind == "bool":
                has_p = has_s = pd.Series(True, index=pv.index)
                agree = pv == sv
            else:
                agree = has_p & has_s & (pv == sv)
            both_filled = has_p & has_s
            differ = both_filled & ~agree
            summary.append((table, name, int(agree.sum()), int(differ.sum()),
                            int((has_p & ~has_s).sum()), int((~has_p & has_s).sum())))
            for i in K.index[differ | (has_p & ~has_s)]:
                row = K.loc[i]
                if kind == "num" and has_s[i]:
                    detail = f"{sv[i]:,.4g} vs {pv[i]:,.4g} ({(sv[i] / pv[i] - 1) * 100:+.1f}%)" if pv[i] else f"{sv[i]} vs {pv[i]}"
                else:
                    detail = f"staging {sv[i]!s} vs published {pv[i]!s}"
                findings.append({"table": table, "field": name, "designer": row.designer, "owner": row.owner,
                                 "chip": row.chip, "quarter": row.quarter, "detail": detail,
                                 "pct": (sv[i] / pv[i] - 1) * 100 if kind == "num" and has_s[i] and pv[i] else np.nan})
    return summary, pd.DataFrame(findings)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mirror", type=Path, default=MIRROR)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--csv", type=Path, help="write every disagreeing cell to this CSV")
    args = ap.parse_args()
    summary, findings = audit(args.mirror, args.tol)

    print(f"{'table':<31}{'field':<16}{'agree':>7}{'differ':>8}{'pub only':>10}{'stg only':>10}")
    for table, field, a, d, po, so in summary:
        if field == "rows":
            print(f"\n{table:<31}{'rows matched':<16}{a:>7}   published-only rows {d}, staging-only rows {po}")
        else:
            print(f"{'':<31}{field:<16}{a:>7}{d:>8}{po:>10}{so:>10}")
    if len(findings):
        print("\nDisagreeing or missing cells by table, field group, designer and chip (median % gap):")
        f = findings.assign(group=findings["field"].str.replace(r"_(p5|med|p95)$", "", regex=True))
        g = f.groupby(["table", "group", "designer", "chip"], dropna=False).agg(
            cells=("detail", "size"), median_gap=("pct", "median"),
            quarters=("quarter", lambda q: f"{min(q, key=cs.parse_quarter)}–{max(q, key=cs.parse_quarter)}"))
        with pd.option_context("display.width", 200, "display.max_rows", 500):
            print(g.round(1).to_string())
    if args.csv:
        findings.to_csv(args.csv, index=False)
        print(f"\nwrote {len(findings)} cells to {args.csv}")


if __name__ == "__main__":
    main()
