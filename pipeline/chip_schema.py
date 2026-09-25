"""The canonical chip sales and owners schema, and the one way to produce it.

Every model ends with sample arrays: for each quarter and chip type, N_SAMPLES draws of
how many chips shipped (or how many an owner acquired). This module turns those arrays
into the six canonical tables described in docs/schema.md. It computes H100e, power and
cost sample by sample before taking percentiles, so the interval of a designer total is
the interval of the sum, not a sum of intervals.

Typical use at the end of a model:

    from pipeline import chip_schema as cs
    tables = cs.sales_tables("Nvidia", quarterly=calendar_quarterly_samples,
                             incomplete={"Q1 2026"})
    cs.write_tables("nvidia", tables)

`quarterly` maps a quarter label ("Q3 2025") to {chip type: sample array}. Chip names may
use any spelling listed in pipeline/chip_type_map.csv; they come out canonical.

Chip specs (8-bit TOPS, TDP, price) are read through the chip-type map from
staging/chip_types.csv, the stand-in for the Airtable chip_types table.
"""
from __future__ import annotations

import datetime as dt
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
CHIP_TYPE_MAP_PATH = REPO / "pipeline" / "chip_type_map.csv"
CHIP_TYPES_PATH = REPO / "staging" / "chip_types.csv"
CANONICAL_EXPORT_DIR = REPO / "canonical_export"

# 8-bit dense operations per second of one H100; H100e = chip ops / this.
H100_OPS = 1979e12

DESIGNERS = ["Nvidia", "AMD", "Google", "Amazon", "Huawei", "Cambricon"]
METRICS = ["units", "h100e", "power_mw", "cost_usd"]
STATS = ["p5", "med", "p95"]
PERCENTILES = {"p5": 5, "med": 50, "p95": 95}

TABLES = [
    "sales_quarterly_by_chip",
    "sales_cumulative_by_chip",
    "sales_cumulative_by_designer",
    "owners_quarterly_by_chip",
    "owners_cumulative_by_chip",
    "owners_cumulative_by_designer",
]

# Owner spellings retired from the public data, mapped on the way in.
OWNER_ALIASES = {"China(smuggled)": "China (smuggled)", "China (official)": "China"}


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------
def metric_columns():
    return [f"{m}_{s}" for m in METRICS for s in STATS]


def table_columns(table):
    """Canonical column order for one of the six tables."""
    owners = table.startswith("owners_")
    quarterly = "_quarterly_" in table
    by_chip = table.endswith("_by_chip")
    cols = ["name", "designer"]
    if owners:
        cols.append("owner")
    if by_chip:
        cols.append("chip_type")
    # The quarter a row covers is given by its dates; the Name spells it out for readers.
    cols += ["start_date", "end_date"] if quarterly else ["series_start", "end_date"]
    return cols + metric_columns() + ["incomplete", "source", "notes"]


_STAT_WORDS = {"p5": "5th percentile", "med": "median", "p95": "95th percentile"}
_METRIC_WORDS = {"units": "Number of units", "h100e": "H100e", "power_mw": "Power (MW)", "cost_usd": "Cost (USD)"}

# canonical -> display name used in Airtable (here, staging/) and the public CSVs.
DISPLAY_NAMES = {
    "name": "Name",
    "designer": "Designer",
    "owner": "Owner",
    "chip_type": "Chip type",
    "series_start": "Series start date",
    "start_date": "Start date",
    "end_date": "End date",
    "incomplete": "Incomplete",
    "source": "Source",
    "notes": "Notes",
    **{f"{m}_{s}": f"{_METRIC_WORDS[m]} ({_STAT_WORDS[s]})" for m in METRICS for s in STATS},
}
CANONICAL_NAMES = {v: k for k, v in DISPLAY_NAMES.items()}


def to_display(df):
    return df.rename(columns=DISPLAY_NAMES)


def from_display(df):
    return df.rename(columns=CANONICAL_NAMES)


# ---------------------------------------------------------------------------
# Chip types
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def chip_type_map():
    """Canonical chip type -> row of pipeline/chip_type_map.csv, plus an alias index."""
    m = pd.read_csv(CHIP_TYPE_MAP_PATH, dtype=str, keep_default_na=False)
    rows = {r.chip_type: r._asdict() for r in m.itertuples(index=False)}
    aliases = {}
    for r in m.itertuples(index=False):
        aliases[r.chip_type] = r.chip_type
        for a in filter(None, r.aliases.split(";")):
            aliases[a] = r.chip_type
    return rows, aliases


def canonical_chip(name):
    """Map any known spelling of a chip type to its canonical label."""
    _, aliases = chip_type_map()
    try:
        return aliases[name]
    except KeyError:
        raise KeyError(f"chip type {name!r} is not in {CHIP_TYPE_MAP_PATH.name}; add a row for it") from None


def canonical_owner(name):
    return OWNER_ALIASES.get(name, name)


def _num(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


@lru_cache(maxsize=1)
def chip_specs():
    """Canonical chip type -> {designer, h100e_per_chip, tdp_w, price_usd, release_date}.

    TDP is the official figure, falling back to the unofficial estimate. A spec missing
    from chip_types comes back as None, and every metric that needs it is left blank.
    """
    rows, _ = chip_type_map()
    ct = pd.read_csv(CHIP_TYPES_PATH, dtype=str).set_index("Name")
    specs = {}
    for chip, r in rows.items():
        s = ct.loc[r["spec_chip"]]
        ops = _num(s.get("8-bit OP/s"))
        tdp = _num(s.get("TDP (W)")) or _num(s.get("Unofficial estimated TDP (W)"))
        specs[chip] = {
            "designer": r["designer"],
            "h100e_per_chip": ops / H100_OPS if ops else None,
            "tdp_w": tdp,
            "price_usd": _num(s.get("Cost per chip (approx.)")),
            "release_date": s.get("Release date"),
        }
    return specs


# ---------------------------------------------------------------------------
# Quarters
# ---------------------------------------------------------------------------
_QUARTER_RE = re.compile(r"^Q([1-4]) (\d{4})$")


def parse_quarter(label):
    m = _QUARTER_RE.match(label)
    if not m:
        raise ValueError(f"quarter labels look like 'Q3 2025', got {label!r}")
    return int(m.group(2)), int(m.group(1))


def quarter_bounds(label):
    """'Q3 2025' -> (date(2025, 7, 1), date(2025, 9, 30))."""
    year, q = parse_quarter(label)
    start = dt.date(year, 3 * q - 2, 1)
    end = (dt.date(year + (q == 4), (3 * q) % 12 + 1, 1) - dt.timedelta(days=1))
    return start, end


def quarter_of(date):
    date = pd.Timestamp(date)
    return f"Q{(date.month - 1) // 3 + 1} {date.year}"


def sort_quarters(labels):
    return sorted(labels, key=parse_quarter)


def incomplete_quarters(quarters, source_first_start, source_last_end):
    """Which quarters a model's source data only partly covers.

    Returns (quarterly, cumulative). A quarterly row is incomplete when its quarter
    starts before the first source period or ends after the last one. A cumulative row
    is incomplete only when it runs past the last source period; an early series start
    is already stated by `series_start`.
    """
    first, last = pd.Timestamp(source_first_start), pd.Timestamp(source_last_end)
    quarterly, cumulative = set(), set()
    for q in quarters:
        start, end = (pd.Timestamp(d) for d in quarter_bounds(q))
        if end > last:
            quarterly.add(q)
            cumulative.add(q)
        elif start < first:
            quarterly.add(q)
    return quarterly, cumulative


# ---------------------------------------------------------------------------
# Metrics from samples
# ---------------------------------------------------------------------------
def _percentiles(samples):
    return {s: float(np.percentile(samples, p)) for s, p in PERCENTILES.items()}


def metric_values(samples_by_chip, prices=None):
    """{chip: unit samples} -> {'units_p5': ..., ..., 'cost_usd_p95': ...}.

    All chips are summed sample by sample. A metric is left blank when any chip with
    units lacks the spec it needs, rather than silently counting that chip as zero.
    `prices` ({chip: USD}) replaces the chip_types price for models that estimate
    their own, such as the TPU model.
    """
    specs = chip_specs()
    prices = prices or {}
    n = len(next(iter(samples_by_chip.values())))
    totals = {m: np.zeros(n) for m in METRICS}
    known = {m: True for m in METRICS}
    for chip, units in samples_by_chip.items():
        units = np.asarray(units, dtype=float)
        spec = specs[chip]
        totals["units"] += units
        per_chip = {
            "h100e": spec["h100e_per_chip"],
            "power_mw": spec["tdp_w"] / 1e6 if spec["tdp_w"] else None,
            "cost_usd": prices.get(chip, spec["price_usd"]),
        }
        for m, factor in per_chip.items():
            if factor is None:
                known[m] = known[m] and not units.any()
            else:
                totals[m] += units * factor
    out = {}
    for m in METRICS:
        pct = _percentiles(totals[m]) if known[m] else dict.fromkeys(STATS)
        for s in STATS:
            v = pct[s]
            if v is None:
                out[f"{m}_{s}"] = None
            elif m == "power_mw":
                out[f"{m}_{s}"] = round(v, 3)
            else:
                out[f"{m}_{s}"] = int(round(v))
    return out


def _canonical_samples(by_quarter):
    """Normalise {quarter: {chip: samples}}: canonical chip names, merged aliases."""
    out = {}
    for q, chips in by_quarter.items():
        parse_quarter(q)
        merged = {}
        for chip, samples in chips.items():
            c = canonical_chip(chip)
            merged[c] = merged[c] + np.asarray(samples, float) if c in merged else np.asarray(samples, float)
        out[q] = merged
    return out


def running_totals(quarterly):
    """Quarterly flows -> cumulative stocks, summed sample by sample from the first quarter."""
    out, acc = {}, {}
    for q in sort_quarters(quarterly):
        for chip, samples in quarterly[q].items():
            acc[chip] = acc.get(chip, 0) + np.asarray(samples, float)
        out[q] = {c: s.copy() for c, s in acc.items()}
    return out


def _has_units(samples):
    return np.asarray(samples).any()


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def row_name(table, designer, quarter, chip=None, owner=None, series_start=None):
    """The row's Name: "Nvidia H100/H200 Q3 2025", or for a cumulative row
    "Nvidia H100/H200 cumulative Q1 2022 through Q3 2025". Owner rows lead with the owner."""
    parts = [owner] if owner else []
    parts.append(designer)
    if chip:
        parts.append(chip)
    if "_cumulative_" in table:
        if not series_start:
            raise ValueError(f"{table}: a cumulative row's Name needs its series start quarter")
        parts += ["cumulative", series_start, "through"]
    parts.append(quarter)
    return " ".join(parts)


def generation_note(extra=None):
    stamp = f"Generated on {dt.datetime.now():%Y-%m-%d %H:%M}"
    return f"{extra} {stamp}" if extra else stamp


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------
def _build(prefix, designer, quarterly, cumulative, incomplete, incomplete_cumulative, note, owner=None, prices=None):
    if designer not in DESIGNERS:
        raise ValueError(f"designer must be one of {DESIGNERS}, got {designer!r}")
    notes = generation_note(note)
    rows = {f"{prefix}_quarterly_by_chip": [], f"{prefix}_cumulative_by_chip": [],
            f"{prefix}_cumulative_by_designer": []}
    base = {"designer": designer, **({"owner": owner} if owner else {}), "source": "", "notes": notes}

    if quarterly:
        quarterly = _canonical_samples(quarterly)
        for q in sort_quarters(quarterly):
            start, end = quarter_bounds(q)
            for chip, samples in quarterly[q].items():
                if not _has_units(samples):
                    continue
                table = f"{prefix}_quarterly_by_chip"
                rows[table].append({
                    **base, "name": row_name(table, designer, q, chip, owner), "chip_type": chip,
                    "start_date": start.isoformat(), "end_date": end.isoformat(),
                    **metric_values({chip: samples}, prices), "incomplete": q in incomplete,
                })

    if cumulative is None:
        cumulative = running_totals(quarterly) if quarterly else {}
    else:
        cumulative = _canonical_samples(cumulative)
    if cumulative:
        order = sort_quarters(cumulative)
        series_start = quarter_bounds(order[0])[0].isoformat()
        for q in order:
            end = quarter_bounds(q)[1].isoformat()
            flagged = q in incomplete_cumulative
            chips = {c: s for c, s in cumulative[q].items() if _has_units(s)}
            common = {"series_start": series_start, "end_date": end, "incomplete": flagged}
            for chip, samples in chips.items():
                table = f"{prefix}_cumulative_by_chip"
                rows[table].append({**base, **common, "chip_type": chip, **metric_values({chip: samples}, prices),
                                    "name": row_name(table, designer, q, chip, owner, order[0])})
            if chips:
                table = f"{prefix}_cumulative_by_designer"
                rows[table].append({**base, **common, **metric_values(chips, prices),
                                    "name": row_name(table, designer, q, None, owner, order[0])})

    return {t: pd.DataFrame(r, columns=table_columns(t)) for t, r in rows.items()}


def sales_tables(designer, quarterly=None, cumulative=None, incomplete=(), incomplete_cumulative=None, note=None,
                 prices=None):
    """The three sales tables for one designer.

    quarterly:  {quarter: {chip: unit samples}}, chips shipped in each quarter.
    cumulative: the same shape, running totals. Omit it to take running sums of
                `quarterly` from its first quarter; pass it when the model defines its
                own series start.
    incomplete: quarters whose source data covers only part of the quarter.
    incomplete_cumulative: quarters whose running total is incomplete (defaults to
                `incomplete`).
    prices:     {chip: USD per chip}, for models with their own price estimates;
                other chips use the chip_types price.
    """
    inc = set(incomplete)
    inc_cum = inc if incomplete_cumulative is None else set(incomplete_cumulative)
    prices = {canonical_chip(c): v for c, v in (prices or {}).items()}
    return _build("sales", designer, quarterly, cumulative, inc, inc_cum, note, prices=prices)


def owners_tables(designer, quarterly_by_owner=None, cumulative_by_owner=None, incomplete=(),
                  incomplete_cumulative=None, note=None, prices=None):
    """The three owners tables for one designer: {owner: {quarter: {chip: samples}}}."""
    inc = set(incomplete)
    inc_cum = inc if incomplete_cumulative is None else set(incomplete_cumulative)
    prices = {canonical_chip(c): v for c, v in (prices or {}).items()}
    owners = list(dict.fromkeys([*(quarterly_by_owner or {}), *(cumulative_by_owner or {})]))
    parts = [
        _build("owners", designer, (quarterly_by_owner or {}).get(o), (cumulative_by_owner or {}).get(o),
               inc, inc_cum, note, owner=canonical_owner(o), prices=prices)
        for o in owners
    ]
    tables = [f"owners_{t}" for t in ("quarterly_by_chip", "cumulative_by_chip", "cumulative_by_designer")]
    return {t: pd.concat([p[t] for p in parts], ignore_index=True) if parts else
            pd.DataFrame(columns=table_columns(t)) for t in tables}


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------
def validate(table, df):
    """Raise if a canonical table breaks the schema. Returns the frame for chaining."""
    expected = table_columns(table)
    if list(df.columns) != expected:
        raise ValueError(f"{table}: columns {list(df.columns)} != {expected}")
    if df["name"].duplicated().any():
        raise ValueError(f"{table}: duplicate names {df.loc[df['name'].duplicated(), 'name'].head().tolist()}")
    bad = set(df["designer"]) - set(DESIGNERS)
    if bad:
        raise ValueError(f"{table}: unknown designers {bad}")
    if "chip_type" in df:
        rows, _ = chip_type_map()
        unknown = set(df["chip_type"]) - set(rows)
        if unknown:
            raise ValueError(f"{table}: chip types missing from the chip-type map: {unknown}")
    # Every row ends on the last day of a quarter; quarterly rows start on its first day.
    for _, r in df.iterrows():
        q = quarter_of(r["end_date"])
        start, end = quarter_bounds(q)
        first = r["start_date"] if "start_date" in df else r["series_start"]
        if r["end_date"] != end.isoformat() or ("start_date" in df and first != start.isoformat()):
            raise ValueError(f"{table}: {r['name']!r} does not span a calendar quarter ({first} to {r['end_date']})")
    for m in METRICS:
        lo, med, hi = (pd.to_numeric(df[f"{m}_{s}"], errors="coerce") for s in STATS)
        ok = lo.isna() | med.isna() | hi.isna() | ((lo <= med + 1e-9) & (med <= hi + 1e-9))
        if not ok.all():
            raise ValueError(f"{table}: {m} percentiles out of order in {df.loc[~ok, 'name'].head().tolist()}")
    return df


def write_tables(family, tables, out_dir=CANONICAL_EXPORT_DIR):
    """Write one family's canonical tables to canonical_export/<family>/<table>.csv."""
    folder = Path(out_dir) / family
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for table, df in tables.items():
        validate(table, df)
        path = folder / f"{table}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    print(f"canonical export: wrote {', '.join(f'{p.name} ({len(tables[p.stem])})' for p in paths)} to {folder}")
    return paths
