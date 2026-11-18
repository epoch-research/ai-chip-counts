#!/usr/bin/env python3.11
"""Push staging/'s six tables into their Airtable tables. Dry run unless --write.

    python3.11 -m pipeline.push_airtable                      # plan: what would change
    python3.11 -m pipeline.push_airtable --write              # do it
    python3.11 -m pipeline.push_airtable --tables sales       # sales tables only
    python3.11 -m pipeline.push_airtable --transition --write # one-time: also seed curated rows

Mirrors ai-chip-components' airtable.py: tables and fields that don't exist yet are created
through the metadata API, rows are upserted on Name, and rows that no longer exist are
deleted. On top of that, following docs/schema.md section 8:

* It writes only rows a model produces. A row belongs to the repo when its designer (and, in
  the owners tables, owner) pair appears in a model family's output. Curated rows, such as
  Huawei, Cambricon, xAI and smuggled China, are Airtable's: never written or deleted,
  except by one --transition run that seeds new tables from staging/curated/.
* It never writes chip_types or organizations, or any field Airtable computes.
* Every write run first saves the in-scope Airtable rows to staging/_snapshots/.
* Tables are found by ID, never by name, so they can be renamed freely in Airtable. The
  names below are used once, when a table is created; its ID is then recorded per base in
  pipeline/airtable_tables.json, which is committed.

Credentials come from the environment or a .env file at the repo root (see .env.example):
AIRTABLE_API_KEY, a personal access token with data.records:read, data.records:write,
schema.bases:read and schema.bases:write on the base; and AIRTABLE_BASE_ID.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import chip_schema as cs  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv(cs.REPO / ".env")
except ImportError:
    pass

API_URL = "https://api.airtable.com/v0"
BATCH_SIZE = 10          # Airtable caps create/update/delete at 10 records per request
STAGING = cs.REPO / "staging"
SNAPSHOTS = STAGING / "_snapshots"
TABLE_IDS_PATH = cs.REPO / "pipeline" / "airtable_tables.json"

# canonical table -> the name a table gets when first created. Afterwards it is found by ID and
# may be renamed in Airtable. The primary field is Name in all six.
TABLE_NAMES = {
    "sales_quarterly_by_chip": "Sales: quarterly by chip",
    "sales_cumulative_by_chip": "Sales: cumulative by chip",
    "sales_cumulative_by_designer": "Sales: cumulative by designer",
    "owners_quarterly_by_chip": "Owners: quarterly by chip",
    "owners_cumulative_by_chip": "Owners: cumulative by chip",
    "owners_cumulative_by_designer": "Owners: cumulative by designer",
}
PRIMARY = "Name"
DATE_FIELDS = {"Start date", "End date", "Series start date"}
# Field types Airtable computes; the push refuses to write a field of these types.
COMPUTED_TYPES = {"formula", "rollup", "multipleLookupValues", "count", "lookup", "autoNumber",
                  "createdTime", "lastModifiedTime", "createdBy", "lastModifiedBy", "button"}


# ---------------------------------------------------------------- HTTP, as in ai-chip-components
def _request(method, url, api_key, **kwargs):
    for attempt in range(5):
        resp = requests.request(method, url, headers={"Authorization": f"Bearer {api_key}",
                                                      "Content-Type": "application/json"},
                                timeout=30, **kwargs)
        if resp.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        if not resp.ok:
            raise RuntimeError(f"{method} {url} -> {resp.status_code}: {resp.text}")
        time.sleep(0.21)     # stay under Airtable's 5 requests per second per base
        return resp.json() if resp.content else {}
    raise RuntimeError(f"rate-limited after retries: {method} {url}")


def list_tables(base_id, api_key):
    """Every table in the base, keyed by table ID."""
    data = _request("GET", f"{API_URL}/meta/bases/{base_id}/tables", api_key)
    return {t["id"]: t for t in data.get("tables", [])}


def recorded_ids(base_id):
    """{canonical table: Airtable table ID} recorded for this base."""
    ids = json.loads(TABLE_IDS_PATH.read_text()) if TABLE_IDS_PATH.exists() else {}
    return ids.get(base_id, {})


def record_id(base_id, table, table_id):
    ids = json.loads(TABLE_IDS_PATH.read_text()) if TABLE_IDS_PATH.exists() else {}
    ids.setdefault(base_id, {})[table] = table_id
    TABLE_IDS_PATH.write_text(json.dumps(ids, indent=2, sort_keys=True) + "\n")


def list_records(base_id, api_key, table_id):
    out, offset = [], None
    while True:
        params = [("pageSize", "100")] + ([("offset", offset)] if offset else [])
        data = _request("GET", f"{API_URL}/{base_id}/{table_id}", api_key, params=params)
        out.extend(data.get("records", []))
        offset = data.get("offset")
        if not offset:
            return out


# ---------------------------------------------------------------- fields and values
def field_spec(name):
    if name == "Notes":
        return {"name": name, "type": "multilineText"}
    if name == "Incomplete":
        return {"name": name, "type": "checkbox", "options": {"icon": "check", "color": "yellowBright"}}
    if name in DATE_FIELDS:
        return {"name": name, "type": "date", "options": {"dateFormat": {"name": "iso", "format": "YYYY-MM-DD"}}}
    if name in cs.CANONICAL_NAMES and cs.CANONICAL_NAMES[name] in cs.metric_columns():
        return {"name": name, "type": "number", "options": {"precision": 3 if "MW" in name else 0}}
    return {"name": name, "type": "singleLineText"}


def cell(name, value):
    """A staging CSV cell as the value Airtable stores, or None for blank."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return False if name == "Incomplete" else None
    if name == "Incomplete":
        return str(value).lower() == "true"
    if field_spec(name)["type"] == "number":
        v = float(value)
        return int(v) if v.is_integer() and "MW" not in name else v
    return str(value)


def same(a, b):
    """Whether a staging value and an Airtable value are the same cell."""
    if a in (None, False, "") and b in (None, False, ""):
        return True
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


# ---------------------------------------------------------------- what the repo owns
def pair_of(fields):
    return (fields.get("Designer") or "", fields.get("Owner") or "")


def load_rows(table, transition):
    """Staging rows this run may write, and the (designer, owner) pairs the repo owns."""
    stg = pd.read_csv(STAGING / f"{table}.csv", dtype=str, keep_default_na=False)
    cur_path = STAGING / "curated" / f"{table}.csv"
    curated = set(pd.read_csv(cur_path, dtype=str, keep_default_na=False)["Name"]) if cur_path.exists() else set()
    rows = {}
    for r in stg.to_dict("records"):
        fields = {k: cell(k, v) for k, v in r.items()}
        rows[r["Name"]] = (fields, r["Name"] in curated)
    owned_pairs = {pair_of(f) for f, is_cur in rows.values() if not is_cur}
    curated_pairs = {pair_of(f) for f, is_cur in rows.values() if is_cur}
    clash = owned_pairs & curated_pairs
    if clash:
        raise SystemExit(f"{table}: designer/owner pairs are both modelled and curated: {sorted(clash)}")
    writable = {n: f for n, (f, is_cur) in rows.items() if transition or not is_cur}
    return writable, owned_pairs | (curated_pairs if transition else set()), list(stg.columns)


def plan_table(table, existing_records, transition):
    """(creates, updates, deletes, unchanged, skipped) for one table."""
    writable, pairs, columns = load_rows(table, transition)
    by_name = {r["fields"].get(PRIMARY): r for r in existing_records}
    creates, updates, unchanged = [], [], 0
    for name, fields in writable.items():
        cur = by_name.get(name)
        if cur is None:
            creates.append(fields)
        elif all(same(v, cur["fields"].get(k)) for k, v in fields.items()):
            unchanged += 1
        else:
            updates.append((cur["id"], fields))
    in_scope = [r for r in existing_records if pair_of(r["fields"]) in pairs]
    deletes = [r["id"] for r in in_scope if r["fields"].get(PRIMARY) not in writable]
    skipped = len(existing_records) - len(in_scope)
    return columns, creates, updates, deletes, unchanged, skipped


# ---------------------------------------------------------------- writes
def ensure_table(base_id, api_key, existing, canonical, columns, write):
    """Find the table by its recorded ID, or create it; add missing fields.

    Returns (table id or None, notes). A recorded ID that no longer exists is an error, not
    a cue to create a new table: the table was deleted, or the ID file points at the wrong base.
    """
    specs = [field_spec(c) for c in [PRIMARY] + [c for c in columns if c != PRIMARY]]
    table_id = recorded_ids(base_id).get(canonical)
    if table_id and table_id not in existing:
        raise SystemExit(f"{canonical}: recorded table {table_id} is not in base {base_id}. If it was deleted "
                         f"on purpose, remove its entry from {TABLE_IDS_PATH.name} to create a new one.")
    if not table_id:
        name = TABLE_NAMES[canonical]
        if any(t["name"] == name for t in existing.values()):
            raise SystemExit(f"{canonical}: base {base_id} already has a table named {name!r} that isn't recorded in "
                             f"{TABLE_IDS_PATH.name}. Add its ID there, or rename it, rather than create a duplicate.")
        if not write:
            return None, [f"would create table {name!r} with {len(specs)} fields"]
        created = _request("POST", f"{API_URL}/meta/bases/{base_id}/tables", api_key,
                           json={"name": name, "fields": specs})
        record_id(base_id, canonical, created["id"])
        return created["id"], [f"created table {name!r} ({created['id']})"]
    table = existing[table_id]
    have = {f["name"]: f for f in table.get("fields", [])}
    computed = [f["name"] for f in table["fields"] if f["type"] in COMPUTED_TYPES and f["name"] in columns]
    if computed:
        raise SystemExit(f"{canonical}: refusing to write computed fields {computed}")
    notes = []
    for spec in specs:
        if spec["name"] not in have:
            notes.append(f"{'added' if write else 'would add'} field {spec['name']}")
            if write:
                _request("POST", f"{API_URL}/meta/bases/{base_id}/tables/{table['id']}/fields", api_key, json=spec)
    return table["id"], notes


def batched(items):
    for i in range(0, len(items), BATCH_SIZE):
        yield items[i:i + BATCH_SIZE]


def apply(base_id, api_key, table_id, creates, updates, deletes):
    url = f"{API_URL}/{base_id}/{table_id}"
    for b in batched(creates):
        # New records leave blank fields out; updates send None, which clears a field.
        _request("POST", url, api_key, json={"records": [{"fields": {k: v for k, v in f.items() if v is not None}}
                                                         for f in b], "typecast": True})
    for b in batched(updates):
        _request("PATCH", url, api_key, json={"records": [{"id": i, "fields": f} for i, f in b], "typecast": True})
    for b in batched(deletes):
        _request("DELETE", url, api_key, params=[("records[]", i) for i in b])


def views_of(base_id, api_key, canonical):
    t = list_tables(base_id, api_key).get(recorded_ids(base_id).get(canonical))
    return (t["id"], t["name"], [(v["id"], v["name"]) for v in t.get("views", [])]) if t else (None, None, [])


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="make the changes; without it, only report them")
    ap.add_argument("--tables", choices=["all", "sales", "owners"], default="all")
    ap.add_argument("--transition", action="store_true",
                    help="one time only: also write the curated rows, to seed new tables")
    ap.add_argument("--base", help="base id, overriding AIRTABLE_BASE_ID (e.g. a rehearsal copy)")
    args = ap.parse_args(argv)

    api_key = os.environ.get("AIRTABLE_API_KEY")
    base_id = args.base or os.environ.get("AIRTABLE_BASE_ID")
    tables = [t for t in cs.TABLES if args.tables == "all" or t.startswith(args.tables)]

    if not api_key or not base_id:
        print("AIRTABLE_API_KEY and AIRTABLE_BASE_ID are not set (see .env.example); showing staging only.\n")
        for t in tables:
            writable, pairs, columns = load_rows(t, args.transition)
            print(f"{TABLE_NAMES[t]:<32} {len(writable):>4} rows the repo would write, {len(columns)} fields, "
                  f"{len(pairs)} designer/owner pairs")
        return 1

    mode = "WRITE" if args.write else "dry run"
    print(f"{mode} to base {base_id}{' (transition: curated rows included)' if args.transition else ''}\n")
    existing = list_tables(base_id, api_key)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    for t in tables:
        table_id, notes = ensure_table(base_id, api_key, existing, t, load_rows(t, args.transition)[2], args.write)
        name = existing[table_id]["name"] if table_id in existing else TABLE_NAMES[t]
        records = list_records(base_id, api_key, table_id) if table_id else []
        _, creates, updates, deletes, unchanged, skipped = plan_table(t, records, args.transition)
        print(f"{name:<32} create {len(creates):>4}  update {len(updates):>4}  delete {len(deletes):>4}  "
              f"unchanged {unchanged:>4}  not the repo's {skipped:>4}" + (f"   [{'; '.join(notes)}]" if notes else ""))
        if args.write and table_id and (creates or updates or deletes):
            if records:
                SNAPSHOTS.mkdir(parents=True, exist_ok=True)
                (SNAPSHOTS / f"{t}-{stamp}.json").write_text(json.dumps(records))
            apply(base_id, api_key, table_id, creates, updates, deletes)
    if args.write:
        print("\nTable and view IDs, for epoch-website-astro's scripts/datahub/config/config.yml:")
        for t in tables:
            table_id, name, views = views_of(base_id, api_key, t)
            print(f"  {t:<31} table {table_id} ({name!r})  views {views}")
    else:
        print("\nNothing was written. Rerun with --write to apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
