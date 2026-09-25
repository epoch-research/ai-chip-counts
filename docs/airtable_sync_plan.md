# Plan: replace the manual Airtable CSV import with a direct push

## Where the manual step actually sits

The chain is already automated on both sides of you:

```
notebooks / *_model.py
  └─ csv_export/, owners_csv_export/   (32 CSVs)
       └─ ***YOU, by hand, in the Airtable import UI***   ← the only manual link
            └─ Airtable (2 bases, 9 tables)
                 └─ epoch-website-astro/scripts/datahub/update_ai_chip_{sales,owners}_databases.py
                      └─ src/public/data/generated/*.csv + .zip
                           └─ S3 → Cloudflare → epoch.ai/data/ai-chip-sales
```

The website side runs hourly on cron (`.github/workflows/data-update.yml`, `AIRTABLE_API_TOKEN`
in the `data-pipeline` environment). Airtable stays the source of truth — it also holds
hand-curated rows the models don't produce (Huawei, Cambricon, the `chip_types` and
`organizations` tables). So the goal is **not** to bypass Airtable; it's to make the models
write into it the same way the website reads out of it.

## Why it's painful today, measured

**1. Picking the right table.** 24 model CSVs fan into 6 Airtable tables, and the names give
you no help — `nvidia_cumulative_totals.csv` goes to *cumulative timelines by designer*,
`nvidia_cumulative_by_chip.csv` goes to *cumulative timelines*.

**2. Column definitions genuinely disagree.** Not sloppiness on import — the exporters emit
different names for the same quantity, and two sibling Airtable tables disagree with each
other. Unmatched columns per file→table pair (15 of 24 pairs need hand-fixing):

| Target table | File | Columns that don't match the table |
|---|---|---|
| `cumulative_timelines` | nvidia / amd / tpu `_cumulative_by_chip` | `Compute estimate in H100e (median/5th/95th)` → table wants `H100e compute power (…)` |
| `cumulative_timelines` | `trainium_cumulative_by_chip` | all 6 metric columns + `Power in MW (…)`, `Source / Link` |
| `cumulative_by_designer` | `trainium_cumulative_totals` | `H100e (5th/95th)`, `Number of Units (…)`, `Source / Link` |
| `owners_quarters_by_chip_type` | `amd_owners_quarters_by_chip` | `Number of Units (median)` → table wants `Number of Units` |
| `owners_cumulative_by_chip_type` | nvidia / tpu / trainium `_owners_cumulative_by_chip` | `Number of Units` → table wants `Number of Units (median)` |
| `owners_cumulative_by_designer` | **all four** `_owners_cumulative_totals` | `Number of Units`, `Total TDP (W) (…)` |
| `timelines_by_chip` | `trainium_chip_timelines` | `Power in MW (…)` (no such field in table) |

Note rows 4 and 5: `owners_quarters_by_chip_type` wants `Number of Units`,
`owners_cumulative_by_chip_type` wants `Number of Units (median)`, and the exporters get it
backwards in opposite directions depending on which notebook wrote them. That is a
guaranteed hand-edit every single run.

**3. Fiddly details with no owner.** Date format (`M/D/YYYY` locally, ISO in Airtable),
chip-name aliases (`H100/H200`↔`H100`, `Instinct MI300X`↔`MI300X` — already documented in
`validate_chip_sales.py`), `Incomplete` booleans, and knowing which Airtable fields are
formulas/rollups you must *not* write (`Chip TDP (W)`, `Cost Estimate (USD)`,
`Last Modified`, `Chip type (linked)`, and in some tables `Total TDP (W)` — but not others).

**4. Merge scoping is on you.** Every table mixes 6 manufacturers, and the owners tables mix
9–10 owners. A bad merge key silently duplicates or clobbers rows belonging to a designer
you weren't touching.

## Two assets you already have

- **`run_chip_model.py`** is already the orchestrator, with a plugin convention: any
  `*_model.py` exposing `FAMILY`, `DESIGNER`, `run_model()`, `export_csvs()`. A `--push`
  step slots straight in after `export_csvs`.
- **`validate_chip_sales.py`'s `TABLE_SPECS` + `DESIGNER_CONFIGS`** is already *half the
  manifest this plan needs* — it maps local CSV → published table → per-column translation,
  and carries the chip aliases. It covers 3 of 6 tables and 3 of 4 designers, and reads the
  published zip (a lagging mirror) rather than Airtable.

And `Name` is unique across every one of the 6 tables — it's a clean merge key, which is
exactly what you're selecting by hand in the import UI today.

## The design

### One manifest, one push tool

Add to `ai-chip-counts`:

```
airtable_schema.yml     # the contract: table ids, keys, scope, field mapping
airtable_push.py        # plan → diff → upsert
```

`airtable_schema.yml`, one block per target table:

```yaml
owners_cumulative_by_chip_type:
  base: app0lYnFRVzEi3lkU
  table: tbl7p2wcXksNLzbjt
  key: [Name]                      # what batch_upsert merges on
  scope: [Chip manufacturer]       # rows this run owns; never touch anything else
  read_only:                       # verified against live schema, not trusted blindly
    - Last Modified
    - Last Modified By
    - Created
  fields:
    Name:                    Name
    Chip manufacturer:       Chip manufacturer
    Owner:                   Owner
    Chip type:               {from: Chip type, alias: chip_aliases}
    Start date:              {from: Start date, type: date}
    End date:                {from: End date, type: date}
    Number of Units (median):  [Number of Units, Number of Units (median)]  # accept either
    ...
```

The `[a, b]` form is the point: the manifest absorbs the dialect differences *once*, so
neither the notebooks nor Airtable have to change on day one.

### Upsert, not import

`pyairtable` 3.3.0 is already a dependency in the website repo and installed on your
`python3.11`. It has server-side merge:

```python
table.batch_upsert(records, key_fields=["Name"])
```

That is precisely the "merge on Name" you tick in the import dialog — same semantics, no UI.
10 records/request, 5 req/s per base; the largest table (515 rows) is ~52 requests, ~11s.

### What makes it safe to run unattended

- **Scoped.** Only rows matching `Chip manufacturer == <designer>` are considered. Huawei,
  Cambricon and hand-curated rows are invisible to the tool.
- **Dry run by default.** `--push` prints a plan — N creates, N updates, N unchanged, with
  per-cell % change reusing `validate_chip_sales.percent_diff` / `classify`. Writes need
  `--yes`. A run that would change >X% of cells or delete anything stops and asks.
- **Never deletes by default.** Orphans (in Airtable, in scope, absent locally) are *reported*.
  `--prune` deletes them, with explicit confirmation.
- **Snapshot before write.** Dump the in-scope rows to `airtable_snapshots/<table>-<ts>.csv`
  first. Rollback is re-upserting the snapshot.
- **Schema-checked.** Fetch `table.schema()` and refuse to write any field whose type is
  `formula`/`rollup`/`multipleLookupValues`/`count`/`lastModifiedTime`/`createdTime`/
  `createdBy`/`lastModifiedBy`/`autoNumber`/`button`. Also fails loudly if a manifest field
  no longer exists — so an Airtable rename surfaces as an error at plan time, not as a
  silently dropped column three weeks later.
- **Preflight on linked records.** `Chip type (linked)` links to the `chip_types` table. A
  brand-new chip (first B300 quarter) needs its `chip_types` row to exist first. The tool
  checks and tells you, rather than half-writing.

### End state

```bash
python run_chip_model.py nvidia --push          # run → export → validate → diff → confirm
python run_chip_model.py all --push --yes       # unattended
```

Optionally kick the website instead of waiting up to an hour for cron:

```bash
gh workflow run data-update.yml -R epoch-research/epoch-website-astro
```

## Sequencing

**Phase 0 — credentials (30 min).** Create an Airtable PAT with `data.records:read`,
`data.records:write`, `schema.bases:read` on `app0lYnFRVzEi3lkU` (chip sales/owners) and
`appGegFvzDKoCZNI0` (components, if you want it later). Keep it in 1Password / a local
`.env`; don't reuse the CI secret and don't commit it. A write-capable token against the
live public dataset is the main new risk surface here — that's what the snapshots and
dry-run default are for.

**Phase 1 — manifest + pilot table (half a day).** Write `airtable_schema.yml` for
`cumulative_by_designer` only (65 rows, columns already match for 3 of 4 designers) and
`airtable_push.py` with plan/diff/snapshot/upsert. Prove it on Nvidia: dry-run, compare
against the current published values, then write. Verify the site picks it up.

**Phase 2 — remaining 5 tables (1–2 days).** Extend the manifest table by table, in
increasing risk order: `timelines_by_chip` → `cumulative_timelines` →
`owners_quarters_by_chip_type` → `owners_cumulative_by_chip_type` →
`owners_cumulative_by_designer`. Each gets a dry-run-only pass first. Wire `--push` into
`run_chip_model.py`. **This is where the manual step dies.**

**Phase 3 — canonicalize the model side (1 day, optional but worth it).** With the manifest
carrying the translation, fix the *exporters* to emit one canonical vocabulary
(`Number of units (median)` / `Compute estimate in H100e (median)` everywhere) and shrink the
alias lists. Also resolve `nvidia_owners_quarters.csv` and `amd_owners_quarters.csv` — they
have no Airtable target today; either wire them up or delete them.

**Do not rename the Airtable fields to match.** The published CSVs are a public product; the
`Number of Units` / `Number of units (median)` split is baked into `epoch.ai/data/ai-chip-sales`
downloads and downstream consumers. The manifest absorbs that inconsistency permanently.

**Phase 4 — fold in validation (half a day).** `validate_chip_sales.py` should read from
Airtable via the same manifest instead of the published zip, so validation sees what you just
pushed rather than what shipped an hour ago. `TABLE_SPECS`/`DESIGNER_CONFIGS` collapse into
`airtable_schema.yml` — one mapping, used by both push and validate.

## Open questions

- **Trainium/Amazon has no `DESIGNER_CONFIGS` entry** — is it currently pushed to Airtable at
  all, and is `Power in MW` on the quarterly tables meant to be dropped or added as a field?
- **`China (smuggled)`** appears as an Owner in `owners_cumulative_by_designer` but I didn't
  trace which local export produces it — needs confirming before scoping deletes.
- **`nvidia_owners_OTHER_*.csv`** presumably merges into the same owners tables as the main
  files; the scope predicate needs to treat a designer's files as a set, not one file per table.
- **Table IDs would live in two repos** (here and the website's `config/config.yml`). Simplest
  is to vendor them here with a small consistency check; worth a nod from whoever owns the
  website datahub.
