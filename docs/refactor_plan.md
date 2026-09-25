# Refactoring the chip sales back end

*Josh You, with Claude. Started 2026-09-11, revised 2026-09-22.
Companions in this folder: `current_schema.md` (what's wrong today), `schema.md` (what replaces
it) and `airtable_sync_plan.md` (an earlier, detailed plan for the Airtable push).*

## Status, 2026-09-23

Milestones 1 and 2 are done against a local stand-in for Airtable. Every model family emits
the canonical schema from its samples through `pipeline/chip_schema.py`,
`pipeline/build_staging.py` assembles the six tables in `staging/`, and `hub/` rebuilds both
explorers on top of them. Staging covers every row the site publishes;
`pipeline/tools/audit_vs_published.py` accounts for every cell that differs. The legacy exports
and the manual import still work unchanged. Next is the push tool, which writes staging into
six new Airtable tables.

## The problem

Every model update ends with someone importing CSVs into Airtable by hand. The import is the
visible pain, but it is hard for a deeper reason: nothing owns the schema.

```
this repo                                Airtable base app0lYnFRVzEi3lkU    epoch-website-astro
notebooks, tpu_model.py, 3 scripts       6 model-fed tables                 hourly cron reads the views
  -> 32 CSVs in about six dialects  ---- + chip_types, organizations ----->  -> public CSVs and zips
                                    hand import, matched on Name             -> the explorer charts
                                                                             -> ai-compute-users
```

The 32 CSVs spell the same quantity several ways, and so do the six Airtable tables they
feed. Each table formats its `Name` key differently, and the format also varies by chip
family. Chip specs are applied twice, once in the repo and again by Airtable lookups. The
owners notebooks pass data to one another through files, in an order only the person running
them knows. `docs/current_schema.md` lists every instance.

Airtable is not a pure pass-through. It holds rows no model produces: Huawei and Cambricon in
every table, xAI and smuggled China in the owners tables, and the `chip_types` and
`organizations` reference tables. Whatever we build has to leave those alone or take them over
deliberately.

## The decision

The repo owns the schema, and Airtable mirrors it. Airtable stays: it is where the curated rows
live and where the website reads from. It stops being a place where column names get decided.

This is how `ai-chip-components` already works. Its `generate_tables.py` returns snake_case
DataFrames, and its `airtable.py` renames them to display names and upserts them on `Name`
from GitHub Actions every hour.

Two alternatives were considered. **Automating the import alone** would kill the manual step
in a few days, but it would encode today's inconsistencies in a mapping file forever. It
survives as a milestone below. **Bypassing Airtable entirely**, as `ai-compute-users` does by
committing CSVs the website fetches from GitHub, is the cleanest end state. It gives up
Airtable as the editing surface for curated rows, though, so it waits until we know whether
anyone edits them there. Nothing in this plan closes that door.

## Existing work to build on

- **`ai-chips-python-test`** (`~/Desktop`, June 2026, never pushed). The Nvidia sales model
  as an importable module, already in the canonical column style, with medians that match the
  notebook.
- **`tpu_model.py` and `run_chip_model.py`**, on `main`. TPU is a module, and the runner
  discovers any `*_model.py`, reruns it and validates the output.
- **`origin/airtable-sync`** (Edu, July 2026, unmerged). An upload script that runs the
  notebooks with papermill, normalises the column dialects by a hard-coded map and syncs each
  table. Its workflow sends pushes on `main` to a production base and manual runs to a test
  base, which is worth keeping. It still references `nvidia_estimates.ipynb`, since renamed.
  Ask Edu where it stands before writing a push tool.
- **`validate_chip_sales.py`**, on `main`. Compares the exports with the published zip.

## Milestones

**1. Sales tables in the canonical schema. Done.** Each sales model (the Nvidia, AMD and
Trainium notebooks, `tpu_model.py`) ends with one export step that hands its unit samples to
`chip_schema`. The model code itself is unchanged, and the legacy CSVs are still written.

**2. Owners tables. Done.** The same for the Nvidia and AMD owners allocations. The Nvidia
"Other" remainder and CoreWeave are converted from their legacy CSVs, since those notebooks
carry no samples. Huawei, Cambricon, xAI and smuggled China live in `staging/curated/`.

**3. Push tool. Written 2026-09-24 as `pipeline/push_airtable.py`, tested against a fake of
Airtable's API; not yet run against Airtable.** A script that writes `staging/`'s six tables into six *new*
Airtable tables (see the next section). Start from Edu's `airtable-sync` branch and the
components repo's `airtable.py`. Upsert on Name; dry-run by default, with a per-table diff;
snapshot rows before writing; never write formula or lookup fields; `--tables sales` to hold
back the owners tables. It writes only rows a model produces (`docs/schema.md`, section 8) and
refuses anything else, including curated rows, `chip_types` and `organizations`. The one
exception is a `--transition` run that seeds the new tables with the cleaned curated rows.
A companion pull step refreshes `staging/chip_types.csv` and, after the transition,
`staging/curated/` from Airtable before each build. It needs a write-scoped Airtable token for
this base.

**4. Website switch, 1 to 2 days plus coordination.** Point the website at the new tables and
the new column names. See the next section.

**5. CI, half a day.** Models run locally and their output is committed to `staging/`, so every
data change is reviewed as a diff. A GitHub Action pushes the committed staging to Airtable on
merge to `main`, gated behind a reviewer, with the dry-run diff in the job summary. Running
the models themselves in CI can come later.

## Cutting over through new tables

The push writes into six new tables with the canonical names and columns, alongside the six
the site reads today. Nothing public changes until the website is pointed at them, and
pointing it back is the rollback.

This replaces the earlier plan of adding new fields beside the old ones in the same tables.
New tables avoid a period of duplicated fields, avoid renaming anything the site is reading,
and keep a change of Name template (which replaces every row) out of the live tables.

| New table | Replaces (website config key) | Current table ID |
|---|---|---|
| Sales: quarterly by chip | `consolidated_timelines` | `tblyL1FwqiMRzCkeX` |
| Sales: cumulative by chip | `culmulative_timelines` | `tblJWvrfdWJJNXlSl` |
| Sales: cumulative by designer | `cumulative_timelines_by_designer` | `tblb8LAUvR9OufJBl` |
| Owners: quarterly by chip | `owners_quarters_by_chip_type` | `tblf8X5qu4r1Hcfjy` |
| Owners: cumulative by chip | `owners_cumulative_totals_by_chip_type` | `tbl7p2wcXksNLzbjt` |
| Owners: cumulative by designer | `owners_cumulative_totals_by_designer` | `tblU5qHjez8cgXkSE` |

`chip_types` and `organizations` stay as they are and are shared. Remove the empty AMD stub
rows from `chip_types` first; the live AMD power bug comes from them.

The sequence:

1. **Rehearse** the push against a duplicate of the base, then run it for real into the new
   tables, once with `--transition` to seed the curated rows, and give each a published view.
   From then on the curated rows are edited in Airtable only.
2. **Change the website on a branch.** Point `scripts/datahub/config/config.yml` at the new
   table and view IDs (and fix the `culmulative` key while there). Update the column names in
   `legacy/vizs/ai-chip-sales` and `ai-chip-owners` and delete their workarounds, such as the
   chip-name mapping and the TDP-to-power conversion; `hub/` in this repo is the same logic
   already written against the new names. Update `ai-compute-users/epoch_data.py`, which reads
   the published zips by column name, and the data pages' records and changelog sections.
3. **Preview** the website branch with the data workflow's per-branch deploy, and run
   `audit_vs_published` against that preview's files.
4. **Merge.** If anything is wrong, revert the config change and the site reads the old tables.
5. **Retire** the hand import, the legacy CSV exports and the old tables after a cycle or two
   runs cleanly.

The files inside the public zips are renamed to match the new tables in step 2, in the same
release as the column changes (`docs/schema.md`, section 7).

## Converging with Edu's epochutils setup

Edu's `origin/airtable-sync` branch is one instance of an org-wide pattern in
[epoch-research/epochutils](https://github.com/epoch-research/epochutils): `sync_dataframe`
(create tables and fields, upsert on the primary field, prune stale rows) plus a standard
deployment (`docs/airtable-upload-setup.md`: production on push to `main`, a separate test base on
manual dispatch, per-environment tokens, branch protection, CODEOWNERS). That is the long-term
delivery mechanism; this repo's contribution is what gets delivered, the canonical schema and
staging.

`pipeline/push_airtable.py` is an interim tool, used to test the new tables and the website
switch without first changing `epochutils`. To converge:

1. Agree the schema direction with Edu: new tables, no lookups, and the ownership rule
   (`docs/schema.md`, section 8). His script targets the old schema, which this supersedes.
2. Add two features to `epochutils`, or keep them as a thin wrapper here: pruning limited to the
   rows a caller owns (plain pruning would delete curated rows), and finding tables by ID
   rather than name.
3. Replace `push_airtable.py` with `sync_dataframe`, keeping the dry-run diff and pre-write
   snapshot as a wrapper or upstreaming them.
4. Adopt the standard workflow, environments and branch protection as they are. Generation runs
   in CI, as the guide requires, rather than pushing committed CSVs.
5. Reduce his `upload_to_airtable.py` to: run the models, build staging, sync the six tables.

Questions for Edu: which base his production environment targets; whether the two features
belong in `epochutils`; how he wants the one-time curated-row transition handled; and that six
new tables already exist in the chip sales base, created by the interim tool.

## Open questions

- Decided 2026-09-24: Airtable owns the curated rows, `chip_types` and `organizations`; the
  repo writes only modelled rows, apart from a one-time transition (`docs/schema.md`, section 8).
- Is `chip_types` linked from other bases, such as ML Hardware? If so, it remains the home of
  chip specs and the models read it.
- The schema choices, decided 2026-09-24 (`docs/schema.md`, end).
