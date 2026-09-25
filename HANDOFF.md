# Handoff

Cross-session state for ai-chip-counts. Keep under about 40 lines; git history is the journal.

## In flight (branch `backend-refactor`, uncommitted as of 2026-09-24)

- **Canonical schema** (`docs/schema.md`, decided 2026-09-24) lives in `pipeline/chip_schema.py`.
  Each model family passes unit samples through it into `canonical_export/<family>/`; legacy
  CSV exports and the hand import are untouched.
- **`pipeline/build_staging.py`** merges those with the converted Nvidia "Other" and CoreWeave
  files and `staging/curated/` into `staging/`, the stand-in for Airtable. `hub/` and
  `staging/viewer-*.html` show it; `pipeline/tools/audit_vs_published.py` checks it.
- **Next:** the website switch. The six new tables are live in the real base (seeded
  2026-09-25; IDs in `pipeline/airtable_tables.json`) but nothing reads them yet
  (`docs/refactor_plan.md`). Airtable owns curated rows, chip_types and organizations; the repo
  writes only modelled rows, after a one-time transition.

## Recent changes

- 2026-09-24: schema decisions settled; ownership rule in `docs/schema.md`, section 8.
  Converted and curated rows now keep their source Notes and generation stamps, so
  rebuilding staging without new model output changes nothing.
- 2026-09-24: dropped the Quarter and Cumulative through columns; cumulative Names now read
  "… cumulative Q1 2022 through Q3 2025". The rehearsal base's tables predate this.
- 2026-09-25: real base seeded with `push_airtable --transition`; read-back matches staging
  cell for cell, and the six old tables are untouched.
- 2026-09-23: TPU cost uses the model's own prices (`tpu_model.tpu_prices`, via `prices=`).
- 2026-09-22: `nvidia_owners.ipynb` drops quarters with no ownership shares (Edu's fix) and
  bases its incomplete flag on the last kept quarter; AMD canonical owners likewise stop at
  the last year with shares. Legacy outputs are unchanged.

## Open decisions

- AMD: the sheet's FY26 Q2 revenue range changed after the 8/28 export and the committed
  owners CSVs date from April; a rerun changes both. Owners updates for 2026 are deferred.
- Cost for Nvidia, AMD and Trainium uses `chip_types` list prices, as production does. Using the
  models' own sampled prices is a methodology change to scope separately.

- Converge with Edu's `epochutils` setup later (`docs/refactor_plan.md`, "Converging");
  `pipeline/push_airtable.py` is interim, for testing the new tables and website switch.

## Known production issues (staging is correct)

- AMD power is 0 in the published cumulative-by-chip table: its chips link to the empty
  `MI300X`-style stub rows in chip_types. Delete the stubs before the transition.
- Amazon's designer-level unit median is its 5th percentile (hand-import column mapping).
- Google owner rows are from a 3/27 run; the June TPU update was never imported.

## Gotchas

- Run notebooks with `python3.11 -m pipeline.tools.run_notebook <nb>` (nbconvert is broken
  here); `python3.11 -m pipeline.tools.diff_exports --restore --only <prefix>` then restores
  legacy files whose only change is the Notes timestamp.
