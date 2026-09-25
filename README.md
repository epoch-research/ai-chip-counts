# ai-chip-counts
Code for estimating quantities of AI chips

For more information, see:
https://epoch.ai/data/ai-chip-sales
https://epoch.ai/data/ai-chip-owners

nvidia_estimates, tpu_estimates, amd_estimates, etc generate estimates for chip sales by designer

nvidia_owners allocates Nvidia chips to hyperscaler and official Chinese owners using a revenue-based model. Other Nvidia owners are modeled separately in nvidia_owners_other, with the exception of smuggled Chinese chips, which are handled separately in v_diversion_and_resale.

## Running and validating models

Chip-family models are being moved out of notebooks into `<family>_model.py` modules
(TPU so far). For those families:

- `python run_chip_model.py tpu` (or `all`) reruns the model, rewrites the
  `csv_export/` and `owners_csv_export/` files, and validates the exports against the
  published dataset, ending with a per-family PASS/REVIEW summary.
- `<family>_estimates.ipynb` is the research notebook: it imports the model module and
  renders tables, charts, and sensitivity checks without owning the model logic.
- `python validate_chip_sales.py --designer all` (or the `validate_chip_sales.ipynb`
  notebook) compares the local CSVs against the currently published dataset on its own.

Families without a model module yet (Nvidia, AMD, Amazon) still run via their notebooks.

## The new back end (branch `backend-refactor`)

The models now also write the public tables themselves, in one consistent schema
(`docs/schema.md`), and push them straight to Airtable. This replaces the hand import. The
legacy `csv_export/` files are still written, so the old process keeps working until the
website switches over.

```
pipeline/            the schema in code (chip_schema.py), the staging build, the Airtable push, checking tools
canonical_export/    each model family's output tables, in the new schema
staging/             local copy of what Airtable should hold: the six public tables plus reference tables
hub/                 HTML rebuild of the AI Chip Sales and AI Chip Owners explorers, reading staging/
docs/                schema.md (the schema), current_schema.md (the old one), refactor_plan.md
```

The data flows in one direction: model → `canonical_export/<family>/` → `staging/` →
Airtable → website. Six new Airtable tables (sales and owners, each quarterly by chip,
cumulative by chip and cumulative by designer) sit in the same base as the old ones. Their
IDs are in `pipeline/airtable_tables.json`, so they can be renamed freely in Airtable.

### One-time setup

Copy `.env.example` to `.env` (git-ignored) and fill in `AIRTABLE_API_KEY`, an Airtable
personal access token with the four scopes listed there, and `AIRTABLE_BASE_ID`. Never
commit the token or paste it anywhere else.

### Updating a model and Airtable

1. **Update the inputs.** Edit the Google Sheet or `data_inputs/` file the model reads, such as a
   new quarter's revenue. For a new chip, first add it to `chip_types` in Airtable. Then
   add it to `staging/chip_types.csv`, which is a snapshot, and to
   `pipeline/chip_type_map.csv`, which says which chip_types row supplies its specs.

2. **Rerun the model.** Every model writes its legacy CSVs and its new-schema tables in the
   same run.

   ```bash
   python3.11 run_chip_model.py tpu
   python3.11 -m pipeline.tools.run_notebook nvidia_chip_estimates.ipynb
   ```
  (note that all chip families will transition to the run_chip_model script; this is in progress)
   Use `run_chip_model.py` for model modules (TPU), and `run_notebook` for notebook families
   (`nvidia_chip_estimates`, `amd_estimates`, `trainium_estimates`, `nvidia_owners`). Or open
   the notebook in Jupyter and run all cells. To see which legacy numbers moved, and to undo
   changes that are only timestamps, run
   `python3.11 -m pipeline.tools.diff_exports --restore --only <prefix>`.

3. **Rebuild staging.**

   ```bash
   python3.11 -m pipeline.build_staging
   ```

   This merges every family with the hand-maintained rows in `staging/curated/`. It refuses to
   write if two sources claim the same row, a chip is missing from the chip-type map, or a
   table breaks the schema.

4. **Check it.**
   - **Look at it.** Open `staging/viewer-sales.html` or `viewer-owners.html` straight from
     disk. The table tab shows every column and flags anything odd.
   - **Compare with the old exports:** `python3.11 -m pipeline.tools.compare_canonical <family>`.
   - **Compare with the live site:** `python3.11 -m pipeline.tools.audit_vs_published`. Expect
     differences where you changed the model, and nowhere else.

5. **Preview the push.**

   ```bash
   python3.11 -m pipeline.push_airtable
   ```

   This is a dry run. It lists, per table, the rows it would create, update and delete, plus
   the rows it leaves alone because a model doesn't produce them. Add `--tables sales` or
   `--tables owners` to limit it. Check that the changes match what you changed in the model.

6. **Push.**

   ```bash
   python3.11 -m pipeline.push_airtable --write
   ```

   It saves the current Airtable rows to `staging/_snapshots/` first, then writes. Running it
   again should report nothing to change.

7. **Commit** the model changes, `canonical_export/` and `staging/` together, so the repo
   records what was pushed.

### Who owns what

- **The repo owns** every row a model produces: a designer, or a designer and owner pair in the
  owners tables. The push creates, updates and deletes only those.
- **Airtable owns** everything else. Edit these in Airtable, and the push never touches them:
  - the hand-curated rows (Huawei, Cambricon, xAI, smuggled chips to China);
  - `chip_types`;
  - `organizations`.
- `staging/curated/`, `staging/chip_types.csv` and `staging/organizations.csv` are read-only
  copies. Refresh them from Airtable after editing there. There is no pull step yet, so this
  is by hand.
- If a designer moves from curated to modelled, delete its curated rows (in Airtable and in
  `staging/curated/`) before pushing. The push refuses to run while both claim the same pair.

### Adding a new model family

In the model, call `chip_schema.sales_tables(...)` or `owners_tables(...)` with the unit
samples (`{quarter: {chip: samples}}`). Then pass the result to
`chip_schema.write_tables("<family>", tables)`. The library computes H100e, power and cost,
the percentiles, Names and dates. See the last cell of `trainium_estimates.ipynb` for a short
example. Then rebuild staging and push as above.

### Status

- The six new tables were seeded in the real base on 2026-09-25 and match staging exactly.
- The website doesn't read them yet. Until it does, the live site still comes from the old
  tables, so a data update still needs the hand import too.
- The owners models aren't ready for 2026 data.

More detail: `pipeline/README.md` (every script), `staging/README.md` (the staging build),
`docs/refactor_plan.md` (the plan), and `HANDOFF.md` (open issues).
