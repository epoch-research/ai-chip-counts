# ai-chip-counts
Code for estimating quantities of AI chips

For more information, see:
https://epoch.ai/data/ai-chip-sales
https://epoch.ai/data/ai-chip-owners

## Layout

```
models/              one module per chip family (tpu.py, trainium.py so far) plus shared helpers (chip_utils.py)
notebooks/           research notebooks; the Nvidia and AMD ones still hold their model logic
scripts/             standalone estimates: xAI chip counts, smuggled chips to China (v_diversion_and_resale.py)
inputs/              hand-maintained input CSVs; most inputs come from Google Sheets
csv_export/, owners_csv_export/   legacy exports for the hand import into Airtable
outputs/canonical/   each family's tables in the new schema (docs/schema.md)
outputs/intermediate/  files one notebook writes for another (the Nvidia "Other" remainder)
pipeline/            the schema, the staging build, validation, the Airtable push, checking tools
staging/             local copy of what Airtable should hold
hub/                 HTML rebuild of the AI Chip Sales and AI Chip Owners explorers, reading staging/
docs/                schema.md (the schema), current_schema.md (the old one), refactor_plan.md
```

Install the dependencies with `pip install -e .` (add `[dev]` to run notebooks headlessly or
push to Airtable). Run everything from the repo root. Notebooks opened in Jupyter move there
in their first cell.

`notebooks/nvidia.ipynb`, `notebooks/amd.ipynb` and the `models/` modules estimate chip
sales by designer. `notebooks/nvidia_owners.ipynb` allocates Nvidia chips to hyperscaler and
official Chinese owners using a revenue-based model. Other Nvidia owners are modeled
separately in `notebooks/nvidia_owners_other.ipynb`, except smuggled Chinese chips, which are
handled in `scripts/v_diversion_and_resale.py`.

## Input data sheets

These are the inputs the models read from outside the repo at run time. Local inputs live in `inputs/`.

**Nvidia** (`notebooks/nvidia.ipynb`, `notebooks/nvidia_owners.ipynb`)
- [Nvidia revenue and prices sheet](https://docs.google.com/spreadsheets/d/1Yhu87Rw--9tviAuBwg_luL3OFAFkdHdVfli6tN215Xk/edit): the [revenue tab](https://docs.google.com/spreadsheets/d/1Yhu87Rw--9tviAuBwg_luL3OFAFkdHdVfli6tN215Xk/edit?gid=0#gid=0) holds quarterly data-center revenue, and the [prices tab](https://docs.google.com/spreadsheets/d/1Yhu87Rw--9tviAuBwg_luL3OFAFkdHdVfli6tN215Xk/edit?gid=1819303346#gid=1819303346) holds GPU prices. Both notebooks read it.
- [Nvidia ownership sheet](https://docs.google.com/spreadsheets/d/13GNAPRlzAm8nvaxCb6NDxu6-ozbZFDlcSJ335fEBsYk/edit?gid=1898856645#gid=1898856645): each hyperscaler's share of Nvidia DC revenue by quarter. Only the owners model reads it.
- `notebooks/nvidia_owners_other.ipynb` reads only local files: `inputs/`, `outputs/intermediate/`, and the Nvidia sales and owners exports.

**AMD** (`notebooks/amd.ipynb`, which covers both sales and owners). All inputs are tabs of the [AMD inputs sheet](https://docs.google.com/spreadsheets/d/1RfjL98gsSe9QeXDnBOraticvS-s13XmKvHv_dwwa3W8/edit):
- [quarterly revenue](https://docs.google.com/spreadsheets/d/1RfjL98gsSe9QeXDnBOraticvS-s13XmKvHv_dwwa3W8/edit?gid=0#gid=0)
- [production mix](https://docs.google.com/spreadsheets/d/1RfjL98gsSe9QeXDnBOraticvS-s13XmKvHv_dwwa3W8/edit?gid=141507805#gid=141507805)
- [GPU prices](https://docs.google.com/spreadsheets/d/1RfjL98gsSe9QeXDnBOraticvS-s13XmKvHv_dwwa3W8/edit?gid=1614409636#gid=1614409636)
- [ownership shares](https://docs.google.com/spreadsheets/d/1RfjL98gsSe9QeXDnBOraticvS-s13XmKvHv_dwwa3W8/edit?gid=1443829275#gid=1443829275)

**TPU** (`models/tpu.py`, `notebooks/tpu.ipynb`). The [TPU inputs sheet](https://docs.google.com/spreadsheets/d/1eGk2AAdewEO81vx-YBRTtdlhZvAMstY7vZuHrf3sgNI/edit) has three tabs, `TPU_Revenue` (Broadcom TPU revenue by fiscal quarter), `Production_Mix` and `prices` (manufacturing costs). The model assigns every TPU to Google, so it has no ownership sheet.

**Shared across all three.** Chip specs (8-bit TOPS and TDP, used for H100e and power) come from `chip_types.csv` inside the published [ai_chip_sales.zip](https://epoch.ai/data/ai_chip_sales.zip). If that download fails, each model falls back to hardcoded specs.

## Running and validating models

Chip-family models are being moved out of notebooks into `models/<family>.py` modules
(TPU and Trainium so far). For those families:

- `python run_chip_model.py tpu` (or `all`) reruns the model, rewrites the
  `csv_export/` and `owners_csv_export/` files and `outputs/canonical/<family>/`, and
  validates the exports against the published dataset, ending with a per-family PASS/REVIEW
  summary.
- `notebooks/<family>.ipynb` is the research notebook: it imports the model module and
  renders tables, charts, and sensitivity checks without owning the model logic.
- `python -m pipeline.validate --designer all` (or `notebooks/validate.ipynb`) compares the
  local CSVs against the currently published dataset on its own.

Families without a model module yet (Nvidia, AMD, the owners notebooks) still run via their
notebooks.

## The new back end (branch `backend-refactor`)

The models now also write the public tables themselves, in one consistent schema
(`docs/schema.md`), and push them straight to Airtable. This replaces the hand import. The
legacy `csv_export/` files are still written, so the old process keeps working until the
website switches over.

The data flows in one direction: model → `outputs/canonical/<family>/` → `staging/` →
Airtable → website. Six new Airtable tables (sales and owners, each quarterly by chip,
cumulative by chip and cumulative by designer) sit in the same base as the old ones. Their
IDs are in `pipeline/airtable_tables.json`, so they can be renamed freely in Airtable.

### One-time setup

Copy `.env.example` to `.env` (git-ignored) and fill in `AIRTABLE_API_KEY`, an Airtable
personal access token with the four scopes listed there, and `AIRTABLE_BASE_ID`. Never
commit the token or paste it anywhere else.

### Updating a model and Airtable

1. **Update the inputs.** Edit the Google Sheet or `inputs/` file the model reads, such as a
   new quarter's revenue. For a new chip, first add it to `chip_types` in Airtable. Then
   add it to `staging/chip_types.csv`, which is a snapshot, and to
   `pipeline/chip_type_map.csv`, which says which chip_types row supplies its specs.

2. **Rerun the model.** Every model writes its legacy CSVs and its new-schema tables in the
   same run.

   ```bash
   python3.11 run_chip_model.py tpu
   python3.11 -m pipeline.tools.run_notebook notebooks/nvidia.ipynb
   ```
  (note that all chip families will transition to the run_chip_model script; this is in progress)
   Use `run_chip_model.py` for model modules (TPU, Trainium), and `run_notebook` for notebook
   families (`notebooks/nvidia.ipynb`, `amd.ipynb`, `nvidia_owners.ipynb`). Or open
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

7. **Commit** the model changes, `outputs/canonical/` and `staging/` together, so the repo
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
the percentiles, Names and dates. See `export_canonical` in `models/trainium.py` for a short
example. Then rebuild staging and push as above.

### Status

- The six new tables were seeded in the real base on 2026-09-25 and match staging exactly.
- The website doesn't read them yet. Until it does, the live site still comes from the old
  tables, so a data update still needs the hand import too.
- The owners models aren't ready for 2026 data.

More detail: `pipeline/README.md` (every script), `staging/README.md` (the staging build),
`docs/refactor_plan.md` (the plan), and `HANDOFF.md` (open issues).
