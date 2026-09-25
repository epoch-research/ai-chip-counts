# pipeline/

The new back end for the chip sales and owners data: the canonical schema, the code that
assembles `staging/`, and the tools that check it. Run everything from the repo root.

| File | Role |
|---|---|
| `chip_schema.py` | The schema in code. Models pass unit samples; it computes H100e, power and cost per sample and writes `canonical_export/<family>/`. |
| `chip_type_map.csv` | Every chip label the results may use, its aliases, and which `staging/chip_types.csv` row supplies its specs. |
| `push_airtable.py` | Pushes staging's six tables to Airtable: creates missing tables and fields, upserts on Name, deletes stale rows, writes only modelled rows. Dry run unless `--write`. Finds tables by the IDs recorded in `airtable_tables.json`, so they can be renamed in Airtable. |
| `build_staging.py` | Merges `canonical_export/`, the converted legacy-only sources and `staging/curated/` into `staging/`. |
| `tools/build_viewer.py` | Writes `staging/viewer-sales.html` and `viewer-owners.html`: the hub explorers with the data built in, openable from disk. Runs at the end of every build. |
| `tools/run_notebook.py` | Runs a model notebook headlessly, without writing outputs into it. |
| `tools/diff_exports.py` | After a rerun, lists legacy CSVs whose numbers moved and restores those that differ only in timestamps. |
| `tools/audit_vs_published.py` | Compares every staging cell with the published data (the website's Airtable mirror), by table, field and designer. |
| `tools/compare_canonical.py` | Checks one family's canonical tables against the legacy CSVs they replace. |
| `tools/snapshot_curated_rows.py` | One-off: copied the Airtable-only rows into `staging/curated/`. |
| `tools/schema_inventory.py` | Regenerates `docs/current_schema.md`, the inventory of the legacy schema. |

```bash
python3.11 -m pipeline.build_staging
python3.11 -m pipeline.tools.compare_canonical nvidia
```

The schema itself is documented in `docs/schema.md`; how `staging/` is refreshed is in
`staging/README.md`.
