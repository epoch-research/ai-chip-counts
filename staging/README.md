# staging/

A local stand-in for the Airtable base behind epoch.ai/data/ai-chip-sales and
epoch.ai/data/ai-chip-owners. It holds what the base would hold after the models push to it,
in the canonical schema of `docs/schema.md`, with display column names. The HTML hub in
`hub/` reads only this folder.

Nothing here feeds the live site. The current process, exporting to `csv_export/` and
`owners_csv_export/` and importing into Airtable by hand, is unchanged and still works.

## Files

| File | What it is | Maintained by |
|---|---|---|
| `sales_quarterly_by_chip.csv` and the other five results tables | The six results tables | `pipeline/build_staging.py`, never by hand |
| `curated/<table>.csv` | Rows no model produces: Huawei, Cambricon, xAI, smuggled China | Airtable. Until the transition, a cleaned snapshot seeds the new tables; after it, a read-only copy |
| `chip_types.csv` | Chip specs: 8-bit OP/s, TDP, price, release date | Airtable; read-only copy here |
| `organizations.csv` | Designers and owners, with country | Airtable; read-only copy here |
| `viewer-sales.html`, `viewer-owners.html` | The hub explorers with every table above built in; open from disk, no server | `pipeline/build_staging.py` |
| `manifest.json` | Build time, and row counts per table and source | `pipeline/build_staging.py` |

`chip_types.csv` and `organizations.csv` are snapshots of the Airtable tables of the same
name, with the empty AMD stub rows (`MI300X` and so on, next to the full `Instinct MI300X`
rows) removed. The models read chip specs from `chip_types.csv` through
`pipeline/chip_type_map.csv`.

## Refreshing it

```bash
python3.11 run_chip_model.py tpu                          # TPU: model, legacy CSVs, canonical tables
python3.11 -m pipeline.tools.run_notebook nvidia_chip_estimates.ipynb amd_estimates.ipynb \
    trainium_estimates.ipynb nvidia_owners.ipynb           # the notebook families
python3.11 -m pipeline.build_staging                      # merge everything into staging/
python3.11 -m http.server 8765                            # then open http://localhost:8765/hub/
```

Each model writes its own canonical tables to `canonical_export/<family>/` alongside its
legacy CSVs. `pipeline/build_staging.py` then:

1. converts the two legacy sources with no samples behind them, the Nvidia "Other"
   remainder and CoreWeave, into `canonical_export/nvidia_other/` and
   `canonical_export/coreweave/`;
2. merges every `canonical_export/` family with `curated/`;
3. refuses to write if two sources claim the same row, a chip type is missing from the
   chip-type map, or a table breaks the schema.

To check one family against the legacy CSVs it replaces:

```bash
python3.11 -m pipeline.tools.compare_canonical nvidia
```

## Known gaps

- **Designer-level cost for converted rows** is the sum of the per-chip cost rows. The
  median is exact only if the chips move together, and each such row says so in Notes.
- **Chips with no price** in `chip_types.csv`, Trainium1 and Siyuan 590, have blank cost.
  A designer total that includes them has blank cost too, rather than an undercount.
- **`curated/` was snapshotted once** by `pipeline/tools/snapshot_curated_rows.py`, which also
  corrected one typo in the published data (see the Notes on the Cambricon Q2 2024 row). It is
  the transition copy that seeds the new Airtable tables; after that, curated rows are edited
  in Airtable and copied back here, never the other way round.
- **`chip_types.csv` is a September 9 snapshot** of the Airtable table, taken from the website's
  mirror. A pull step that refreshes it from Airtable before each build is still to be written.
