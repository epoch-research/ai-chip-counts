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

## Website data

epoch.ai reads the AI chip sales and owners tables from S3, not from Airtable directly.
`.github/workflows/data.yml` runs `build.py` every hour (at :20), which exports the published
Airtable views to `output/`, checks them against `datapackage.yaml` (`tests/`), and uploads
them to `s3://epoch-datahub/ai-chip-counts/`. The website's hourly data update picks them up.

Airtable is still where the data is edited: upload the notebooks' estimates and edit the
hand-kept tables (Huawei, Cambricon, chip types, organizations) there as before.
If you add, rename or remove a column in a published Airtable view, update `datapackage.yaml`
to match, or the check fails and the upload stops until it does.

To run the export locally: `AIRTABLE_READ_TOKEN=... uv run python build.py && uv run pytest`.
