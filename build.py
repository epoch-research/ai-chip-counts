"""Export the AI chip sales and owners tables from Airtable to output/ for the S3 upload.

Airtable stays where the data is kept: the notebooks' estimates are uploaded there, and the
Huawei and Cambricon rows, chip types and organizations are edited there by hand. This copies
the views epoch.ai publishes, flattened the way the website's data pipeline flattens them
(linked records and lookups become their names), so its files don't change when it reads
them from S3 instead of Airtable.

    AIRTABLE_READ_TOKEN=... uv run python build.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from pyairtable import Api

OUTPUT = Path(__file__).resolve().parent / "output"
BASE_ID = "app0lYnFRVzEi3lkU"

# Output file: (table id, view id) of the view the website publishes.
TABLES = {
    "ai_chip_sales_organizations.csv": ("tblJc5m8qIWdc8ZQL", "viwaaWj7R5Fa5HbAP"),
    "ai_chip_sales_chip_types.csv": ("tblKrWNRsoC0QotV3", "viwVErvSeu05jgYCm"),
    "ai_chip_sales_timelines_by_chip.csv": ("tblyL1FwqiMRzCkeX", "viwHS1MAiCc65ajw8"),
    "ai_chip_sales_cumulative_timelines.csv": ("tblJWvrfdWJJNXlSl", "viwPR5kldSm6Xxs3l"),
    "ai_chip_sales_cumulative_timelines_by_designer.csv": ("tblb8LAUvR9OufJBl", "viw9Zr1fYLdcDXoiE"),
    "ai_chip_owners_cumulative_by_designer.csv": ("tblU5qHjez8cgXkSE", "viwJViMiOyY4wZnsi"),
    "ai_chip_owners_cumulative_by_chip_type.csv": ("tbl7p2wcXksNLzbjt", "viwhqC4xceQvAo1Og"),
    "ai_chip_owners_quarters_by_chip_type.csv": ("tblf8X5qu4r1Hcfjy", "viwJ0x6YIW3UegLF9"),
}


class Base:
    """Every table's schema and records, so linked records and lookups resolve to names."""

    def __init__(self, api: Api, base_id: str):
        self.fields = {}
        self.records = {}
        self.primary_field = {}
        self.schemas = {}
        for table in api.base(base_id).tables():
            schema = table.schema()
            self.schemas[table.id] = schema
            self.primary_field[table.id] = schema.primary_field_id
            self.fields.update({field.id: field for field in schema.fields})
            self.records.update({record["id"]: record for record in table.all(use_field_ids=True)})

    def resolve(self, record, field):
        value = record["fields"].get(field.id)
        if value is None:
            return None
        if field.type == "lastModifiedTime":
            return pd.to_datetime(value)
        if field.type == "aiText":
            return value["value"] if isinstance(value, dict) and "value" in value else value
        if field.type == "multipleRecordLinks":
            primary = self.primary_field[field.options.linked_table_id]
            linked = (self.records.get(id) for id in value)
            return [r["fields"][primary] for r in linked if r and primary in r["fields"]]
        if field.type == "multipleLookupValues":
            out = []
            for id in record["fields"].get(field.options.record_link_field_id, []):
                linked = self.records.get(id)
                if not linked:
                    continue
                item = self.resolve(linked, self.fields[field.options.field_id_in_linked_table])
                if isinstance(item, list):
                    out.extend(x for x in item if x is not None)
                elif item is not None:
                    out.append(item)
            # Airtable shows text lookups joined with ", ".
            return ", ".join(map(str, out)) if field.options.result.type == "singleLineText" else out
        return value


def to_cell(value):
    if isinstance(value, list):
        return ",".join(str(to_cell_item(x)) for x in value if x is not None)
    if isinstance(value, dict):
        return value["name"] if "name" in value else next(iter(value.values()))
    return value


def to_cell_item(value):
    if isinstance(value, dict) and "filename" in value and "url" in value:
        return f'{value["filename"]} ({value["url"]})'
    return str(value)


def read_view(api: Api, base: Base, table_id: str, view_id: str) -> pd.DataFrame:
    fields = [base.fields[id] for id in base.schemas[table_id].view(view_id).visible_field_ids if id in base.fields]
    records = api.table(BASE_ID, table_id).all(use_field_ids=True, view=view_id)
    return pd.DataFrame.from_records(
        [{field.name: to_cell(base.resolve(record, field)) for field in fields} for record in records]
    )


def main() -> None:
    api = Api(os.environ["AIRTABLE_READ_TOKEN"])
    base = Base(api, BASE_ID)
    OUTPUT.mkdir(exist_ok=True)
    for path, (table_id, view_id) in TABLES.items():
        df = read_view(api, base, table_id, view_id)
        df.to_csv(OUTPUT / path, index=False)
        print(f"{path}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
