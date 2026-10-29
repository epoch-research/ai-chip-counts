"""Checks that the CSVs in output/ match datapackage.yaml.

Every data repo made from the template shares this file: change it in the template, not here.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = yaml.safe_load((ROOT / "datapackage.yaml").read_text()) or {}
RESOURCES = PACKAGE.get("resources") or []
MAX_EXAMPLES = 5
REQUIRED = {
    "package": ("name", "title", "resources"),
    "table": ("name", "path", "title", "description"),
    "column": ("name", "title", "type", "description"),
}


def is_date(value: str) -> bool:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def is_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return False
    return True


CHECKS = {
    "string": lambda value: True,
    "integer": re.compile(r"[-+]?\d+").fullmatch,
    "number": re.compile(r"[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?").fullmatch,
    "boolean": lambda value: value.lower() in ("true", "false"),
    "date": is_date,
    "datetime": is_datetime,
    "year": re.compile(r"\d{4}").fullmatch,
}
EXPECTED = {
    "integer": "whole numbers",
    "number": "numbers",
    "boolean": "true or false",
    "date": "dates (YYYY-MM-DD)",
    "datetime": "ISO 8601 date-times",
    "year": "years (YYYY)",
}


def fields_of(resource: dict) -> list[dict]:
    return (resource.get("schema") or {}).get("fields") or []


def missing(item: dict, kind: str) -> list[str]:
    return [key for key in REQUIRED[kind] if not item.get(key)]


def test_every_table_and_column_is_described():
    problems = [f"datapackage.yaml needs a `{key}`" for key in missing(PACKAGE, "package")]
    for resource in RESOURCES:
        table = resource.get("name") or "<unnamed table>"
        problems += [f"{table}: needs a `{key}`" for key in missing(resource, "table")]
        fields = fields_of(resource)
        if not fields:
            problems.append(f"{table}: list its columns under schema.fields")
        names = [field.get("name") for field in fields]
        duplicates = sorted({name for name in names if name and names.count(name) > 1})
        problems += [f"{table}: column {name!r} is listed twice" for name in duplicates]
        for field in fields:
            column = f"{table}.{field.get('name') or '<unnamed column>'}"
            problems += [f"{column}: needs a `{key}`" for key in missing(field, "column")]
            if field.get("type") and field["type"] not in CHECKS:
                problems.append(f"{column}: unknown type {field['type']!r}; use one of {', '.join(CHECKS)}")
    assert not problems, "\n".join(problems)


@pytest.fixture(params=RESOURCES, ids=lambda resource: resource.get("name") or "?")
def table(request):
    resource = request.param
    path = ROOT / "output" / resource["path"]
    if not path.is_file():
        pytest.fail(f"output/{resource['path']} is missing: build.py must write every table in datapackage.yaml")
    return resource, pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def test_columns_match_the_package(table):
    resource, df = table
    declared = [field["name"] for field in fields_of(resource) if field.get("name")]
    absent = [name for name in declared if name not in df.columns]
    extra = [name for name in df.columns if name not in declared]
    assert not absent, f"{resource['name']}: in datapackage.yaml but not in the CSV: {', '.join(absent)}"
    assert not extra, f"{resource['name']}: in the CSV but not described in datapackage.yaml: {', '.join(extra)}"


def test_values_match_their_types(table):
    resource, df = table
    problems = []
    for field in fields_of(resource):
        check = CHECKS.get(field.get("type"))
        if check is None or field.get("name") not in df.columns:
            continue
        bad = [(line, value) for line, value in enumerate(df[field["name"]], start=2) if value and not check(value)]
        if not bad:
            continue
        examples = ", ".join(f"{value!r} (line {line})" for line, value in bad[:MAX_EXAMPLES])
        count = "1 value doesn't" if len(bad) == 1 else f"{len(bad)} values don't"
        expected = EXPECTED[field["type"]]
        problems.append(f"{resource['name']}.{field['name']} should hold {expected}, but {count}: {examples}")
        if field["type"] == "integer" and any(re.fullmatch(r"[-+]?\d+\.0+", value) for _, value in bad):
            problems.append("  Whole numbers are being written as 12.0: give the column pandas' Int64 dtype.")
    assert not problems, "\n".join(problems)
