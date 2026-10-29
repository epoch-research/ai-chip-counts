"""Upload the tables listed in datapackage.yaml, and datapackage.yaml itself, to a folder in S3.

Files whose content hasn't changed are skipped, so each file's last-modified date in S3 is
when it last changed. Files in the folder that the package no longer lists are deleted.

    uv run --group upload python .github/scripts/upload_to_s3.py s3://<bucket>/<folder> [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import boto3
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTENT_TYPES = {".csv": "text/csv; charset=utf-8", ".gz": "application/gzip", ".yaml": "application/yaml"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("destination", help="s3://<bucket>/<folder>")
    parser.add_argument("--dry-run", action="store_true", help="show what would change without changing it")
    args = parser.parse_args()

    bucket, _, prefix = args.destination.removeprefix("s3://").partition("/")
    prefix = prefix.strip("/")
    if not bucket or not prefix:
        sys.exit("The destination must look like s3://<bucket>/<folder>")
    upload(boto3.client("s3"), bucket, prefix, files_to_upload(), dry_run=args.dry_run)


def files_to_upload() -> dict[str, Path]:
    package = yaml.safe_load((ROOT / "datapackage.yaml").read_text())
    files = {resource["path"]: ROOT / "output" / resource["path"] for resource in package["resources"]}
    missing = [str(path.relative_to(ROOT)) for path in files.values() if not path.is_file()]
    if missing:
        sys.exit(f"Missing {', '.join(missing)}: run scripts/build.py first")
    files["datapackage.yaml"] = ROOT / "datapackage.yaml"
    return files


def upload(s3, bucket: str, prefix: str, files: dict[str, Path], *, dry_run: bool = False) -> None:
    existing = existing_keys(s3, bucket, prefix)
    for name, path in files.items():
        key = f"{prefix}/{name}"
        body = path.read_bytes()
        digest = hashlib.sha256(body).hexdigest()
        if key in existing and s3.head_object(Bucket=bucket, Key=key)["Metadata"].get("sha256") == digest:
            print(f"unchanged  {key}")
            continue
        print(f"upload     {key}")
        if not dry_run:
            content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
            s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type, Metadata={"sha256": digest})
    for key in sorted(existing - {f"{prefix}/{name}" for name in files}):
        print(f"delete     {key}")
        if not dry_run:
            s3.delete_object(Bucket=bucket, Key=key)


def existing_keys(s3, bucket: str, prefix: str) -> set[str]:
    keys = set()
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{prefix}/"):
        keys.update(item["Key"] for item in page.get("Contents", []))
    return keys


if __name__ == "__main__":
    main()
