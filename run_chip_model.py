#!/usr/bin/env python3
"""Run chip-family models end to end: rerun the model, export CSVs, validate.

This is the "one thing to run" for refreshing published estimates. For each requested
family it reruns the Monte Carlo model, rewrites the csv_export/ (and owners_csv_export/)
files, and validates the exports against the currently published dataset at
https://epoch.ai/data/ai-chip-sales, finishing with a per-family summary table.

Families are discovered by convention: any *_model.py in the repo root that exposes
    FAMILY        - family name used on the command line (e.g. "tpu")
    DESIGNER      - key into validate_chip_sales.DESIGNER_CONFIGS (e.g. "google")
    run_model()   - run the simulation, return a results dict
    export_csvs(results) - write the CSV exports, return the written paths
Adding a chip family means dropping in a new <family>_model.py; nothing to register here.

Usage:
    python run_chip_model.py tpu    # one family
    python run_chip_model.py all    # every family with a *_model.py module

One family failing (a moved Google Sheet, a schema change) does not stop the others;
it shows up as ERROR in the summary. Exit code is 0 only if every family runs and
validates clean.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import sys
import traceback

import validate_chip_sales as vcs


def discover_model_modules():
    """Import every *_model.py in the repo root; return {family: module}."""
    modules = {}
    for path in sorted(glob.glob("*_model.py")):
        # This script's own name matches the glob; it is not a model module.
        if path == "run_chip_model.py":
            continue
        module = importlib.import_module(path[:-3])
        required = ("FAMILY", "DESIGNER", "run_model", "export_csvs")
        missing = [attr for attr in required if not hasattr(module, attr)]
        if missing:
            print(f"warning: ignoring {path}: missing {', '.join(missing)}")
            continue
        modules[module.FAMILY] = module
    return modules


def run_family(module, web_tables):
    """Run one family's model, export its CSVs, and validate the exports.

    Returns a status dict for the summary table. Raises nothing: errors are caught
    by the caller so one family cannot abort the others.
    """
    print("\n" + "#" * 78)
    print(f"#  {module.FAMILY}: running model")
    print("#" * 78)
    results = module.run_model()
    exported = module.export_csvs(results)
    # The canonical tables (docs/schema.md) feed staging/ and the HTML hub in hub/.
    if hasattr(module, "export_canonical"):
        module.export_canonical(results)

    if module.DESIGNER not in vcs.DESIGNER_CONFIGS:
        print(f"\n  {module.FAMILY}: no validation config for designer '{module.DESIGNER}' - skipping validation")
        return {"family": module.FAMILY, "model": "ok", "files": len(exported), "validation": "no config"}
    if web_tables is None:
        return {"family": module.FAMILY, "model": "ok", "files": len(exported), "validation": "skipped (download failed)"}

    print("\n" + "#" * 78)
    print(f"#  {module.FAMILY}: validating against published data (designer: {module.DESIGNER})")
    print("#" * 78)
    validation_results = vcs.validate_designer(module.DESIGNER, web_tables)
    for result in validation_results:
        vcs.print_table_report(result)
    v = vcs.verdict(validation_results)
    print(f"\n  {module.FAMILY}: {v['major_cells']} major cell diff(s), {v['unmatched_rows']} unmatched row(s)")
    return {
        "family": module.FAMILY,
        "model": "ok",
        "files": len(exported),
        "validation": "PASS" if v["ok"] else "REVIEW",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("family", help="chip family to run (from *_model.py modules), or 'all'")
    args = parser.parse_args(argv)

    modules = discover_model_modules()
    if not modules:
        parser.error("no *_model.py modules found in the repo root")

    if args.family == "all":
        selected = list(modules.values())
    elif args.family in modules:
        selected = [modules[args.family]]
    else:
        parser.error(f"unknown family '{args.family}'. Available: {', '.join(modules)} (or 'all')")

    # Designers that have validation wiring but no model module yet (their models are
    # still notebook-based) - worth a reminder when running everything.
    if args.family == "all":
        model_designers = {m.DESIGNER for m in modules.values()}
        config_only = [d for d in vcs.DESIGNER_CONFIGS if d not in model_designers]
        if config_only:
            print(f"note: validation configs without a model module (run their notebooks instead): "
                  f"{', '.join(config_only)}")

    # Fetch the published dataset once for all families; models still run if this fails.
    try:
        web_tables = vcs.fetch_website_tables()
    except Exception as exc:
        print(f"warning: could not fetch published data ({exc}); validation will be skipped")
        web_tables = None

    statuses = []
    for module in selected:
        try:
            statuses.append(run_family(module, web_tables))
        except Exception:
            traceback.print_exc()
            statuses.append({"family": module.FAMILY, "model": "ERROR", "files": 0, "validation": "-"})

    print("\n" + "=" * 78)
    print(f"  {'family':<12}{'model':<10}{'files exported':<16}{'validation'}")
    print("  " + "-" * 74)
    for s in statuses:
        print(f"  {s['family']:<12}{s['model']:<10}{s['files']:<16}{s['validation']}")
    print("=" * 78)

    all_ok = all(s["model"] == "ok" and s["validation"] == "PASS" for s in statuses)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
