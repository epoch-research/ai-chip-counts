#!/usr/bin/env python3.11
"""Compare working-tree export CSVs with git HEAD, ignoring the Notes timestamp.

    python3.11 -m pipeline.tools.diff_exports [--restore] [--only PREFIX ...]

Prints every file whose numbers moved. With --restore, files that differ only in
their Notes column are checked out from HEAD, so a deterministic rerun leaves no noise.
"""
import io
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parents[2])

# --only nvidia_ limits the check to files whose name starts with that prefix, so
# parallel reruns of different families never touch each other's files.
prefixes = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
changed = subprocess.run(["git", "diff", "--name-only", "--", "csv_export", "owners_csv_export", "data_inputs"],
                         capture_output=True, text=True).stdout.split()
if prefixes:
    changed = [f for f in changed if any(f.split("/")[-1].startswith(p) for p in prefixes)]
noise = []
for f in changed:
    if not f.endswith(".csv"):
        continue
    old = pd.read_csv(io.StringIO(subprocess.run(["git", "show", f"HEAD:{f}"], capture_output=True, text=True).stdout), dtype=str)
    new = pd.read_csv(f, dtype=str)
    drop = [c for c in ("Notes",) if c in old.columns]
    o, n = old.drop(columns=drop, errors="ignore"), new.drop(columns=drop, errors="ignore")
    if o.shape == n.shape and list(o.columns) == list(n.columns) and o.fillna("").equals(n.fillna("")):
        noise.append(f)
    else:
        print(f"NUMBERS CHANGED: {f} ({len(old)} -> {len(new)} rows)")
print(f"{len(noise)} file(s) differ only in Notes timestamps")
if "--restore" in sys.argv and noise:
    subprocess.run(["git", "checkout", "--", *noise], check=True)
    print("restored them from HEAD")
