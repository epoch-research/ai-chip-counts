#!/usr/bin/env python3.11
"""Execute a notebook headlessly and save its outputs back into it.

    python3.11 -m pipeline.tools.run_notebook notebooks/nvidia.ipynb [more.ipynb ...]

Each notebook runs from the repo root (its exports use relative paths) on a kernel
backed by the same interpreter as this script, so squigglepy is available whatever
`python` means on the PATH. The executed notebook, outputs and charts included, is
written over the source file, so the outputs committed with a notebook always come from
the same run as the CSVs it wrote. If a cell fails, the notebook is still saved, showing
every output up to and including the error, and the run exits nonzero.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

REPO = Path(__file__).resolve().parents[2]


def run(path, timeout=1800):
    """Run one notebook from the repo root and write the executed copy back to it.

    Returns True if every cell ran, False if one failed.
    """
    # The stock kernelspec launches a bare `python`; point that at this interpreter.
    shim = Path(tempfile.mkdtemp(prefix="nbshim-"))
    (shim / "python").symlink_to(sys.executable)
    os.environ["PATH"] = f"{shim}{os.pathsep}{os.environ['PATH']}"
    # Use Jupyter's inline backend so charts are captured as outputs, not silently dropped.
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"

    source = REPO / path
    nb = nbformat.read(source, as_version=4)
    start = time.time()
    # Skip per-cell timing metadata so reruns don't add noise to the diff.
    client = NotebookClient(nb, timeout=timeout, kernel_name="python3", record_timing=False,
                            resources={"metadata": {"path": str(REPO)}})
    ok = True
    try:
        client.execute()
    except CellExecutionError as err:
        ok = False
        print(f"FAILED {path}: {str(err).strip().splitlines()[-1]}", file=sys.stderr)
    finally:
        nbformat.write(nb, source)
    status = "ran" if ok else "saved partial run of"
    print(f"{status} {path} in {time.time() - start:.0f}s; outputs written back to the notebook")
    return ok


if __name__ == "__main__":
    results = [run(p) for p in sys.argv[1:]]
    sys.exit(0 if all(results) else 1)
