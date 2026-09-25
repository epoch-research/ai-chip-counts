#!/usr/bin/env python3.11
"""Execute a notebook headlessly without writing outputs back into it.

    python3.11 -m pipeline.tools.run_notebook nvidia_chip_estimates.ipynb [more.ipynb ...]

Each notebook runs from the repo root (its exports use relative paths) on a kernel
backed by the same interpreter as this script, so squigglepy is available whatever
`python` means on the PATH. The executed copy goes to /tmp-style scratch space and the
source notebook is left untouched: only the CSVs the notebook writes change.
"""
import os
import sys
import tempfile
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

REPO = Path(__file__).resolve().parents[2]


def run(path, timeout=1800):
    # The stock kernelspec launches a bare `python`; point that at this interpreter.
    shim = Path(tempfile.mkdtemp(prefix="nbshim-"))
    (shim / "python").symlink_to(sys.executable)
    os.environ["PATH"] = f"{shim}{os.pathsep}{os.environ['PATH']}"
    os.environ.setdefault("MPLBACKEND", "Agg")

    nb = nbformat.read(REPO / path, as_version=4)
    start = time.time()
    NotebookClient(nb, timeout=timeout, kernel_name="python3",
                   resources={"metadata": {"path": str(REPO)}}).execute()
    out = Path(tempfile.gettempdir()) / f"executed-{Path(path).name}"
    nbformat.write(nb, out)
    print(f"ran {path} in {time.time() - start:.0f}s (executed copy: {out})")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p)
