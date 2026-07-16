#!/usr/bin/env python3
"""Execute the tutorial notebook headlessly, persist it, and scan every cell."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from scan_notebook import validate_notebook


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "notebooks" / "alphago_2016_tutorial.ipynb"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", choices=("smoke", "tutorial"), default="smoke")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--timeout", type=int, default=900, help="seconds allowed per cell")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    notebook_path = args.notebook.resolve()
    if not notebook_path.is_file():
        print(f"Notebook not found: {notebook_path}", file=sys.stderr)
        return 2
    output = args.output or (
        ROOT / "artifacts" / "notebooks" / f"{notebook_path.stem}.{args.profile}.{args.device}.ipynb"
    )
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    os.environ["ALPHAGO_PROFILE"] = args.profile
    os.environ["ALPHAGO_DEVICE"] = args.device
    notebook = nbformat.read(notebook_path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=args.timeout,
        kernel_name="python3",
        allow_errors=False,
        resources={"metadata": {"path": str(ROOT)}},
    )

    started = time.monotonic()
    try:
        client.execute()
    except Exception:
        failed = output.with_suffix(".failed.ipynb")
        nbformat.write(notebook, failed)
        print(f"Partial failed notebook written to {failed}", file=sys.stderr)
        raise
    nbformat.write(notebook, output)

    required_device = None if args.device == "auto" else args.device
    code_cells, _ = validate_notebook(
        output,
        required_text=("✓ Gymnasium API",),
        required_device=required_device,
    )
    elapsed = time.monotonic() - started
    print(
        f"Validated {code_cells} code cells in {elapsed:.1f}s; "
        f"executed notebook: {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

