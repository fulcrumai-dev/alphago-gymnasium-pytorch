#!/usr/bin/env python3
"""Fail unless an executed notebook completed without hidden cell errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat


def output_text(notebook: nbformat.NotebookNode) -> str:
    """Collect human-readable text emitted by every output cell."""

    chunks: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                chunks.append(str(output.get("text", "")))
            elif output.output_type in {"display_data", "execute_result"}:
                text = output.get("data", {}).get("text/plain")
                if text is not None:
                    chunks.append(str(text))
    return "\n".join(chunks)


def validate_notebook(
    path: Path,
    *,
    required_text: tuple[str, ...] = (),
    required_device: str | None = None,
    allow_output_only_execution: bool = False,
) -> tuple[int, str]:
    """Validate outputs and return ``(executed_code_cells, collected_text)``."""

    notebook = nbformat.read(path, as_version=4)
    errors: list[str] = []
    code_cells = 0
    unexecuted: list[int] = []

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or not cell.source.strip():
            continue
        code_cells += 1
        # google-colab-cli 0.6.0 writes outputs but leaves execution_count at
        # null.  Its output-only mode remains strict: every non-empty code cell
        # must have produced at least one saved output.
        has_saved_output = bool(cell.get("outputs", []))
        if cell.execution_count is None and not (
            allow_output_only_execution and has_saved_output
        ):
            unexecuted.append(index)
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append(
                    f"cell {index}: {output.get('ename', 'Error')}: "
                    f"{output.get('evalue', '')}"
                )

    if errors:
        raise RuntimeError("notebook contains error outputs:\n" + "\n".join(errors))
    if unexecuted:
        raise RuntimeError(f"notebook has unexecuted code cells: {unexecuted}")
    if code_cells == 0:
        raise RuntimeError("notebook contains no executable code cells")

    text = output_text(notebook)
    missing = [required for required in required_text if required not in text]
    if missing:
        raise RuntimeError(f"notebook output is missing required text: {missing}")
    if required_device is not None:
        marker = f"{required_device.upper()} tensor execution"
        if marker not in text:
            raise RuntimeError(f"notebook did not prove accelerator use: missing {marker!r}")
    return code_cells, text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument(
        "--require-text",
        action="append",
        default=[],
        help="output sentinel that must be present (repeatable)",
    )
    parser.add_argument("--require-device", choices=("cpu", "cuda", "mps"))
    parser.add_argument(
        "--allow-output-only-execution",
        action="store_true",
        help=(
            "accept a saved output in every code cell as execution proof; "
            "needed for google-colab-cli, which omits execution_count"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.notebook.is_file():
        print(f"Notebook not found: {args.notebook}", file=sys.stderr)
        return 2
    try:
        count, _ = validate_notebook(
            args.notebook,
            required_text=tuple(args.require_text),
            required_device=args.require_device,
            allow_output_only_execution=args.allow_output_only_execution,
        )
    except (RuntimeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    device = f", {args.require_device.upper()} proven" if args.require_device else ""
    print(f"Notebook validation passed: {count} executed code cells, zero errors{device}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
