#!/usr/bin/env python3
"""Fetch and integrity-check the official Google DeepMind copy of the paper."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


PAPER_URL = (
    "https://storage.googleapis.com/deepmind-media/alphago/"
    "AlphaGoNaturePaper.pdf"
)
PAPER_SHA256 = "9c9184385a3d37b4f4e9d9715270986c43172747b1d08f29093128c1ef878b60"
PAPER_SIZE_BYTES = 2_682_222
DEFAULT_OUTPUT = Path(__file__).with_name("AlphaGoNaturePaper.pdf")


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 digest of *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_paper(
    destination: str | Path = DEFAULT_OUTPUT,
    *,
    url: str = PAPER_URL,
    expected_sha256: str = PAPER_SHA256,
    force: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download *url* atomically and return a checksum-verified local path.

    An already-valid file is reused. An existing invalid file is never replaced
    unless ``force=True``. A failed or interrupted transfer leaves no partial
    destination behind.
    """

    output = Path(destination).expanduser()
    if output.exists():
        actual = sha256_file(output)
        if actual == expected_sha256:
            return output
        if not force:
            raise RuntimeError(
                f"existing file has SHA-256 {actual}, expected {expected_sha256}; "
                "use --force to replace it"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "alphago-research-reference-fetcher/1.0"},
        )
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output.parent,
            prefix=f"{output.name}.",
            suffix=".part",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                while chunk := response.read(1024 * 1024):
                    temporary.write(chunk)

        actual = sha256_file(temporary_path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {url}: got {actual}, expected {expected_sha256}"
            )
        os.replace(temporary_path, output)
        temporary_path = None
        return output
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"local output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing file whose checksum does not match",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = download_paper(args.output, force=args.force)
    print(f"Verified {output}")
    print(f"SHA-256 {sha256_file(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
