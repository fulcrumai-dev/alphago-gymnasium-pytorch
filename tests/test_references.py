from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FETCH_SCRIPT = ROOT / "references" / "fetch_paper.py"
PAPER_README = ROOT / "references" / "README.md"


def _load_fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_paper", FETCH_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def test_download_is_atomic_and_checksum_verified(tmp_path, monkeypatch) -> None:
    fetch = _load_fetch_module()
    body = b"test-only paper bytes"
    expected = hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *_a, **_k: _FakeResponse(body))

    destination = tmp_path / "paper.pdf"
    result = fetch.download_paper(
        destination,
        url="https://example.invalid/paper.pdf",
        expected_sha256=expected,
    )

    assert result == destination
    assert destination.read_bytes() == body
    assert not list(tmp_path.glob("*.part"))


def test_checksum_failure_leaves_no_output(tmp_path, monkeypatch) -> None:
    fetch = _load_fetch_module()
    monkeypatch.setattr(
        fetch.urllib.request,
        "urlopen",
        lambda *_a, **_k: _FakeResponse(b"unexpected bytes"),
    )

    destination = tmp_path / "paper.pdf"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        fetch.download_paper(
            destination,
            url="https://example.invalid/paper.pdf",
            expected_sha256="0" * 64,
        )

    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_an_existing_verified_copy_avoids_network(tmp_path, monkeypatch) -> None:
    fetch = _load_fetch_module()
    body = b"already downloaded"
    destination = tmp_path / "paper.pdf"
    destination.write_bytes(body)

    def unexpected_network_call(*_args, **_kwargs):
        raise AssertionError("a verified local copy should not be downloaded again")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", unexpected_network_call)
    result = fetch.download_paper(
        destination,
        expected_sha256=hashlib.sha256(body).hexdigest(),
    )

    assert result == destination


def test_reference_docs_pin_primary_sources_and_integrity_metadata() -> None:
    readme = PAPER_README.read_text(encoding="utf-8")

    assert "https://doi.org/10.1038/nature16961" in readme
    assert (
        "https://storage.googleapis.com/deepmind-media/alphago/"
        "AlphaGoNaturePaper.pdf"
    ) in readme
    assert "9c9184385a3d37b4f4e9d9715270986c43172747b1d08f29093128c1ef878b60" in readme
