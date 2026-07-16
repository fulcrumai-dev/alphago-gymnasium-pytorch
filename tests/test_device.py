"""Tests for portable PyTorch device selection."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from alphago_gym.device import select_device


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_device_priority(
    cuda_available: bool, mps_available: bool, expected: str
) -> None:
    with (
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch("torch.backends.mps.is_available", return_value=mps_available),
    ):
        assert select_device().type == expected


def test_explicit_cpu_is_always_available() -> None:
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.backends.mps.is_available", return_value=True),
    ):
        assert select_device("cpu") == torch.device("cpu")


@pytest.mark.parametrize("preference", ["cuda", "cuda:0"])
def test_explicit_cuda_is_returned_when_available(preference: str) -> None:
    with patch("torch.cuda.is_available", return_value=True):
        assert select_device(preference) == torch.device(preference)


def test_explicit_mps_is_returned_when_available() -> None:
    with patch("torch.backends.mps.is_available", return_value=True):
        assert select_device("mps") == torch.device("mps")


@pytest.mark.parametrize(
    ("preference", "patched_name", "message"),
    [
        ("cuda", "torch.cuda.is_available", "CUDA"),
        ("mps", "torch.backends.mps.is_available", "MPS"),
    ],
)
def test_unavailable_explicit_accelerator_has_clear_error(
    preference: str, patched_name: str, message: str
) -> None:
    with patch(patched_name, return_value=False):
        with pytest.raises(RuntimeError, match=message):
            select_device(preference)


@pytest.mark.parametrize("preference", ["gpu", "tpu", "cuda:not-an-index", ""])
def test_invalid_device_preference_has_clear_error(preference: str) -> None:
    with pytest.raises(ValueError, match="auto.*cpu.*cuda.*mps"):
        select_device(preference)


def test_device_object_is_accepted() -> None:
    assert select_device(torch.device("cpu")) == torch.device("cpu")

