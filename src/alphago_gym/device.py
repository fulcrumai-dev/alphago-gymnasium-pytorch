"""Portable PyTorch device selection.

The project deliberately keeps device policy at its boundary: models contain no
device-specific branches and can be moved with the normal ``nn.Module.to`` API.
"""

from __future__ import annotations

from typing import TypeAlias

import torch


DevicePreference: TypeAlias = str | torch.device


def _mps_is_available() -> bool:
    """Return whether PyTorch can currently execute on Apple's MPS backend."""

    backend = getattr(torch.backends, "mps", None)
    return bool(backend is not None and backend.is_available())


def select_device(preference: DevicePreference = "auto") -> torch.device:
    """Resolve a requested PyTorch device.

    ``"auto"`` chooses CUDA first, then Apple MPS, and finally CPU. Explicit
    accelerator requests fail loudly when that accelerator is unavailable;
    silently falling back in that case tends to hide configuration mistakes in
    Colab and local training runs.

    Args:
        preference: ``"auto"``, ``"cpu"``, ``"cuda"`` (optionally with a
            numeric index), ``"mps"``, or the equivalent ``torch.device``.

    Returns:
        A resolved ``torch.device``.

    Raises:
        TypeError: If ``preference`` is not a string or ``torch.device``.
        ValueError: If the requested device syntax/backend is unsupported.
        RuntimeError: If an explicitly requested accelerator is unavailable.
    """

    if isinstance(preference, torch.device):
        requested = str(preference)
    elif isinstance(preference, str):
        requested = preference.strip().lower()
    else:
        raise TypeError("device preference must be a string or torch.device")

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError(
            "device preference must be one of: auto, cpu, cuda, cuda:N, or mps"
        ) from error

    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            "device preference must be one of: auto, cpu, cuda, cuda:N, or mps"
        )

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false"
        )
    if device.type == "mps" and not _mps_is_available():
        raise RuntimeError(
            "MPS was requested, but torch.backends.mps.is_available() is false"
        )
    return device


__all__ = ["select_device"]
