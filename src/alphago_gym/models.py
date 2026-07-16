"""AlphaGo-inspired policy, rollout, and value networks.

These are deliberately configurable, scaled versions of the networks described
in Silver et al. (2016). They preserve the important architectural roles while
remaining small enough for the repository's CPU smoke profile:

* a deep, spatially preserving convolutional policy network;
* a single-convolution rollout policy for inexpensive simulations; and
* a deep convolutional value network with a bounded scalar output.

All modules are ordinary PyTorch modules. They create no tensors on a hardcoded
device, so ``model.to("cuda")`` and ``model.to("mps")`` work normally.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


DEFAULT_INPUT_CHANNELS = 8


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _convolutional_trunk(
    input_channels: int, channels: int, depth: int
) -> nn.Sequential:
    """Build an AlphaGo-style trunk that preserves board dimensions."""

    layers: list[nn.Module] = [
        nn.Conv2d(input_channels, channels, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
    ]
    for _ in range(depth - 1):
        layers.extend(
            [
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ]
        )
    return nn.Sequential(*layers)


class _BoardModel(nn.Module):
    """Shared input-contract validation for fixed-size Go networks."""

    def __init__(self, board_size: int, input_channels: int) -> None:
        super().__init__()
        self.board_size = _positive_int("board_size", board_size)
        self.input_channels = _positive_int("input_channels", input_channels)
        self.num_actions = self.board_size**2 + 1

    def _validate_input(self, observations: torch.Tensor) -> None:
        if observations.ndim != 4:
            raise ValueError(
                "observations must have shape (batch, channels, board, board); "
                f"got {tuple(observations.shape)}"
            )
        expected = (self.input_channels, self.board_size, self.board_size)
        if tuple(observations.shape[1:]) != expected:
            raise ValueError(
                "expected observation shape "
                f"(batch, {expected[0]}, {expected[1]}, {expected[2]}), got "
                f"{tuple(observations.shape)}"
            )


class PolicyNetwork(_BoardModel):
    """Deep convolutional policy producing board-move and pass logits.

    ``depth`` counts the spatial trunk convolutions. The first uses the 5x5
    receptive field from the paper and subsequent layers use 3x3 kernels. A
    1x1 head produces one logit per intersection; a learned scalar supplies the
    extra pass action used by this Gymnasium environment.
    """

    def __init__(
        self,
        board_size: int,
        input_channels: int = DEFAULT_INPUT_CHANNELS,
        channels: int = 64,
        depth: int = 6,
    ) -> None:
        super().__init__(board_size, input_channels)
        self.channels = _positive_int("channels", channels)
        self.depth = _positive_int("depth", depth)
        self.trunk = _convolutional_trunk(
            self.input_channels, self.channels, self.depth
        )
        self.board_head = nn.Conv2d(self.channels, 1, kernel_size=1)
        self.pass_logit = nn.Parameter(torch.zeros(1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self._validate_input(observations)
        features = self.trunk(observations)
        board_logits = self.board_head(features).flatten(start_dim=1)
        pass_logits = self.pass_logit.expand(observations.shape[0], 1)
        return torch.cat((board_logits, pass_logits), dim=-1)


class RolloutPolicy(_BoardModel):
    """A fast, shallow policy intended for many inexpensive rollouts.

    The single 3x3 convolution is a spatial analogue of the paper's fast
    linear rollout policy. It has no hidden trunk, normalization, or dropout.
    """

    def __init__(
        self,
        board_size: int,
        input_channels: int = DEFAULT_INPUT_CHANNELS,
    ) -> None:
        super().__init__(board_size, input_channels)
        self.board_head = nn.Conv2d(
            self.input_channels, 1, kernel_size=3, padding=1
        )
        self.pass_logit = nn.Parameter(torch.zeros(1))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self._validate_input(observations)
        board_logits = self.board_head(observations).flatten(start_dim=1)
        pass_logits = self.pass_logit.expand(observations.shape[0], 1)
        return torch.cat((board_logits, pass_logits), dim=-1)


class ValueNetwork(_BoardModel):
    """Deep convolutional evaluator returning values in ``[-1, 1]``.

    Values are from the encoded side-to-move's perspective. The shape is
    ``(batch,)`` so it composes directly with one-dimensional game-outcome
    targets in standard PyTorch losses.
    """

    def __init__(
        self,
        board_size: int,
        input_channels: int = DEFAULT_INPUT_CHANNELS,
        channels: int = 64,
        depth: int = 6,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__(board_size, input_channels)
        self.channels = _positive_int("channels", channels)
        self.depth = _positive_int("depth", depth)
        self.hidden_channels = _positive_int("hidden_channels", hidden_channels)
        self.trunk = _convolutional_trunk(
            self.input_channels, self.channels, self.depth
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(self.channels, 1, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(self.board_size**2, self.hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, 1),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self._validate_input(observations)
        values = self.value_head(self.trunk(observations))
        return values.squeeze(-1)


def _expanded_legal_mask(
    logits: torch.Tensor,
    legal_mask: torch.Tensor | Sequence[bool],
    dim: int,
) -> torch.Tensor:
    if not logits.is_floating_point():
        raise TypeError("logits must be a floating-point tensor")
    if logits.ndim == 0:
        raise ValueError("logits must have at least one dimension")

    normalized_dim = dim if dim >= 0 else logits.ndim + dim
    if normalized_dim < 0 or normalized_dim >= logits.ndim:
        raise IndexError(
            f"dimension {dim} is out of range for {logits.ndim}-D logits"
        )

    mask = torch.as_tensor(legal_mask, dtype=torch.bool, device=logits.device)
    try:
        expanded = torch.broadcast_to(mask, logits.shape)
    except RuntimeError as error:
        raise ValueError(
            f"legal mask shape {tuple(mask.shape)} cannot broadcast to logits "
            f"shape {tuple(logits.shape)}"
        ) from error

    has_legal_action = expanded.any(dim=normalized_dim)
    if not bool(has_legal_action.all().item()):
        raise ValueError("each distribution must contain at least one legal action")
    return expanded


def mask_logits(
    logits: torch.Tensor,
    legal_mask: torch.Tensor | Sequence[bool],
    dim: int = -1,
) -> torch.Tensor:
    """Replace illegal logits with negative infinity.

    A one-dimensional action mask is broadcast across batched logits, while a
    batched mask can encode distinct legal actions for each position. Every
    distribution must retain at least one legal action.
    """

    expanded_mask = _expanded_legal_mask(logits, legal_mask, dim)
    return logits.masked_fill(~expanded_mask, -torch.inf)


def masked_softmax(
    logits: torch.Tensor,
    legal_mask: torch.Tensor | Sequence[bool],
    dim: int = -1,
) -> torch.Tensor:
    """Compute a softmax with exactly zero probability for illegal actions."""

    return torch.softmax(mask_logits(logits, legal_mask, dim=dim), dim=dim)


__all__ = [
    "PolicyNetwork",
    "RolloutPolicy",
    "ValueNetwork",
    "mask_logits",
    "masked_softmax",
]
