"""Focused tests for the AlphaGo-style PyTorch networks."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from alphago_gym.models import (
    PolicyNetwork,
    RolloutPolicy,
    ValueNetwork,
    mask_logits,
    masked_softmax,
)


BOARD_SIZE = 5
INPUT_CHANNELS = 8
BATCH_SIZE = 3


def observations(batch_size: int = BATCH_SIZE, *, device: str = "cpu") -> torch.Tensor:
    return torch.randn(
        batch_size, INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE, device=device
    )


def test_policy_network_is_configurable_deep_trunk_with_finite_logits() -> None:
    model = PolicyNetwork(
        board_size=BOARD_SIZE,
        input_channels=INPUT_CHANNELS,
        channels=16,
        depth=4,
    )

    logits = model(observations())

    assert logits.shape == (BATCH_SIZE, BOARD_SIZE**2 + 1)
    assert torch.isfinite(logits).all()
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) >= 4


def test_rollout_policy_is_shallow_fast_and_includes_pass() -> None:
    model = RolloutPolicy(board_size=BOARD_SIZE, input_channels=INPUT_CHANNELS)

    logits = model(observations())

    assert logits.shape == (BATCH_SIZE, BOARD_SIZE**2 + 1)
    assert torch.isfinite(logits).all()
    assert sum(isinstance(layer, nn.Conv2d) for layer in model.modules()) == 1


def test_value_network_returns_one_bounded_value_per_position() -> None:
    model = ValueNetwork(
        board_size=BOARD_SIZE,
        input_channels=INPUT_CHANNELS,
        channels=16,
        depth=3,
        hidden_channels=32,
    )

    values = model(observations())

    assert values.shape == (BATCH_SIZE,)
    assert torch.isfinite(values).all()
    assert torch.all(values >= -1.0)
    assert torch.all(values <= 1.0)


def test_default_input_channel_contract_is_eight_planes() -> None:
    policy = PolicyNetwork(board_size=BOARD_SIZE, channels=8, depth=2)
    rollout = RolloutPolicy(board_size=BOARD_SIZE)
    value = ValueNetwork(
        board_size=BOARD_SIZE, channels=8, depth=2, hidden_channels=16
    )
    batch = observations(batch_size=1)

    assert policy(batch).shape == (1, BOARD_SIZE**2 + 1)
    assert rollout(batch).shape == (1, BOARD_SIZE**2 + 1)
    assert value(batch).shape == (1,)


def test_mask_logits_supports_unbatched_logits() -> None:
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    legal = torch.tensor([True, False, True, False])

    masked = mask_logits(logits, legal)
    probabilities = masked_softmax(logits, legal)

    assert masked.shape == logits.shape
    assert torch.isneginf(masked[~legal]).all()
    assert torch.equal(masked[legal], logits[legal])
    assert torch.equal(probabilities[~legal], torch.zeros(2))
    assert probabilities.sum() == pytest.approx(1.0)


def test_masked_softmax_broadcasts_unbatched_mask_over_batch() -> None:
    logits = torch.randn(3, 6)
    legal = torch.tensor([True, False, True, False, False, True])

    probabilities = masked_softmax(logits, legal)

    assert probabilities.shape == logits.shape
    assert torch.equal(probabilities[:, ~legal], torch.zeros(3, 3))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(3))


def test_masked_softmax_supports_a_distinct_mask_per_batch_item() -> None:
    logits = torch.randn(2, 4)
    legal = torch.tensor([[True, False, False, True], [False, True, False, False]])

    probabilities = masked_softmax(logits, legal)

    assert torch.equal(probabilities[~legal], torch.zeros(5))
    torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2))


@pytest.mark.parametrize(
    "legal",
    [
        torch.tensor([False, False, False]),
        torch.tensor([[True, False, False], [False, False, False]]),
    ],
)
def test_mask_helpers_reject_any_distribution_without_a_legal_action(
    legal: torch.Tensor,
) -> None:
    logits = torch.randn(legal.shape)
    with pytest.raises(ValueError, match="at least one legal action"):
        mask_logits(logits, legal)
    with pytest.raises(ValueError, match="at least one legal action"):
        masked_softmax(logits, legal)


def test_mask_helpers_reject_non_broadcastable_masks() -> None:
    with pytest.raises(ValueError, match="broadcast"):
        masked_softmax(torch.randn(2, 4), torch.ones(3, dtype=torch.bool))


@pytest.mark.parametrize(
    "model",
    [
        PolicyNetwork(
            board_size=BOARD_SIZE,
            input_channels=INPUT_CHANNELS,
            channels=8,
            depth=2,
        ),
        RolloutPolicy(board_size=BOARD_SIZE, input_channels=INPUT_CHANNELS),
        ValueNetwork(
            board_size=BOARD_SIZE,
            input_channels=INPUT_CHANNELS,
            channels=8,
            depth=2,
            hidden_channels=16,
        ),
    ],
    ids=["policy", "rollout", "value"],
)
def test_gradients_flow_through_every_trainable_parameter(model: nn.Module) -> None:
    loss = model(observations(batch_size=2)).sum()

    loss.backward()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None for parameter in trainable)
    assert all(torch.isfinite(parameter.grad).all() for parameter in trainable)


@pytest.mark.parametrize(
    ("factory", "input_batch"),
    [
        (
            lambda: PolicyNetwork(
                board_size=BOARD_SIZE,
                input_channels=INPUT_CHANNELS,
                channels=8,
                depth=2,
            ),
            observations(batch_size=2),
        ),
        (
            lambda: RolloutPolicy(
                board_size=BOARD_SIZE, input_channels=INPUT_CHANNELS
            ),
            observations(batch_size=2),
        ),
        (
            lambda: ValueNetwork(
                board_size=BOARD_SIZE,
                input_channels=INPUT_CHANNELS,
                channels=8,
                depth=2,
                hidden_channels=16,
            ),
            observations(batch_size=2),
        ),
    ],
    ids=["policy", "rollout", "value"],
)
def test_state_dict_save_reload_is_deterministic(factory, input_batch: torch.Tensor) -> None:
    torch.manual_seed(11)
    original = factory().eval()
    expected = original(input_batch)
    saved_state = copy.deepcopy(original.state_dict())

    torch.manual_seed(22)
    restored = factory().eval()
    restored.load_state_dict(saved_state)

    torch.testing.assert_close(restored(input_batch), expected, rtol=0.0, atol=0.0)


@pytest.mark.mps
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable on this machine"
)
def test_available_mps_smoke() -> None:
    device = torch.device("mps")
    model = PolicyNetwork(
        board_size=BOARD_SIZE,
        input_channels=INPUT_CHANNELS,
        channels=8,
        depth=2,
    ).to(device)
    logits = model(observations(batch_size=1, device="mps"))
    legal = torch.ones(BOARD_SIZE**2 + 1, dtype=torch.bool, device=device)

    probabilities = masked_softmax(logits, legal)

    assert probabilities.device.type == "mps"
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(1, device=device)
    )
