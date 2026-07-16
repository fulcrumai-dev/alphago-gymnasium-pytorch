"""Tests for supervised, reinforcement, and value-data training helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

from alphago_gym.training import (
    OpponentPool,
    PolicyExample,
    PolicyGradientEpisode,
    PolicyGradientStep,
    ValueExample,
    dihedral_policy_augmentations,
    dihedral_value_augmentations,
    generate_policy_gradient_episode,
    generate_value_examples,
    legal_policy_probabilities,
    sample_legal_action,
    train_policy_epoch,
    train_reinforce_epoch,
    train_value_epoch,
)


class BiasPolicy(nn.Module):
    """Small policy whose learning direction is easy to inspect."""

    def __init__(self, logits: tuple[float, ...] = (0.0, 0.0, 0.0)) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.logits.unsqueeze(0).expand(observations.shape[0], -1)


class ScalarValue(nn.Module):
    def __init__(self, initial: float = 0.0) -> None:
        super().__init__()
        self.raw_value = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.raw_value).expand(observations.shape[0])


def test_dihedral_policy_augmentation_keeps_action_aligned_with_board() -> None:
    observation = np.zeros((2, 3, 3), dtype=np.float32)
    observation[0, 0, 0] = 1.0
    observation[1] = np.arange(9, dtype=np.float32).reshape(3, 3)
    example = PolicyExample(observation=observation, action=0)

    augmented = dihedral_policy_augmentations(example)

    assert len(augmented) == 8
    assert {item.action for item in augmented} == {0, 2, 6, 8}
    for item in augmented:
        assert item.observation.flags.c_contiguous
        assert int(np.argmax(item.observation[0])) == item.action


def test_dihedral_policy_augmentation_preserves_pass_action() -> None:
    observation = np.arange(18, dtype=np.float32).reshape(2, 3, 3)

    augmented = dihedral_policy_augmentations(
        PolicyExample(observation=observation, action=9)
    )

    assert len(augmented) == 8
    assert all(item.action == 9 for item in augmented)


def test_dihedral_value_augmentation_preserves_outcome() -> None:
    example = ValueExample(
        observation=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
        outcome=-1.0,
    )

    augmented = dihedral_value_augmentations(example)

    assert len(augmented) == 8
    assert all(item.outcome == -1.0 for item in augmented)
    assert all(item.observation.flags.c_contiguous for item in augmented)


def test_supervised_policy_epoch_reduces_loss_and_moves_probability() -> None:
    model = BiasPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.35)
    examples = [
        PolicyExample(np.zeros((1, 1, 1), dtype=np.float32), action=1)
        for _ in range(12)
    ]
    before = torch.softmax(model.logits.detach(), dim=-1)[1].item()

    losses = [
        train_policy_epoch(
            model,
            examples,
            optimizer,
            batch_size=4,
            device="cpu",
            shuffle=True,
            rng=np.random.default_rng(epoch),
        )
        for epoch in range(8)
    ]
    after = torch.softmax(model.logits.detach(), dim=-1)[1].item()

    assert losses[-1] < losses[0]
    assert after > before
    assert after > 0.8
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()


def test_supervised_value_epoch_reduces_error_and_moves_value() -> None:
    model = ScalarValue(initial=-0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.2)
    examples = [
        ValueExample(np.zeros((1, 1, 1), dtype=np.float32), outcome=0.75)
        for _ in range(10)
    ]
    before = model(torch.zeros(1, 1, 1, 1)).item()

    losses = [
        train_value_epoch(
            model,
            examples,
            optimizer,
            batch_size=5,
            device="cpu",
            shuffle=True,
            rng=np.random.default_rng(epoch),
        )
        for epoch in range(8)
    ]
    after = model(torch.zeros(1, 1, 1, 1)).item()

    assert losses[-1] < losses[0]
    assert after > before
    assert abs(after - 0.75) < abs(before - 0.75)
    assert model.raw_value.grad is not None
    assert torch.isfinite(model.raw_value.grad)


@dataclass(frozen=True)
class LegalPosition:
    """Non-terminal structural position for policy adapter tests."""

    to_play: int = 1
    action_size: int = 3
    pass_action: int = 2
    is_terminal: bool = False

    def legal_actions_mask(self) -> np.ndarray:
        return np.array([False, True, False], dtype=np.bool_)

    def encode(self) -> np.ndarray:
        return np.zeros((1, 1, 1), dtype=np.float32)

    def play(self, action: int) -> "LegalPosition":
        raise NotImplementedError

    def outcome(self, player: int) -> float:
        raise NotImplementedError


def test_legal_policy_helper_removes_illegal_mass_and_preserves_mode() -> None:
    model = BiasPolicy((20.0, -3.0, 30.0)).train()
    position = LegalPosition()

    probabilities = legal_policy_probabilities(model, position, device="cpu")

    assert model.training
    assert probabilities.shape == (3,)
    assert probabilities.dtype == np.float64
    assert probabilities.tolist() == [0.0, 1.0, 0.0]
    assert sample_legal_action(
        model, position, np.random.default_rng(4), device="cpu"
    ) == 1


def test_legal_policy_helper_sanitizes_callable_probabilities() -> None:
    probabilities = legal_policy_probabilities(
        lambda _: np.array([np.inf, 2.0, -1.0]), LegalPosition()
    )

    assert probabilities.tolist() == [0.0, 1.0, 0.0]


def test_opponent_pool_snapshots_are_cpu_frozen_and_immutable() -> None:
    source = BiasPolicy((1.0, 2.0, 3.0))
    pool = OpponentPool(model_factory=BiasPolicy, max_size=2)
    pool.add(source)

    with torch.no_grad():
        source.logits.fill_(99.0)
    first_sample = pool.sample(np.random.default_rng(0), device="cpu")

    torch.testing.assert_close(
        first_sample.logits, torch.tensor([1.0, 2.0, 3.0])
    )
    assert not first_sample.training
    assert all(not parameter.requires_grad for parameter in first_sample.parameters())
    assert all(parameter.device.type == "cpu" for parameter in first_sample.parameters())

    with torch.no_grad():
        first_sample.logits.fill_(-50.0)
    second_sample = pool.sample(np.random.default_rng(0), device="cpu")
    torch.testing.assert_close(
        second_sample.logits, torch.tensor([1.0, 2.0, 3.0])
    )


def test_opponent_pool_is_fifo_when_bounded() -> None:
    pool = OpponentPool(model_factory=BiasPolicy, max_size=2)
    for value in (1.0, 2.0, 3.0):
        pool.add(BiasPolicy((value, value, value)))

    observed = {
        float(pool.sample(np.random.default_rng(seed)).logits[0])
        for seed in range(20)
    }

    assert len(pool) == 2
    assert observed == {2.0, 3.0}


class ConstantBaseline(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value.expand(observations.shape[0])


def test_reinforce_epoch_uses_baseline_entropy_and_legal_log_probability() -> None:
    model = BiasPolicy((0.0, 0.0, 20.0))
    baseline = ConstantBaseline(0.25)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.25)
    step = PolicyGradientStep(
        observation=np.zeros((1, 1, 1), dtype=np.float32),
        action=1,
        legal_mask=np.array([True, True, False], dtype=np.bool_),
        player=1,
    )
    episodes = [
        PolicyGradientEpisode(steps=(step, step), outcome=1.0, learner_player=1)
        for _ in range(4)
    ]
    before = legal_policy_probabilities(
        model,
        _TwoActionView(),
    )[1]

    losses = [
        train_reinforce_epoch(
            model,
            episodes,
            optimizer,
            device="cpu",
            value_baseline=baseline,
            entropy_coefficient=0.01,
        )
        for _ in range(6)
    ]
    after = legal_policy_probabilities(model, _TwoActionView())[1]

    assert np.isfinite(losses).all()
    assert after > before
    assert after > 0.7
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()
    assert baseline.value.grad is None
    assert baseline.training


def test_reinforce_keeps_gradient_for_extremely_unlikely_legal_action() -> None:
    model = BiasPolicy((-1000.0, 0.0))
    optimizer = torch.optim.SGD(model.parameters(), lr=200.0)
    step = PolicyGradientStep(
        observation=np.zeros((1, 1, 1), dtype=np.float32),
        action=0,
        legal_mask=np.array([True, True], dtype=np.bool_),
        player=1,
    )
    episode = PolicyGradientEpisode(
        steps=(step,), outcome=1.0, learner_player=1
    )
    before_logits = model.logits.detach().clone()
    before_probability = torch.softmax(before_logits.double(), dim=-1)[0]

    loss = train_reinforce_epoch(
        model,
        [episode],
        optimizer,
        batch_size=1,
        shuffle=False,
    )
    after_probability = torch.softmax(model.logits.detach().double(), dim=-1)[0]

    assert np.isfinite(loss)
    assert model.logits.grad is not None
    assert torch.isfinite(model.logits.grad).all()
    assert torch.count_nonzero(model.logits.grad) == 2
    assert model.logits[0] > before_logits[0]
    assert after_probability > before_probability


def test_reinforce_update_is_symmetric_for_negative_outcome() -> None:
    positive_model = BiasPolicy((0.0, 0.0))
    negative_model = BiasPolicy((0.0, 0.0))
    positive_optimizer = torch.optim.SGD(positive_model.parameters(), lr=0.2)
    negative_optimizer = torch.optim.SGD(negative_model.parameters(), lr=0.2)
    step = PolicyGradientStep(
        observation=np.zeros((1, 1, 1), dtype=np.float32),
        action=0,
        legal_mask=np.array([True, True], dtype=np.bool_),
        player=1,
    )

    train_reinforce_epoch(
        positive_model,
        [PolicyGradientEpisode((step,), outcome=1.0, learner_player=1)],
        positive_optimizer,
        shuffle=False,
    )
    train_reinforce_epoch(
        negative_model,
        [PolicyGradientEpisode((step,), outcome=-1.0, learner_player=1)],
        negative_optimizer,
        shuffle=False,
    )

    torch.testing.assert_close(positive_model.logits, -negative_model.logits)
    assert torch.softmax(positive_model.logits.detach(), dim=-1)[0] > 0.5
    assert torch.softmax(negative_model.logits.detach(), dim=-1)[0] < 0.5


@dataclass(frozen=True)
class _TwoActionView(LegalPosition):
    def legal_actions_mask(self) -> np.ndarray:
        return np.array([True, True, False], dtype=np.bool_)


@dataclass(frozen=True)
class AlternatingPosition:
    """Two-ply game where each side has a distinct forced action."""

    to_play: int = 1
    history: tuple[int, ...] = ()

    @property
    def action_size(self) -> int:
        return 3

    @property
    def pass_action(self) -> int:
        return 2

    @property
    def is_terminal(self) -> bool:
        return len(self.history) == 2

    def legal_actions_mask(self) -> np.ndarray:
        if self.is_terminal:
            return np.zeros(3, dtype=np.bool_)
        if self.to_play == 1:
            return np.array([True, False, True], dtype=np.bool_)
        return np.array([False, True, True], dtype=np.bool_)

    def play(self, action: int) -> "AlternatingPosition":
        if not self.legal_actions_mask()[action]:
            raise ValueError("illegal action")
        return AlternatingPosition(to_play=-self.to_play, history=self.history + (action,))

    def outcome(self, player: int) -> float:
        if not self.is_terminal:
            raise RuntimeError("game is not terminal")
        black_outcome = 1.0 if self.history == (0, 1) else -1.0
        return black_outcome * player

    def encode(self) -> np.ndarray:
        return np.array(
            [[[self.to_play]], [[len(self.history)]]], dtype=np.float32
        )


def test_alternating_episode_records_only_learner_legal_steps_and_perspective() -> None:
    learner_calls: list[int] = []
    opponent_calls: list[int] = []

    def learner(position: AlternatingPosition) -> np.ndarray:
        learner_calls.append(position.to_play)
        return np.array([1.0, 0.0, 0.0])

    def opponent(position: AlternatingPosition) -> np.ndarray:
        opponent_calls.append(position.to_play)
        return np.array([0.0, 1.0, 0.0])

    episode = generate_policy_gradient_episode(
        AlternatingPosition(),
        learner,
        opponent,
        learner_player=1,
        rng=np.random.default_rng(9),
    )

    assert learner_calls == [1]
    assert opponent_calls == [-1]
    assert episode.learner_player == 1
    assert episode.outcome == 1.0
    assert len(episode.steps) == 1
    assert episode.steps[0].player == 1
    assert episode.steps[0].action == 0
    assert episode.steps[0].legal_mask[episode.steps[0].action]


def test_alternating_episode_scores_white_learner_from_white_perspective() -> None:
    learner_calls: list[int] = []
    opponent_calls: list[int] = []

    def white_learner(position: AlternatingPosition) -> np.ndarray:
        learner_calls.append(position.to_play)
        return np.array([0.0, 1.0, 0.0])

    def black_opponent(position: AlternatingPosition) -> np.ndarray:
        opponent_calls.append(position.to_play)
        return np.array([1.0, 0.0, 0.0])

    episode = generate_policy_gradient_episode(
        AlternatingPosition(),
        white_learner,
        black_opponent,
        learner_player=-1,
        rng=np.random.default_rng(3),
    )

    assert learner_calls == [-1]
    assert opponent_calls == [1]
    assert episode.learner_player == -1
    assert episode.outcome == -1.0
    assert len(episode.steps) == 1
    assert episode.steps[0].player == -1
    assert episode.steps[0].action == 1


@dataclass(frozen=True)
class PassCappedPosition:
    """Go-like game that terminates only after two consecutive passes."""

    to_play: int = 1
    history: tuple[int, ...] = ()

    @property
    def action_size(self) -> int:
        return 3

    @property
    def pass_action(self) -> int:
        return 2

    @property
    def is_terminal(self) -> bool:
        return len(self.history) >= 2 and self.history[-2:] == (2, 2)

    def legal_actions_mask(self) -> np.ndarray:
        if self.is_terminal:
            return np.zeros(3, dtype=np.bool_)
        return np.ones(3, dtype=np.bool_)

    def play(self, action: int) -> "PassCappedPosition":
        if action < 0 or action >= self.action_size:
            raise ValueError("invalid action")
        if not self.legal_actions_mask()[action]:
            raise ValueError("illegal action")
        return PassCappedPosition(
            to_play=-self.to_play,
            history=self.history + (action,),
        )

    def outcome(self, player: int) -> float:
        if not self.is_terminal:
            raise RuntimeError("game is not terminal")
        return float(player)  # Black wins this deterministic teaching game.

    def encode(self) -> np.ndarray:
        last_action = self.history[-1] if self.history else -1
        return np.array(
            [[[self.to_play]], [[len(self.history)]], [[last_action]]],
            dtype=np.float32,
        )


def test_policy_gradient_episode_reserves_unrecorded_forced_passes() -> None:
    learner_calls: list[int] = []
    opponent_calls: list[int] = []

    def learner(position: PassCappedPosition) -> np.ndarray:
        learner_calls.append(len(position.history))
        return np.array([1.0, 0.0, 0.0])

    def opponent(position: PassCappedPosition) -> np.ndarray:
        opponent_calls.append(len(position.history))
        return np.array([1.0, 0.0, 0.0])

    episode = generate_policy_gradient_episode(
        PassCappedPosition(),
        learner,
        opponent,
        learner_player=1,
        rng=np.random.default_rng(0),
        max_moves=4,
    )

    assert episode.outcome == 1.0
    assert learner_calls == [0]
    assert opponent_calls == [1]
    assert len(episode.steps) == 1
    assert episode.steps[0].action == 0
    assert all(
        step.action != PassCappedPosition().pass_action for step in episode.steps
    )


def test_policy_gradient_episode_requires_budget_for_two_cap_passes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        generate_policy_gradient_episode(
            PassCappedPosition(),
            lambda _: np.array([1.0, 0.0, 0.0]),
            lambda _: np.array([1.0, 0.0, 0.0]),
            rng=np.random.default_rng(0),
            max_moves=1,
        )


@dataclass(frozen=True)
class DecorrelatedPosition:
    """Three-ply game exposing game id, sample ply, and random move in encode."""

    game_id: int
    to_play: int = 1
    history: tuple[int, ...] = ()

    @property
    def action_size(self) -> int:
        return 3

    @property
    def pass_action(self) -> int:
        return 2

    @property
    def is_terminal(self) -> bool:
        return len(self.history) == 3

    def legal_actions_mask(self) -> np.ndarray:
        if self.is_terminal:
            return np.zeros(3, dtype=np.bool_)
        if len(self.history) == 0:
            return np.array([True, False, True], dtype=np.bool_)
        if len(self.history) == 1:
            return np.array([True, True, True], dtype=np.bool_)
        return np.array([True, False, True], dtype=np.bool_)

    def play(self, action: int) -> "DecorrelatedPosition":
        if action < 0 or action >= self.action_size:
            raise ValueError("invalid action")
        if not self.legal_actions_mask()[action]:
            raise ValueError("illegal action")
        return DecorrelatedPosition(
            game_id=self.game_id,
            to_play=-self.to_play,
            history=self.history + (action,),
        )

    def outcome(self, player: int) -> float:
        if not self.is_terminal:
            raise RuntimeError("game is not terminal")
        black_outcome = 1.0 if self.history[1] == 0 else -1.0
        return black_outcome * player

    def encode(self) -> np.ndarray:
        random_action = self.history[1] if len(self.history) >= 2 else -1
        return np.array(
            [
                [[self.game_id]],
                [[self.to_play]],
                [[len(self.history)]],
                [[random_action]],
            ],
            dtype=np.float32,
        )


def test_value_examples_follow_sl_random_move_rl_and_one_per_game_contract() -> None:
    next_game_id = 0
    sl_calls: list[tuple[int, int]] = []
    rl_calls: list[tuple[int, int]] = []

    def factory() -> DecorrelatedPosition:
        nonlocal next_game_id
        position = DecorrelatedPosition(game_id=next_game_id)
        next_game_id += 1
        return position

    def sl_policy(position: DecorrelatedPosition) -> np.ndarray:
        sl_calls.append((position.game_id, len(position.history)))
        return np.array([1.0, 0.0, 0.0])

    def rl_policy(position: DecorrelatedPosition) -> np.ndarray:
        rl_calls.append((position.game_id, len(position.history)))
        return np.array([1.0, 0.0, 0.0])

    examples = generate_value_examples(
        factory,
        sl_policy,
        rl_policy,
        num_games=200,
        opening_moves=1,
        rng=np.random.default_rng(1234),
    )

    game_ids = [int(example.observation[0, 0, 0]) for example in examples]
    sample_players = [int(example.observation[1, 0, 0]) for example in examples]
    sample_depths = [int(example.observation[2, 0, 0]) for example in examples]
    random_actions = np.array(
        [int(example.observation[3, 0, 0]) for example in examples]
    )
    outcomes = np.array([example.outcome for example in examples])

    assert len(examples) == 200
    assert game_ids == list(range(200))
    assert len(set(game_ids)) == 200
    assert sample_players == [1] * 200
    assert sample_depths == [2] * 200
    assert set(random_actions) == {0, 1}
    assert abs(int((random_actions == 0).sum()) - 100) < 30
    np.testing.assert_array_equal(outcomes, np.where(random_actions == 0, 1.0, -1.0))
    assert sl_calls == [(game_id, 0) for game_id in range(200)]
    assert rl_calls == [(game_id, 2) for game_id in range(200)]


def test_value_completion_reserves_forced_passes_and_keeps_one_target() -> None:
    sl_calls: list[int] = []
    rl_calls: list[int] = []

    def no_pass_sl(position: PassCappedPosition) -> np.ndarray:
        sl_calls.append(len(position.history))
        return np.array([1.0, 0.0, 0.0])

    def no_pass_rl(position: PassCappedPosition) -> np.ndarray:
        rl_calls.append(len(position.history))
        return np.array([1.0, 0.0, 0.0])

    examples = generate_value_examples(
        PassCappedPosition,
        no_pass_sl,
        no_pass_rl,
        num_games=1,
        opening_moves=1,
        rng=np.random.default_rng(5),
        max_moves=5,
    )

    assert len(examples) == 1
    assert examples[0].outcome == 1.0
    assert int(examples[0].observation[1, 0, 0]) == 2
    assert sl_calls == [0]
    assert rl_calls == [2]


def test_value_opening_range_varies_per_game_and_is_reproducible() -> None:
    def no_pass(position: PassCappedPosition) -> np.ndarray:
        del position
        return np.array([1.0, 0.0, 0.0])

    def generate(seed: int) -> list[ValueExample]:
        return generate_value_examples(
            PassCappedPosition,
            no_pass,
            no_pass,
            num_games=40,
            opening_moves=(0, 3),
            rng=np.random.default_rng(seed),
            max_moves=7,
        )

    first = generate(812)
    second = generate(812)
    depths = [int(example.observation[1, 0, 0]) for example in first]

    assert len(first) == 40
    assert set(depths) == {1, 2, 3, 4}  # inclusive prefix range, plus random move
    for first_example, second_example in zip(first, second, strict=True):
        np.testing.assert_array_equal(
            first_example.observation, second_example.observation
        )
        assert first_example.outcome == second_example.outcome


@pytest.mark.parametrize(
    "opening_moves",
    [(-1, 2), (3, 2), (0,), (True, 2), (0, 2.5)],
)
def test_value_opening_range_validation(opening_moves: object) -> None:
    with pytest.raises(ValueError, match="opening_moves"):
        generate_value_examples(
            PassCappedPosition,
            lambda _: np.array([1.0, 0.0, 0.0]),
            lambda _: np.array([1.0, 0.0, 0.0]),
            num_games=1,
            opening_moves=opening_moves,  # type: ignore[arg-type]
            rng=np.random.default_rng(0),
            max_moves=8,
        )


def test_value_opening_range_budget_covers_largest_prefix() -> None:
    with pytest.raises(ValueError, match="opening.*random.*two.*pass"):
        generate_value_examples(
            PassCappedPosition,
            lambda _: np.array([1.0, 0.0, 0.0]),
            lambda _: np.array([1.0, 0.0, 0.0]),
            num_games=1,
            opening_moves=(0, 3),
            rng=np.random.default_rng(0),
            max_moves=5,
        )


def test_value_generation_requires_budget_for_opening_random_move_and_passes() -> None:
    with pytest.raises(ValueError, match="opening.*random.*two.*pass"):
        generate_value_examples(
            PassCappedPosition,
            lambda _: np.array([1.0, 0.0, 0.0]),
            lambda _: np.array([1.0, 0.0, 0.0]),
            num_games=1,
            opening_moves=1,
            rng=np.random.default_rng(0),
            max_moves=3,
        )


def test_value_example_generation_requires_a_legal_non_pass_random_move() -> None:
    @dataclass(frozen=True)
    class PassOnly(AlternatingPosition):
        def legal_actions_mask(self) -> np.ndarray:
            return np.array([False, False, True], dtype=np.bool_)

    with pytest.raises(ValueError, match="non-pass"):
        generate_value_examples(
            PassOnly,
            lambda _: np.array([0.0, 0.0, 1.0]),
            lambda _: np.array([0.0, 0.0, 1.0]),
            num_games=1,
            opening_moves=0,
            rng=np.random.default_rng(0),
        )
