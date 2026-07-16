from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from alphago_gym.mcts import AlphaGoMCTS, MCTSConfig, PolicyRolloutEvaluator


@dataclass(frozen=True)
class OneMovePosition:
    """Tiny deterministic zero-sum game used to test search invariants."""

    to_play: int = 1
    result_for_black: int | None = None

    @property
    def action_size(self) -> int:
        return 2

    @property
    def is_terminal(self) -> bool:
        return self.result_for_black is not None

    def legal_actions_mask(self) -> np.ndarray:
        return np.array([not self.is_terminal, not self.is_terminal], dtype=np.bool_)

    def play(self, action: int) -> "OneMovePosition":
        if self.is_terminal or action not in (0, 1):
            raise ValueError("illegal action")
        # Action 0 wins for the actor; action 1 loses for the actor.
        black_result = self.to_play if action == 0 else -self.to_play
        return OneMovePosition(to_play=-self.to_play, result_for_black=black_result)

    def outcome(self, player: int) -> float:
        if not self.is_terminal:
            raise RuntimeError("game is not over")
        assert self.result_for_black is not None
        return float(self.result_for_black * player)

    def encode(self) -> np.ndarray:
        return np.full((1, 1, 1), self.to_play, dtype=np.float32)


def uniform_policy(position: OneMovePosition) -> np.ndarray:
    del position
    return np.array([0.5, 0.5], dtype=np.float64)


def zero_value(position: OneMovePosition) -> float:
    del position
    return 0.0


def terminal_rollout(position: OneMovePosition, rng: np.random.Generator) -> float:
    del rng
    if position.is_terminal:
        return position.outcome(position.to_play)
    return position.play(0).outcome(position.to_play)


def test_config_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="num_simulations"):
        MCTSConfig(num_simulations=0)
    with pytest.raises(ValueError, match="mixing_lambda"):
        MCTSConfig(mixing_lambda=1.1)
    with pytest.raises(ValueError, match="c_puct"):
        MCTSConfig(c_puct=0.0)


def test_search_visits_only_legal_edges_and_does_not_mutate_position() -> None:
    position = OneMovePosition()
    mcts = AlphaGoMCTS(
        policy=uniform_policy,
        value=zero_value,
        rollout=terminal_rollout,
        config=MCTSConfig(num_simulations=12, mixing_lambda=1.0),
        seed=7,
    )

    result = mcts.search(position)

    assert position == OneMovePosition()
    assert result.visit_counts.shape == (2,)
    assert int(result.visit_counts.sum()) == 12
    assert result.action in (0, 1)
    assert np.isclose(result.search_policy.sum(), 1.0)
    assert result.search_policy[result.action] > 0


def test_backup_uses_each_nodes_player_perspective() -> None:
    mcts = AlphaGoMCTS(
        policy=uniform_policy,
        value=zero_value,
        rollout=terminal_rollout,
        config=MCTSConfig(num_simulations=40, c_puct=1.0, mixing_lambda=1.0),
        seed=3,
    )

    result = mcts.search(OneMovePosition())

    assert result.action == 0
    assert result.q_values[0] == pytest.approx(1.0)
    assert result.q_values[1] == pytest.approx(-1.0)
    assert result.visit_counts[0] > result.visit_counts[1]


def test_leaf_evaluation_mixes_value_and_rollout_separately() -> None:
    @dataclass(frozen=True)
    class TwoMovePosition:
        to_play: int = 1
        depth: int = 0
        first_action: int | None = None

        @property
        def action_size(self) -> int:
            return 2

        @property
        def is_terminal(self) -> bool:
            return self.depth == 2

        def legal_actions_mask(self) -> np.ndarray:
            return np.array([not self.is_terminal] * 2, dtype=np.bool_)

        def play(self, action: int) -> "TwoMovePosition":
            return TwoMovePosition(
                to_play=-self.to_play,
                depth=self.depth + 1,
                first_action=action if self.depth == 0 else self.first_action,
            )

        def outcome(self, player: int) -> float:
            if not self.is_terminal:
                raise RuntimeError("game is not over")
            black_result = 1 if self.first_action == 0 else -1
            return float(black_result * player)

        def encode(self) -> np.ndarray:
            return np.full((1, 1, 1), self.to_play, dtype=np.float32)

    def pessimistic_value(position: TwoMovePosition) -> float:
        # The child player loses after root action 0 and wins after action 1.
        return -1.0 if position.first_action == 0 else 1.0

    def optimistic_rollout(position: TwoMovePosition, rng: np.random.Generator) -> float:
        del rng
        return 1.0 if position.first_action == 0 else -1.0

    mcts = AlphaGoMCTS(
        policy=lambda _: np.array([0.5, 0.5]),
        value=pessimistic_value,
        rollout=optimistic_rollout,
        config=MCTSConfig(num_simulations=1, c_puct=1.0, mixing_lambda=0.25),
        seed=11,
    )
    result = mcts.search(TwoMovePosition())

    visited = np.flatnonzero(result.visit_counts)
    assert len(visited) >= 1
    for action in visited:
        # From the root actor's perspective the learned value is +1 for action
        # 0 / -1 for action 1, while this synthetic rollout says the opposite.
        expected = 0.5 if action == 0 else -0.5
        assert result.q_values[action] == pytest.approx(expected)


def test_illegal_policy_mass_is_removed_and_priors_are_renormalized() -> None:
    @dataclass(frozen=True)
    class OnlySecondAction(OneMovePosition):
        def legal_actions_mask(self) -> np.ndarray:
            return np.array([False, not self.is_terminal], dtype=np.bool_)

    mcts = AlphaGoMCTS(
        policy=lambda _: np.array([0.999, 0.001]),
        value=zero_value,
        rollout=terminal_rollout,
        config=MCTSConfig(num_simulations=3),
        seed=0,
    )

    result = mcts.search(OnlySecondAction())

    assert result.action == 1
    assert result.priors.tolist() == [0.0, 1.0]
    assert result.visit_counts.tolist() == [0, 3]


def test_seed_makes_tied_search_reproducible() -> None:
    kwargs = dict(
        policy=uniform_policy,
        value=zero_value,
        rollout=terminal_rollout,
        config=MCTSConfig(num_simulations=1),
        seed=123,
    )
    first = AlphaGoMCTS(**kwargs).search(OneMovePosition())
    second = AlphaGoMCTS(**kwargs).search(OneMovePosition())
    assert np.array_equal(first.visit_counts, second.visit_counts)


def test_rollout_evaluator_returns_leaf_player_outcome() -> None:
    rollout = PolicyRolloutEvaluator(
        policy=lambda position: np.array([1.0, 0.0]), max_moves=2
    )
    root = OneMovePosition(to_play=-1)

    assert rollout(root, np.random.default_rng(0)) == 1.0
    terminal = root.play(1)
    assert rollout(terminal, np.random.default_rng(0)) == terminal.outcome(terminal.to_play)
