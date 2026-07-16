"""Tests for the small synthetic expert-data replacement for KGS records."""

from __future__ import annotations

import numpy as np
import pytest

from alphago_gym.data import (
    ExpertDataset,
    ExpertGame,
    ExpertStep,
    capture_and_liberty_policy,
    generate_expert_games,
    heuristic_policy,
    uniform_random_policy,
)
from alphago_gym.go import BLACK, EMPTY, WHITE, GoPosition


def test_uniform_policy_normalizes_over_exactly_the_legal_actions() -> None:
    board = np.array(
        [
            [BLACK, EMPTY, EMPTY],
            [EMPTY, WHITE, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    position = GoPosition(size=3, board=board, to_play=BLACK)
    board_before = position.board.copy()
    legal = position.legal_actions_mask()

    probabilities = uniform_random_policy(position)

    assert probabilities.shape == (position.action_size,)
    assert probabilities.dtype == np.float64
    assert np.isclose(probabilities.sum(), 1.0)
    np.testing.assert_array_equal(probabilities > 0.0, legal)
    np.testing.assert_allclose(
        probabilities[legal], np.full(legal.sum(), 1.0 / legal.sum())
    )
    np.testing.assert_array_equal(position.board, board_before)


@pytest.mark.parametrize("policy", [uniform_random_policy, heuristic_policy])
def test_policies_reject_a_terminal_position(policy: object) -> None:
    terminal = GoPosition(size=3).play(9).play(9)

    with pytest.raises(ValueError, match="terminal|legal"):
        policy(terminal)  # type: ignore[operator]


def test_heuristic_policy_is_deterministic_normalized_and_center_seeking() -> None:
    position = GoPosition(size=5)
    board_before = position.board.copy()

    first = heuristic_policy(position)
    second = heuristic_policy(position)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (26,)
    assert first.dtype == np.float64
    assert np.isclose(first.sum(), 1.0)
    assert np.all(np.isfinite(first))
    assert np.all(first[position.legal_actions_mask()] > 0.0)
    assert np.argmax(first) == 12
    # Passing is a safety action, not an opening move for synthetic experts.
    assert first[position.pass_action] < first[:-1].min() * 1e-4
    np.testing.assert_array_equal(position.board, board_before)
    assert capture_and_liberty_policy is heuristic_policy


def test_lower_temperature_makes_heuristic_distribution_sharper() -> None:
    position = GoPosition(size=5)

    cold = heuristic_policy(position, temperature=0.25)
    warm = heuristic_policy(position, temperature=2.0)

    assert np.argmax(cold) == np.argmax(warm) == 12
    assert cold.max() > warm.max()


def test_smallest_positive_temperature_remains_finite_and_normalized() -> None:
    probabilities = heuristic_policy(
        GoPosition(size=3), temperature=np.nextafter(0.0, 1.0)
    )

    assert np.all(np.isfinite(probabilities))
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities > 0.0)


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.inf, np.nan, True, "1"])
def test_heuristic_policy_validates_temperature(temperature: object) -> None:
    with pytest.raises((TypeError, ValueError), match="temperature"):
        heuristic_policy(
            GoPosition(size=3), temperature=temperature  # type: ignore[arg-type]
        )


def test_heuristic_strongly_favors_a_capture_over_quiet_moves() -> None:
    board = np.array(
        [
            [WHITE, EMPTY, WHITE],
            [BLACK, WHITE, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    position = GoPosition(size=3, board=board, to_play=BLACK)

    probabilities = heuristic_policy(position)
    capture = 1

    assert np.argmax(probabilities) == capture
    assert probabilities[capture] > 5.0 * np.partition(probabilities[:-1], -2)[-2]
    assert position.play(capture).board[0, 0] == EMPTY


def test_heuristic_favors_the_only_liberty_of_an_atari_group() -> None:
    board = np.array(
        [
            [BLACK, EMPTY, EMPTY],
            [WHITE, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    position = GoPosition(size=3, board=board, to_play=BLACK)

    probabilities = heuristic_policy(position)

    assert np.argmax(probabilities) == 1
    assert probabilities[1] > probabilities[4]


def test_heuristic_assigns_zero_mass_to_occupied_suicide_and_superko_moves() -> None:
    ko_board = np.array(
        [
            [WHITE, EMPTY, WHITE],
            [BLACK, WHITE, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    ko_position = GoPosition(size=3, board=ko_board, to_play=BLACK).play(1)
    suicide_board = np.array(
        [
            [EMPTY, WHITE, EMPTY],
            [WHITE, EMPTY, WHITE],
            [EMPTY, WHITE, EMPTY],
        ],
        dtype=np.int8,
    )
    suicide_position = GoPosition(size=3, board=suicide_board, to_play=BLACK)

    ko_probabilities = heuristic_policy(ko_position)
    suicide_probabilities = heuristic_policy(suicide_position)

    assert ko_probabilities[0] == 0.0  # positional-superko recapture
    assert ko_probabilities[1] == 0.0  # occupied by Black
    assert suicide_probabilities[4] == 0.0


def test_heuristic_uses_pass_when_it_is_the_only_legal_action() -> None:
    # Playing the sole point on a 1x1 board is suicide, so only pass is legal.
    position = GoPosition(size=1)

    probabilities = heuristic_policy(position)

    np.testing.assert_array_equal(probabilities, np.array([0.0, 1.0]))


def test_generate_expert_games_returns_typed_terminal_records_and_flat_steps() -> None:
    dataset = generate_expert_games(
        num_games=3,
        size=3,
        komi=0.5,
        seed=17,
        max_moves=14,
    )

    assert isinstance(dataset, ExpertDataset)
    assert isinstance(dataset.games, tuple)
    assert isinstance(dataset.steps, tuple)
    assert dataset.expert_steps is dataset.steps
    assert dataset.game_records is dataset.games
    assert len(dataset.games) == 3
    assert len(dataset) == len(dataset.steps)
    assert len(dataset.steps) == sum(len(game.steps) for game in dataset.games)

    offset = 0
    for game in dataset.games:
        assert isinstance(game, ExpertGame)
        assert game.final_position.is_terminal
        assert 2 <= len(game.steps) <= 14
        assert game.black_outcome == game.final_position.outcome(BLACK)
        assert game.white_outcome == game.final_position.outcome(WHITE)
        assert game.black_outcome == -game.white_outcome
        assert game.outcomes == {
            BLACK: game.black_outcome,
            WHITE: game.white_outcome,
        }
        expected_winner = (
            BLACK
            if game.black_outcome > 0
            else WHITE
            if game.black_outcome < 0
            else EMPTY
        )
        assert game.winner == expected_winner

        for index, step in enumerate(game.steps):
            assert isinstance(step, ExpertStep)
            assert step.observation.shape == (8, 3, 3)
            assert step.observation.dtype == np.float32
            assert step.observation.flags.c_contiguous
            assert step.legal_mask.shape == (10,)
            assert step.legal_mask.dtype == np.bool_
            assert step.legal_mask[step.action]
            assert step.player in (BLACK, WHITE)
            expected_colour_plane = 1.0 if step.player == BLACK else 0.0
            assert np.all(step.observation[7] == expected_colour_plane)

            flat_step = dataset.steps[offset + index]
            assert flat_step.action == step.action
            assert flat_step.player == step.player
            np.testing.assert_array_equal(flat_step.observation, step.observation)
            np.testing.assert_array_equal(flat_step.legal_mask, step.legal_mask)
            assert not np.shares_memory(flat_step.observation, step.observation)
            assert not np.shares_memory(flat_step.legal_mask, step.legal_mask)
        offset += len(game.steps)


def test_each_game_can_be_replayed_exactly_from_its_snapshots() -> None:
    dataset = generate_expert_games(5, size=3, komi=0.5, seed=91, max_moves=15)

    for game in dataset.games:
        position = GoPosition(size=3, komi=0.5)
        for step in game.steps:
            np.testing.assert_array_equal(step.observation, position.encode())
            np.testing.assert_array_equal(
                step.legal_mask, position.legal_actions_mask()
            )
            assert step.player == position.to_play
            position = position.play(step.action)
        np.testing.assert_array_equal(position.board, game.final_position.board)
        assert position.history == game.final_position.history


def test_expert_snapshots_are_deeply_immutable_and_never_share_memory() -> None:
    dataset = generate_expert_games(2, size=3, seed=2, max_moves=10)
    all_game_steps = [step for game in dataset.games for step in game.steps]

    for step in (*all_game_steps, *dataset.steps):
        assert not step.observation.flags.writeable
        assert not step.legal_mask.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            step.observation.setflags(write=True)
        with pytest.raises(ValueError, match="WRITEABLE"):
            step.legal_mask.setflags(write=True)

    for left, right in zip(all_game_steps, all_game_steps[1:]):
        assert not np.shares_memory(left.observation, right.observation)
        assert not np.shares_memory(left.legal_mask, right.legal_mask)


def test_expert_step_constructor_copies_inputs_and_validates_contract() -> None:
    observation = np.zeros((8, 2, 2), dtype=np.float64)
    legal_mask = np.ones(5, dtype=np.bool_)
    step = ExpertStep(observation, legal_mask, action=2, player=BLACK)
    observation.fill(7.0)
    legal_mask.fill(False)

    assert step.observation.dtype == np.float32
    assert np.count_nonzero(step.observation) == 0
    assert step.legal_mask.all()
    with pytest.raises(ValueError, match="observation"):
        ExpertStep(np.zeros((2, 2)), np.ones(5), action=0, player=BLACK)
    with pytest.raises(ValueError, match="legal_mask"):
        ExpertStep(np.zeros((8, 2, 2)), np.ones(4), action=0, player=BLACK)
    with pytest.raises(ValueError, match="smaller"):
        ExpertStep(np.zeros((8, 2, 2)), np.ones(5), action=5, player=BLACK)
    illegal = np.ones(5, dtype=np.bool_)
    illegal[2] = False
    with pytest.raises(ValueError, match="legal move"):
        ExpertStep(np.zeros((8, 2, 2)), illegal, action=2, player=BLACK)
    with pytest.raises(ValueError, match="player"):
        ExpertStep(np.zeros((8, 2, 2)), np.ones(5), action=2, player=EMPTY)


def test_game_and_dataset_records_reject_inconsistent_contents() -> None:
    valid = generate_expert_games(1, size=2, seed=4, max_moves=2)
    game = valid.games[0]
    common = {
        "steps": game.steps,
        "winner": game.winner,
        "black_outcome": game.black_outcome,
        "white_outcome": game.white_outcome,
        "final_position": game.final_position,
    }

    with pytest.raises(TypeError, match="ExpertStep"):
        ExpertGame(**{**common, "steps": (object(),)})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="GoPosition"):
        ExpertGame(**{**common, "final_position": object()})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="terminal"):
        ExpertGame(**{**common, "final_position": GoPosition(size=2)})
    with pytest.raises(ValueError, match="opposites"):
        ExpertGame(**{**common, "black_outcome": 1, "white_outcome": 1})
    with pytest.raises(ValueError, match="winner"):
        ExpertGame(**{**common, "winner": BLACK})
    with pytest.raises(ValueError, match="final_position"):
        ExpertGame(
            **{
                **common,
                "winner": BLACK,
                "black_outcome": 1,
                "white_outcome": -1,
            }
        )

    with pytest.raises(TypeError, match="ExpertGame"):
        ExpertDataset(games=(object(),), steps=())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ExpertStep"):
        ExpertDataset(games=(), steps=(object(),))  # type: ignore[arg-type]


def _dataset_signature(dataset: ExpertDataset) -> tuple[object, ...]:
    return tuple(
        (
            tuple(step.action for step in game.steps),
            tuple(step.player for step in game.steps),
            game.winner,
            game.black_outcome,
            game.final_position.board.tobytes(),
        )
        for game in dataset.games
    )


def test_seeded_generation_is_reproducible_but_returns_independent_arrays() -> None:
    first = generate_expert_games(4, size=3, komi=0.5, seed=123, max_moves=16)
    second = generate_expert_games(4, size=3, komi=0.5, seed=123, max_moves=16)
    different = generate_expert_games(4, size=3, komi=0.5, seed=124, max_moves=16)

    assert _dataset_signature(first) == _dataset_signature(second)
    assert _dataset_signature(first) != _dataset_signature(different)
    for left, right in zip(first.steps, second.steps):
        np.testing.assert_array_equal(left.observation, right.observation)
        np.testing.assert_array_equal(left.legal_mask, right.legal_mask)
        assert not np.shares_memory(left.observation, right.observation)
        assert not np.shares_memory(left.legal_mask, right.legal_mask)


def test_max_move_guard_forces_two_passes_and_respects_total_cap() -> None:
    dataset = generate_expert_games(1, size=3, komi=0.5, seed=4, max_moves=2)
    game = dataset.games[0]

    assert tuple(step.action for step in game.steps) == (9, 9)
    assert tuple(step.player for step in game.steps) == (BLACK, WHITE)
    assert game.final_position.is_terminal
    assert game.winner == WHITE
    assert game.white_outcome == 1


def test_zero_games_returns_an_empty_dataset() -> None:
    dataset = generate_expert_games(0, size=3, seed=0, max_moves=2)

    assert dataset.games == ()
    assert dataset.steps == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_games": -1}, "num_games"),
        ({"num_games": True}, "num_games"),
        ({"num_games": 1, "size": 0}, "size"),
        ({"num_games": 1, "komi": np.inf}, "komi"),
        ({"num_games": 1, "max_moves": 1}, "max_moves"),
        ({"num_games": 1, "max_moves": 2.5}, "max_moves"),
    ],
)
def test_generate_expert_games_validates_inputs(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        generate_expert_games(**kwargs)  # type: ignore[arg-type]
