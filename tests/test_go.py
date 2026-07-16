from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from alphago_gym.go import BLACK, EMPTY, WHITE, GoPosition


def point(row: int, col: int, size: int) -> int:
    return row * size + col


def test_new_position_has_expected_defaults_and_action_contract() -> None:
    position = GoPosition(size=3, komi=5.5)

    assert position.size == 3
    assert position.komi == 5.5
    assert position.to_play == BLACK
    assert position.current_player == BLACK
    assert position.action_size == 10
    assert position.pass_action == 9
    assert position.move_count == 0
    assert position.consecutive_passes == 0
    assert not position.is_terminal
    np.testing.assert_array_equal(position.board, np.full((3, 3), EMPTY))


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_position_rejects_invalid_board_size(size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="size"):
        GoPosition(size=size)  # type: ignore[arg-type]


def test_position_validates_custom_board_and_player() -> None:
    with pytest.raises(ValueError, match="shape"):
        GoPosition(size=3, board=np.zeros((2, 2), dtype=np.int8))
    with pytest.raises(ValueError, match="values"):
        GoPosition(size=2, board=np.array([[0, 2], [0, 0]]))
    with pytest.raises(ValueError, match="to_play"):
        GoPosition(size=2, to_play=0)
    with pytest.raises(ValueError, match="consecutive_passes"):
        GoPosition(size=2, consecutive_passes=-1)


def test_position_is_deeply_immutable() -> None:
    source = np.zeros((2, 2), dtype=np.int8)
    position = GoPosition(size=2, board=source)
    source[0, 0] = BLACK

    assert position.board[0, 0] == EMPTY
    assert not position.board.flags.writeable
    assert isinstance(position.history, frozenset)
    with pytest.raises(ValueError, match="read-only"):
        position.board[0, 0] = BLACK
    with pytest.raises(FrozenInstanceError):
        position.to_play = WHITE  # type: ignore[misc]


def test_play_returns_a_new_position_without_mutating_the_parent() -> None:
    position = GoPosition(size=3, komi=0.5)
    child = position.play(point(1, 1, 3))

    assert child is not position
    assert position.board[1, 1] == EMPTY
    assert child.board[1, 1] == BLACK
    assert position.to_play == BLACK
    assert child.to_play == WHITE
    assert child.komi == position.komi
    assert position.move_count == 0
    assert child.move_count == 1


def test_single_stone_capture() -> None:
    board = np.array(
        [
            [WHITE, EMPTY, WHITE],
            [BLACK, WHITE, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    position = GoPosition(size=3, board=board, to_play=BLACK)

    captured = position.play(point(0, 1, 3))

    assert captured.board[0, 0] == EMPTY
    assert captured.board[0, 1] == BLACK
    assert position.board[0, 0] == WHITE


def test_multi_stone_group_capture() -> None:
    board = np.array(
        [
            [WHITE, WHITE, EMPTY],
            [BLACK, BLACK, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    position = GoPosition(size=3, board=board, to_play=BLACK)

    captured = position.play(point(0, 2, 3))

    assert captured.board[0, 0] == EMPTY
    assert captured.board[0, 1] == EMPTY
    assert captured.board[0, 2] == BLACK


def test_suicide_is_illegal_and_capture_that_creates_liberties_is_legal() -> None:
    surrounded = np.array(
        [
            [EMPTY, WHITE, EMPTY],
            [WHITE, EMPTY, WHITE],
            [EMPTY, WHITE, EMPTY],
        ],
        dtype=np.int8,
    )
    suicidal = GoPosition(size=3, board=surrounded, to_play=BLACK)

    assert not suicidal.legal_actions_mask()[point(1, 1, 3)]
    with pytest.raises(ValueError, match="suicide"):
        suicidal.play(point(1, 1, 3))

    capturable = np.array(
        [
            [BLACK, WHITE, BLACK],
            [WHITE, EMPTY, WHITE],
            [BLACK, WHITE, BLACK],
        ],
        dtype=np.int8,
    )
    capture = GoPosition(size=3, board=capturable, to_play=BLACK).play(
        point(1, 1, 3)
    )
    assert capture.board[1, 1] == BLACK
    assert np.count_nonzero(capture.board == WHITE) == 0


def test_positional_superko_rejects_recapture() -> None:
    board = np.array(
        [
            [WHITE, EMPTY, WHITE],
            [BLACK, WHITE, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    before_capture = GoPosition(size=3, board=board, to_play=BLACK)
    after_capture = before_capture.play(point(0, 1, 3))

    assert not after_capture.legal_actions_mask()[point(0, 0, 3)]
    with pytest.raises(ValueError, match="superko"):
        after_capture.play(point(0, 0, 3))


def test_pass_is_always_legal_despite_repeating_the_board() -> None:
    position = GoPosition(size=3)

    assert position.legal_actions_mask()[position.pass_action]
    after_one_pass = position.play(position.pass_action)
    assert after_one_pass.legal_actions_mask()[after_one_pass.pass_action]
    after_two_passes = after_one_pass.play(after_one_pass.pass_action)

    np.testing.assert_array_equal(after_one_pass.board, position.board)
    assert after_one_pass.to_play == WHITE
    assert after_one_pass.consecutive_passes == 1
    assert not after_one_pass.is_terminal
    assert after_two_passes.to_play == BLACK
    assert after_two_passes.consecutive_passes == 2
    assert after_two_passes.is_terminal
    assert not after_two_passes.legal_actions_mask().any()
    with pytest.raises(ValueError, match="terminal"):
        after_two_passes.play(after_two_passes.pass_action)


def test_playing_a_stone_resets_consecutive_passes() -> None:
    position = GoPosition(size=3).play(9)
    continued = position.play(point(1, 1, 3))

    assert continued.consecutive_passes == 0
    assert not continued.is_terminal


@pytest.mark.parametrize("action", [-1, 10, 2.5, True, "0"])
def test_play_rejects_invalid_action_values(action: object) -> None:
    with pytest.raises((TypeError, ValueError), match="action"):
        GoPosition(size=3).play(action)  # type: ignore[arg-type]


def test_occupied_intersection_is_illegal() -> None:
    position = GoPosition(size=3).play(point(0, 0, 3))

    assert not position.legal_actions_mask()[point(0, 0, 3)]
    with pytest.raises(ValueError, match="occupied"):
        position.play(point(0, 0, 3))


def test_legal_action_mask_shape_dtype_and_independence() -> None:
    position = GoPosition(size=2)
    mask = position.legal_actions_mask()

    assert mask.shape == (5,)
    assert mask.dtype == np.bool_
    assert mask.all()
    mask[:] = False
    assert position.legal_actions_mask().all()


def test_chinese_area_scoring_counts_stones_and_surrounded_territory() -> None:
    black_enclosure = np.array(
        [
            [BLACK, BLACK, BLACK],
            [BLACK, EMPTY, BLACK],
            [BLACK, BLACK, BLACK],
        ],
        dtype=np.int8,
    )
    position = GoPosition(size=3, board=black_enclosure, komi=8.5)

    # Black has eight stones plus one territory point; White has 8.5 komi.
    assert position.outcome(BLACK) == 1
    assert position.outcome(WHITE) == -1


def test_chinese_area_scoring_leaves_dame_neutral_and_applies_komi() -> None:
    mixed = np.array(
        [
            [BLACK, EMPTY, WHITE],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )

    tie = GoPosition(size=3, board=mixed, komi=0.0)
    assert tie.outcome(BLACK) == 0
    assert tie.outcome(WHITE) == 0

    empty_with_komi = GoPosition(size=3, komi=0.5)
    assert empty_with_komi.outcome(BLACK) == -1
    assert empty_with_komi.outcome(WHITE) == 1
    with pytest.raises(ValueError, match="player"):
        empty_with_komi.outcome(EMPTY)


def test_encode_has_eight_float32_current_player_feature_planes() -> None:
    board = np.array(
        [
            [BLACK, BLACK, EMPTY],
            [EMPTY, WHITE, EMPTY],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    black_view = GoPosition(size=3, board=board, to_play=BLACK).encode()
    white_view = GoPosition(size=3, board=board, to_play=WHITE).encode()

    assert black_view.shape == (8, 3, 3)
    assert black_view.dtype == np.float32
    np.testing.assert_array_equal(black_view[0], board == BLACK)
    np.testing.assert_array_equal(black_view[1], board == WHITE)
    np.testing.assert_array_equal(black_view[2], board == EMPTY)
    # The connected black pair has exactly two liberties.
    np.testing.assert_array_equal(black_view[3], np.zeros((3, 3)))
    np.testing.assert_array_equal(black_view[4], board == BLACK)
    # The white singleton has three liberties.
    np.testing.assert_array_equal(black_view[5], np.zeros((3, 3)))
    np.testing.assert_array_equal(black_view[6], np.zeros((3, 3)))
    np.testing.assert_array_equal(black_view[7], np.ones((3, 3)))

    np.testing.assert_array_equal(white_view[0], board == WHITE)
    np.testing.assert_array_equal(white_view[1], board == BLACK)
    np.testing.assert_array_equal(white_view[2], black_view[2])
    np.testing.assert_array_equal(white_view[5], np.zeros((3, 3)))
    np.testing.assert_array_equal(white_view[6], board == BLACK)
    np.testing.assert_array_equal(white_view[7], np.zeros((3, 3)))


def test_encode_marks_every_stone_in_one_liberty_groups() -> None:
    board = np.array(
        [
            [BLACK, BLACK, EMPTY],
            [WHITE, WHITE, WHITE],
            [EMPTY, EMPTY, EMPTY],
        ],
        dtype=np.int8,
    )
    encoded = GoPosition(size=3, board=board, to_play=BLACK).encode()

    expected = np.zeros((3, 3), dtype=np.float32)
    expected[0, :2] = 1.0
    np.testing.assert_array_equal(encoded[3], expected)

