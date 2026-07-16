"""Immutable Go rules used by the tutorial environment and search code.

The implementation deliberately favours clarity over tournament-engine speed.  It
implements the rules that matter to AlphaGo-style self play: captures, suicide,
positional superko, passing, and Chinese area scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Iterable

import numpy as np

BLACK = 1
WHITE = -1
EMPTY = 0

_PLAYERS = (BLACK, WHITE)


class _ImmutableBoard(np.ndarray):
    """Read-only ndarray whose shape/type metadata cannot be reassigned.

    A bytes-backed ndarray protects its elements but NumPy still permits
    metadata-only operations such as ``shape = ...`` and same-size ``resize``.
    This zero-copy subclass closes those mutation paths while retaining normal
    ndarray indexing, ufuncs, and view performance.
    """

    __slots__ = ()

    def __new__(cls, board_bytes: bytes, size: int) -> _ImmutableBoard:
        board = np.frombuffer(board_bytes, dtype=np.int8).reshape(size, size)
        return board.view(cls)

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("GoPosition board metadata is immutable")

    def resize(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise ValueError("GoPosition board shape is immutable; resize is disabled")

    def setflags(
        self,
        write: bool | None = None,
        align: bool | None = None,
        uic: bool | None = None,
    ) -> None:
        del write, align, uic
        raise ValueError("GoPosition board flags are immutable; WRITEABLE is false")


@dataclass(frozen=True, slots=True, eq=False)
class GoPosition:
    """An immutable, alternating-turn Go position.

    Actions are flattened row-major board coordinates.  ``size * size`` is the
    pass action.  ``history`` stores board-only keys, which makes the ko rule
    *positional* superko.  Passes are intentionally exempt from that check.

    The eight encoded planes are a compact educational reduction of AlphaGo's
    48 input planes: current stones, opponent stones, empty points, exact one-
    and two-liberty groups for each player, and the absolute colour to play.
    """

    size: int = 5
    komi: float = 5.5
    board: np.ndarray | None = field(default=None, repr=False, compare=False)
    to_play: int = BLACK
    history: frozenset[bytes] | Iterable[bytes] | None = field(
        default=None, repr=False, compare=False
    )
    consecutive_passes: int = 0
    move_count: int = 0

    def __post_init__(self) -> None:
        size = _validated_integer(self.size, "size", minimum=1)
        if not isinstance(self.komi, Real) or isinstance(self.komi, (bool, np.bool_)):
            raise TypeError("komi must be a finite real number")
        komi = float(self.komi)
        if not np.isfinite(komi):
            raise ValueError("komi must be a finite real number")

        to_play = _validated_player(self.to_play, "to_play")
        consecutive_passes = _validated_integer(
            self.consecutive_passes, "consecutive_passes", minimum=0
        )
        if consecutive_passes > 2:
            raise ValueError("consecutive_passes must be 0, 1, or 2")
        move_count = _validated_integer(self.move_count, "move_count", minimum=0)

        if self.board is None:
            board = np.full((size, size), EMPTY, dtype=np.int8)
        else:
            raw_board = np.asarray(self.board)
            if raw_board.shape != (size, size):
                raise ValueError(
                    f"board shape must be {(size, size)}, got {raw_board.shape}"
                )
            try:
                values_are_valid = bool(
                    np.all(np.isin(raw_board, (BLACK, WHITE, EMPTY)))
                )
            except (TypeError, ValueError):
                values_are_valid = False
            if not values_are_valid:
                raise ValueError(
                    "board values must be BLACK (1), WHITE (-1), or EMPTY (0)"
                )
            board = np.array(raw_board, dtype=np.int8, order="C", copy=True)

        # Back the public array with immutable ``bytes``.  Merely clearing an
        # owning ndarray's WRITEABLE flag is reversible via ``setflags`` and
        # would let callers mutate a supposedly frozen position.
        current_key = board.tobytes(order="C")
        board = _ImmutableBoard(current_key, size)
        if self.history is None:
            history = frozenset((current_key,))
        else:
            try:
                history = frozenset(bytes(key) for key in self.history)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "history must be an iterable of board byte keys"
                ) from error
            history = history | {current_key}

        object.__setattr__(self, "size", size)
        object.__setattr__(self, "komi", komi)
        object.__setattr__(self, "board", board)
        object.__setattr__(self, "to_play", to_play)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "consecutive_passes", consecutive_passes)
        object.__setattr__(self, "move_count", move_count)

    @property
    def current_player(self) -> int:
        """Alias that reads naturally in environment and notebook code."""

        return self.to_play

    @property
    def action_size(self) -> int:
        """Number of board actions including pass."""

        return self.size * self.size + 1

    @property
    def pass_action(self) -> int:
        """The action index representing pass."""

        return self.size * self.size

    @property
    def is_terminal(self) -> bool:
        """Whether both players have passed consecutively."""

        return self.consecutive_passes >= 2

    @property
    def position_key(self) -> bytes:
        """Board-only key used for positional superko."""

        return self.board.tobytes(order="C")

    def legal_actions_mask(self) -> np.ndarray:
        """Return a fresh Boolean mask of legal board moves plus pass."""

        mask = np.zeros(self.action_size, dtype=np.bool_)
        if self.is_terminal:
            return mask

        flat_board = self.board.reshape(-1)
        for action in np.flatnonzero(flat_board == EMPTY):
            try:
                self._board_after_stone(int(action))
            except ValueError:
                continue
            mask[action] = True
        # A pass repeats the board, but is explicitly exempt from superko.
        mask[self.pass_action] = True
        return mask

    def play(self, action: int) -> GoPosition:
        """Play ``action`` and return a new position.

        Raises:
            ValueError: if the game is over, the point is occupied, the move is
                suicide, repeats a historical board, or is out of range.
            TypeError: if ``action`` is not an integer action index.
        """

        action = _validated_integer(action, "action", minimum=0)
        if action >= self.action_size:
            raise ValueError(
                f"action must be between 0 and {self.pass_action}, got {action}"
            )
        if self.is_terminal:
            raise ValueError("cannot play an action in a terminal position")

        if action == self.pass_action:
            return GoPosition(
                size=self.size,
                komi=self.komi,
                board=self.board,
                to_play=-self.to_play,
                history=self.history,
                consecutive_passes=self.consecutive_passes + 1,
                move_count=self.move_count + 1,
            )

        next_board = self._board_after_stone(action)
        return GoPosition(
            size=self.size,
            komi=self.komi,
            board=next_board,
            to_play=-self.to_play,
            history=self.history,
            consecutive_passes=0,
            move_count=self.move_count + 1,
        )

    def area_score(self) -> tuple[float, float]:
        """Return ``(black_score, white_score)`` under Chinese area scoring."""

        black_score = float(np.count_nonzero(self.board == BLACK))
        white_score = float(np.count_nonzero(self.board == WHITE)) + self.komi
        visited: set[int] = set()
        flat_board = self.board.reshape(-1)

        for start in np.flatnonzero(flat_board == EMPTY):
            start = int(start)
            if start in visited:
                continue
            region, bordering_colours = self._empty_region(start)
            visited.update(region)
            if bordering_colours == {BLACK}:
                black_score += len(region)
            elif bordering_colours == {WHITE}:
                white_score += len(region)

        return black_score, white_score

    def outcome(self, player: int) -> int:
        """Return the Chinese-area result from ``player``'s perspective."""

        player = _validated_player(player, "player")
        black_score, white_score = self.area_score()
        black_result = int(black_score > white_score) - int(black_score < white_score)
        return black_result if player == BLACK else -black_result

    def encode(self) -> np.ndarray:
        """Encode the position as eight ``float32`` feature planes.

        Planes 0--2 use the side-to-move perspective.  Planes 3--6 mark every
        stone belonging to a group with exactly one or two liberties.  Plane 7
        is all ones when Black is to play and zeros when White is to play.
        """

        current = self.to_play
        opponent = -current
        planes = np.zeros((8, self.size, self.size), dtype=np.float32)
        planes[0] = self.board == current
        planes[1] = self.board == opponent
        planes[2] = self.board == EMPTY
        self._encode_liberties(planes[3], planes[4], current)
        self._encode_liberties(planes[5], planes[6], opponent)
        if current == BLACK:
            planes[7].fill(1.0)
        return planes

    def _board_after_stone(self, action: int) -> np.ndarray:
        """Simulate a non-pass move, raising ``ValueError`` when illegal."""

        if self.board.reshape(-1)[action] != EMPTY:
            raise ValueError(f"action {action} selects an occupied intersection")

        board = np.array(self.board, copy=True)
        flat_board = board.reshape(-1)
        flat_board[action] = self.to_play

        examined: set[int] = set()
        for neighbour in _neighbours(action, self.size):
            if flat_board[neighbour] != -self.to_play or neighbour in examined:
                continue
            group, liberties = _group_and_liberties(board, neighbour, self.size)
            examined.update(group)
            if not liberties:
                flat_board[list(group)] = EMPTY

        _, own_liberties = _group_and_liberties(board, action, self.size)
        if not own_liberties:
            raise ValueError(f"action {action} is suicide")

        if board.tobytes(order="C") in self.history:
            raise ValueError(f"action {action} violates positional superko")
        return board

    def _empty_region(self, start: int) -> tuple[set[int], set[int]]:
        region: set[int] = set()
        bordering_colours: set[int] = set()
        stack = [start]
        flat_board = self.board.reshape(-1)
        while stack:
            point = stack.pop()
            if point in region:
                continue
            region.add(point)
            for neighbour in _neighbours(point, self.size):
                colour = int(flat_board[neighbour])
                if colour == EMPTY and neighbour not in region:
                    stack.append(neighbour)
                elif colour != EMPTY:
                    bordering_colours.add(colour)
        return region, bordering_colours

    def _encode_liberties(
        self,
        one_liberty_plane: np.ndarray,
        two_liberty_plane: np.ndarray,
        colour: int,
    ) -> None:
        visited: set[int] = set()
        flat_board = self.board.reshape(-1)
        flat_one = one_liberty_plane.reshape(-1)
        flat_two = two_liberty_plane.reshape(-1)
        for start in np.flatnonzero(flat_board == colour):
            start = int(start)
            if start in visited:
                continue
            group, liberties = _group_and_liberties(self.board, start, self.size)
            visited.update(group)
            if len(liberties) == 1:
                flat_one[list(group)] = 1.0
            elif len(liberties) == 2:
                flat_two[list(group)] = 1.0


def _validated_integer(value: object, name: str, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _validated_player(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be BLACK (1) or WHITE (-1)")
    result = int(value)
    if result not in _PLAYERS:
        raise ValueError(f"{name} must be BLACK (1) or WHITE (-1)")
    return result


def _neighbours(point: int, size: int) -> tuple[int, ...]:
    row, column = divmod(point, size)
    neighbours: list[int] = []
    if row > 0:
        neighbours.append(point - size)
    if row + 1 < size:
        neighbours.append(point + size)
    if column > 0:
        neighbours.append(point - 1)
    if column + 1 < size:
        neighbours.append(point + 1)
    return tuple(neighbours)


def _group_and_liberties(
    board: np.ndarray, start: int, size: int
) -> tuple[set[int], set[int]]:
    flat_board = board.reshape(-1)
    colour = int(flat_board[start])
    if colour == EMPTY:
        raise ValueError("cannot collect a group from an empty point")

    group: set[int] = set()
    liberties: set[int] = set()
    stack = [start]
    while stack:
        point = stack.pop()
        if point in group:
            continue
        group.add(point)
        for neighbour in _neighbours(point, size):
            neighbour_colour = int(flat_board[neighbour])
            if neighbour_colour == EMPTY:
                liberties.add(neighbour)
            elif neighbour_colour == colour and neighbour not in group:
                stack.append(neighbour)
    return group, liberties


__all__ = ["BLACK", "WHITE", "EMPTY", "GoPosition"]
