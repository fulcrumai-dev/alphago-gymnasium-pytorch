"""Small, reproducible expert-like Go data for the educational notebook.

AlphaGo's supervised policy was trained on millions of positions from the KGS
Go Server.  Shipping and preprocessing that corpus is unsuitable for a small
Colab tutorial, so this module provides an explicit synthetic replacement: a
capture-, liberty-, and centre-aware policy plays short legal games that can be
used to exercise the same supervised-learning stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import numpy as np

from .go import BLACK, EMPTY, WHITE, GoPosition


@dataclass(frozen=True, slots=True, eq=False)
class ExpertStep:
    """One supervised policy example captured before its expert action."""

    observation: np.ndarray
    legal_mask: np.ndarray
    action: int
    player: int

    def __post_init__(self) -> None:
        observation = _immutable_array(self.observation, np.float32)
        legal_mask = _immutable_array(self.legal_mask, np.bool_)
        if observation.ndim != 3 or observation.shape[1] != observation.shape[2]:
            raise ValueError(
                "observation must have shape (planes, board_size, board_size)"
            )
        expected_actions = observation.shape[1] ** 2 + 1
        if legal_mask.shape != (expected_actions,):
            raise ValueError(
                f"legal_mask must have shape {(expected_actions,)}, "
                f"got {legal_mask.shape}"
            )
        action = _integer(self.action, "action", minimum=0)
        if action >= expected_actions:
            raise ValueError(f"action must be smaller than {expected_actions}")
        if not legal_mask[action]:
            raise ValueError("action must select a legal move")
        player = _player(self.player)

        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "legal_mask", legal_mask)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "player", player)


@dataclass(frozen=True, slots=True, eq=False)
class ExpertGame:
    """A complete synthetic game and its terminal outcomes."""

    steps: tuple[ExpertStep, ...] | Iterable[ExpertStep]
    winner: int
    black_outcome: int
    white_outcome: int
    final_position: GoPosition

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        if not all(isinstance(step, ExpertStep) for step in steps):
            raise TypeError("steps must contain ExpertStep records")
        if not isinstance(self.final_position, GoPosition):
            raise TypeError("final_position must be a GoPosition")
        if not self.final_position.is_terminal:
            raise ValueError("final_position must be terminal")

        winner = _winner(self.winner)
        black_outcome = _outcome(self.black_outcome, "black_outcome")
        white_outcome = _outcome(self.white_outcome, "white_outcome")
        if black_outcome != -white_outcome:
            raise ValueError("black and white outcomes must be opposites")
        expected_winner = (
            BLACK if black_outcome > 0 else WHITE if black_outcome < 0 else EMPTY
        )
        if winner != expected_winner:
            raise ValueError("winner must agree with terminal outcomes")
        if black_outcome != self.final_position.outcome(BLACK):
            raise ValueError("outcomes must agree with final_position")

        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "winner", winner)
        object.__setattr__(self, "black_outcome", black_outcome)
        object.__setattr__(self, "white_outcome", white_outcome)

    @property
    def outcomes(self) -> dict[int, int]:
        """Return a fresh player-to-result mapping."""

        return {BLACK: self.black_outcome, WHITE: self.white_outcome}


@dataclass(frozen=True, slots=True, eq=False)
class ExpertDataset:
    """Independent per-game records and a training-friendly flat step view."""

    games: tuple[ExpertGame, ...] | Iterable[ExpertGame]
    steps: tuple[ExpertStep, ...] | Iterable[ExpertStep]

    def __post_init__(self) -> None:
        games = tuple(self.games)
        steps = tuple(self.steps)
        if not all(isinstance(game, ExpertGame) for game in games):
            raise TypeError("games must contain ExpertGame records")
        if not all(isinstance(step, ExpertStep) for step in steps):
            raise TypeError("steps must contain ExpertStep records")
        object.__setattr__(self, "games", games)
        object.__setattr__(self, "steps", steps)

    @property
    def expert_steps(self) -> tuple[ExpertStep, ...]:
        """Descriptive alias for the flattened ``steps`` tuple."""

        return self.steps

    @property
    def game_records(self) -> tuple[ExpertGame, ...]:
        """Descriptive alias for the per-game ``games`` tuple."""

        return self.games

    def __len__(self) -> int:
        return len(self.steps)


def uniform_random_policy(position: GoPosition) -> np.ndarray:
    """Return a uniform probability distribution over legal actions."""

    legal = _legal_mask(position)
    count = int(legal.sum())
    if count == 0:
        raise ValueError("cannot build a policy for a terminal position")
    return legal.astype(np.float64) / count


def heuristic_policy(
    position: GoPosition,
    *,
    temperature: float = 1.0,
) -> np.ndarray:
    """Return a deterministic distribution from compact Go heuristics.

    Legal moves receive logits for captures, saving friendly groups in atari,
    liberties gained by the newly played group, and proximity to the centre.
    Self-atari is discouraged.  Pass remains legal but receives negligible mass
    while a board move exists.  ``temperature`` controls how probabilistically
    games sample these deterministic preferences.
    """

    if not isinstance(temperature, Real) or isinstance(
        temperature, (bool, np.bool_)
    ):
        raise TypeError("temperature must be a finite positive real number")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be a finite positive real number")

    legal = _legal_mask(position)
    if not legal.any():
        raise ValueError("cannot build a policy for a terminal position")

    board_actions = np.flatnonzero(legal[: position.pass_action])
    if len(board_actions) == 0:
        probabilities = np.zeros(position.action_size, dtype=np.float64)
        probabilities[position.pass_action] = 1.0
        return probabilities

    scores = np.full(position.action_size, -np.inf, dtype=np.float64)
    escape_bonuses = _atari_escape_bonuses(position, legal)
    opponent_before = int(np.count_nonzero(position.board == -position.to_play))
    centre = (position.size - 1) / 2.0
    maximum_distance = np.sqrt(2.0) * centre

    for action_value in board_actions:
        action = int(action_value)
        child = position.play(action)
        opponent_after = int(np.count_nonzero(child.board == -position.to_play))
        captured = opponent_before - opponent_after
        _, liberties = _group_and_liberties(child.board, action, position.size)

        row, column = divmod(action, position.size)
        distance = float(np.hypot(row - centre, column - centre))
        centrality = (
            1.0 if maximum_distance == 0.0 else 1.0 - distance / maximum_distance
        )
        score = 14.0 * captured
        score += escape_bonuses.get(action, 0.0)
        score += 0.35 * min(len(liberties), 4)
        score += 1.5 * centrality
        if len(liberties) == 1 and captured == 0:
            score -= 2.5
        scores[action] = score

    # Tying pass to the weakest legal board move makes the ratio predictable
    # across board sizes while retaining a non-zero chance before the cap.
    scores[position.pass_action] = float(scores[board_actions].min()) - 12.0
    return _softmax_legal(scores, legal, temperature)


# A notebook-friendly descriptive name without maintaining two implementations.
capture_and_liberty_policy = heuristic_policy


def generate_expert_games(
    num_games: int,
    size: int = 5,
    komi: float = 5.5,
    seed: int | None = 0,
    max_moves: int = 200,
) -> ExpertDataset:
    """Generate reproducible, bounded expert-like games.

    ``max_moves`` includes the terminating passes.  Once only two action slots
    remain, the generator forces pass actions.  Every returned game is thus
    terminal and contains at most ``max_moves`` records, even when the heuristic
    would otherwise keep playing indefinitely.
    """

    num_games = _integer(num_games, "num_games", minimum=0)
    max_moves = _integer(max_moves, "max_moves", minimum=2)
    # Validate size and komi even for the useful zero-game case.
    prototype = GoPosition(size=size, komi=komi)
    if seed is not None:
        seed = _integer(seed, "seed", minimum=0)
    rng = np.random.default_rng(seed)

    games: list[ExpertGame] = []
    flat_steps: list[ExpertStep] = []
    for _ in range(num_games):
        position = GoPosition(size=prototype.size, komi=prototype.komi)
        game_steps: list[ExpertStep] = []

        while not position.is_terminal:
            legal_mask = position.legal_actions_mask()
            remaining = max_moves - len(game_steps)
            if remaining <= 2:
                action = position.pass_action
            else:
                probabilities = heuristic_policy(position)
                action = int(rng.choice(position.action_size, p=probabilities))

            step = ExpertStep(
                observation=position.encode(),
                legal_mask=legal_mask,
                action=action,
                player=position.to_play,
            )
            game_steps.append(step)
            position = position.play(action)

        black_outcome = position.outcome(BLACK)
        white_outcome = position.outcome(WHITE)
        winner = (
            BLACK if black_outcome > 0 else WHITE if black_outcome < 0 else EMPTY
        )
        game = ExpertGame(
            steps=game_steps,
            winner=winner,
            black_outcome=black_outcome,
            white_outcome=white_outcome,
            final_position=position,
        )
        games.append(game)
        # Flat records own separate immutable buffers so callers cannot create
        # aliases between the two dataset views.
        flat_steps.extend(_copy_step(step) for step in game.steps)

    return ExpertDataset(games=games, steps=flat_steps)


def _legal_mask(position: GoPosition) -> np.ndarray:
    if not isinstance(position, GoPosition):
        raise TypeError("position must be a GoPosition")
    legal = np.asarray(position.legal_actions_mask(), dtype=np.bool_)
    if legal.shape != (position.action_size,):
        raise ValueError("position returned an invalid legal action mask")
    return legal


def _softmax_legal(
    scores: np.ndarray, legal: np.ndarray, temperature: float
) -> np.ndarray:
    probabilities = np.zeros_like(scores, dtype=np.float64)
    legal_scores = scores[legal]
    shifted_scores = legal_scores - legal_scores.max()
    with np.errstate(over="ignore"):
        scaled_scores = shifted_scores / temperature
    # Clipping retains strictly positive mass for every legal action without
    # allowing an extreme tactical score or temperature to underflow to zero.
    weights = np.exp(np.clip(scaled_scores, -700.0, 0.0))
    probabilities[legal] = weights / weights.sum()
    return probabilities


def _atari_escape_bonuses(
    position: GoPosition, legal: np.ndarray
) -> dict[int, float]:
    bonuses: dict[int, float] = {}
    visited: set[int] = set()
    flat_board = position.board.reshape(-1)
    for start_value in np.flatnonzero(flat_board == position.to_play):
        start = int(start_value)
        if start in visited:
            continue
        group, liberties = _group_and_liberties(position.board, start, position.size)
        visited.update(group)
        if len(liberties) == 1:
            liberty = next(iter(liberties))
            if legal[liberty]:
                bonuses[liberty] = bonuses.get(liberty, 0.0) + 10.0 + 2.0 * len(
                    group
                )
    return bonuses


def _group_and_liberties(
    board: np.ndarray, start: int, size: int
) -> tuple[set[int], set[int]]:
    flat_board = board.reshape(-1)
    colour = int(flat_board[start])
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


def _neighbours(point: int, size: int) -> tuple[int, ...]:
    row, column = divmod(point, size)
    result: list[int] = []
    if row > 0:
        result.append(point - size)
    if row + 1 < size:
        result.append(point + size)
    if column > 0:
        result.append(point - 1)
    if column + 1 < size:
        result.append(point + 1)
    return tuple(result)


def _copy_step(step: ExpertStep) -> ExpertStep:
    return ExpertStep(
        observation=step.observation,
        legal_mask=step.legal_mask,
        action=step.action,
        player=step.player,
    )


def _immutable_array(value: np.ndarray, dtype: np.dtype[object]) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    immutable_bytes = array.tobytes(order="C")
    return np.frombuffer(immutable_bytes, dtype=array.dtype).reshape(array.shape)


def _integer(value: object, name: str, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _player(value: object) -> int:
    result = _integer(value, "player", minimum=WHITE)
    if result not in (BLACK, WHITE):
        raise ValueError("player must be BLACK (1) or WHITE (-1)")
    return result


def _winner(value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError("winner must be BLACK (1), WHITE (-1), or EMPTY (0)")
    result = int(value)
    if result not in (BLACK, WHITE, EMPTY):
        raise ValueError("winner must be BLACK (1), WHITE (-1), or EMPTY (0)")
    return result


def _outcome(value: object, name: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be -1, 0, or 1")
    result = int(value)
    if result not in (-1, 0, 1):
        raise ValueError(f"{name} must be -1, 0, or 1")
    return result


__all__ = [
    "ExpertStep",
    "ExpertGame",
    "ExpertDataset",
    "uniform_random_policy",
    "heuristic_policy",
    "capture_and_liberty_policy",
    "generate_expert_games",
]
