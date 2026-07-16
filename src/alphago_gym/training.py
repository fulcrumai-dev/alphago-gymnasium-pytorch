"""Training and data-generation utilities for the scaled AlphaGo pipeline.

The functions in this module deliberately depend on a small structural Go
position protocol rather than a concrete environment class. This keeps the
supervised-learning, policy-gradient, opponent-pool, and value-data stages easy
to test and reusable with immutable Go positions, Gymnasium wrappers, or tiny
teaching games.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import mask_logits, masked_softmax


class TrainingPosition(Protocol):
    """Structural position interface used by self-play data generation."""

    to_play: int
    action_size: int
    pass_action: int
    is_terminal: bool

    def legal_actions_mask(self) -> np.ndarray: ...

    def play(self, action: int) -> "TrainingPosition": ...

    def outcome(self, player: int) -> float: ...

    def encode(self) -> np.ndarray: ...


PolicyCallable: TypeAlias = Callable[[TrainingPosition], np.ndarray]
Policy: TypeAlias = nn.Module | PolicyCallable
Device: TypeAlias = str | torch.device


@dataclass(frozen=True, slots=True)
class PolicyExample:
    """One supervised state/expert-action pair."""

    observation: np.ndarray
    action: int


@dataclass(frozen=True, slots=True)
class ValueExample:
    """One state and final outcome from that state's side-to-move perspective."""

    observation: np.ndarray
    outcome: float


@dataclass(frozen=True, slots=True)
class PolicyGradientStep:
    """One learner move retained for a REINFORCE update."""

    observation: np.ndarray
    action: int
    legal_mask: np.ndarray
    player: int


@dataclass(frozen=True, slots=True)
class PolicyGradientEpisode:
    """Learner steps and terminal result from ``learner_player``'s perspective."""

    steps: tuple[PolicyGradientStep, ...]
    outcome: float
    learner_player: int


def _validate_observation(observation: np.ndarray) -> tuple[np.ndarray, int]:
    array = np.asarray(observation)
    if array.ndim != 3 or array.shape[-2] != array.shape[-1]:
        raise ValueError(
            "observation must have shape (channels, board_size, board_size)"
        )
    board_size = int(array.shape[-1])
    if board_size <= 0:
        raise ValueError("board_size must be positive")
    return array, board_size


def _dihedral_arrays(array: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return the four rotations of an array and of one reflection."""

    transformed: list[np.ndarray] = []
    for reflected in (False, True):
        base = np.flip(array, axis=-1) if reflected else array
        for rotations in range(4):
            transformed.append(
                np.ascontiguousarray(np.rot90(base, rotations, axes=(-2, -1)))
            )
    return tuple(transformed)


def dihedral_policy_augmentations(
    example: PolicyExample,
) -> tuple[PolicyExample, ...]:
    """Generate all eight D4 symmetries of a policy example.

    Board actions follow the same rotations/reflections as the encoded feature
    planes. The extra action at ``board_size**2`` is pass and remains pass under
    every symmetry.
    """

    observation, board_size = _validate_observation(example.observation)
    pass_action = board_size**2
    action = int(example.action)
    if action < 0 or action > pass_action:
        raise ValueError(
            f"action must be in [0, {pass_action}] for a {board_size}x{board_size} board"
        )

    observations = _dihedral_arrays(observation)
    if action == pass_action:
        actions = (pass_action,) * 8
    else:
        action_plane = np.zeros((board_size, board_size), dtype=np.bool_)
        action_plane.flat[action] = True
        actions = tuple(
            int(np.flatnonzero(transformed)[0])
            for transformed in _dihedral_arrays(action_plane)
        )
    return tuple(
        PolicyExample(observation=transformed, action=transformed_action)
        for transformed, transformed_action in zip(observations, actions, strict=True)
    )


def dihedral_value_augmentations(
    example: ValueExample,
) -> tuple[ValueExample, ...]:
    """Generate all eight board symmetries while preserving the outcome."""

    observation, _ = _validate_observation(example.observation)
    return tuple(
        ValueExample(observation=transformed, outcome=float(example.outcome))
        for transformed in _dihedral_arrays(observation)
    )


def _epoch_indices(
    length: int,
    shuffle: bool,
    rng: np.random.Generator | None,
) -> np.ndarray:
    if length <= 0:
        raise ValueError("training examples must not be empty")
    indices = np.arange(length, dtype=np.int64)
    if shuffle:
        (rng if rng is not None else np.random.default_rng()).shuffle(indices)
    return indices


def _validate_batch_size(batch_size: int) -> None:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")


def _observation_batch(
    examples: Sequence[PolicyExample] | Sequence[ValueExample],
    indices: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    observations = np.stack(
        [np.asarray(examples[int(index)].observation) for index in indices]
    )
    return torch.as_tensor(observations, dtype=torch.float32, device=device)


def train_policy_epoch(
    model: nn.Module,
    examples: Sequence[PolicyExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 128,
    device: Device = "cpu",
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> float:
    """Train one supervised-policy epoch and return mean cross-entropy."""

    _validate_batch_size(batch_size)
    indices = _epoch_indices(len(examples), shuffle, rng)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.train()
    total_loss = 0.0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        observations = _observation_batch(examples, batch_indices, resolved_device)
        targets = torch.as_tensor(
            [int(examples[int(index)].action) for index in batch_indices],
            dtype=torch.long,
            device=resolved_device,
        )

        optimizer.zero_grad(set_to_none=True)
        logits = model(observations)
        if logits.ndim != 2 or logits.shape[0] != len(batch_indices):
            raise ValueError("policy model must return shape (batch, actions)")
        if bool(((targets < 0) | (targets >= logits.shape[-1])).any().item()):
            raise ValueError("a supervised action is outside the policy action space")
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(batch_indices)

    return total_loss / len(indices)


def train_value_epoch(
    model: nn.Module,
    examples: Sequence[ValueExample],
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 128,
    device: Device = "cpu",
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> float:
    """Train one value-network epoch and return mean squared error."""

    _validate_batch_size(batch_size)
    indices = _epoch_indices(len(examples), shuffle, rng)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.train()
    total_loss = 0.0

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        observations = _observation_batch(examples, batch_indices, resolved_device)
        outcomes = np.asarray(
            [float(examples[int(index)].outcome) for index in batch_indices],
            dtype=np.float32,
        )
        if not np.isfinite(outcomes).all() or np.any(np.abs(outcomes) > 1.0):
            raise ValueError("value outcomes must be finite and in [-1, 1]")
        targets = torch.as_tensor(outcomes, device=resolved_device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(observations).reshape(-1)
        if predictions.shape != targets.shape:
            raise ValueError("value model must return one scalar per observation")
        loss = F.mse_loss(predictions, targets)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().item()) * len(batch_indices)

    return total_loss / len(indices)


def _position_legal_mask(position: TrainingPosition) -> np.ndarray:
    legal = np.asarray(position.legal_actions_mask(), dtype=np.bool_)
    if legal.shape != (int(position.action_size),):
        raise ValueError("legal action mask has the wrong shape")
    if not legal.any():
        raise ValueError("non-terminal position has no legal actions")
    return legal


def _force_passes_to_termination(position: TrainingPosition) -> TrainingPosition:
    """End a Go game with at most two legal passes, outside policy sampling."""

    current = position
    for _ in range(2):
        if current.is_terminal:
            return current
        legal = _position_legal_mask(current)
        pass_action = int(current.pass_action)
        if pass_action < 0 or pass_action >= int(current.action_size):
            raise ValueError("pass_action is outside the action space")
        if not legal[pass_action]:
            raise ValueError("pass action must be legal when forcing game termination")
        current = current.play(pass_action)
    if not current.is_terminal:
        raise RuntimeError("two forced passes did not terminate the game")
    return current


def legal_policy_probabilities(
    policy: Policy,
    position: TrainingPosition,
    *,
    device: Device = "cpu",
    temperature: float = 1.0,
) -> np.ndarray:
    """Return a normalized legal distribution from a model or policy callable.

    Neural policies are interpreted as returning logits. Plain callables are
    interpreted as returning probabilities, matching the MCTS evaluator
    protocol. Invalid/negative callable mass is removed, with a uniform legal
    fallback when no usable mass remains.
    """

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    legal = _position_legal_mask(position)

    if isinstance(policy, nn.Module):
        resolved_device = torch.device(device)
        policy.to(resolved_device)
        was_training = policy.training
        policy.eval()
        try:
            with torch.no_grad():
                observation = torch.as_tensor(
                    np.asarray(position.encode()),
                    dtype=torch.float32,
                    device=resolved_device,
                ).unsqueeze(0)
                logits = policy(observation)
                if logits.shape != (1, int(position.action_size)):
                    raise ValueError("policy model returned the wrong number of actions")
                legal_tensor = torch.as_tensor(
                    legal, dtype=torch.bool, device=resolved_device
                )
                probabilities = masked_softmax(
                    logits / float(temperature), legal_tensor
                ).squeeze(0)
                result = probabilities.detach().cpu().numpy().astype(np.float64)
        finally:
            policy.train(was_training)
        # Re-normalize after float32 -> float64 conversion: NumPy's sampler has
        # a deliberately tight sum-to-one check for float64 probabilities.
        result = np.where(np.isfinite(result) & (result >= 0.0) & legal, result, 0.0)
        result_total = float(result.sum())
        if result_total <= 0.0:
            return legal.astype(np.float64) / float(legal.sum())
        result /= result_total
        return result

    raw_probabilities: Any = policy(position)
    if isinstance(raw_probabilities, torch.Tensor):
        raw_probabilities = raw_probabilities.detach().cpu().numpy()
    probabilities = np.asarray(raw_probabilities, dtype=np.float64)
    if probabilities.shape != legal.shape:
        raise ValueError("policy callable returned the wrong number of actions")
    probabilities = np.where(
        np.isfinite(probabilities) & (probabilities > 0.0) & legal,
        probabilities,
        0.0,
    )
    total = float(probabilities.sum())
    if total <= 0.0:
        return legal.astype(np.float64) / float(legal.sum())
    probabilities /= total
    if temperature != 1.0:
        probabilities = np.power(probabilities, 1.0 / float(temperature))
        powered_total = float(probabilities.sum())
        if not np.isfinite(powered_total) or powered_total <= 0.0:
            return legal.astype(np.float64) / float(legal.sum())
        probabilities /= powered_total
    return probabilities


def sample_legal_action(
    policy: Policy,
    position: TrainingPosition,
    rng: np.random.Generator,
    *,
    device: Device = "cpu",
    temperature: float = 1.0,
) -> int:
    """Sample one legal action using only the supplied NumPy generator."""

    probabilities = legal_policy_probabilities(
        policy, position, device=device, temperature=temperature
    )
    return int(rng.choice(len(probabilities), p=probabilities))


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


class OpponentPool:
    """FIFO pool of immutable CPU state-dict snapshots.

    ``sample`` always constructs a fresh frozen module, so changing either the
    source model after ``add`` or a previously sampled opponent cannot mutate a
    stored checkpoint.
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module] | None = None,
        max_size: int | None = None,
    ) -> None:
        if max_size is not None and (
            isinstance(max_size, bool)
            or not isinstance(max_size, int)
            or max_size <= 0
        ):
            raise ValueError("max_size must be a positive integer or None")
        self.model_factory = model_factory
        self.max_size = max_size
        self._prototype: nn.Module | None = None
        self._snapshots: list[dict[str, torch.Tensor]] = []

    def __len__(self) -> int:
        return len(self._snapshots)

    def add(self, model: nn.Module) -> None:
        """Store a detached CPU clone of ``model``'s current parameters/buffers."""

        if self.model_factory is None and self._prototype is None:
            self._prototype = copy.deepcopy(model).cpu()
        self._snapshots.append(_cpu_state_dict(model))
        if self.max_size is not None and len(self._snapshots) > self.max_size:
            del self._snapshots[: len(self._snapshots) - self.max_size]

    def sample(
        self,
        rng: np.random.Generator,
        *,
        device: Device = "cpu",
    ) -> nn.Module:
        """Instantiate a fresh frozen opponent from a uniformly sampled snapshot."""

        if not self._snapshots:
            raise ValueError("cannot sample from an empty opponent pool")
        index = int(rng.integers(len(self._snapshots)))
        if self.model_factory is not None:
            opponent = self.model_factory()
        else:
            if self._prototype is None:  # Defensive; add always initializes it.
                raise RuntimeError("opponent pool has no model prototype")
            opponent = copy.deepcopy(self._prototype)
        state = {
            name: tensor.clone() for name, tensor in self._snapshots[index].items()
        }
        opponent.load_state_dict(state)
        opponent.to(torch.device(device))
        opponent.eval()
        for parameter in opponent.parameters():
            parameter.requires_grad_(False)
        return opponent


def _flatten_policy_gradient_episodes(
    episodes: Sequence[PolicyGradientEpisode],
) -> tuple[list[PolicyGradientStep], np.ndarray]:
    if not episodes:
        raise ValueError("policy-gradient episodes must not be empty")
    steps: list[PolicyGradientStep] = []
    outcomes: list[float] = []
    for episode in episodes:
        outcome = float(episode.outcome)
        if not np.isfinite(outcome) or abs(outcome) > 1.0:
            raise ValueError("episode outcomes must be finite and in [-1, 1]")
        for step in episode.steps:
            if int(step.player) != int(episode.learner_player):
                raise ValueError("an episode may contain only learner policy steps")
            steps.append(step)
            outcomes.append(outcome)
    if not steps:
        raise ValueError("policy-gradient episodes contain no learner steps")
    return steps, np.asarray(outcomes, dtype=np.float32)


def train_reinforce_epoch(
    model: nn.Module,
    episodes: Sequence[PolicyGradientEpisode],
    optimizer: torch.optim.Optimizer,
    *,
    device: Device = "cpu",
    value_baseline: nn.Module | Callable[[torch.Tensor], torch.Tensor] | None = None,
    entropy_coefficient: float = 0.0,
    batch_size: int = 128,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
) -> float:
    """Run one legal-action REINFORCE epoch with baseline and entropy options."""

    if not np.isfinite(entropy_coefficient) or entropy_coefficient < 0:
        raise ValueError("entropy_coefficient must be finite and non-negative")
    _validate_batch_size(batch_size)
    steps, outcomes = _flatten_policy_gradient_episodes(episodes)
    indices = _epoch_indices(len(steps), shuffle, rng)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    model.train()

    baseline_module = (
        value_baseline if isinstance(value_baseline, nn.Module) else None
    )
    baseline_was_training: bool | None = None
    if baseline_module is not None:
        baseline_module.to(resolved_device)
        baseline_was_training = baseline_module.training
        baseline_module.eval()

    total_loss = 0.0
    try:
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            observations_np = np.stack(
                [np.asarray(steps[int(index)].observation) for index in batch_indices]
            )
            masks_np = np.stack(
                [
                    np.asarray(steps[int(index)].legal_mask, dtype=np.bool_)
                    for index in batch_indices
                ]
            )
            actions_np = np.asarray(
                [int(steps[int(index)].action) for index in batch_indices],
                dtype=np.int64,
            )
            batch_outcomes = torch.as_tensor(
                outcomes[batch_indices], dtype=torch.float32, device=resolved_device
            )
            observations = torch.as_tensor(
                observations_np, dtype=torch.float32, device=resolved_device
            )
            legal_masks = torch.as_tensor(
                masks_np, dtype=torch.bool, device=resolved_device
            )
            actions = torch.as_tensor(
                actions_np, dtype=torch.long, device=resolved_device
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(observations)
            if logits.ndim != 2 or logits.shape != legal_masks.shape:
                raise ValueError(
                    "policy logits and legal masks must share shape (batch, actions)"
                )
            if bool(((actions < 0) | (actions >= logits.shape[-1])).any().item()):
                raise ValueError("a policy-gradient action is outside the action space")
            chosen_is_legal = legal_masks.gather(1, actions.unsqueeze(1)).squeeze(1)
            if not bool(chosen_is_legal.all().item()):
                raise ValueError("a policy-gradient action is illegal in its state")

            # Work in log space so a legal move remains trainable even when its
            # float32 probability underflows to zero. Exponentiating the masked
            # log probabilities still gives exactly zero probability to illegal
            # moves. Replacing their -inf log values before the entropy product
            # avoids the undefined 0 * -inf operation.
            log_probabilities = F.log_softmax(
                mask_logits(logits, legal_masks), dim=-1
            )
            probabilities = log_probabilities.exp()
            chosen_log_probabilities = log_probabilities.gather(
                1, actions.unsqueeze(1)
            ).squeeze(1)
            finite_log_probabilities = log_probabilities.masked_fill(
                ~legal_masks, 0.0
            )
            entropy = -(probabilities * finite_log_probabilities).sum(dim=-1)

            if value_baseline is None:
                baseline = torch.zeros_like(batch_outcomes)
            else:
                with torch.no_grad():
                    raw_baseline = value_baseline(observations)
                    baseline = torch.as_tensor(
                        raw_baseline,
                        dtype=torch.float32,
                        device=resolved_device,
                    ).reshape(-1)
                if baseline.shape != batch_outcomes.shape:
                    raise ValueError("value baseline must return one scalar per step")

            advantages = batch_outcomes - baseline
            loss = -(advantages * chosen_log_probabilities).mean()
            loss = loss - float(entropy_coefficient) * entropy.mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * len(batch_indices)
    finally:
        if baseline_module is not None and baseline_was_training is not None:
            baseline_module.train(baseline_was_training)

    return total_loss / len(indices)


def generate_policy_gradient_episode(
    initial_position: TrainingPosition,
    learner_policy: Policy,
    opponent_policy: Policy,
    *,
    learner_player: int | None = None,
    rng: np.random.Generator,
    device: Device = "cpu",
    temperature: float = 1.0,
    max_moves: int = 1_000,
) -> PolicyGradientEpisode:
    """Play learner and frozen-opponent policies alternately to termination.

    Only learner actions are retained because the opponent is not updated. The
    final return is explicitly converted to the learner's player perspective.
    The last two move slots are reserved for legal forced passes, guaranteeing
    bounded Go episodes even when both sampled policies assign pass zero mass.
    Those cap passes are deliberately absent from the REINFORCE trajectory
    because they were not sampled from the learner policy.
    """

    if (
        isinstance(max_moves, bool)
        or not isinstance(max_moves, int)
        or max_moves < 2
    ):
        raise ValueError("max_moves must allow at least two forced pass actions")
    if initial_position.is_terminal:
        raise ValueError("cannot generate an episode from a terminal position")
    chosen_player = (
        int(initial_position.to_play)
        if learner_player is None
        else int(learner_player)
    )
    current = initial_position
    steps: list[PolicyGradientStep] = []

    sampled_move_budget = max_moves - 2
    for _ in range(sampled_move_budget):
        if current.is_terminal:
            break
        actor = int(current.to_play)
        acting_policy = learner_policy if actor == chosen_player else opponent_policy
        legal = _position_legal_mask(current)
        probabilities = legal_policy_probabilities(
            acting_policy,
            current,
            device=device,
            temperature=temperature,
        )
        action = int(rng.choice(len(probabilities), p=probabilities))
        if actor == chosen_player:
            steps.append(
                PolicyGradientStep(
                    observation=np.ascontiguousarray(
                        np.asarray(current.encode(), dtype=np.float32)
                    ),
                    action=action,
                    legal_mask=legal.copy(),
                    player=actor,
                )
            )
        current = current.play(action)

    if not current.is_terminal:
        current = _force_passes_to_termination(current)
    outcome = float(current.outcome(chosen_player))
    if not np.isfinite(outcome) or abs(outcome) > 1.0:
        raise ValueError("terminal outcome must be finite and in [-1, 1]")
    return PolicyGradientEpisode(
        steps=tuple(steps), outcome=outcome, learner_player=chosen_player
    )


def _opening_move_bounds(
    opening_moves: int | tuple[int, int],
) -> tuple[int, int, bool]:
    """Normalize a fixed/ranged SL prefix to inclusive bounds and range mode."""

    if isinstance(opening_moves, int) and not isinstance(opening_moves, bool):
        if opening_moves < 0:
            raise ValueError("opening_moves must be non-negative")
        return opening_moves, opening_moves, False

    if not isinstance(opening_moves, tuple) or len(opening_moves) != 2:
        raise ValueError(
            "opening_moves must be a non-negative integer or an inclusive "
            "(min_moves, max_moves) tuple"
        )
    minimum, maximum = opening_moves
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, int)
        or not isinstance(maximum, int)
        or minimum < 0
        or maximum < minimum
    ):
        raise ValueError(
            "opening_moves range must contain non-negative integers with min <= max"
        )
    return minimum, maximum, True


def generate_value_examples(
    initial_position_factory: Callable[[], TrainingPosition],
    supervised_policy: Policy,
    reinforcement_policy: Policy,
    *,
    num_games: int,
    opening_moves: int | tuple[int, int],
    rng: np.random.Generator,
    device: Device = "cpu",
    temperature: float = 1.0,
    max_moves: int = 1_000,
) -> list[ValueExample]:
    """Generate paper-style decorrelated self-play value examples.

    Each distinct game has three phases:

    1. play an SL prefix using the supervised policy. ``opening_moves`` may be
       a fixed non-negative integer or an inclusive ``(min_moves, max_moves)``
       range sampled independently for every game (the paper samples U and
       therefore uses an SL prefix of U - 1 moves);
    2. choose one uniformly random legal *board* move (never pass), and retain
       exactly the resulting post-move state; and
    3. complete the game with the reinforcement policy for both players,
       reserving the final two move slots for legal forced passes.

    The retained outcome is scored for the side to move in the retained state,
    matching :class:`~alphago_gym.models.ValueNetwork`'s perspective contract.
    Forced cap passes cannot alter the exactly-one-example-per-game contract.
    """

    if isinstance(num_games, bool) or not isinstance(num_games, int) or num_games <= 0:
        raise ValueError("num_games must be a positive integer")
    minimum_opening, maximum_opening, sample_opening = _opening_move_bounds(
        opening_moves
    )
    minimum_move_budget = maximum_opening + 3  # opening + random + two passes
    if (
        isinstance(max_moves, bool)
        or not isinstance(max_moves, int)
        or max_moves < minimum_move_budget
    ):
        raise ValueError(
            "max_moves must cover the opening, random move, and two forced pass "
            "actions"
        )

    examples: list[ValueExample] = []
    for _ in range(num_games):
        game_opening_moves = (
            int(rng.integers(minimum_opening, maximum_opening + 1))
            if sample_opening
            else minimum_opening
        )
        current = initial_position_factory()
        if current.is_terminal:
            raise ValueError("initial position factory returned a terminal position")
        moves_played = 0

        for _ in range(game_opening_moves):
            if current.is_terminal:
                raise RuntimeError("game ended during the supervised opening")
            if moves_played >= max_moves:
                raise RuntimeError("value game exceeded max_moves during its opening")
            action = sample_legal_action(
                supervised_policy,
                current,
                rng,
                device=device,
                temperature=temperature,
            )
            current = current.play(action)
            moves_played += 1

        if current.is_terminal:
            raise RuntimeError("no random move can be played from a terminal opening")
        legal = _position_legal_mask(current)
        pass_action = int(current.pass_action)
        if pass_action < 0 or pass_action >= int(current.action_size):
            raise ValueError("pass_action is outside the action space")
        legal_board_actions = np.flatnonzero(legal)
        legal_board_actions = legal_board_actions[legal_board_actions != pass_action]
        if len(legal_board_actions) == 0:
            raise ValueError("randomization point has no legal non-pass board move")
        random_action = int(rng.choice(legal_board_actions))
        current = current.play(random_action)
        moves_played += 1

        sample_player = int(current.to_play)
        sample_observation = np.ascontiguousarray(
            np.asarray(current.encode(), dtype=np.float32)
        )

        sampled_completion_limit = max_moves - 2
        while not current.is_terminal and moves_played < sampled_completion_limit:
            action = sample_legal_action(
                reinforcement_policy,
                current,
                rng,
                device=device,
                temperature=temperature,
            )
            current = current.play(action)
            moves_played += 1

        if not current.is_terminal:
            current = _force_passes_to_termination(current)

        outcome = float(current.outcome(sample_player))
        if not np.isfinite(outcome) or abs(outcome) > 1.0:
            raise ValueError("terminal outcome must be finite and in [-1, 1]")
        examples.append(ValueExample(sample_observation, outcome))

    return examples


__all__ = [
    "OpponentPool",
    "PolicyExample",
    "PolicyGradientEpisode",
    "PolicyGradientStep",
    "TrainingPosition",
    "ValueExample",
    "dihedral_policy_augmentations",
    "dihedral_value_augmentations",
    "generate_policy_gradient_episode",
    "generate_value_examples",
    "legal_policy_probabilities",
    "sample_legal_action",
    "train_policy_epoch",
    "train_reinforce_epoch",
    "train_value_epoch",
]
