"""Synchronous policy/value Monte Carlo tree search in the style of AlphaGo.

The 2016 system used asynchronous workers and separate value/rollout backups.
This module keeps the latter—the scientifically important part—while making the
execution deterministic and small enough for a teaching notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


class Position(Protocol):
    """Structural protocol required by the search implementation."""

    to_play: int
    action_size: int
    is_terminal: bool

    def legal_actions_mask(self) -> np.ndarray: ...

    def play(self, action: int) -> "Position": ...

    def outcome(self, player: int) -> float: ...

    def encode(self) -> np.ndarray: ...


PolicyEvaluator = Callable[[Position], np.ndarray]
ValueEvaluator = Callable[[Position], float]
RolloutEvaluator = Callable[[Position, np.random.Generator], float]


@dataclass(frozen=True)
class MCTSConfig:
    """Configuration for AlphaGo-style APV-MCTS.

    ``mixing_lambda`` is lambda in the paper: zero uses only the
    learned value and one uses only the fast rollout outcome.
    """

    num_simulations: int = 100
    c_puct: float = 1.5
    mixing_lambda: float = 0.5

    def __post_init__(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.c_puct <= 0:
            raise ValueError("c_puct must be positive")
        if not 0.0 <= self.mixing_lambda <= 1.0:
            raise ValueError("mixing_lambda must be between 0 and 1")


@dataclass
class EdgeStats:
    """Paper-style statistics for one state/action edge."""

    prior: float
    value_sum: float = 0.0
    value_visits: int = 0
    rollout_sum: float = 0.0
    rollout_visits: int = 0
    child: "SearchNode | None" = None

    @property
    def visit_count(self) -> int:
        # Both estimates are backed up once per synchronous simulation.
        return self.rollout_visits

    def q_value(self, mixing_lambda: float) -> float:
        value_mean = self.value_sum / self.value_visits if self.value_visits else 0.0
        rollout_mean = (
            self.rollout_sum / self.rollout_visits if self.rollout_visits else 0.0
        )
        return (1.0 - mixing_lambda) * value_mean + mixing_lambda * rollout_mean


@dataclass
class SearchNode:
    position: Position
    edges: dict[int, EdgeStats] = field(default_factory=dict)
    expanded: bool = False


@dataclass(frozen=True)
class MCTSResult:
    """Root statistics returned after a search."""

    action: int
    visit_counts: np.ndarray
    search_policy: np.ndarray
    q_values: np.ndarray
    priors: np.ndarray
    root: SearchNode = field(repr=False, compare=False)


class AlphaGoMCTS:
    """Synchronous version of AlphaGo's policy/value-guided MCTS.

    The evaluators are ordinary callables, which makes the algorithm testable
    without PyTorch and lets the notebook swap in neural adapters later.
    """

    def __init__(
        self,
        policy: PolicyEvaluator,
        value: ValueEvaluator,
        rollout: RolloutEvaluator,
        config: MCTSConfig | None = None,
        seed: int | None = None,
    ) -> None:
        self.policy = policy
        self.value = value
        self.rollout = rollout
        self.config = config or MCTSConfig()
        self.rng = np.random.default_rng(seed)

    def search(self, position: Position) -> MCTSResult:
        """Run simulations from ``position`` and return immutable root arrays."""

        if position.is_terminal:
            raise ValueError("cannot search from a terminal position")

        root = SearchNode(position=position)
        self._expand(root)
        if not root.edges:
            raise ValueError("non-terminal position has no legal actions")

        for _ in range(self.config.num_simulations):
            self._simulate(root)

        visits = np.zeros(position.action_size, dtype=np.int64)
        q_values = np.zeros(position.action_size, dtype=np.float64)
        priors = np.zeros(position.action_size, dtype=np.float64)
        for action, edge in root.edges.items():
            visits[action] = edge.visit_count
            q_values[action] = edge.q_value(self.config.mixing_lambda)
            priors[action] = edge.prior

        total = int(visits.sum())
        if total <= 0:  # Defensive; positive simulations should make this unreachable.
            search_policy = priors.copy()
        else:
            search_policy = visits.astype(np.float64) / total
        action = self._choose_root_action(visits, priors)
        return MCTSResult(
            action=action,
            visit_counts=visits,
            search_policy=search_policy,
            q_values=q_values,
            priors=priors,
            root=root,
        )

    def _expand(self, node: SearchNode) -> None:
        if node.expanded or node.position.is_terminal:
            return
        legal = np.asarray(node.position.legal_actions_mask(), dtype=np.bool_)
        if legal.shape != (node.position.action_size,):
            raise ValueError("legal action mask has the wrong shape")
        if not legal.any():
            raise ValueError("non-terminal position has no legal actions")

        raw_priors = np.asarray(self.policy(node.position), dtype=np.float64)
        if raw_priors.shape != legal.shape:
            raise ValueError("policy evaluator returned the wrong number of actions")
        raw_priors = np.where(np.isfinite(raw_priors) & (raw_priors > 0), raw_priors, 0)
        raw_priors[~legal] = 0.0
        prior_sum = float(raw_priors.sum())
        priors = raw_priors / prior_sum if prior_sum > 0 else legal / legal.sum()
        node.edges = {
            int(action): EdgeStats(prior=float(priors[action]))
            for action in np.flatnonzero(legal)
        }
        node.expanded = True

    def _simulate(self, root: SearchNode) -> None:
        node = root
        # Each tuple stores the player whose action the edge represents.
        path: list[tuple[SearchNode, int, int]] = []

        while not node.position.is_terminal and node.expanded:
            action = self._select_edge(node)
            edge = node.edges[action]
            actor = int(node.position.to_play)
            path.append((node, action, actor))
            if edge.child is None:
                edge.child = SearchNode(position=node.position.play(action))
            node = edge.child

        leaf_player = int(node.position.to_play)
        if node.position.is_terminal:
            learned_value = float(node.position.outcome(leaf_player))
            rollout_value = learned_value
        else:
            self._expand(node)
            learned_value = float(self.value(node.position))
            rollout_value = float(self.rollout(node.position, self.rng))
            if not np.isfinite(learned_value) or not -1.000001 <= learned_value <= 1.000001:
                raise ValueError("value evaluator must return a finite value in [-1, 1]")
            if not np.isfinite(rollout_value) or not -1.000001 <= rollout_value <= 1.000001:
                raise ValueError("rollout evaluator must return a finite value in [-1, 1]")

        for parent, action, actor in reversed(path):
            sign = 1.0 if actor == leaf_player else -1.0
            edge = parent.edges[action]
            edge.value_visits += 1
            edge.value_sum += sign * learned_value
            edge.rollout_visits += 1
            edge.rollout_sum += sign * rollout_value

    def _select_edge(self, node: SearchNode) -> int:
        total_visits = sum(edge.visit_count for edge in node.edges.values())
        sqrt_total = float(np.sqrt(total_visits))
        actions = np.fromiter(node.edges, dtype=np.int64)
        scores = np.empty(len(actions), dtype=np.float64)
        for index, action in enumerate(actions):
            edge = node.edges[int(action)]
            exploration = (
                self.config.c_puct
                * edge.prior
                * sqrt_total
                / (1 + edge.visit_count)
            )
            scores[index] = edge.q_value(self.config.mixing_lambda) + exploration
        best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=1e-12, atol=1e-12))
        return int(actions[int(self.rng.choice(best))])

    def _choose_root_action(self, visits: np.ndarray, priors: np.ndarray) -> int:
        maximum = int(visits.max())
        candidates = np.flatnonzero(visits == maximum)
        if len(candidates) > 1:
            candidate_priors = priors[candidates]
            best_prior = float(candidate_priors.max())
            candidates = candidates[np.isclose(candidate_priors, best_prior)]
        return int(self.rng.choice(candidates))


class PolicyRolloutEvaluator:
    """Play a fast policy to termination and score from the leaf player."""

    def __init__(self, policy: PolicyEvaluator, max_moves: int = 1_000) -> None:
        if max_moves <= 0:
            raise ValueError("max_moves must be positive")
        self.policy = policy
        self.max_moves = max_moves

    def __call__(self, position: Position, rng: np.random.Generator) -> float:
        start_player = int(position.to_play)
        current = position
        for _ in range(self.max_moves):
            if current.is_terminal:
                return float(current.outcome(start_player))
            legal = np.asarray(current.legal_actions_mask(), dtype=np.bool_)
            probabilities = np.asarray(self.policy(current), dtype=np.float64)
            if probabilities.shape != legal.shape:
                raise ValueError("rollout policy returned the wrong number of actions")
            probabilities = np.where(
                np.isfinite(probabilities) & (probabilities > 0) & legal,
                probabilities,
                0.0,
            )
            total = float(probabilities.sum())
            probabilities = probabilities / total if total > 0 else legal / legal.sum()
            current = current.play(int(rng.choice(len(probabilities), p=probabilities)))

        # Go rollouts can always be ended by two passes. This also makes a badly
        # initialized policy unable to hang the notebook indefinitely.
        pass_action = getattr(current, "pass_action", None)
        if pass_action is not None:
            for _ in range(2):
                if current.is_terminal:
                    break
                current = current.play(int(pass_action))
        if not current.is_terminal:
            raise RuntimeError("rollout did not reach a terminal state")
        return float(current.outcome(start_player))


class NeuralPolicyEvaluator:
    """Adapt a PyTorch policy network to the NumPy search interface."""

    def __init__(self, model: Any, device: Any = "cpu", temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.model = model
        self.device = device
        self.temperature = temperature

    def __call__(self, position: Position) -> np.ndarray:
        import torch

        from .models import masked_softmax

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            features = torch.as_tensor(position.encode(), dtype=torch.float32)
            features = features.unsqueeze(0).to(self.device)
            logits = self.model(features) / self.temperature
            mask = torch.as_tensor(position.legal_actions_mask(), dtype=torch.bool)
            probabilities = masked_softmax(logits, mask.to(self.device)).squeeze(0)
        self.model.train(was_training)
        return probabilities.detach().cpu().numpy().astype(np.float64)


class NeuralValueEvaluator:
    """Adapt a PyTorch value network to the scalar search interface."""

    def __init__(self, model: Any, device: Any = "cpu") -> None:
        self.model = model
        self.device = device

    def __call__(self, position: Position) -> float:
        import torch

        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            features = torch.as_tensor(position.encode(), dtype=torch.float32)
            value = self.model(features.unsqueeze(0).to(self.device)).reshape(-1)[0]
        self.model.train(was_training)
        return float(value.detach().cpu())
