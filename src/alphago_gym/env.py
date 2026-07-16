"""Gymnasium adapter for :mod:`alphago_gym.go`."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .go import BLACK, EMPTY, WHITE, GoPosition


class GoEnv(gym.Env[np.ndarray, int]):
    """A deterministic two-player Go environment.

    Observations are always encoded for the next player to move.  Intermediate
    rewards are zero; after the second pass, reward is the result from the
    perspective of the player who made that final pass.
    """

    metadata = {"render_modes": ["ansi", "human"], "render_fps": 1}
    reward_range = (-1.0, 1.0)

    def __init__(
        self,
        size: int = 5,
        komi: float = 5.5,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (*self.metadata["render_modes"], None):
            raise ValueError(
                f"render_mode must be one of {self.metadata['render_modes']} or None"
            )

        # Let GoPosition perform the canonical validation once.
        initial_position = GoPosition(size=size, komi=komi)
        self.size = initial_position.size
        self.komi = initial_position.komi
        self.render_mode = render_mode
        self.position = initial_position
        self.action_space = spaces.Discrete(initial_position.action_size)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8, self.size, self.size),
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh empty game and optionally seed Gymnasium spaces."""

        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        # ``options`` is accepted for the Gymnasium API; there are currently no
        # alternate reset configurations.
        del options
        self.position = GoPosition(size=self.size, komi=self.komi)
        observation = self.position.encode()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the game by one legal action."""

        actor = self.position.to_play
        next_position = self.position.play(action)
        self.position = next_position
        terminated = next_position.is_terminal
        reward = float(next_position.outcome(actor)) if terminated else 0.0
        observation = next_position.encode()
        info = self._info()
        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def render(self) -> str | None:
        """Render a small text board, returning it for ``ansi`` mode."""

        symbols = {BLACK: "X", WHITE: "O", EMPTY: "."}
        rows = [
            " ".join(symbols[int(value)] for value in row)
            for row in self.position.board
        ]
        player_name = "black" if self.position.to_play == BLACK else "white"
        rendered = "\n".join((*rows, f"to play: {player_name}"))
        if self.render_mode == "human":
            print(rendered)
            return None
        return rendered

    def close(self) -> None:
        """Release environment resources (there are none for text rendering)."""

    def _info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "legal_actions_mask": self.position.legal_actions_mask(),
            "to_play": self.position.to_play,
        }
        if self.position.is_terminal:
            black_result = self.position.outcome(BLACK)
            info["winner"] = (
                BLACK if black_result > 0 else WHITE if black_result < 0 else EMPTY
            )
            info["score"] = self.position.area_score()
        return info


__all__ = ["GoEnv"]
