from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from alphago_gym.env import GoEnv
from alphago_gym.go import BLACK, WHITE


def test_environment_declares_gymnasium_spaces_and_metadata() -> None:
    env = GoEnv(size=5, komi=5.5)

    assert isinstance(env, gym.Env)
    assert env.action_space == gym.spaces.Discrete(26)
    assert env.observation_space == gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(8, 5, 5),
        dtype=np.float32,
    )
    assert "render_modes" in env.metadata


def test_reset_returns_valid_observation_and_an_independent_legal_mask() -> None:
    env = GoEnv(size=3, komi=0.5)
    observation, info = env.reset(seed=7)

    assert env.observation_space.contains(observation)
    assert observation.dtype == np.float32
    assert observation.shape == (8, 3, 3)
    assert env.position.to_play == BLACK
    assert info["to_play"] == BLACK
    assert info["legal_actions_mask"].dtype == np.bool_
    assert info["legal_actions_mask"].shape == (10,)
    assert info["legal_actions_mask"].all()
    info["legal_actions_mask"][:] = False
    assert env.position.legal_actions_mask().all()


def test_step_observation_switches_to_the_next_players_perspective() -> None:
    env = GoEnv(size=3, komi=0.5)
    env.reset()

    observation, reward, terminated, truncated, info = env.step(0)

    assert reward == 0.0
    assert not terminated
    assert not truncated
    assert env.position.to_play == WHITE
    assert info["to_play"] == WHITE
    # Black's new stone is an opponent stone from White's perspective.
    assert observation[0, 0, 0] == 0.0
    assert observation[1, 0, 0] == 1.0
    assert not info["legal_actions_mask"][0]
    assert info["legal_actions_mask"][env.position.pass_action]


def test_intermediate_rewards_are_zero_and_terminal_reward_is_for_actor() -> None:
    env = GoEnv(size=2, komi=0.5)
    env.reset()

    _, first_reward, first_terminal, first_truncated, _ = env.step(0)
    _, second_reward, second_terminal, second_truncated, _ = env.step(4)
    _, reward, terminated, truncated, info = env.step(4)

    assert first_reward == 0.0
    assert second_reward == 0.0
    assert not first_terminal and not first_truncated
    assert not second_terminal and not second_truncated
    assert terminated
    assert not truncated
    # Black just made the second pass and wins by Chinese area scoring.
    assert reward == 1.0
    assert info["winner"] == BLACK
    assert not info["legal_actions_mask"].any()


def test_terminal_reward_can_be_negative_for_the_player_who_acted() -> None:
    env = GoEnv(size=2, komi=4.5)
    env.reset()
    env.step(0)
    env.step(4)

    _, reward, terminated, truncated, info = env.step(4)

    assert terminated and not truncated
    assert reward == -1.0
    assert info["winner"] == WHITE


def test_two_immediate_passes_end_game_and_komi_rewards_white_actor() -> None:
    env = GoEnv(size=3, komi=0.5)
    env.reset()

    _, reward1, terminated1, _, _ = env.step(9)
    _, reward2, terminated2, truncated2, info = env.step(9)

    assert reward1 == 0.0
    assert not terminated1
    assert reward2 == 1.0
    assert terminated2 and not truncated2
    assert info["winner"] == WHITE


def test_illegal_step_raises_without_changing_environment_state() -> None:
    env = GoEnv(size=3)
    env.reset()
    env.step(0)
    before = env.position

    with pytest.raises(ValueError, match="occupied"):
        env.step(0)

    assert env.position is before
    with pytest.raises(ValueError, match="action"):
        env.step(10)


def test_reset_after_terminal_starts_a_fresh_game() -> None:
    env = GoEnv(size=2)
    env.reset()
    env.step(4)
    env.step(4)

    observation, info = env.reset()

    assert not env.position.is_terminal
    assert env.position.move_count == 0
    assert env.position.to_play == BLACK
    assert env.observation_space.contains(observation)
    assert info["legal_actions_mask"].all()


def test_reset_seed_reproducibly_seeds_both_rng_and_action_space() -> None:
    first = GoEnv(size=3)
    second = GoEnv(size=3)
    first.reset(seed=1234)
    second.reset(seed=1234)

    assert first.np_random.integers(0, 1_000_000) == second.np_random.integers(
        0, 1_000_000
    )
    assert [first.action_space.sample() for _ in range(8)] == [
        second.action_space.sample() for _ in range(8)
    ]


def test_environment_passes_official_gymnasium_checker() -> None:
    check_env(GoEnv(size=3, komi=0.5), skip_render_check=True)


def test_ansi_render_contains_board_and_player_information() -> None:
    env = GoEnv(size=2, render_mode="ansi")
    env.reset()
    env.step(0)

    rendered = env.render()

    assert isinstance(rendered, str)
    assert "X" in rendered
    assert "to play: white" in rendered.lower()


def test_invalid_render_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="render_mode"):
        GoEnv(render_mode="rgb_array")
