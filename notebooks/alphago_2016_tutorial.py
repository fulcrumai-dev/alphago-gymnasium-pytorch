# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # AlphaGo 2016, reimplemented at teaching scale
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fulcrumai-dev/alphago-gymnasium-pytorch/blob/main/notebooks/alphago_2016_tutorial.ipynb)
# [![Paper](https://img.shields.io/badge/Nature-paper-0f766e)](https://doi.org/10.1038/nature16961)
#
# Welcome! In this notebook we will assemble the complete algorithmic pipeline
# from Silver et al., *Mastering the game of Go with deep neural networks and
# tree search* (Nature, 2016):
#
# 1. learn a **supervised policy** from expert-like positions;
# 2. learn a separate, cheap **rollout policy**;
# 3. improve a copy of the policy with **REINFORCE self-play** against a pool of
#    older opponents;
# 4. learn a **value network** from one decorrelated position per self-play game;
# 5. combine frozen supervised priors, value estimates, and fast rollouts in
#    AlphaGo-style Monte Carlo tree search (MCTS).
#
# > **Honest scope.** This is an algorithmic reimplementation, not a claim to
# > reproduce AlphaGo's professional playing strength. The paper used 19×19 Go,
# > 48/49 feature planes, tens of millions of positions, 50 GPUs, distributed
# > asynchronous search, and weeks of training. Our default `smoke` profile uses
# > a tiny board and minutes/seconds of compute so every stage can run end to end.
# > Every deviation is catalogued in `docs/PAPER_FIDELITY.md`.

# %% [markdown]
# ## 0 · Runtime setup
#
# The setup cell makes the same notebook portable across:
#
# - Google Colab with an NVIDIA CUDA runtime;
# - an Apple-silicon Mac through PyTorch MPS;
# - CPU as a safe fallback.
#
# In Colab, select **Runtime → Change runtime type → T4 GPU**. The repository's
# automated release gate also executes this notebook on a fresh managed T4 with
# the official `google-colab-cli`.

# %%
from __future__ import annotations

import copy
import hashlib
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPOSITORY = "https://github.com/fulcrumai-dev/alphago-gymnasium-pytorch.git"

if IN_COLAB:
    PROJECT_ROOT = Path("/content/alphago-gymnasium-pytorch")
    if not PROJECT_ROOT.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", REPOSITORY, str(PROJECT_ROOT)],
            check=True,
        )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "gymnasium>=1.0,<2"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(PROJECT_ROOT)],
        check=True,
    )
else:
    PROJECT_ROOT = Path.cwd().resolve()
    if PROJECT_ROOT.name == "notebooks":
        PROJECT_ROOT = PROJECT_ROOT.parent
    if not (PROJECT_ROOT / "pyproject.toml").exists():
        candidates = [parent for parent in Path.cwd().resolve().parents if (parent / "pyproject.toml").exists()]
        if not candidates:
            raise RuntimeError("Run this notebook from the cloned project directory.")
        PROJECT_ROOT = candidates[0]

os.chdir(PROJECT_ROOT)
print(f"Project: {PROJECT_ROOT}")
print(f"Hosted Colab: {IN_COLAB}")

# %%
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from alphago_gym.data import generate_expert_games, heuristic_policy
from alphago_gym.device import select_device
from alphago_gym.env import GoEnv
from alphago_gym.go import BLACK, WHITE, GoPosition
from alphago_gym.mcts import (
    AlphaGoMCTS,
    MCTSConfig,
    NeuralPolicyEvaluator,
    NeuralValueEvaluator,
    PolicyRolloutEvaluator,
)
from alphago_gym.models import PolicyNetwork, RolloutPolicy, ValueNetwork
from alphago_gym.training import (
    OpponentPool,
    PolicyExample,
    dihedral_policy_augmentations,
    dihedral_value_augmentations,
    generate_policy_gradient_episode,
    generate_value_examples,
    legal_policy_probabilities,
    train_policy_epoch,
    train_reinforce_epoch,
    train_value_epoch,
)
from alphago_gym.visualization import plot_board, plot_search_summary

SEED = 7
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)
device = select_device(os.getenv("ALPHAGO_DEVICE", "auto"))

print(f"Python {sys.version.split()[0]} · PyTorch {torch.__version__} · Gymnasium {gym.__version__}")
print(f"Selected device: {device}")
if device.type == "cuda":
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
elif device.type == "mps":
    print("Apple Metal Performance Shaders is active.")

# %% [markdown]
# ### Pick a profile
#
# `smoke` is the reproducible validation path and the notebook default.
# `tutorial` is still modest, but gives the networks and search more experience.
# You can scale any field independently after the first successful run.

# %%
PROFILES = {
    "smoke": dict(
        board_size=3,
        komi=0.5,
        expert_games=5,
        channels=16,
        depth=2,
        sl_epochs=2,
        rl_games=3,
        value_games=4,
        value_epochs=2,
        mcts_simulations=12,
    ),
    "tutorial": dict(
        board_size=5,
        komi=2.5,
        expert_games=30,
        channels=64,
        depth=6,
        sl_epochs=5,
        rl_games=16,
        value_games=32,
        value_epochs=5,
        mcts_simulations=100,
    ),
}

PROFILE = os.getenv("ALPHAGO_PROFILE", "smoke")
if PROFILE not in PROFILES:
    raise ValueError(f"ALPHAGO_PROFILE must be one of {tuple(PROFILES)}")
config = PROFILES[PROFILE]
EXPERT_MAX_MOVES = config["board_size"] ** 2 * 4 + 2
SELF_PLAY_MAX_MOVES = max(500, config["board_size"] ** 2 * 12 + 2)
ROLLOUT_MAX_MOVES = config["board_size"] ** 2 * 4 + 2
VALUE_OPENING_RANGE = (0, min(config["board_size"] ** 2 - 1, 2 * config["board_size"]))
print(f"Profile: {PROFILE} → {config}")

# %% [markdown]
# ## 1 · The paper's map
#
# The arrows matter. AlphaGo did **not** use its stronger RL policy as the MCTS
# prior. It kept the supervised policy because its human-like distribution
# preserved a broader, more useful search beam. The RL policy instead defined
# the self-play distribution learned by the value network.
#
# <img src="https://raw.githubusercontent.com/fulcrumai-dev/alphago-gymnasium-pytorch/main/assets/alphago_pipeline.svg" alt="AlphaGo training and search pipeline" width="100%"/>

# %% [markdown]
# ## 2 · Go as a Gymnasium environment
#
# A Gymnasium environment is convenient for reset/step tooling, but Go is a
# two-player alternating, zero-sum game. Here each observation is encoded from
# the **next side to move**. Rewards are zero until both players pass; the final
# reward is from the perspective of the player who made the terminating move.
#
# The immutable `GoPosition` underneath the environment implements captures,
# suicide, positional superko, pass, and a board-as-is Chinese-style area
# formula with komi. There is no separate dead-stone agreement phase: agents
# must capture dead groups before both pass. The final action index
# (`board_size²`) always means pass.

# %%
env = GoEnv(size=config["board_size"], komi=config["komi"], render_mode="ansi")
observation, info = env.reset(seed=SEED)

print("Observation shape:", observation.shape)
print("Action count (including pass):", env.action_space.n)
print("Legal opening actions:", int(info["legal_actions_mask"].sum()))
print(env.render())

assert env.observation_space.contains(observation)
assert info["legal_actions_mask"].shape == (env.action_space.n,)

# %% [markdown]
# ### A capture, visually
#
# `X` is Black and `O` is White. White's centre stone has one liberty before
# Black fills it. Try changing the last move and inspect `legal_actions_mask()`.

# %%
capture = GoPosition(size=5, komi=0.5)
for action in (7, 12, 11, 0, 13, 1):
    capture = capture.play(action)
captured = capture.play(17)

figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
plot_board(capture, ax=axes[0], title="Before: White has one liberty")
plot_board(captured, ax=axes[1], last_action=17, title="After: Black captures")
plt.show()

# %% [markdown]
# ## 3 · Feature planes: local Go knowledge for convolutions
#
# The paper preprocessed a 19×19 position into 48 binary policy planes (49 for
# value), including stone colour, liberties, captures, legality, move age, and a
# ladder feature. Our compact eight-plane encoding keeps the same
# side-to-move-relative idea:
#
# | Plane | Meaning |
# |---:|---|
# | 0–2 | current stones, opponent stones, empty intersections |
# | 3–4 | current groups with exactly 1 or 2 liberties |
# | 5–6 | opponent groups with exactly 1 or 2 liberties |
# | 7 | constant 1 if Black is to move, else 0 |
#
# This reduction is a compute choice, not a paper claim.

# %%
encoded = capture.encode()
plane_names = [
    "current", "opponent", "empty", "current: 1 liberty",
    "current: 2 liberties", "opponent: 1 liberty",
    "opponent: 2 liberties", "black to play",
]

figure, axes = plt.subplots(2, 4, figsize=(11, 5), constrained_layout=True)
for axis, plane, name in zip(axes.flat, encoded, plane_names, strict=True):
    axis.imshow(plane, vmin=0, vmax=1, cmap="Blues")
    axis.set_title(name, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
plt.show()

# %% [markdown]
# ## 4 · Three networks, three jobs
#
# - `PolicyNetwork`: spatial 5×5 then 3×3 convolutions and action logits. The
#   paper used a 13-layer, 192-filter match network.
# - `RolloutPolicy`: one shallow convolution, standing in for the paper's fast
#   linear softmax over local response/non-response patterns.
# - `ValueNetwork`: a convolutional trunk and tanh scalar in `[-1, 1]`, always
#   from the encoded side-to-move perspective.

# %%
size = config["board_size"]
model_kwargs = dict(
    board_size=size,
    input_channels=8,
    channels=config["channels"],
    depth=config["depth"],
)
sl_policy = PolicyNetwork(**model_kwargs).to(device)
rollout_policy = RolloutPolicy(board_size=size, input_channels=8).to(device)
value_network = ValueNetwork(
    **model_kwargs,
    hidden_channels=max(32, config["channels"] * 2),
).to(device)

def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

print(f"SL policy parameters: {parameter_count(sl_policy):,}")
print(f"Rollout policy parameters: {parameter_count(rollout_policy):,}")
print(f"Value network parameters: {parameter_count(value_network):,}")

sample = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
with torch.no_grad():
    assert sl_policy(sample).shape == (1, size * size + 1)
    assert rollout_policy(sample).shape == (1, size * size + 1)
    assert value_network(sample).shape == (1,)

# %% [markdown]
# ## 5 · Stage I — supervised policy and fast rollout learning
#
# AlphaGo's supervised policy maximized the log-likelihood of expert KGS moves
# and augmented each position with all eight rotations/reflections. The original
# data is not redistributed with the paper, so this runnable lesson generates a
# small **expert-like curriculum** using a transparent capture/liberty heuristic.
# Substitute parsed SGF games here for a closer data replication.
# The tiny lesson uses Adam for fast, stable feedback; the paper used
# asynchronous SGD at its distributed scale. The likelihood objective is the
# same, while the optimizer is an explicit teaching substitution.

# %%
expert_dataset = generate_expert_games(
    num_games=config["expert_games"],
    size=size,
    komi=config["komi"],
    seed=SEED,
    max_moves=EXPERT_MAX_MOVES,
)
base_examples = [
    PolicyExample(step.observation, step.action) for step in expert_dataset.steps
]
expert_examples = [
    transformed
    for example in base_examples
    for transformed in dihedral_policy_augmentations(example)
]

print(f"Games: {len(expert_dataset.games)}")
print(f"Raw positions: {len(base_examples)}")
print(f"After D4 augmentation: {len(expert_examples)}")
assert len(expert_examples) == 8 * len(base_examples)

# %%
sl_optimizer = torch.optim.Adam(sl_policy.parameters(), lr=2e-3)
rollout_optimizer = torch.optim.Adam(rollout_policy.parameters(), lr=3e-3)
sl_losses, rollout_losses = [], []

for epoch in range(config["sl_epochs"]):
    sl_losses.append(
        train_policy_epoch(
            sl_policy,
            expert_examples,
            sl_optimizer,
            batch_size=64,
            device=device,
            rng=np.random.default_rng(SEED + epoch),
        )
    )
    rollout_losses.append(
        train_policy_epoch(
            rollout_policy,
            expert_examples,
            rollout_optimizer,
            batch_size=64,
            device=device,
            rng=np.random.default_rng(SEED + 100 + epoch),
        )
    )

plt.figure(figsize=(6, 3))
plt.plot(sl_losses, marker="o", label="SL policy")
plt.plot(rollout_losses, marker="o", label="rollout policy")
plt.xlabel("epoch")
plt.ylabel("cross-entropy")
plt.title("Expert-move classification")
plt.legend()
plt.show()
print("Final losses:", {"SL": sl_losses[-1], "rollout": rollout_losses[-1]})

# %% [markdown]
# ## 6 · Stage II — policy-gradient reinforcement learning
#
# Initialize `pρ` from `pσ`, then play it against a uniformly sampled older
# checkpoint. For a learner move at time *t*, REINFORCE follows
#
# $$\Delta\rho \propto z_t\,\nabla_\rho \log p_\rho(a_t\mid s_t),$$
#
# where the terminal result is converted to the player-at-that-state
# perspective. The opponent pool avoids chasing only the newest policy.
# For a hard runtime bound, this teaching runner reserves the final two move
# slots for pass and excludes those forced actions from the gradient; normal
# sampled games terminate earlier.

# %%
rl_policy = copy.deepcopy(sl_policy).to(device)

def policy_factory() -> PolicyNetwork:
    return PolicyNetwork(**model_kwargs)

opponent_pool = OpponentPool(model_factory=policy_factory, max_size=6)
opponent_pool.add(rl_policy)
episodes = []
rl_losses = []
rl_optimizer = torch.optim.Adam(rl_policy.parameters(), lr=5e-4)

def checkpoint_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for tensor in model.state_dict().values():
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:12]

checkpoint_fingerprints = [checkpoint_fingerprint(rl_policy)]

for game_index in range(config["rl_games"]):
    opponent = opponent_pool.sample(rng, device=device)
    learner_player = BLACK if game_index % 2 == 0 else WHITE
    episode = generate_policy_gradient_episode(
        GoPosition(size=size, komi=config["komi"]),
        rl_policy,
        opponent,
        learner_player=learner_player,
        rng=rng,
        device=device,
        temperature=1.1,
        max_moves=SELF_PLAY_MAX_MOVES,
    )
    episodes.append(episode)
    # Update before snapshotting, so future games can really face an earlier,
    # distinct learner rather than several copies of the initial SL weights.
    rl_losses.append(
        train_reinforce_epoch(
            rl_policy,
            [episode],
            rl_optimizer,
            device=device,
            entropy_coefficient=0.0,
            rng=rng,
        )
    )
    opponent_pool.add(rl_policy)
    checkpoint_fingerprints.append(checkpoint_fingerprint(rl_policy))

print(f"Episodes: {len(episodes)} · learner outcomes: {[e.outcome for e in episodes]}")
print(f"REINFORCE losses: {[round(loss, 4) for loss in rl_losses]}")
print(f"Opponent snapshots: {len(opponent_pool)} · checkpoint hashes: {checkpoint_fingerprints}")
assert len(set(checkpoint_fingerprints)) > 1

# %% [markdown]
# ## 7 · Stage III — value learning from decorrelated self-play
#
# Training on every adjacent state in a game gives highly correlated examples.
# AlphaGo instead generated a new game for each value example: play an SL
# opening, inject one uniformly random legal move, finish with the RL policy,
# and retain only the state immediately after that random move. The paper draws
# an independent SL-prefix length from 0…449; we draw independently from the
# scaled range below, then apply board symmetries.

# %%
value_examples = generate_value_examples(
    lambda: GoPosition(size=size, komi=config["komi"]),
    sl_policy,
    rl_policy,
    num_games=config["value_games"],
    opening_moves=VALUE_OPENING_RANGE,
    rng=rng,
    device=device,
    temperature=1.05,
    max_moves=SELF_PLAY_MAX_MOVES,
)
augmented_value_examples = [
    transformed
    for example in value_examples
    for transformed in dihedral_value_augmentations(example)
]

value_optimizer = torch.optim.Adam(value_network.parameters(), lr=2e-3)
value_losses = []
for epoch in range(config["value_epochs"]):
    value_losses.append(
        train_value_epoch(
            value_network,
            augmented_value_examples,
            value_optimizer,
            batch_size=64,
            device=device,
            rng=np.random.default_rng(SEED + 200 + epoch),
        )
    )

plt.figure(figsize=(5.5, 3))
plt.plot(value_losses, marker="o", color="#16a34a")
plt.xlabel("epoch")
plt.ylabel("mean squared error")
plt.title("Self-play outcome regression")
plt.show()
print(
    f"Distinct games/examples: {len(value_examples)} · "
    f"SL-prefix range: {VALUE_OPENING_RANGE} · final MSE: {value_losses[-1]:.4f}"
)

# %% [markdown]
# ## 8 · Stage IV — policy + value + rollout MCTS
#
# <img src="https://raw.githubusercontent.com/fulcrumai-dev/alphago-gymnasium-pytorch/main/assets/mcts_cycle.svg" alt="Selection expansion evaluation and backup in AlphaGo MCTS" width="100%"/>
#
# Each edge stores a prior `P`, separate value/rollout visit counts and totals,
# and the mixed action value
#
# $$Q(s,a)=(1-\lambda)\frac{W_v(s,a)}{N_v(s,a)}
#           +\lambda\frac{W_r(s,a)}{N_r(s,a)}.$$
#
# Selection uses the 2016 prior-guided term (a PUCT variant)
#
# $$u(s,a)=c_{puct}P(s,a)\frac{\sqrt{\sum_b N_r(s,b)}}{1+N_r(s,a)}.$$
#
# This implementation is synchronous and single-process, so it omits virtual
# loss and asynchronous GPU queues while preserving the search statistics and
# leaf-evaluation mixture.

# %%
# Build a small non-empty position so the root heatmaps are interesting.
search_position = GoPosition(size=size, komi=config["komi"])
for _ in range(min(3, size)):
    probabilities = heuristic_policy(search_position, temperature=1.2)
    action = int(rng.choice(search_position.action_size, p=probabilities))
    search_position = search_position.play(action)

# Paper-faithful routing: frozen SL policy for priors; RL policy is NOT used here.
# The paper raises probabilities to beta=0.67. Since our adapter uses the
# conventional softmax temperature T, softmax(logits/T) == normalize(p**(1/T)),
# so T must be the reciprocal of beta.
PAPER_POLICY_BETA = 0.67
tree_policy = NeuralPolicyEvaluator(
    sl_policy, device=device, temperature=1.0 / PAPER_POLICY_BETA
)
fast_policy = NeuralPolicyEvaluator(rollout_policy, device=device)
rollout_evaluator = PolicyRolloutEvaluator(fast_policy, max_moves=ROLLOUT_MAX_MOVES)
learned_value = NeuralValueEvaluator(value_network, device=device)

mcts = AlphaGoMCTS(
    policy=tree_policy,
    value=learned_value,
    rollout=rollout_evaluator,
    config=MCTSConfig(
        num_simulations=config["mcts_simulations"],
        c_puct=1.5,
        mixing_lambda=0.5,
    ),
    seed=SEED,
)
search_result = mcts.search(search_position)

plot_board(search_position, title=f"Search root · {'Black' if search_position.to_play == BLACK else 'White'} to play")
plt.show()
plot_search_summary(search_position, search_result.visit_counts, search_result.q_values)
plt.show()

row, column = divmod(search_result.action, size)
move_name = "pass" if search_result.action == search_position.pass_action else f"({row}, {column})"
print(f"MCTS chooses {move_name}; root visits = {search_result.visit_counts.sum()}")

# %% [markdown]
# ## 9 · Executable correctness checks
#
# These are notebook-level integration checks, not a replacement for the
# repository's full unit suite. They verify Gymnasium compliance, legal masking,
# accelerator forward passes, bounded values, and MCTS visit accounting in the
# exact runtime executing this notebook.

# %%
from gymnasium.utils.env_checker import check_env

check_env(GoEnv(size=size, komi=config["komi"]), skip_render_check=True)

legal = search_position.legal_actions_mask()
sl_probabilities = legal_policy_probabilities(sl_policy, search_position, device=device)
assert np.isclose(sl_probabilities.sum(), 1.0)
assert np.all(sl_probabilities[~legal] == 0.0)
assert legal[search_result.action]
assert int(search_result.visit_counts.sum()) == config["mcts_simulations"]
assert np.isfinite(search_result.q_values).all()

with torch.no_grad():
    accelerator_batch = torch.as_tensor(search_position.encode(), dtype=torch.float32, device=device).unsqueeze(0)
    accelerator_policy = sl_policy(accelerator_batch)
    accelerator_value = value_network(accelerator_batch)
assert accelerator_policy.device.type == device.type
assert accelerator_value.device.type == device.type
assert -1.0 <= float(accelerator_value.detach().cpu()[0]) <= 1.0

print(f"✓ Gymnasium API · legal masks · MCTS invariants · {device.type.upper()} tensor execution")

# %% [markdown]
# ## 10 · What is faithful—and what is deliberately scaled
#
# | Component | AlphaGo (2016) | This runnable lesson |
# |---|---|---|
# | Go | 19×19, Chinese match rules/referee | configurable 3×3/5×5, captures/suicide/superko, board-as-is area scoring |
# | Expert data | 29.4M KGS positions | generated capture/liberty curriculum; SGF-ready API |
# | Features | 48 policy / 49 value planes | 8 explicit binary planes |
# | SL policy | 13 layers, 192 filters in match version | same spatial-convolution role; configurable depth/width |
# | Rollout | linear local-pattern softmax, ~2 μs | learned single-convolution fast policy |
# | RL | 10,000 mini-batches × 128 games, opponent pool | same REINFORCE/pool logic, tiny game count |
# | Value data | >30M distinct games, U−1 sampled in 0…449, one state/game | same sampled-prefix → random move → RL phases, scaled range/count |
# | Optimizer | asynchronous SGD | local Adam for quick teaching feedback; same losses/targets |
# | Termination | learned play to terminal, match adjudication | normal two-pass play plus two forced cap passes as a bounded fallback |
# | Search | asynchronous APV-MCTS, CPU/GPU workers, virtual loss | synchronous APV statistics, same prior/value/rollout roles |
#
# Read the line-by-line audit in
# [`docs/PAPER_FIDELITY.md`](https://github.com/fulcrumai-dev/alphago-gymnasium-pytorch/blob/main/docs/PAPER_FIDELITY.md).

# %% [markdown]
# ## 11 · Try it yourself
#
# 1. Switch to `ALPHAGO_PROFILE=tutorial` (or edit `PROFILE`) and compare loss
#    curves and root visit concentration.
# 2. Replace `generate_expert_games` with SGF positions and keep the same
#    `PolicyExample` interface.
# 3. Run two searches with `mixing_lambda=0` and `1`. Where do value and rollout
#    disagree?
# 4. Expand the eight-plane encoder toward Extended Data Table 2 in the paper.
# 5. Add dihedral inference ensembling, which AlphaGo used explicitly for raw
#    networks and implicitly inside search.

# %% [markdown]
# ## References
#
# - Silver, D. et al. (2016), [*Mastering the game of Go with deep neural
#   networks and tree search*](https://doi.org/10.1038/nature16961), Nature 529,
#   484–489. [Official DeepMind-hosted PDF](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf).
# - The repository includes a checksum-verifying paper fetcher and complete
#   citation in [`references/`](https://github.com/fulcrumai-dev/alphago-gymnasium-pytorch/tree/main/references).
# - Farama Foundation, [Gymnasium documentation](https://gymnasium.farama.org/).
# - PyTorch, [MPS backend notes](https://pytorch.org/docs/stable/notes/mps.html)
#   and [CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html).
# - Google Colab, [official Colab CLI](https://github.com/googlecolab/google-colab-cli).
#
# You now have every algorithmic piece of the original AlphaGo training/search
# pipeline in a small, inspectable implementation. Scale is a configuration;
# correctness starts with making the roles and perspectives explicit.
