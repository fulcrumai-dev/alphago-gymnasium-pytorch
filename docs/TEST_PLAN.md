# Test plan

The test pyramid follows three gates.

## Unit tests

- Go rules: liberties, single/multi-stone captures, suicide, positional
  superko, passes, area scoring, observation perspective, immutability.
- Gymnasium: spaces, reset/step protocol, masks, seeding, terminal rewards.
- Networks: output shapes/ranges, illegal-action masking, gradients, device
  movement, serialisation.
- Search: expansion, legal selection, value-sign backup, visit accounting,
  terminal handling, deterministic seeded behaviour.
- Training: loss direction, replay examples, self-play labels, frozen-opponent
  sampling, value targets.

## Integration tests

- Gymnasium environment checker.
- Short supervised-policy, RL-policy, and value-training loops.
- MCTS completes games without illegal moves.
- Save/reload preserves outputs.
- CPU and available accelerator smoke paths.

## End-to-end tests

- Execute the notebook headlessly with the `smoke` profile.
- Run a clean-environment install and CLI notebook validator.
- Run an MPS smoke test locally on Apple silicon.
- Exercise a CUDA-selected path in a Colab-compatible environment when an
  authenticated Colab runtime is available; otherwise report that external
  runtime limitation explicitly and retain a one-click Colab entry point.

Target: all critical rules and search invariants directly covered; at least 90%
branch coverage for rules/search modules and 80% project branch coverage.

