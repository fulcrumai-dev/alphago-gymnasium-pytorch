# Architecture and interface contracts

The code is split so the Go rules, Gymnasium adapter, neural networks, search,
training stages, and notebook can be tested independently.

## Go state contract

`alphago_gym.go.GoPosition` is an immutable alternating-turn position:

- board values are `BLACK = 1`, `WHITE = -1`, and `EMPTY = 0`;
- action indices `0 .. size*size-1` are board intersections;
- action `size*size` is pass;
- `legal_actions_mask()` returns a Boolean vector of length `size*size + 1`;
- `play(action)` returns a new position and enforces captures, suicide, and
  positional superko;
- two consecutive passes terminate the game;
- `outcome(player)` returns `-1`, `0`, or `1` using the Chinese-style area
  formula and komi, from the requested player's perspective. Scoring is
  board-as-is: there is no separate dead-stone agreement/adjudication phase,
  so self-play agents must capture dead groups before both pass;
- `encode()` returns float32 planes from the side-to-move perspective.

`alphago_gym.env.GoEnv` is a Gymnasium adapter over that state. Its observation
is always from the next player-to-move perspective. At termination, `reward`
is the game result from the player who just acted; intermediate rewards are 0.
Illegal actions raise `ValueError` and the legal mask is provided in `info`.

## Neural contract

Networks consume tensors shaped `(batch, planes, size, size)`:

- `PolicyNetwork` returns unnormalised logits for `size*size + 1` actions;
- `RolloutPolicy` is deliberately shallow and fast;
- `ValueNetwork` returns a scalar in `[-1, 1]` for the encoded side to move;
- masking helpers assign zero probability to illegal moves;
- all models and training helpers support CPU, CUDA, and MPS.

## Search contract

AlphaGo-style MCTS stores policy priors, visit counts, total/action values, and
uses a prior-guided upper-confidence selection term. At a leaf, the backed-up
evaluation mixes the learned value estimate and a fast rollout result. Values
change sign at each ply. Search never mutates the caller's position.

## Educational profiles

`smoke` runs in a few minutes on CPU for automated validation. `tutorial` uses
more examples/simulations but stays Colab-friendly. All scale parameters are
exposed so readers can increase board size, depth, games, and search budget.
