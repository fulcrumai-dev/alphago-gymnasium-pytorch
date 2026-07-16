# AlphaGo with Gymnasium and PyTorch

An educational, scaled reimplementation of *Mastering the game of Go with deep
neural networks and tree search* (Silver et al., 2016).

The implementation is being developed test-first. It targets Google Colab
(CUDA), Apple silicon (MPS), and CPU. The final tutorial notebook will connect
supervised policy learning, self-play policy-gradient reinforcement learning,
value learning, fast rollouts, and policy/value-guided Monte Carlo tree search.

> This project reproduces the paper's algorithmic pipeline, not DeepMind's
> distributed compute scale or playing strength. See `docs/PAPER_FIDELITY.md`
> for the exact fidelity boundary.

## Development

```bash
uv sync --extra dev --python 3.12
uv run pytest
```

The paper citation and official links are kept in `references/README.md`.

