from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "alphago_2016_tutorial.ipynb"


def test_notebook_is_parseable_clean_and_tutorial_shaped() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert len(notebook.cells) >= 28
    assert all(
        cell.execution_count is None and not cell.outputs
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    for section in (
        "Runtime setup",
        "Go as a Gymnasium environment",
        "supervised policy",
        "policy-gradient reinforcement learning",
        "value learning",
        "policy + value + rollout MCTS",
        "Executable correctness checks",
        "References",
    ):
        assert section in markdown


def test_notebook_has_colab_paper_diagrams_and_honest_scope() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert "colab.research.google.com/github/fulcrumai-dev" in markdown
    assert "https://doi.org/10.1038/nature16961" in markdown
    assert "AlphaGoNaturePaper.pdf" in markdown
    assert "assets/alphago_pipeline.svg" in markdown
    assert "assets/mcts_cycle.svg" in markdown
    assert "Honest scope" in markdown
    assert "professional playing strength" in markdown


def test_notebook_routes_checkpoints_and_checks_accelerator() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )

    assert "rl_policy = copy.deepcopy(sl_policy)" in code
    assert "tree_policy = NeuralPolicyEvaluator(\n    sl_policy" in code
    assert "tree_policy = NeuralPolicyEvaluator(\n    rl_policy" not in code
    assert "PAPER_POLICY_BETA = 0.67" in code
    assert "temperature=1.0 / PAPER_POLICY_BETA" in code
    assert "generate_value_examples(" in code
    assert "sl_policy,\n    rl_policy," in code
    rl_loop = code[code.index("for game_index in range(config[\"rl_games\"])") :]
    assert rl_loop.index("train_reinforce_epoch(") < rl_loop.index(
        "opponent_pool.add(rl_policy)"
    )
    assert "len(set(checkpoint_fingerprints)) > 1" in rl_loop
    assert "select_device(" in code
    assert "check_env(" in code
    assert "MPS tensor execution" not in code  # success text is backend-neutral
    assert "tensor execution" in code


def test_scanner_rejects_error_outputs_and_accepts_success(tmp_path: Path) -> None:
    clean = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "print('ok')",
                execution_count=1,
                outputs=[nbformat.v4.new_output("stream", name="stdout", text="✓ sentinel · CUDA tensor execution\n")],
            )
        ]
    )
    clean_path = tmp_path / "clean.ipynb"
    nbformat.write(clean, clean_path)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "scan_notebook.py"),
        str(clean_path),
        "--require-text",
        "✓ sentinel",
        "--require-device",
        "cuda",
    ]
    accepted = subprocess.run(command, text=True, capture_output=True)
    assert accepted.returncode == 0, accepted.stderr

    broken = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(
                "raise RuntimeError('boom')",
                execution_count=1,
                outputs=[
                    nbformat.v4.new_output(
                        "error",
                        ename="RuntimeError",
                        evalue="boom",
                        traceback=["RuntimeError: boom"],
                    )
                ],
            )
        ]
    )
    broken_path = tmp_path / "broken.ipynb"
    nbformat.write(broken, broken_path)
    rejected = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scan_notebook.py"), str(broken_path)],
        text=True,
        capture_output=True,
    )
    assert rejected.returncode != 0
    assert "RuntimeError: boom" in rejected.stderr
