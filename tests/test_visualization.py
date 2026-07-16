from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from alphago_gym.go import GoPosition
from alphago_gym.visualization import plot_board, plot_search_summary


def test_plot_board_returns_axes_and_draws_stones() -> None:
    position = GoPosition(size=3, komi=0.5).play(0).play(4)

    axes = plot_board(position, title="Capture lesson")

    assert axes.get_title() == "Capture lesson"
    # One PathCollection holds the two stone markers.
    assert sum(len(collection.get_offsets()) for collection in axes.collections) == 2
    plt.close(axes.figure)


def test_plot_board_can_highlight_last_move() -> None:
    position = GoPosition(size=3).play(4)
    axes = plot_board(position, last_action=4)
    assert len(axes.patches) >= 1
    plt.close(axes.figure)


def test_plot_search_summary_masks_illegal_actions_and_labels_pass() -> None:
    position = GoPosition(size=3).play(0)
    visits = np.arange(position.action_size)
    q_values = np.linspace(-1, 1, position.action_size)

    figure, axes = plot_search_summary(position, visits, q_values)

    assert len(axes) == 2
    assert "Pass visits" in axes[0].get_title()
    board_image = axes[0].images[0].get_array()
    assert np.ma.is_masked(board_image[0, 0])
    plt.close(figure)


def test_plot_search_summary_validates_action_vector_shape() -> None:
    position = GoPosition(size=3)
    with pytest.raises(ValueError, match="action_size"):
        plot_search_summary(position, np.ones(3), np.ones(position.action_size))

