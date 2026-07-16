"""Notebook-friendly visualizations for Go positions and MCTS statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from .go import BLACK, WHITE

if TYPE_CHECKING:
    from .go import GoPosition


def plot_board(
    position: "GoPosition",
    *,
    ax: Axes | None = None,
    last_action: int | None = None,
    title: str | None = None,
) -> Axes:
    """Draw a Go position with coordinates and an optional last-move marker."""

    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
    size = position.size
    ax.set_facecolor("#d8a45d")
    for coordinate in range(size):
        ax.plot([0, size - 1], [coordinate, coordinate], color="#4a3523", lw=1.2)
        ax.plot([coordinate, coordinate], [0, size - 1], color="#4a3523", lw=1.2)

    _draw_star_points(ax, size)
    rows, columns = np.nonzero(position.board == BLACK)
    if len(rows):
        ax.scatter(
            columns,
            rows,
            s=_stone_size(size),
            c="#111827",
            edgecolors="#020617",
            linewidths=1.0,
            zorder=3,
        )
    rows, columns = np.nonzero(position.board == WHITE)
    if len(rows):
        ax.scatter(
            columns,
            rows,
            s=_stone_size(size),
            c="#f8fafc",
            edgecolors="#334155",
            linewidths=1.2,
            zorder=3,
        )

    if last_action is not None and last_action != position.pass_action:
        if not 0 <= last_action < position.pass_action:
            raise ValueError("last_action is outside the board")
        row, column = divmod(last_action, size)
        ax.add_patch(
            Circle(
                (column, row),
                radius=0.12,
                facecolor="none",
                edgecolor="#ef4444",
                linewidth=2.2,
                zorder=4,
            )
        )

    labels = [_column_label(column) for column in range(size)]
    ax.set_xticks(range(size), labels)
    ax.set_yticks(range(size), [str(row + 1) for row in range(size)])
    ax.set_xlim(-0.55, size - 0.45)
    ax.set_ylim(size - 0.45, -0.55)
    ax.set_aspect("equal")
    ax.tick_params(length=0, labelsize=9, colors="#475569")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, fontweight="semibold", pad=12)
    return ax


def plot_search_summary(
    position: "GoPosition",
    visit_counts: Sequence[float] | np.ndarray,
    q_values: Sequence[float] | np.ndarray,
) -> tuple[Figure, np.ndarray]:
    """Plot legal root visits and action values as aligned board heatmaps."""

    visits = _action_vector(position, visit_counts, "visit_counts")
    values = _action_vector(position, q_values, "q_values")
    legal = position.legal_actions_mask()
    board_legal = legal[:-1].reshape(position.size, position.size)
    board_visits = np.ma.array(
        visits[:-1].reshape(position.size, position.size), mask=~board_legal
    )
    board_values = np.ma.array(
        values[:-1].reshape(position.size, position.size), mask=~board_legal
    )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    _heatmap(
        axes[0],
        board_visits,
        cmap="Blues",
        title=f"Root visits · Pass visits: {int(visits[-1])}",
        integer_labels=True,
    )
    _heatmap(
        axes[1],
        board_values,
        cmap="coolwarm",
        title=f"Action value Q · Pass Q: {values[-1]:+.2f}",
        integer_labels=False,
        vmin=-1.0,
        vmax=1.0,
    )
    return figure, axes


def _action_vector(
    position: "GoPosition", values: Sequence[float] | np.ndarray, name: str
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (position.action_size,):
        raise ValueError(f"{name} must have shape (action_size,)")
    return result


def _heatmap(
    ax: Axes,
    values: np.ma.MaskedArray,
    *,
    cmap: str,
    title: str,
    integer_labels: bool,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    size = values.shape[0]
    for row in range(size):
        for column in range(size):
            if np.ma.is_masked(values[row, column]):
                continue
            value = float(values[row, column])
            label = f"{int(value)}" if integer_labels else f"{value:+.2f}"
            ax.text(column, row, label, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(size), [_column_label(index) for index in range(size)])
    ax.set_yticks(range(size), [str(index + 1) for index in range(size)])
    ax.set_title(title, fontsize=12, fontweight="semibold")
    ax.tick_params(length=0, labelsize=8)
    figure = ax.figure
    figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _draw_star_points(ax: Axes, size: int) -> None:
    if size >= 19:
        points = (3, size // 2, size - 4)
    elif size >= 9:
        points = (2, size // 2, size - 3)
    elif size >= 5:
        points = (size // 2,)
    else:
        points = ()
    if points:
        coordinates = [(column, row) for row in points for column in points]
        ax.scatter(
            [point[0] for point in coordinates],
            [point[1] for point in coordinates],
            s=12,
            c="#3f2d1d",
            zorder=2,
        )


def _column_label(column: int) -> str:
    # Go coordinates conventionally skip I.
    alphabet = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    return alphabet[column] if column < len(alphabet) else str(column + 1)


def _stone_size(size: int) -> float:
    return max(90.0, min(1_300.0, 5_000.0 / max(size, 1)))


__all__ = ["plot_board", "plot_search_summary"]

