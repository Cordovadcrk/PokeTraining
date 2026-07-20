"""Small, headless figures derived from structured episode metrics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


class EpisodeSummary(Protocol):
    """Fields needed to visualize one episode without coupling to the trainer."""

    episode: int
    total_reward: float
    mean_loss: float | None
    action_counts: dict[str, int]


def save_training_figures(
    metrics: Sequence[EpisodeSummary],
    action_names: Sequence[str],
    output_dir: Path,
) -> dict[str, Path]:
    """Save reward, mean-loss and action-count PNGs for one training run."""

    if not metrics:
        raise ValueError("metrics must contain at least one episode")
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = [item.episode for item in metrics]

    reward_figure = Figure(figsize=(8, 4.5), layout="constrained")
    reward_axis = reward_figure.subplots()
    reward_axis.plot(
        episodes,
        [item.total_reward for item in metrics],
        color="tab:green",
        marker="o",
    )
    reward_axis.set(title="Recompensa por episodio", xlabel="Episodio", ylabel="Recompensa")
    reward_axis.grid(alpha=0.25)

    loss_figure = Figure(figsize=(8, 4.5), layout="constrained")
    loss_axis = loss_figure.subplots()
    loss_points = [(item.episode, item.mean_loss) for item in metrics if item.mean_loss is not None]
    if loss_points:
        loss_axis.plot(
            [point[0] for point in loss_points],
            [point[1] for point in loss_points],
            color="tab:blue",
            marker="o",
        )
    else:
        loss_axis.text(
            0.5,
            0.5,
            "Sin actualizaciones de optimización",
            ha="center",
            va="center",
            transform=loss_axis.transAxes,
        )
    loss_axis.set(
        title="Pérdida media por episodio",
        xlabel="Episodio",
        ylabel="Huber loss",
    )
    loss_axis.grid(alpha=0.25)

    totals = {
        name: sum(item.action_counts.get(name, 0) for item in metrics) for name in action_names
    }
    action_figure = Figure(figsize=(8, 4.5), layout="constrained")
    action_axis = action_figure.subplots()
    action_axis.bar(totals.keys(), totals.values(), color="tab:purple")
    action_axis.set(
        title="Distribución de acciones",
        xlabel="Acción",
        ylabel="Frecuencia",
    )

    paths = {
        "rewards": output_dir / "training_rewards.png",
        "loss": output_dir / "training_loss.png",
        "actions": output_dir / "training_actions.png",
    }
    for figure, path in (
        (reward_figure, paths["rewards"]),
        (loss_figure, paths["loss"]),
        (action_figure, paths["actions"]),
    ):
        FigureCanvasAgg(figure).print_png(path)
        figure.clear()
    return paths
