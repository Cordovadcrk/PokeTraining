"""Checks for portable, headless run-summary figures."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from poketraining.reporting import save_training_figures


@dataclass
class Metric:
    episode: int
    total_reward: float
    mean_loss: float | None
    action_counts: dict[str, int]


def test_training_summary_writes_three_valid_pngs(tmp_path) -> None:
    metrics = [
        Metric(0, 1.5, None, {"left": 2, "right": 1}),
        Metric(1, 2.5, 0.25, {"left": 1, "right": 3}),
    ]

    paths = save_training_figures(metrics, ("left", "right"), tmp_path / "figures")

    assert set(paths) == {"rewards", "loss", "actions"}
    for path in paths.values():
        assert path.is_file()
        with Image.open(path) as image:
            assert image.format == "PNG"
            assert image.width > 100
            assert image.height > 100
