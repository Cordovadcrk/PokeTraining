"""ROM-free smoke tests for the unified trainer lifecycle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PIL")
pytest.importorskip("pyboy")
pytest.importorskip("tensorflow")

from poketraining.config import TrainerConfig
from poketraining.emulator import EnvironmentStep
from poketraining.memory import POKEDEX_OWNED_LENGTH, MemorySnapshot
from poketraining.trainer import PokemonAITrainer


class FakeEnvironment:
    """Minimal deterministic environment; it never opens PyBoy or a ROM."""

    num_actions = 2
    action_names = ("left", "right")

    def __init__(self, observation_shape: tuple[int, int, int]) -> None:
        self.observation_shape = observation_shape
        self.started = False
        self.closed = False
        self.reset_calls = 0
        self.step_calls = 0

    @staticmethod
    def _snapshot(*, block_x: int) -> MemorySnapshot:
        return MemorySnapshot(
            map_id=1,
            block_y=2,
            block_x=block_x,
            tileset_id=3,
            battle_type=0,
            party_count=1,
            pokedex_owned=bytes(POKEDEX_OWNED_LENGTH),
            enemy_level=0,
        )

    def start(self) -> None:
        assert not self.started
        self.started = True

    def reset(self) -> tuple[np.ndarray, MemorySnapshot]:
        assert self.started
        self.reset_calls += 1
        return (
            np.zeros(self.observation_shape, dtype=np.uint8),
            self._snapshot(block_x=1),
        )

    def step(self, action_index: int) -> EnvironmentStep:
        assert self.started
        assert 0 <= action_index < self.num_actions
        self.step_calls += 1
        return EnvironmentStep(
            observation=np.ones(self.observation_shape, dtype=np.uint8),
            snapshot=self._snapshot(block_x=2),
            terminated=True,
        )

    def save_screenshot(self, path: Path) -> None:
        raise AssertionError(f"screenshots are disabled in this smoke test: {path}")

    def close(self) -> None:
        self.closed = True
        self.started = False


def tiny_config() -> TrainerConfig:
    """Use the smallest spatial input accepted by the three convolutions."""

    return TrainerConfig(
        frame_height=36,
        frame_width=36,
        frame_stack=1,
        replay_capacity=2,
        batch_size=1,
        episodes=1,
        max_steps=2,
        screenshot_interval=0,
        checkpoint_every=0,
        update_target_every=1,
        expected_rom_sha1=None,
        expected_cartridge_title=None,
        seed=23,
    )


def test_training_smoke_runs_one_fake_episode_and_writes_metadata(tmp_path) -> None:
    config = tiny_config()
    environment = FakeEnvironment(config.observation_shape)
    run_dir = tmp_path / "run"
    trainer = PokemonAITrainer(environment, config, run_dir)  # type: ignore[arg-type]

    metrics = trainer.train()

    assert environment.reset_calls == 1
    assert environment.step_calls == 1
    assert environment.closed is True
    assert len(metrics) == 1
    episode = metrics[0]
    assert episode.steps == 1
    assert episode.terminated is True
    assert episode.truncated is False
    assert episode.mean_loss is not None
    assert np.isfinite(episode.mean_loss)
    assert episode.epsilon == pytest.approx(
        max(config.epsilon_min, config.epsilon_start * config.epsilon_decay)
    )
    assert trainer.epsilon == episode.epsilon

    run_config = json.loads((run_dir / "run_config.json").read_text("utf-8"))
    assert run_config["actions"] == ["left", "right"]
    assert run_config["rom_fingerprints"] == {}
    assert run_config["runtime"]["python"]
    assert run_config["created_at"].endswith("+00:00")
    assert run_config["source_revision"]
    metric_lines = (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()
    assert len(metric_lines) == 1
    assert json.loads(metric_lines[0])["terminated"] is True
    assert (run_dir / "checkpoints" / "dqn_final.weights.h5").is_file()
    assert (run_dir / "checkpoints" / "training_state.json").is_file()
    assert (run_dir / "figures" / "training_rewards.png").is_file()
    assert (run_dir / "figures" / "training_loss.png").is_file()
    assert (run_dir / "figures" / "training_actions.png").is_file()


def test_training_refuses_to_reuse_an_existing_run_directory(tmp_path) -> None:
    config = tiny_config()
    environment = FakeEnvironment(config.observation_shape)
    run_dir = tmp_path / "existing"
    run_dir.mkdir()
    (run_dir / "keep.txt").write_text("do not overwrite", encoding="utf-8")
    trainer = PokemonAITrainer(environment, config, run_dir)  # type: ignore[arg-type]

    with pytest.raises(FileExistsError):
        trainer.train()

    assert (run_dir / "keep.txt").read_text(encoding="utf-8") == "do not overwrite"
    assert environment.started is False


def test_trainer_starts_with_synchronized_models(tmp_path) -> None:
    config = tiny_config()
    environment = FakeEnvironment(config.observation_shape)
    trainer = PokemonAITrainer(environment, config, tmp_path / "run")  # type: ignore[arg-type]

    assert trainer.replay is None

    for online_weight, target_weight in zip(
        trainer.online_model.get_weights(),
        trainer.target_model.get_weights(),
        strict=True,
    ):
        np.testing.assert_array_equal(online_weight, target_weight)


def test_play_uses_no_replay_and_records_weights_fingerprint(tmp_path) -> None:
    config = tiny_config()
    environment = FakeEnvironment(config.observation_shape)
    trainer = PokemonAITrainer(environment, config, tmp_path / "play-run")  # type: ignore[arg-type]
    weights_path = tmp_path / "fixture.weights.h5"
    trainer.online_model.save_weights(weights_path)

    metrics = trainer.play(weights_path, episodes=1, max_steps=1)

    assert len(metrics) == 1
    assert metrics[0].terminated is True
    assert trainer.replay is None
    assert environment.closed is True
    run_config = json.loads((tmp_path / "play-run" / "run_config.json").read_text("utf-8"))
    assert (
        run_config["input_artifacts"]["weights_sha256"]
        == hashlib.sha256(weights_path.read_bytes()).hexdigest()
    )
