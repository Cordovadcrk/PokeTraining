"""CLI tests that avoid constructing PyBoy or reading a real ROM."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("pyboy")
pytest.importorskip("tensorflow")

from poketraining import cli
from poketraining.config import TrainerConfig


def test_train_parser_and_overrides() -> None:
    args = cli.build_parser().parse_args(
        [
            "train",
            "--rom",
            "legal.gb",
            "--reward-mode",
            "advanced",
            "--episodes",
            "3",
            "--max-steps",
            "7",
            "--seed",
            "19",
        ]
    )

    config = cli._apply_overrides(TrainerConfig(), args)

    assert args.command == "train"
    assert args.rom == Path("legal.gb")
    assert config.reward_mode == "advanced"
    assert config.episodes == 3
    assert config.max_steps == 7
    assert config.seed == 19


def test_run_directory_rejects_unsafe_or_existing_names(tmp_path) -> None:
    with pytest.raises(ValueError, match="run-name"):
        cli._new_run_directory(
            tmp_path,
            "../escape",
            command="train",
            reward_mode="exploration",
        )

    existing = tmp_path / "already-there"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        cli._new_run_directory(
            tmp_path,
            existing.name,
            command="train",
            reward_mode="exploration",
        )


def test_cli_train_smoke_delegates_without_opening_rom(
    monkeypatch,
    tmp_path,
) -> None:
    calls: dict[str, object] = {}

    class FakeTrainer:
        def __init__(self, environment, config, run_dir) -> None:
            calls["environment"] = environment
            calls["config"] = config
            calls["run_dir"] = run_dir

        def train(self) -> None:
            calls["trained"] = True

    class FakeEnvironment:
        def validate_rom(self) -> None:
            calls["validated"] = True

    fake_environment = FakeEnvironment()

    def environment_factory(rom, config):
        calls.update({"rom": rom, "environment_config": config})
        return fake_environment

    monkeypatch.setattr(
        cli,
        "_runtime_classes",
        lambda: (environment_factory, FakeTrainer),
    )

    status = cli.main(
        [
            "train",
            "--rom",
            "synthetic.gb",
            "--output-root",
            str(tmp_path),
            "--run-name",
            "smoke",
            "--episodes",
            "1",
            "--max-steps",
            "1",
        ]
    )

    assert status == 0
    assert calls["rom"] == Path("synthetic.gb")
    assert calls["environment"] is fake_environment
    assert calls["validated"] is True
    assert calls["trained"] is True
    assert calls["run_dir"] == (tmp_path / "smoke").resolve()
    assert isinstance(calls["config"], TrainerConfig)


def test_apply_overrides_leaves_config_unchanged_without_values() -> None:
    config = TrainerConfig()
    args = Namespace(
        reward_mode=None,
        seed=None,
        episodes=None,
        max_steps=None,
    )

    assert cli._apply_overrides(config, args) is config


def test_play_rejects_missing_weights_before_importing_runtime(monkeypatch) -> None:
    def unexpected_runtime_import():
        raise AssertionError("runtime classes should not be imported")

    monkeypatch.setattr(cli, "_runtime_classes", unexpected_runtime_import)

    with pytest.raises(FileNotFoundError, match="Weights not found"):
        cli.main(
            [
                "play",
                "--rom",
                "synthetic.gb",
                "--weights",
                "missing.weights.h5",
            ]
        )
