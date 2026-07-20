"""Tests for configuration validation and ROM fingerprint checks."""

from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("PIL")
pytest.importorskip("pyboy")

from poketraining.config import TrainerConfig, load_config
from poketraining.emulator import PokemonRedEnvironment


def test_default_shapes_and_aliases() -> None:
    config = TrainerConfig()

    assert config.frame_shape == (84, 84)
    assert config.observation_shape == (84, 84, 4)
    assert config.reward_mode == "exploration"
    assert load_config(None) == config


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"reward_mode": "unknown"}, "reward_mode"),
        ({"frame_height": 0}, "frame_height"),
        ({"replay_capacity": 31}, "replay_capacity"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"gamma": 1.01}, "gamma"),
        ({"epsilon_min": 0.6, "epsilon_start": 0.5}, "epsilon"),
        ({"epsilon_decay": 0.0}, "epsilon_decay"),
        ({"screenshot_interval": -1}, "screenshot_interval"),
        ({"expected_rom_sha1": "not-a-digest"}, "expected_rom_sha1"),
        ({"expected_rom_sha256": "f" * 63}, "expected_rom_sha256"),
        ({"frame_height": 35}, "at least 36"),
        ({"seed": -1}, "seed"),
        ({"learning_rate": float("inf")}, "finite"),
    ],
)
def test_invalid_config_values_are_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainerConfig(**overrides)  # type: ignore[arg-type]


def test_mapping_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown configuration fields: typo"):
        TrainerConfig.from_mapping({"typo": 1})


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"episodes": True}, "episodes"),
        ({"batch_size": 1.5}, "batch_size"),
        ({"learning_rate": "fast"}, "learning_rate"),
        ({"include_start_action": 1}, "include_start_action"),
        ({"expected_rom_sha1": 1}, "expected_rom_sha1"),
    ],
)
def test_invalid_config_types_are_rejected(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        TrainerConfig(**overrides)  # type: ignore[arg-type]


def test_json_round_trip_and_non_object_rejection(tmp_path) -> None:
    source = TrainerConfig(
        reward_mode="advanced",
        batch_size=4,
        replay_capacity=8,
        seed=17,
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(source.to_dict()), encoding="utf-8")

    assert load_config(path) == source

    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        TrainerConfig.from_json(path)


def test_rom_hash_validation_uses_a_synthetic_file(tmp_path) -> None:
    payload = b"synthetic test bytes, not a game ROM"
    rom_path = tmp_path / "fixture.gb"
    rom_path.write_bytes(payload)
    config = TrainerConfig(
        expected_rom_sha1=hashlib.sha1(payload).hexdigest(),
        expected_rom_sha256=hashlib.sha256(payload).hexdigest(),
    )

    environment = PokemonRedEnvironment(rom_path, config)

    assert environment.validate_rom() == {
        "sha1": hashlib.sha1(payload).hexdigest(),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


@pytest.mark.parametrize(
    "hash_overrides, expected_message",
    [
        (
            {"expected_rom_sha1": "0" * 40, "expected_rom_sha256": None},
            "SHA-1",
        ),
        (
            {"expected_rom_sha1": None, "expected_rom_sha256": "0" * 64},
            "SHA-256",
        ),
    ],
)
def test_rom_hash_mismatch_is_rejected(
    tmp_path,
    hash_overrides: dict[str, object],
    expected_message: str,
) -> None:
    rom_path = tmp_path / "fixture.gb"
    rom_path.write_bytes(b"synthetic")
    config = TrainerConfig(**hash_overrides)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=expected_message):
        PokemonRedEnvironment(rom_path, config)._validate_rom_file()


def test_missing_rom_is_rejected_without_opening_pyboy(tmp_path) -> None:
    environment = PokemonRedEnvironment(
        tmp_path / "missing.gb",
        TrainerConfig(expected_rom_sha1=None, expected_rom_sha256=None),
    )

    with pytest.raises(FileNotFoundError, match="ROM not found"):
        environment._validate_rom_file()
