"""Configuration objects for reproducible PokeTraining runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

REWARD_MODES = frozenset({"exploration", "advanced"})


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Hyperparameters and emulator settings shared by training and play.

    Defaults retain the notebook's DQN architecture and core training
    hyperparameters. A configuration can be loaded from JSON without adding a
    YAML dependency.
    """

    reward_mode: str = "exploration"
    frame_height: int = 84
    frame_width: int = 84
    frame_stack: int = 4
    replay_capacity: int = 20_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.999
    batch_size: int = 32
    update_target_every: int = 1_000
    episodes: int = 500
    max_steps: int = 5_000
    screenshot_interval: int = 200
    checkpoint_every: int = 50
    press_ticks: int = 4
    release_ticks: int = 4
    skip_intro_start_presses: int = 25
    skip_intro_a_presses: int = 10_000
    initial_random_ticks_max: int = 10
    include_start_action: bool = False
    emulation_speed: int = 0
    expected_cartridge_title: str | None = "POKEMON RED"
    expected_rom_sha1: str | None = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"
    expected_rom_sha256: str | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid settings early, before allocating large buffers."""

        if not isinstance(self.reward_mode, str):
            raise TypeError("reward_mode must be a string")
        if self.reward_mode not in REWARD_MODES:
            allowed = ", ".join(sorted(REWARD_MODES))
            raise ValueError(f"reward_mode must be one of: {allowed}")
        positive_ints = {
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "frame_stack": self.frame_stack,
            "replay_capacity": self.replay_capacity,
            "batch_size": self.batch_size,
            "update_target_every": self.update_target_every,
            "episodes": self.episodes,
            "max_steps": self.max_steps,
            "press_ticks": self.press_ticks,
            "release_ticks": self.release_ticks,
        }
        all_ints = {
            **positive_ints,
            "screenshot_interval": self.screenshot_interval,
            "checkpoint_every": self.checkpoint_every,
            "skip_intro_start_presses": self.skip_intro_start_presses,
            "skip_intro_a_presses": self.skip_intro_a_presses,
            "initial_random_ticks_max": self.initial_random_ticks_max,
            "emulation_speed": self.emulation_speed,
        }
        if self.seed is not None:
            all_ints["seed"] = self.seed
        for name, value in all_ints.items():
            if type(value) is not int:
                raise TypeError(f"{name} must be an integer")
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        nonnegative_ints = {
            "screenshot_interval": self.screenshot_interval,
            "checkpoint_every": self.checkpoint_every,
            "skip_intro_start_presses": self.skip_intro_start_presses,
            "skip_intro_a_presses": self.skip_intro_a_presses,
            "initial_random_ticks_max": self.initial_random_ticks_max,
            "emulation_speed": self.emulation_speed,
        }
        for name, value in nonnegative_ints.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.frame_height < 36 or self.frame_width < 36:
            raise ValueError("frame_height and frame_width must be at least 36 for the DQN")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        real_values = {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        for name, value in real_values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= epsilon_min <= epsilon_start <= 1")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1]")
        if type(self.include_start_action) is not bool:
            raise TypeError("include_start_action must be a boolean")
        if self.expected_cartridge_title is not None and not isinstance(
            self.expected_cartridge_title, str
        ):
            raise TypeError("expected_cartridge_title must be a string or null")
        for field_name, digest, length in (
            ("expected_rom_sha1", self.expected_rom_sha1, 40),
            ("expected_rom_sha256", self.expected_rom_sha256, 64),
        ):
            if digest is None:
                continue
            if not isinstance(digest, str):
                raise TypeError(f"{field_name} must be a string or null")
            normalized = digest.lower()
            if len(normalized) != length or any(ch not in "0123456789abcdef" for ch in normalized):
                raise ValueError(f"{field_name} must be a {length}-character hex digest")

    @property
    def frame_shape(self) -> tuple[int, int]:
        """Return the processed frame shape as ``(height, width)``."""

        return (self.frame_height, self.frame_width)

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Return the stacked observation shape expected by the DQN."""

        return (*self.frame_shape, self.frame_stack)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TrainerConfig:
        """Build a configuration and reject misspelled or unknown keys."""

        allowed = {item.name for item in fields(cls)}
        unknown = set(values) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown configuration fields: {names}")
        return cls(**dict(values))

    @classmethod
    def from_json(cls, path: Path) -> TrainerConfig:
        """Load configuration values from a UTF-8 JSON file."""

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Configuration JSON must contain an object")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def load_config(path: Path | None) -> TrainerConfig:
    """Load ``path`` when supplied, otherwise return verified defaults."""

    return TrainerConfig.from_json(path) if path is not None else TrainerConfig()


# Public compatibility name used by repository documentation and downstream code.
TrainingConfig = TrainerConfig
