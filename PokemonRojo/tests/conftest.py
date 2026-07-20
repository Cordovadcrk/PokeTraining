"""Shared deterministic fixtures for the PokeTraining test suite."""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest

from poketraining.memory import POKEDEX_OWNED_LENGTH, MemorySnapshot

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


@pytest.fixture
def snapshot_factory() -> Callable[..., MemorySnapshot]:
    """Build concise, valid memory snapshots with overridable fields."""

    def make_snapshot(**overrides: object) -> MemorySnapshot:
        values: dict[str, object] = {
            "map_id": 1,
            "block_y": 2,
            "block_x": 3,
            "tileset_id": 4,
            "battle_type": 0,
            "party_count": 1,
            "pokedex_owned": bytes(POKEDEX_OWNED_LENGTH),
            "enemy_level": 0,
        }
        values.update(overrides)
        return MemorySnapshot(**values)  # type: ignore[arg-type]

    return make_snapshot
