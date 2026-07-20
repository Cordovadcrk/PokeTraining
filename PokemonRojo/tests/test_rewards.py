"""Tests for event-based exploration and advanced rewards."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from poketraining.memory import POKEDEX_OWNED_LENGTH, MemorySnapshot
from poketraining.rewards import RewardCalculator

Frame = np.ndarray
SnapshotFactory = Callable[..., MemorySnapshot]


def blank_frame() -> Frame:
    return np.zeros((3, 3), dtype=np.uint8)


def owned_with_bits(*bit_indices: int) -> bytes:
    owned = bytearray(POKEDEX_OWNED_LENGTH)
    for bit_index in bit_indices:
        owned[bit_index // 8] |= 1 << (bit_index % 8)
    return bytes(owned)


def test_calculate_requires_reset(snapshot_factory: SnapshotFactory) -> None:
    calculator = RewardCalculator("exploration")

    with pytest.raises(RuntimeError, match=r"reset\(\)"):
        calculator.calculate(blank_frame(), blank_frame(), snapshot_factory())


def test_exploration_rewards_initial_block_once(
    snapshot_factory: SnapshotFactory,
) -> None:
    initial = snapshot_factory()
    calculator = RewardCalculator("exploration")
    calculator.reset(initial)

    first = calculator.calculate(blank_frame(), blank_frame(), initial)
    second = calculator.calculate(blank_frame(), blank_frame(), initial)

    assert first.components == {
        "visual_change": 0.0,
        "new_block": 1.0,
        "new_map": 0.0,
        "pokedex_progress": 0.0,
        "stagnation": 0.0,
    }
    assert first.total == pytest.approx(1.0)
    assert second.components["new_block"] == 0.0
    assert second.components["stagnation"] == pytest.approx(-0.05)


def test_exploration_identity_includes_map_and_block_coordinates(
    snapshot_factory: SnapshotFactory,
) -> None:
    initial = snapshot_factory(map_id=1, block_x=5, block_y=6)
    calculator = RewardCalculator("exploration")
    calculator.reset(initial)
    calculator.calculate(blank_frame(), blank_frame(), initial)

    moved = snapshot_factory(map_id=1, block_x=7, block_y=6)
    moved_reward = calculator.calculate(blank_frame(), blank_frame(), moved)
    new_map_same_coordinates = snapshot_factory(map_id=2, block_x=7, block_y=6)
    map_reward = calculator.calculate(
        blank_frame(),
        blank_frame(),
        new_map_same_coordinates,
    )

    assert moved_reward.components["new_block"] == 1.0
    assert moved_reward.components["new_map"] == 0.0
    assert map_reward.components["new_block"] == 1.0
    assert map_reward.components["new_map"] == 5.0


def test_exploration_rewards_visual_change_and_each_new_pokedex_bit(
    snapshot_factory: SnapshotFactory,
) -> None:
    initial = snapshot_factory()
    calculator = RewardCalculator("exploration")
    calculator.reset(initial)
    current = snapshot_factory(pokedex_owned=owned_with_bits(72, 73))

    reward = calculator.calculate(
        blank_frame(),
        np.full((3, 3), 5, dtype=np.uint8),
        current,
    )

    assert reward.components["visual_change"] == 1.0
    assert reward.components["pokedex_progress"] == 20.0


def test_advanced_rewards_battle_entry_but_not_battle_duration(
    snapshot_factory: SnapshotFactory,
) -> None:
    calculator = RewardCalculator("advanced")
    calculator.reset(snapshot_factory(battle_type=0))
    entered = snapshot_factory(battle_type=1, enemy_level=12)

    first = calculator.calculate(blank_frame(), blank_frame(), entered)
    second = calculator.calculate(blank_frame(), blank_frame(), entered)

    assert first.components["battle_entry"] == 12.0
    assert first.total == pytest.approx(12.0)
    assert second.components["battle_entry"] == 0.0
    assert second.total == pytest.approx(0.0)


def test_advanced_collection_does_not_double_count_same_event(
    snapshot_factory: SnapshotFactory,
) -> None:
    calculator = RewardCalculator("advanced")
    calculator.reset(snapshot_factory(party_count=1))
    captured = snapshot_factory(
        party_count=2,
        pokedex_owned=owned_with_bits(4),
    )

    reward = calculator.calculate(blank_frame(), blank_frame(), captured)

    assert reward.components["collection_progress"] == 50.0


def test_advanced_counts_multiple_new_pokedex_entries(
    snapshot_factory: SnapshotFactory,
) -> None:
    calculator = RewardCalculator("advanced")
    calculator.reset(snapshot_factory())
    captured = snapshot_factory(pokedex_owned=owned_with_bits(4, 81))

    reward = calculator.calculate(blank_frame(), blank_frame(), captured)

    assert reward.components["collection_progress"] == 100.0


def test_blackout_suppresses_visual_map_battle_and_capture_rewards(
    snapshot_factory: SnapshotFactory,
) -> None:
    for mode in ("exploration", "advanced"):
        calculator = RewardCalculator(mode)
        calculator.reset(snapshot_factory())
        blackout = snapshot_factory(
            map_id=99,
            block_x=99,
            battle_type=0xFF,
            party_count=2,
            pokedex_owned=owned_with_bits(1),
            enemy_level=50,
        )

        reward = calculator.calculate(
            blank_frame(),
            np.full((3, 3), 255, dtype=np.uint8),
            blackout,
        )

        assert reward.total == 0.0
        assert reward.components == {"blackout": 0.0}


def test_frame_shapes_must_match(snapshot_factory: SnapshotFactory) -> None:
    calculator = RewardCalculator("exploration")
    calculator.reset(snapshot_factory())

    with pytest.raises(ValueError, match="equal shapes"):
        calculator.calculate(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((3, 3), dtype=np.uint8),
            snapshot_factory(),
        )


def test_invalid_reward_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        RewardCalculator("invalid")
