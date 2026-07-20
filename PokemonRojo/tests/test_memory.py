"""Tests for verified WRAM snapshots and Pokédex bitfields."""

from __future__ import annotations

import pytest

from poketraining.memory import (
    POKEDEX_OWNED_LENGTH,
    MemorySnapshot,
    count_new_pokedex_entries,
)


def make_snapshot(*, battle_type: int = 0, owned: bytes | None = None) -> MemorySnapshot:
    return MemorySnapshot(
        map_id=7,
        block_y=11,
        block_x=13,
        tileset_id=2,
        battle_type=battle_type,
        party_count=3,
        pokedex_owned=owned or bytes(POKEDEX_OWNED_LENGTH),
        enemy_level=9,
    )


@pytest.mark.parametrize("battle_type", [1, 2])
def test_battle_types_are_active(battle_type: int) -> None:
    snapshot = make_snapshot(battle_type=battle_type)

    assert snapshot.in_battle is True
    assert snapshot.lost_battle is False


@pytest.mark.parametrize("battle_type", [0, 3, 0xFE])
def test_non_battle_values_are_inactive(battle_type: int) -> None:
    snapshot = make_snapshot(battle_type=battle_type)

    assert snapshot.in_battle is False
    assert snapshot.lost_battle is False


def test_blackout_marker_is_loss_but_not_active_battle() -> None:
    snapshot = make_snapshot(battle_type=0xFF)

    assert snapshot.lost_battle is True
    assert snapshot.in_battle is False


def test_block_id_is_map_aware_and_orders_x_before_y() -> None:
    assert make_snapshot().block_id == (7, 13, 11)


def test_snapshot_requires_complete_owned_pokedex() -> None:
    with pytest.raises(ValueError, match="19 bytes"):
        make_snapshot(owned=bytes(POKEDEX_OWNED_LENGTH - 1))


def test_new_pokedex_bits_count_only_zero_to_one_transitions() -> None:
    previous = bytearray(POKEDEX_OWNED_LENGTH)
    previous[0] = 0b0000_0011
    previous[9] = 0b1000_0000
    current = previous.copy()
    current[0] = 0b0000_0101  # One new bit and one removed bit.
    current[9] = 0b1000_0011  # Two additional bits at D300.
    current[10] = 0b0000_1000  # One additional bit at D301.

    assert count_new_pokedex_entries(bytes(previous), bytes(current)) == 4


def test_removed_pokedex_bits_do_not_count_as_progress() -> None:
    previous = bytes([0xFF]) + bytes(POKEDEX_OWNED_LENGTH - 1)
    current = bytes(POKEDEX_OWNED_LENGTH)

    assert count_new_pokedex_entries(previous, current) == 0


def test_unused_padding_bit_after_species_151_is_ignored() -> None:
    previous = bytes(POKEDEX_OWNED_LENGTH)
    mew_and_padding = bytearray(POKEDEX_OWNED_LENGTH)
    mew_and_padding[-1] = 0b1100_0000

    assert count_new_pokedex_entries(previous, bytes(mew_and_padding)) == 1


@pytest.mark.parametrize(
    "previous,current",
    [
        (b"", bytes(POKEDEX_OWNED_LENGTH)),
        (bytes(POKEDEX_OWNED_LENGTH), b""),
    ],
)
def test_pokedex_bitfields_require_exact_length(previous: bytes, current: bytes) -> None:
    with pytest.raises(ValueError, match="19 bytes"):
        count_new_pokedex_entries(previous, current)
