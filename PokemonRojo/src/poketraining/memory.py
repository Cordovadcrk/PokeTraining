"""Verified Pokémon Red WRAM addresses and immutable memory snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class WramAddress(IntEnum):
    """Addresses used by the international Pokémon Red memory layout."""

    CURRENT_MAP = 0xD35E
    BLOCK_Y = 0xD363
    BLOCK_X = 0xD364
    TILESET = 0xD367
    IS_IN_BATTLE = 0xD057
    PARTY_COUNT = 0xD163
    POKEDEX_OWNED_START = 0xD2F7
    ENEMY_LEVEL = 0xCFF3


POKEDEX_OWNED_LENGTH = 19
POKEDEX_SPECIES_COUNT = 151
_POKEDEX_LAST_BYTE_MASK = (1 << (POKEDEX_SPECIES_COUNT % 8)) - 1


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    """Small, testable view of game memory needed by reward functions."""

    map_id: int
    block_y: int
    block_x: int
    tileset_id: int
    battle_type: int
    party_count: int
    pokedex_owned: bytes
    enemy_level: int

    def __post_init__(self) -> None:
        if len(self.pokedex_owned) != POKEDEX_OWNED_LENGTH:
            raise ValueError(f"pokedex_owned must contain {POKEDEX_OWNED_LENGTH} bytes")

    @property
    def in_battle(self) -> bool:
        """Return whether a wild or trainer battle is active."""

        return self.battle_type in (1, 2)

    @property
    def lost_battle(self) -> bool:
        """Return whether the game reports a battle loss/blackout."""

        return self.battle_type == 0xFF

    @property
    def block_id(self) -> tuple[int, int, int]:
        """Return a map-aware block identifier."""

        return (self.map_id, self.block_x, self.block_y)


def count_new_pokedex_entries(previous: bytes, current: bytes) -> int:
    """Count bits that changed from not-owned to owned between snapshots."""

    if len(previous) != POKEDEX_OWNED_LENGTH or len(current) != POKEDEX_OWNED_LENGTH:
        raise ValueError(f"Pokédex bitfields must contain {POKEDEX_OWNED_LENGTH} bytes")
    new_entries = 0
    for index, (old, new) in enumerate(zip(previous, current, strict=True)):
        new_bits = new & ~old & 0xFF
        if index == POKEDEX_OWNED_LENGTH - 1:
            new_bits &= _POKEDEX_LAST_BYTE_MASK
        new_entries += new_bits.bit_count()
    return new_entries
