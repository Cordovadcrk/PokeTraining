"""Reward strategies shared by the exploration and advanced trainers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .memory import MemorySnapshot, count_new_pokedex_entries


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Total reward and named components for diagnostics."""

    total: float
    components: dict[str, float]


@dataclass(slots=True)
class RewardCalculator:
    """Stateful reward calculator with two notebook-compatible modes.

    ``exploration`` retains the visual-change, block, map and milestone intent.
    ``advanced`` rewards new maps, battle entry and captures. Battle reward is
    granted once on entry rather than every step, avoiding a stalling incentive.
    """

    mode: str
    visited_blocks: set[tuple[int, int, int]] = field(default_factory=set)
    visited_maps: set[int] = field(default_factory=set)
    previous: MemorySnapshot | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"exploration", "advanced"}:
            raise ValueError("mode must be 'exploration' or 'advanced'")

    def reset(self, initial: MemorySnapshot) -> None:
        """Reset episode-local novelty state from the initial game snapshot."""

        # The exploration notebook awarded the initial block once; retain that
        # behavior while correctly excluding the initial map from map novelty.
        self.visited_blocks = set() if self.mode == "exploration" else {initial.block_id}
        self.visited_maps = {initial.map_id}
        self.previous = initial

    def calculate(
        self,
        previous_frame: NDArray[np.uint8],
        current_frame: NDArray[np.uint8],
        snapshot: MemorySnapshot,
    ) -> RewardBreakdown:
        """Calculate reward for a transition and update internal state."""

        if self.previous is None:
            raise RuntimeError("reset() must be called before calculate()")
        if previous_frame.shape != current_frame.shape:
            raise ValueError("previous_frame and current_frame must have equal shapes")

        if snapshot.lost_battle:
            # A blackout can animate the screen or move the player to another
            # map. Neither side effect should be mistaken for progress.
            components = {"blackout": 0.0}
        elif self.mode == "exploration":
            components = self._exploration(previous_frame, current_frame, snapshot)
        else:
            components = self._advanced(snapshot)

        self.previous = snapshot
        self.visited_blocks.add(snapshot.block_id)
        self.visited_maps.add(snapshot.map_id)
        total = float(sum(components.values()))
        return RewardBreakdown(total=total, components=components)

    def _exploration(
        self,
        previous_frame: NDArray[np.uint8],
        current_frame: NDArray[np.uint8],
        snapshot: MemorySnapshot,
    ) -> dict[str, float]:
        """Reward visual activity, map-aware exploration and Pokédex progress."""

        assert self.previous is not None
        difference = float(
            np.mean(np.abs(current_frame.astype(np.float32) - previous_frame.astype(np.float32)))
        )
        visual_change = 1.0 if difference > 1.0 else difference * 0.1
        new_block = snapshot.block_id not in self.visited_blocks
        new_map = snapshot.map_id not in self.visited_maps
        new_pokedex = count_new_pokedex_entries(
            self.previous.pokedex_owned,
            snapshot.pokedex_owned,
        )
        return {
            "visual_change": visual_change,
            "new_block": 1.0 if new_block else 0.0,
            "new_map": 5.0 if new_map else 0.0,
            "pokedex_progress": 10.0 * new_pokedex,
            "stagnation": 0.0 if new_block else -0.05,
        }

    def _advanced(self, snapshot: MemorySnapshot) -> dict[str, float]:
        """Reward meaningful events without paying for battle duration."""

        assert self.previous is not None
        entered_battle = snapshot.in_battle and not self.previous.in_battle
        new_pokedex = count_new_pokedex_entries(
            self.previous.pokedex_owned,
            snapshot.pokedex_owned,
        )
        party_increased = snapshot.party_count > self.previous.party_count
        collection_events = max(new_pokedex, int(party_increased))
        return {
            "new_map": 10.0 if snapshot.map_id not in self.visited_maps else 0.0,
            "battle_entry": float(snapshot.enemy_level) if entered_battle else 0.0,
            # This is collection progress, not proof of a capture: gifts,
            # trades or evolutions can change the same WRAM signals.
            "collection_progress": 50.0 * collection_events,
        }
