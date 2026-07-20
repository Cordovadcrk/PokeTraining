"""Experience replay storage with explicit episode-safe transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

UInt8Array = NDArray[np.uint8]
Int32Array = NDArray[np.int32]
Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    """A sampled mini-batch of complete replay transitions."""

    observations: UInt8Array
    actions: Int32Array
    rewards: Float32Array
    next_observations: UInt8Array
    terminated: BoolArray
    truncated: BoolArray

    @property
    def dones(self) -> BoolArray:
        """Combine environment termination and time-limit truncation."""

        return np.logical_or(self.terminated, self.truncated)


class ReplayBuffer:
    """Fixed-size circular replay buffer.

    Both observations are stored explicitly. This costs more memory than the
    notebook's adjacency shortcut, but prevents transitions from crossing
    episode boundaries or becoming invalid when the ring wraps.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: Sequence[int],
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        shape = tuple(int(size) for size in observation_shape)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("observation_shape must contain positive dimensions")

        self.capacity = capacity
        self.observation_shape = shape
        self._rng = rng if rng is not None else np.random.default_rng()
        self._position = 0
        self._size = 0

        self._observations = np.empty((capacity, *shape), dtype=np.uint8)
        self._next_observations = np.empty((capacity, *shape), dtype=np.uint8)
        self._actions = np.empty(capacity, dtype=np.int32)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.bool_)
        self._truncated = np.empty(capacity, dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    @property
    def position(self) -> int:
        """Return the index that will be overwritten by the next insertion."""

        return self._position

    def store(
        self,
        observation: UInt8Array,
        action: int,
        reward: float,
        next_observation: UInt8Array,
        *,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Insert one complete transition and advance the circular pointer."""

        if observation.shape != self.observation_shape:
            raise ValueError(
                f"observation has shape {observation.shape}; expected {self.observation_shape}"
            )
        if next_observation.shape != self.observation_shape:
            raise ValueError(
                f"next_observation has shape {next_observation.shape}; "
                f"expected {self.observation_shape}"
            )

        index = self._position
        self._observations[index] = observation
        self._next_observations[index] = next_observation
        self._actions[index] = action
        self._rewards[index] = reward
        self._terminated[index] = terminated
        self._truncated[index] = truncated

        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample distinct stored transitions without relying on adjacency."""

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {self._size}"
            )
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return TransitionBatch(
            observations=self._observations[indices].copy(),
            actions=self._actions[indices].copy(),
            rewards=self._rewards[indices].copy(),
            next_observations=self._next_observations[indices].copy(),
            terminated=self._terminated[indices].copy(),
            truncated=self._truncated[indices].copy(),
        )
