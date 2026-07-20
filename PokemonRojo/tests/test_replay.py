"""Tests for complete, episode-safe replay transitions."""

from __future__ import annotations

import numpy as np
import pytest

from poketraining.replay import ReplayBuffer


def transition(value: int) -> tuple[np.ndarray, np.ndarray]:
    observation = np.array([value], dtype=np.uint8)
    next_observation = np.array([100 + value], dtype=np.uint8)
    return observation, next_observation


def test_store_keeps_explicit_next_observation_and_copies_inputs() -> None:
    buffer = ReplayBuffer(1, (1,), rng=np.random.default_rng(0))
    observation, next_observation = transition(4)
    buffer.store(
        observation,
        action=2,
        reward=1.25,
        next_observation=next_observation,
        terminated=False,
        truncated=False,
    )
    observation[0] = 99
    next_observation[0] = 99

    batch = buffer.sample(1)

    assert batch.observations.tolist() == [[4]]
    assert batch.next_observations.tolist() == [[104]]
    assert batch.actions.tolist() == [2]
    assert batch.rewards.tolist() == pytest.approx([1.25])


def test_ring_wrap_preserves_complete_recent_transition_pairs() -> None:
    buffer = ReplayBuffer(2, (1,), rng=np.random.default_rng(0))
    for value in range(3):
        observation, next_observation = transition(value)
        buffer.store(
            observation,
            action=value,
            reward=float(value),
            next_observation=next_observation,
            terminated=False,
            truncated=False,
        )

    batch = buffer.sample(2)
    pairs = set(
        zip(
            batch.observations[:, 0].tolist(),
            batch.next_observations[:, 0].tolist(),
            strict=True,
        )
    )

    assert len(buffer) == 2
    assert buffer.position == 1
    assert pairs == {(1, 101), (2, 102)}


def test_dones_combines_termination_and_time_limit_truncation() -> None:
    buffer = ReplayBuffer(3, (1,), rng=np.random.default_rng(2))
    endings = [(False, False), (True, False), (False, True)]
    for action, (terminated, truncated) in enumerate(endings):
        observation, next_observation = transition(action)
        buffer.store(
            observation,
            action=action,
            reward=0.0,
            next_observation=next_observation,
            terminated=terminated,
            truncated=truncated,
        )

    batch = buffer.sample(3)
    done_by_action = dict(zip(batch.actions.tolist(), batch.dones.tolist(), strict=True))

    assert done_by_action == {0: False, 1: True, 2: True}


def test_sample_returns_copies_not_views() -> None:
    buffer = ReplayBuffer(1, (1,), rng=np.random.default_rng(0))
    observation, next_observation = transition(7)
    buffer.store(
        observation,
        action=0,
        reward=0.0,
        next_observation=next_observation,
        terminated=False,
        truncated=False,
    )
    first = buffer.sample(1)
    first.observations[0, 0] = 42

    assert buffer.sample(1).observations.tolist() == [[7]]


@pytest.mark.parametrize("batch_size", [0, -1])
def test_sample_rejects_nonpositive_batch_size(batch_size: int) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ReplayBuffer(1, (1,)).sample(batch_size)


def test_sample_rejects_more_items_than_stored() -> None:
    with pytest.raises(ValueError, match="Cannot sample"):
        ReplayBuffer(2, (1,)).sample(1)


@pytest.mark.parametrize("which", ["observation", "next_observation"])
def test_store_rejects_wrong_observation_shape(which: str) -> None:
    buffer = ReplayBuffer(2, (2,))
    observation = np.zeros((2,), dtype=np.uint8)
    next_observation = np.ones((2,), dtype=np.uint8)
    if which == "observation":
        observation = np.zeros((1,), dtype=np.uint8)
    else:
        next_observation = np.zeros((1,), dtype=np.uint8)

    with pytest.raises(ValueError, match=which):
        buffer.store(
            observation,
            action=0,
            reward=0.0,
            next_observation=next_observation,
            terminated=False,
            truncated=False,
        )
