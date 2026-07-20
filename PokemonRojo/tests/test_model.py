"""Small TensorFlow checks for model shape and target synchronization."""

from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
model_module = pytest.importorskip("poketraining.model")
DQN = model_module.DQN
create_synchronized_models = model_module.create_synchronized_models


def test_dqn_forward_returns_one_q_value_per_action() -> None:
    tf.keras.utils.set_random_seed(7)
    model = DQN(num_actions=6)
    observations = tf.zeros((2, 84, 84, 4), dtype=tf.uint8)

    output = model(observations, training=False)

    assert tuple(output.shape) == (2, 6)
    assert output.dtype == tf.float32
    assert bool(tf.reduce_all(tf.math.is_finite(output)))


def test_online_and_target_models_start_synchronized_but_independent() -> None:
    tf.keras.utils.set_random_seed(11)
    online, target = create_synchronized_models(6, (84, 84, 4))

    for online_weight, target_weight in zip(
        online.get_weights(),
        target.get_weights(),
        strict=True,
    ):
        np.testing.assert_array_equal(online_weight, target_weight)

    first_online = online.trainable_variables[0]
    first_target_before = target.trainable_variables[0].numpy().copy()
    first_online.assign_add(tf.ones_like(first_online))

    np.testing.assert_array_equal(
        target.trainable_variables[0].numpy(),
        first_target_before,
    )
    assert not np.array_equal(
        online.trainable_variables[0].numpy(),
        target.trainable_variables[0].numpy(),
    )


def test_model_factory_rejects_invalid_shape_or_action_count() -> None:
    with pytest.raises(ValueError, match="num_actions"):
        DQN(0)
    with pytest.raises(ValueError, match="observation_shape"):
        create_synchronized_models(6, (84, 0, 4))
    with pytest.raises(ValueError, match="at least 36"):
        create_synchronized_models(6, (35, 84, 4))
