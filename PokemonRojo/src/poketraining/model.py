"""TensorFlow implementation of the notebook's Deep Q-Network."""

from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DQN(keras.Model):
    """Convolutional Q-network equivalent to the original notebook model."""

    def __init__(self, num_actions: int) -> None:
        if num_actions <= 0:
            raise ValueError("num_actions must be greater than zero")
        super().__init__(name="dqn")
        self.conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")
        self.conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")
        self.conv3 = layers.Conv2D(64, 3, strides=1, activation="relu")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(512, activation="relu")
        self.output_layer = layers.Dense(num_actions)

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
    ) -> tf.Tensor:
        """Return one Q-value per action for each stacked observation."""

        del training  # The architecture has no mode-dependent layers.
        x = tf.cast(inputs, tf.float32) / 255.0
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return self.output_layer(x)


def create_synchronized_models(
    num_actions: int,
    observation_shape: Sequence[int],
) -> tuple[DQN, DQN]:
    """Build online and target networks, then synchronize their weights."""

    shape = tuple(int(size) for size in observation_shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("observation_shape must contain three positive dimensions")
    if shape[0] < 36 or shape[1] < 36:
        raise ValueError("observation height and width must be at least 36 for the DQN")
    online = DQN(num_actions)
    target = DQN(num_actions)
    dummy = tf.zeros((1, *shape), dtype=tf.uint8)
    online(dummy, training=False)
    target(dummy, training=False)
    target.set_weights(online.get_weights())
    return online, target
