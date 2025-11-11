"""Channel attention blocks for 2D and 3D feature maps."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras  # type: ignore[attr-defined]


class ChannelAttention2D(keras.layers.Layer):
    """Squeeze-and-excitation style channel attention for 2D inputs."""

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined for ChannelAttention2D.")
        reduced_channels = max(int(channels) // self.reduction_ratio, 1)
        self.global_pool = keras.layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = keras.layers.Dense(reduced_channels, activation="gelu", use_bias=False)
        self.fc2 = keras.layers.Dense(int(channels), activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        squeeze = self.global_pool(inputs)
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        return inputs * excitation

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class ChannelAttention3D(keras.layers.Layer):
    """Channel attention for volumetric features."""

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined for ChannelAttention3D.")
        reduced_channels = max(int(channels) // self.reduction_ratio, 1)
        self.global_pool = keras.layers.GlobalAveragePooling3D(keepdims=True)
        self.fc1 = keras.layers.Dense(reduced_channels, activation="gelu", use_bias=False)
        self.fc2 = keras.layers.Dense(int(channels), activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        squeeze = self.global_pool(inputs)
        excitation = self.fc1(squeeze)
        excitation = self.fc2(excitation)
        return inputs * excitation

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config
