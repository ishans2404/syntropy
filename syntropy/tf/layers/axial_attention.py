"""Axial attention layers for 2D and 3D feature tensors."""

from __future__ import annotations

from typing import Sequence, cast

import tensorflow as tf
from tensorflow import keras  # type: ignore[attr-defined]


class AxialAttention2D(keras.layers.Layer):
    """Self-attention applied along a single spatial axis of 2D feature maps."""

    def __init__(self, axis: int = 0, num_heads: int = 4, mlp_dim: int = 128, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if axis not in (0, 1):
            raise ValueError("axis must be 0 (height) or 1 (width) for AxialAttention2D")
        self.axis = axis
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined for AxialAttention2D.")
        self.channels = int(channels)
        if self.channels % self.num_heads != 0:
            raise ValueError(
                f"channels ({self.channels}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim = self.channels // self.num_heads
        self.qkv_dense = keras.layers.Dense(self.channels * 3, use_bias=False)
        self.proj_dense = keras.layers.Dense(self.channels, use_bias=False)
        
        super().build(input_shape)

    def _reshape_inputs(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]  # type: ignore[index]
        height = shape[1]  # type: ignore[index]
        width = shape[2]  # type: ignore[index]

        if self.axis == 0:
            x_reshaped = inputs
            seq_len = height
        else:
            x_reshaped = tf.reshape(inputs, (batch_size * height, width, self.channels))
            seq_len = width

        return x_reshaped, seq_len, batch_size, height, width

    def call(self, inputs, training=None):
        x_reshaped, seq_len, batch_size, height, width = self._reshape_inputs(inputs)
        x_norm = self.layer_norm(x_reshaped)
        qkv = self.qkv_dense(x_norm)
        qkv = tf.reshape(qkv, (-1, seq_len, self.num_heads, 3 * self.head_dim))
        qkv = tf.transpose(qkv, (0, 2, 1, 3))
        components = cast(Sequence[tf.Tensor], tf.split(qkv, num_or_size_splits=3, axis=-1))
        q, k, v = components[0], components[1], components[2]

        scale = tf.cast(tf.math.rsqrt(tf.cast(self.head_dim, tf.float32)), inputs.dtype)
        attn = tf.matmul(q, k, transpose_b=True) * scale
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn, training=training)

        out = tf.matmul(attn, v)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (-1, seq_len, self.channels))
        out = self.proj_dense(out)
        out = self.dropout(out, training=training)
        out = tf.reshape(out, (batch_size, height, width, self.channels))

        return inputs + out

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


class AxialAttention3D(keras.layers.Layer):
    """Axial attention layer for volumetric feature tensors."""

    def __init__(self, axis: int = 0, num_heads: int = 4, mlp_dim: int = 128, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0 (depth), 1 (height), or 2 (width) for AxialAttention3D")
        self.axis = axis
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout_rate
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = keras.layers.Dropout(self.dropout_rate)

    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("Channel dimension must be defined for AxialAttention3D.")
        self.channels = int(channels)
        if self.channels % self.num_heads != 0:
            raise ValueError(
                f"channels ({self.channels}) must be divisible by num_heads ({self.num_heads})"
            )
        self.head_dim = self.channels // self.num_heads
        self.qkv_dense = keras.layers.Dense(self.channels * 3, use_bias=False)
        self.proj_dense = keras.layers.Dense(self.channels, use_bias=False)
        super().build(input_shape)

    def call(self, inputs, training=None):
        shape = tf.shape(inputs)
        batch_size = shape[0]  # type: ignore[index]
        depth = shape[1]  # type: ignore[index]
        height = shape[2]  # type: ignore[index]
        width = shape[3]  # type: ignore[index]
        compute_dtype = inputs.dtype

        if self.axis == 0:
            x_reshaped = tf.transpose(inputs, (0, 2, 3, 1, 4))
            x_reshaped = tf.reshape(x_reshaped, (batch_size * height * width, depth, self.channels))
            seq_len = depth
            undo_shape = (batch_size, height, width, depth, self.channels)
            undo_permutation = (0, 3, 1, 2, 4)
        elif self.axis == 1:
            x_reshaped = tf.transpose(inputs, (0, 1, 3, 2, 4))
            x_reshaped = tf.reshape(x_reshaped, (batch_size * depth * width, height, self.channels))
            seq_len = height
            undo_shape = (batch_size, depth, width, height, self.channels)
            undo_permutation = (0, 1, 3, 2, 4)
        else:
            x_reshaped = tf.reshape(inputs, (batch_size * depth * height, width, self.channels))
            seq_len = width
            undo_shape = (batch_size, depth, height, width, self.channels)
            undo_permutation = None

        x_norm = self.layer_norm(x_reshaped)
        qkv = self.qkv_dense(x_norm)
        qkv = tf.reshape(qkv, (-1, seq_len, self.num_heads, 3 * self.head_dim))
        qkv = tf.transpose(qkv, (0, 2, 1, 3))
        components = cast(Sequence[tf.Tensor], tf.split(qkv, num_or_size_splits=3, axis=-1))
        q, k, v = components[0], components[1], components[2]

        scale = tf.cast(tf.math.rsqrt(tf.cast(self.head_dim, tf.float32)), compute_dtype)
        attn_scores = tf.matmul(q, k, transpose_b=True) * scale
        attn_weights = tf.nn.softmax(attn_scores, axis=-1)
        attn_weights = self.dropout(attn_weights, training=training)

        out = tf.matmul(attn_weights, v)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (-1, seq_len, self.channels))
        out = self.proj_dense(out)
        out = self.dropout(out, training=training)

        out = tf.reshape(out, undo_shape)
        if undo_permutation is not None:
            out = tf.transpose(out, undo_permutation)

        return inputs + out

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config
