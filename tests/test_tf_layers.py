from __future__ import annotations

import tensorflow as tf

from axiom.tf.layers import AxialAttention2D, ChannelAttention2D, efficient_2d_convblock


def test_axial_attention_2d_preserves_shape():
    layer = AxialAttention2D(axis=0, num_heads=4)
    inputs = tf.random.normal([2, 16, 16, 32])
    outputs = layer(inputs, training=False)
    assert outputs.shape == inputs.shape


def test_channel_attention_2d_scales_channels():
    layer = ChannelAttention2D(reduction_ratio=8)
    inputs = tf.random.normal([2, 16, 16, 32])
    outputs = layer(inputs)
    assert outputs.shape == inputs.shape


def test_efficient_convblock_changes_channels():
    inputs = tf.keras.Input(shape=(32, 32, 16))
    outputs = efficient_2d_convblock(inputs, filters=24)
    model = tf.keras.Model(inputs, outputs)
    result = model(tf.random.normal([2, 32, 32, 16]))
    assert result.shape[-1] == 24
