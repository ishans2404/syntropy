"""TensorFlow reference implementation for the 2D Efficient Axial Network."""

from __future__ import annotations

from typing import Optional

from tensorflow import keras  # type: ignore[attr-defined]

from ..layers import AxialAttention2D, ChannelAttention2D, efficient_2d_convblock


def build_model(input_shape, num_classes: int, name: Optional[str] = None) -> keras.Model:
    """Builds the EffAxNet 2D architecture."""

    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, 4, strides=4, padding="valid", use_bias=False, name="stem_conv")(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.Activation("gelu", name="stem_activation")(x)

    x = efficient_2d_convblock(x, 48, kernel_size=3, strides=1, name="stage1")
    x = ChannelAttention2D(reduction_ratio=16, name="channel_attn_1")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, padding="same", name="pool_1")(x)

    x = efficient_2d_convblock(x, 64, kernel_size=3, strides=1, name="stage2")
    x = ChannelAttention2D(reduction_ratio=16, name="channel_attn_2")(x)
    x = AxialAttention2D(axis=0, num_heads=4, mlp_dim=128, dropout_rate=0.15, name="axial_attn_height")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, padding="same", name="pool_2")(x)

    x = efficient_2d_convblock(x, 128, kernel_size=3, strides=1, name="stage3")
    x = ChannelAttention2D(reduction_ratio=16, name="channel_attn_3")(x)
    x = AxialAttention2D(axis=1, num_heads=4, mlp_dim=256, dropout_rate=0.15, name="axial_attn_width")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, padding="same", name="pool_3")(x)

    x = efficient_2d_convblock(x, 256, kernel_size=3, strides=1, name="stage4")
    x = ChannelAttention2D(reduction_ratio=16, name="channel_attn_4")(x)

    x = keras.layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
    x = keras.layers.Dense(512, activation="gelu", name="fc1")(x)
    x = keras.layers.Dropout(0.5, name="dropout1")(x)
    x = keras.layers.Dense(256, activation="gelu", name="fc2")(x)
    x = keras.layers.Dropout(0.4, name="dropout2")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classification")(x)

    model = keras.Model(inputs, outputs, name=name or "axial_efficientnet_2d")
    return model
