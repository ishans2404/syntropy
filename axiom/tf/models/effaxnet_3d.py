"""TensorFlow reference implementation for the 3D Efficient Axial Network."""

from __future__ import annotations

from typing import Optional

from tensorflow import keras  # type: ignore[attr-defined]

from ..layers import AxialAttention3D, ChannelAttention3D, efficient_3d_convblock


def build_model(input_shape, num_classes: int, name: Optional[str] = None) -> keras.Model:
    """Builds the volumetric EffAxNet architecture."""

    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv3D(32, 4, strides=4, padding="valid", use_bias=False, name="stem_conv")(inputs)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.Activation("gelu", name="stem_activation")(x)

    x = efficient_3d_convblock(x, 48, kernel_size=3, strides=1, name="stage1")
    x = ChannelAttention3D(reduction_ratio=16, name="channel_attn_1")(x)
    x = keras.layers.MaxPooling3D(2, strides=2, padding="same", name="pool_1")(x)

    x = efficient_3d_convblock(x, 64, kernel_size=3, strides=1, name="stage2")
    x = ChannelAttention3D(reduction_ratio=16, name="channel_attn_2")(x)
    x = AxialAttention3D(axis=1, num_heads=4, mlp_dim=128, dropout_rate=0.1, name="axial_attn_height")(x)
    x = keras.layers.MaxPooling3D(2, strides=2, padding="same", name="pool_2")(x)

    x = efficient_3d_convblock(x, 128, kernel_size=3, strides=1, name="stage3")
    x = ChannelAttention3D(reduction_ratio=16, name="channel_attn_3")(x)
    x = AxialAttention3D(axis=0, num_heads=4, mlp_dim=256, dropout_rate=0.1, name="axial_attn_depth")(x)
    x = keras.layers.MaxPooling3D(2, strides=2, padding="same", name="pool_3")(x)

    x = efficient_3d_convblock(x, 256, kernel_size=3, strides=1, name="stage4")
    x = ChannelAttention3D(reduction_ratio=16, name="channel_attn_4")(x)

    x = keras.layers.GlobalAveragePooling3D(name="global_pool")(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
    x = keras.layers.Dense(512, activation="gelu", name="fc1", dtype="float32")(x)
    x = keras.layers.Dropout(0.5, name="dropout1")(x)
    x = keras.layers.Dense(256, activation="gelu", name="fc2", dtype="float32")(x)
    x = keras.layers.Dropout(0.3, name="dropout2")(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classification", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name=name or "axial_efficientnet_3d")
    return model
