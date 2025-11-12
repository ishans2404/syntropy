"""Convolutional building blocks inspired by EfficientNetV2."""

from __future__ import annotations

from typing import Optional

from tensorflow import keras  # type: ignore[attr-defined]


def efficient_2d_convblock(
    inputs,
    filters: int,
    kernel_size: int | tuple[int, int] = 3,
    strides: int | tuple[int, int] = 1,
    use_bn: bool = True,
    use_activation: bool = True,
    name: Optional[str] = None,
):
    """Depthwise-separable 2D convolution block with GELU activation."""

    x = keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding="same", use_bias=False, name=_fullname(name, "dw"))(inputs)
    if use_bn:
        x = keras.layers.BatchNormalization(name=_fullname(name, "dw_bn"))(x)
    if use_activation:
        x = keras.layers.Activation("gelu", name=_fullname(name, "dw_act"))(x)

    x = keras.layers.Conv2D(filters, 1, padding="same", use_bias=False, name=_fullname(name, "pw"))(x)
    if use_bn:
        x = keras.layers.BatchNormalization(name=_fullname(name, "pw_bn"))(x)
    if use_activation:
        x = keras.layers.Activation("gelu", name=_fullname(name, "pw_act"))(x)

    return x


def efficient_3d_convblock(
    inputs,
    filters: int,
    kernel_size: int | tuple[int, int, int] = 3,
    strides: int | tuple[int, int, int] = 1,
    use_bn: bool = True,
    use_activation: bool = True,
    name: Optional[str] = None,
):
    """Depthwise-separable 3D convolution block using grouped convolutions."""

    input_channels = inputs.shape[-1]
    if input_channels is None:
        raise ValueError("Input channel dimension must be known for efficient_3d_convblock.")

    x = keras.layers.Conv3D(
        filters=int(input_channels),
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        groups=int(input_channels),
        use_bias=False,
        name=_fullname(name, "dw"),
    )(inputs)

    if use_bn:
        x = keras.layers.BatchNormalization(name=_fullname(name, "dw_bn"))(x)
    if use_activation:
        x = keras.layers.Activation("gelu", name=_fullname(name, "dw_act"))(x)

    x = keras.layers.Conv3D(filters, 1, padding="same", use_bias=False, name=_fullname(name, "pw"))(x)
    if use_bn:
        x = keras.layers.BatchNormalization(name=_fullname(name, "pw_bn"))(x)
    if use_activation:
        x = keras.layers.Activation("gelu", name=_fullname(name, "pw_act"))(x)

    return x


def _fullname(root: Optional[str], suffix: str) -> Optional[str]:
    if root is None:
        return None
    return f"{root}_{suffix}"
