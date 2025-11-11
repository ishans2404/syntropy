"""TensorFlow reference implementation for the 2D Efficient Axial Network."""

from __future__ import annotations

from os import PathLike
from typing import Optional, Tuple, Union

from tensorflow import keras  # type: ignore[attr-defined]

from ..layers import AxialAttention2D, ChannelAttention2D, efficient_2d_convblock

DEFAULT_INPUT_SHAPE_2D: Tuple[int, int, int] = (128, 128, 3)


def _construct_inputs(
    input_shape: Tuple[int, ...],
    input_tensor: Optional[keras.Tensor],
) -> keras.Tensor:
    """Normalises user provided `input_tensor`/`input_shape` into a Keras input."""

    if input_tensor is None:
        return keras.Input(shape=input_shape, name="input")

    if not keras.backend.is_keras_tensor(input_tensor):
        return keras.Input(tensor=input_tensor, shape=input_shape, name="input")

    return input_tensor


def _maybe_apply_weight_loading(model: keras.Model, weights: Optional[Union[str, PathLike]]) -> None:
    if weights is None:
        return

    if isinstance(weights, (str, PathLike)):
        weights_path = str(weights)
        if weights_path.lower() == "imagenet":  # Placeholder until pretrained weights exist.
            msg = "Pretrained EffAxNet 2D weights are not bundled; provide a file path instead."
            raise ValueError(msg)
        model.load_weights(weights_path)
    else:  # pragma: no cover - defensive programming, should be unreachable.
        msg = "`weights` must be a string or os.PathLike pointing to a weights file."
        raise TypeError(msg)


def build_model(
    input_shape,
    num_classes: int,
    name: Optional[str] = None,
    *,
    include_top: bool = True,
    input_tensor: Optional[keras.Tensor] = None,
    pooling: Optional[str] = None,
    classifier_activation: Optional[str] = "softmax",
    weights: Optional[Union[str, PathLike]] = None,
) -> keras.Model:
    """Builds the EffAxNet 2D architecture with optional Keras-style customisation."""

    if input_shape is None:
        input_shape = DEFAULT_INPUT_SHAPE_2D
    input_shape = tuple(input_shape)

    if include_top and num_classes is None:
        msg = "`num_classes` must be provided when `include_top=True`."
        raise ValueError(msg)

    if pooling not in {None, "avg", "max"}:
        msg = "`pooling` must be one of {None, 'avg', 'max'}."
        raise ValueError(msg)

    inputs = _construct_inputs(input_shape, input_tensor)

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

    features = efficient_2d_convblock(x, 256, kernel_size=3, strides=1, name="stage4")
    features = ChannelAttention2D(reduction_ratio=16, name="channel_attn_4")(features)

    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
        x = keras.layers.Dense(512, activation="gelu", name="fc1")(x)
        x = keras.layers.Dropout(0.5, name="dropout1")(x)
        x = keras.layers.Dense(256, activation="gelu", name="fc2")(x)
        x = keras.layers.Dropout(0.4, name="dropout2")(x)
        outputs = keras.layers.Dense(num_classes, activation=classifier_activation, name="classification")(x)
    else:
        if pooling == "avg":
            outputs = keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
        elif pooling == "max":
            outputs = keras.layers.GlobalMaxPooling2D(name="global_pool")(features)
        else:
            outputs = features

    model_inputs = keras.utils.get_source_inputs(inputs)
    model = keras.Model(model_inputs, outputs, name=name or "effaxnet_2d")

    _maybe_apply_weight_loading(model, weights)

    return model
