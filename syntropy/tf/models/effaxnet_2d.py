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
        if weights_path.lower() == "imagenet":
            msg = "Pretrained EffAxNet 2D weights are not bundled; provide a file path instead."
            raise ValueError(msg)
        model.load_weights(weights_path)
    else:
        msg = "`weights` must be a string or os.PathLike pointing to a weights file."
        raise TypeError(msg)


def _get_stem_config(input_size: int) -> dict:
    """Adaptive stem configuration based on input size."""
    if input_size <= 32:
        # Small images (MNIST, CIFAR): gentle downsampling
        return {
            "kernel_size": 3,
            "stride": 1,
            "use_multi_conv": False
        }
    elif input_size <= 64:
        # Medium-small images: moderate downsampling
        return {
            "kernel_size": 3,
            "stride": 2,
            "use_multi_conv": False
        }
    elif input_size <= 128:
        # Medium images: standard downsampling
        return {
            "kernel_size": 3,
            "stride": 2,
            "use_multi_conv": True  # Use 3x3 stack like ResNet-RS
        }
    else:
        # Large images (224+): aggressive downsampling
        return {
            "kernel_size": 4,
            "stride": 4,
            "use_multi_conv": False
        }


def _get_classifier_config(input_size: int, num_classes: int) -> dict:
    """Adaptive classifier configuration based on input size."""
    if input_size <= 32:
        # Lightweight classifier for small inputs
        return {
            "fc1_units": 128,
            "fc2_units": None,  # Skip second FC layer
            "dropout1": 0.2,
            "dropout2": None
        }
    elif input_size <= 64:
        # Medium classifier
        return {
            "fc1_units": 256,
            "fc2_units": 128,
            "dropout1": 0.3,
            "dropout2": 0.2
        }
    else:
        # Full classifier for large inputs
        return {
            "fc1_units": 512,
            "fc2_units": 256,
            "dropout1": 0.5,
            "dropout2": 0.4
        }


def _adaptive_stem(inputs, input_size: int) -> keras.layers.Layer:
    """Build adaptive stem based on input size."""
    config = _get_stem_config(input_size)
    
    if config["use_multi_conv"]:
        # ResNet-RS style: stack of 3x3 convs for better gradient flow
        x = keras.layers.Conv2D(
            32, 3, strides=2, padding="same", use_bias=False, name="stem_conv1"
        )(inputs)
        x = keras.layers.BatchNormalization(name="stem_bn1")(x)
        x = keras.layers.Activation("gelu", name="stem_activation1")(x)
        
        x = keras.layers.Conv2D(
            32, 3, strides=1, padding="same", use_bias=False, name="stem_conv2"
        )(x)
        x = keras.layers.BatchNormalization(name="stem_bn2")(x)
        x = keras.layers.Activation("gelu", name="stem_activation2")(x)
        
        x = keras.layers.Conv2D(
            32, 3, strides=1, padding="same", use_bias=False, name="stem_conv3"
        )(x)
    else:
        # Single conv stem
        padding = "same" if config["stride"] <= 2 else "valid"
        x = keras.layers.Conv2D(
            32, config["kernel_size"], strides=config["stride"], 
            padding=padding, use_bias=False, name="stem_conv"
        )(inputs)
    
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.Activation("gelu", name="stem_activation")(x)
    
    return x


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
    """Builds the EffAxNet 2D architecture with adaptive configuration."""

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
    input_size = input_shape[0]  # Assume square or use min(height, width)

    # Adaptive stem
    x = _adaptive_stem(inputs, input_size)

    # Stage 1
    x = efficient_2d_convblock(x, 48, kernel_size=3, strides=1, name="stage1")
    x = ChannelAttention2D(reduction_ratio=16, name="channel_attn_1")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, padding="same", name="pool_1")(x)

    # Stage 2 - Add axial attention only if spatial dims > 4x4
    x = efficient_2d_convblock(x, 64, kernel_size=3, strides=1, name="stage2")
    x = ChannelAttention2D(reduction_ratio=16, name="channel_attn_2")(x)
    
    # Calculate approximate spatial size after stem + pool1
    spatial_size_stage2 = input_size // (_get_stem_config(input_size)["stride"] * 2)
    if spatial_size_stage2 >= 4:  # Only apply if meaningful spatial dims
        x = AxialAttention2D(
            axis=0, num_heads=4, mlp_dim=128, dropout_rate=0.15, name="axial_attn_height"
        )(x)
    
    x = keras.layers.MaxPooling2D(2, strides=2, padding="same", name="pool_2")(x)

    # Stage 3
    x = efficient_2d_convblock(x, 128, kernel_size=3, strides=1, name="stage3")
    x = ChannelAttention2D(reduction_ratio=16, name="channel_attn_3")(x)
    
    spatial_size_stage3 = spatial_size_stage2 // 2
    if spatial_size_stage3 >= 4:
        x = AxialAttention2D(
            axis=1, num_heads=4, mlp_dim=256, dropout_rate=0.15, name="axial_attn_width"
        )(x)
    
    x = keras.layers.MaxPooling2D(2, strides=2, padding="same", name="pool_3")(x)

    # Stage 4
    features = efficient_2d_convblock(x, 256, kernel_size=3, strides=1, name="stage4")
    features = ChannelAttention2D(reduction_ratio=16, name="channel_attn_4")(features)

    # Adaptive classifier head
    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
        x = keras.layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)
        
        classifier_config = _get_classifier_config(input_size, num_classes)
        
        # First FC layer
        x = keras.layers.Dense(
            classifier_config["fc1_units"], activation="gelu", name="fc1"
        )(x)
        x = keras.layers.Dropout(classifier_config["dropout1"], name="dropout1")(x)
        
        # Optional second FC layer
        if classifier_config["fc2_units"] is not None:
            x = keras.layers.Dense(
                classifier_config["fc2_units"], activation="gelu", name="fc2"
            )(x)
            x = keras.layers.Dropout(classifier_config["dropout2"], name="dropout2")(x)
        
        outputs = keras.layers.Dense(
            num_classes, activation=classifier_activation, name="classification"
        )(x)
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
