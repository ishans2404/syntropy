"""Factory helpers for constructing TensorFlow EffAxNet models."""

from __future__ import annotations

from os import PathLike
from typing import Callable, Dict, Literal, Optional, Tuple, Union

from tensorflow import keras  # type: ignore[attr-defined]

from syntropy.core.config import EffAxNetConfig
from syntropy.core.registry import ModelRegistry
from . import effaxnet_2d, effaxnet_3d

MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register("effaxnet_2d", effaxnet_2d.build_model)
MODEL_REGISTRY.register("effaxnet_3d", effaxnet_3d.build_model)

FrameType = Literal["2d", "3d"]
DEFAULT_INPUT_SHAPES: Dict[FrameType, Tuple[int, ...]] = {
    "2d": effaxnet_2d.DEFAULT_INPUT_SHAPE_2D,
    "3d": effaxnet_3d.DEFAULT_INPUT_SHAPE_3D,
}


def _normalise_weights_argument(
    include_weights: Optional[Union[str, PathLike, bool]],
    explicit_weights: Optional[Union[str, PathLike]],
    variant: FrameType,
) -> Optional[Union[str, PathLike]]:
    if explicit_weights is not None and include_weights not in {None, False}:
        msg = "Pass either `include_weights` or `weights`, not both."
        raise ValueError(msg)

    if explicit_weights is not None:
        return explicit_weights

    if include_weights in (None, False):
        return None

    if include_weights is True:
        msg = (
            "Pretrained EffAxNet %s weights are not bundled. Provide a file path or set the argument to None."
            % variant
        )
        raise ValueError(msg)

    return include_weights


def EffAxNetV1(
    variant: FrameType = "2d",
    include_weights: Optional[Union[str, PathLike, bool]] = None,
    *,
    include_top: bool = True,
    input_tensor: Optional[keras.Tensor] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    num_classes: int = 1000,
    classifier_activation: Optional[str] = "softmax",
    pooling: Optional[str] = None,
    name: Optional[str] = None,
    weights: Optional[Union[str, PathLike]] = None,
    **kwargs,
) -> keras.Model:
    """Keras-like constructor mirroring `tf.keras.applications` ergonomics for EffAxNet."""

    frame = variant.lower()
    if frame not in DEFAULT_INPUT_SHAPES:
        msg = f"Unsupported EffAxNet variant '{variant}'. Expected one of {list(DEFAULT_INPUT_SHAPES)}."
        raise ValueError(msg)

    resolved_shape = tuple(input_shape) if input_shape is not None else DEFAULT_INPUT_SHAPES[frame]  # type: ignore[index]
    builder = MODEL_REGISTRY.get(f"effaxnet_{frame}")

    resolved_weights = _normalise_weights_argument(include_weights, weights, frame)  # type: ignore[arg-type]

    return builder(
        resolved_shape,
        num_classes,
        name=name,
        include_top=include_top,
        input_tensor=input_tensor,
        pooling=pooling,
        classifier_activation=classifier_activation,
        weights=resolved_weights,
        **kwargs,
    )


def build_from_config(config: EffAxNetConfig):
    """Builds a TensorFlow model from a configuration object."""

    builder = MODEL_REGISTRY.get(config.framework_key())
    kwargs: Dict[str, object] = dict(config.extra_kwargs)
    kwargs.setdefault("include_top", config.include_top)
    kwargs.setdefault("classifier_activation", config.classifier_activation)
    kwargs.setdefault("pooling", config.pooling)
    if config.weights is not None:
        kwargs.setdefault("weights", config.weights)
    if config.name:
        kwargs.setdefault("name", config.name)
    return builder(config.input_shape, config.num_classes, **kwargs)


def available_models() -> Dict[str, Callable[..., object]]:
    """Returns the registered TensorFlow models."""

    return MODEL_REGISTRY.available()
