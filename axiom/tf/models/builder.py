"""Factory helpers for constructing TensorFlow EffAxNet models."""

from __future__ import annotations

from typing import Callable, Dict

from ..core.config import EffAxNetConfig
from ..core.registry import ModelRegistry
from . import effaxnet_2d, effaxnet_3d

MODEL_REGISTRY = ModelRegistry()
MODEL_REGISTRY.register("effaxnet_2d", effaxnet_2d.build_model)
MODEL_REGISTRY.register("effaxnet_3d", effaxnet_3d.build_model)


def build_from_config(config: EffAxNetConfig):
    """Builds a TensorFlow model from a configuration object."""

    builder = MODEL_REGISTRY.get(config.framework_key())
    kwargs: Dict[str, object] = dict(config.extra_kwargs)
    if config.name:
        kwargs.setdefault("name", config.name)
    return builder(config.input_shape, config.num_classes, **kwargs)


def available_models() -> Dict[str, Callable[..., object]]:
    """Returns the registered TensorFlow models."""

    return MODEL_REGISTRY.available()
