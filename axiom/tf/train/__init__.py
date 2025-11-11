"""Training utilities for TensorFlow EffAxNet models."""

from .train_2d import train as train_2d
from .train_3d import train as train_3d

__all__ = ["train_2d", "train_3d"]
