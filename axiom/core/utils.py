"""Utility helpers shared across frameworks."""

from __future__ import annotations

from typing import Iterable, Tuple


def compute_spatial_reduction(input_shape: Iterable[int], stride: int) -> Tuple[int, ...]:
    """Returns the spatial shape after applying an integer stride."""

    dims = list(input_shape)
    if stride <= 0:
        raise ValueError("Stride must be positive.")
    return tuple(max(d // stride, 1) for d in dims)
