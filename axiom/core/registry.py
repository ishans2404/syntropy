"""Simple registry for model factories keyed by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Dict


class ModelRegistry:
    """Keeps a mapping between model identifiers and build callables."""

    def __init__(self) -> None:
        self._registry: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str, builder: Callable[..., Any]) -> None:
        if name in self._registry:
            msg = f"A builder named '{name}' is already registered."
            raise ValueError(msg)
        self._registry[name] = builder

    def get(self, name: str) -> Callable[..., Any]:
        try:
            return self._registry[name]
        except KeyError as exc:
            msg = f"No model builder registered under '{name}'."
            raise KeyError(msg) from exc

    def available(self) -> Dict[str, Callable[..., Any]]:
        return dict(self._registry)
