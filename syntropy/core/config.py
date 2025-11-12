"""Dataclasses defining configuration for Efficient Axial Networks."""

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional, Tuple

FrameType = Literal["2d", "3d"]


@dataclass
class EffAxNetConfig:
    """Configuration describing an EffAxNet variant."""

    variant: FrameType = "2d"
    input_shape: Tuple[int, ...] = (128, 128, 3)
    num_classes: int = 2
    include_top: bool = True
    classifier_activation: Optional[str] = "softmax"
    pooling: Optional[Literal["avg", "max"]] = None
    weights: Optional[str] = None
    name: Optional[str] = None
    extra_kwargs: Dict[str, object] = field(default_factory=dict)

    def framework_key(self) -> str:
        """Returns a stable key for registry lookups."""

        frame = self.variant.lower()
        if frame not in {"2d", "3d"}:
            msg = f"Unsupported EffAxNet variant '{self.variant}'."
            raise ValueError(msg)
        return f"effaxnet_{frame}"
