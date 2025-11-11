"""PyTorch axial attention layers (placeholders for future releases)."""

from __future__ import annotations

import torch
from torch import nn


class AxialAttention2D(nn.Module):
    """Placeholder axial attention for 2D tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("PyTorch AxialAttention2D is planned for a future release.")


class AxialAttention3D(nn.Module):
    """Placeholder axial attention for 3D tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("PyTorch AxialAttention3D is planned for a future release.")
