"""PyTorch channel attention placeholders."""

from __future__ import annotations

from torch import nn


class ChannelAttention2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("PyTorch ChannelAttention2D is planned for a future release.")


class ChannelAttention3D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError("PyTorch ChannelAttention3D is planned for a future release.")
