from __future__ import annotations

import pytest

from syntropy.torch.layers import AxialAttention2D, AxialAttention3D, ChannelAttention2D, ChannelAttention3D


def test_torch_layer_placeholders_raise():
    for cls in (AxialAttention2D, AxialAttention3D, ChannelAttention2D, ChannelAttention3D):
        with pytest.raises(NotImplementedError):
            cls()
