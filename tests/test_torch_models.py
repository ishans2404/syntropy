from __future__ import annotations

import pytest

from axiom.torch.models import effaxnet_2d, effaxnet_3d


def test_torch_model_placeholders_raise():
    with pytest.raises(NotImplementedError):
        effaxnet_2d.build_model()
    with pytest.raises(NotImplementedError):
        effaxnet_3d.build_model()
