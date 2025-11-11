from __future__ import annotations

from axiom.core.utils import compute_spatial_reduction


def test_compute_spatial_reduction():
    assert compute_spatial_reduction((32, 32), stride=2) == (16, 16)
    assert compute_spatial_reduction((3, 3, 3), stride=4) == (1, 1, 1)
