from __future__ import annotations

import tensorflow as tf

from axiom.tf.models import effaxnet_2d, effaxnet_3d


def test_effaxnet_2d_output_shape():
    model = effaxnet_2d.build_model((64, 64, 3), num_classes=5)
    outputs = model(tf.random.normal([2, 64, 64, 3]))
    assert outputs.shape == (2, 5)


def test_effaxnet_3d_output_shape():
    model = effaxnet_3d.build_model((32, 32, 32, 1), num_classes=3)
    outputs = model(tf.random.normal([2, 32, 32, 32, 1]))
    assert outputs.shape == (2, 3)
