from __future__ import annotations

import tensorflow as tf

from axiom.tf.models import EffAxNetV1, effaxnet_2d, effaxnet_3d


def test_effaxnet_2d_output_shape():
    model = effaxnet_2d.build_model((64, 64, 3), num_classes=5)
    outputs = model(tf.random.normal([2, 64, 64, 3]))
    assert outputs.shape == (2, 5)


def test_effaxnet_3d_output_shape():
    model = effaxnet_3d.build_model((32, 32, 32, 1), num_classes=3)
    outputs = model(tf.random.normal([2, 32, 32, 32, 1]))
    assert outputs.shape == (2, 3)


def test_effaxnet_v1_without_top_avg_pool(tmp_path):
    model = EffAxNetV1(
        variant="2d",
        include_top=False,
        pooling="avg",
        input_shape=(64, 64, 3),
        num_classes=5,
    )
    outputs = model(tf.random.normal([2, 64, 64, 3]))
    assert outputs.shape == (2, 256)

    weights_path = tmp_path / "effaxnet2d-transfer.weights.h5"
    model.save_weights(weights_path)

    restored = EffAxNetV1(
        variant="2d",
        include_top=False,
        pooling="avg",
        input_shape=(64, 64, 3),
        num_classes=5,
        include_weights=weights_path,
    )
    restored_outputs = restored(tf.random.normal([1, 64, 64, 3]))
    assert restored_outputs.shape == (1, 256)


def test_effaxnet_v1_3d_variant_pooling():
    model = EffAxNetV1(
        variant="3d",
        include_top=False,
        pooling="avg",
        input_shape=(32, 32, 32, 1),
        num_classes=3,
    )
    outputs = model(tf.random.normal([1, 32, 32, 32, 1]))
    assert outputs.shape == (1, 256)
