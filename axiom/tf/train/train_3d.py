"""Simple training loop helpers for 3D EffAxNet models."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from tensorflow import keras  # type: ignore[attr-defined]


def train(
    model: keras.Model,
    train_dataset,
    epochs: int,
    steps_per_epoch: Optional[int] = None,
    validation_data=None,
    validation_steps: Optional[int] = None,
    optimizer: Optional[keras.optimizers.Optimizer] = None,
    loss: Optional[keras.losses.Loss] = None,
    metrics: Optional[Sequence[keras.metrics.Metric]] = None,
    callbacks: Optional[Iterable[keras.callbacks.Callback]] = None,
):
    """Compiles and fits a 3D EffAxNet model."""

    if optimizer is None:
        optimizer = keras.optimizers.Adam()
    if loss is None:
        loss = keras.losses.CategoricalCrossentropy()
    if metrics is None:
        metrics = (keras.metrics.CategoricalAccuracy(name="acc"),)

    model.compile(optimizer=optimizer, loss=loss, metrics=list(metrics))
    history = model.fit(
        train_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
        validation_steps=validation_steps,
        callbacks=list(callbacks or ()),
    )
    return history
