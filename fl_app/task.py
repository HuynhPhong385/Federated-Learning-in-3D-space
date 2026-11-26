"""Tiny logistic model used to exercise the strategy in the terminal.

Real MNIST / CIFAR partitions belong to phase-1 dataset work. This module
only needs a few local steps so client metrics (loss, accuracy) are real.
"""

from __future__ import annotations

import numpy as np

WEIGHTS = np.array([1.5, -0.8], dtype=np.float64)


def initial_model() -> list[np.ndarray]:
    return [np.zeros(2, dtype=np.float64), np.zeros(1, dtype=np.float64)]


def local_dataset(partition_id: int, num_examples: int) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic 2D points. Each partition is shifted so clients are not identical."""
    rng = np.random.default_rng(partition_id + 11)
    features = rng.normal(size=(num_examples, 2))
    features[:, 0] += partition_id * 0.2
    logits = features @ WEIGHTS + rng.normal(scale=0.25, size=num_examples)
    labels = (logits > 0).astype(np.float64)
    return features, labels


def _predict(weights: np.ndarray, bias: np.ndarray, features: np.ndarray) -> np.ndarray:
    scores = features @ weights + bias
    return 1.0 / (1.0 + np.exp(-np.clip(scores, -20.0, 20.0)))


def evaluate_model(
    weights: np.ndarray, bias: np.ndarray, features: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    probs = _predict(weights, bias, features)
    loss = float(-np.mean(labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8)))
    accuracy = float(np.mean((probs >= 0.5) == labels))
    return loss, accuracy


def train_local(
    arrays: list[np.ndarray],
    features: np.ndarray,
    labels: np.ndarray,
    lr: float,
    epochs: int,
) -> tuple[list[np.ndarray], float, float]:
    weights = np.array(arrays[0], dtype=np.float64, copy=True)
    bias = np.array(arrays[1], dtype=np.float64, copy=True)
    count = max(len(labels), 1)
    for _ in range(epochs):
        probs = _predict(weights, bias, features)
        error = probs - labels
        weights -= lr * (features.T @ error) / count
        bias -= lr * np.array([error.mean()])
    loss, accuracy = evaluate_model(weights, bias, features, labels)
    return [weights, bias], loss, accuracy
