from __future__ import annotations

import numpy as np


def power_normalize(values, power: float = 2.0) -> np.ndarray:
    """Clip negatives, apply a power transform, and normalize to sum to 1.

    The transform emphasizes larger semantic similarity values while preserving
    each vector as a proportional distribution. If every clipped value is zero,
    the all-zero vector is returned.
    """
    vector = np.asarray(values, dtype=np.float64)
    transformed = np.power(np.maximum(vector, 0.0), power)
    denominator = float(transformed.sum())
    if denominator == 0.0:
        return np.zeros_like(transformed)
    return transformed / denominator
