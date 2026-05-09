from __future__ import annotations

import numpy as np


def power_normalize(values, power: float = 2.0) -> np.ndarray:
    """Apply a clipped power transform and normalize the vector.

    Negative values are clipped to zero before the power transform is applied.
    The transformed values are then divided by their sum so the output can be
    interpreted as a proportional distribution. If every clipped value is zero,
    this returns an all-zero vector with the same shape as the input.

    Args:
        values: Sequence or array of raw similarity values.
        power: Exponent to apply after clipping values to zero. Values greater
            than 1.0 emphasize larger dimensions and suppress smaller ones.

    Returns:
        A NumPy float64 array containing the power-transformed normalized
        values. The returned vector sums to 1.0 unless all clipped input values
        are zero, in which case it sums to 0.0.
    """
    vector = np.asarray(values, dtype=np.float64)
    transformed = np.power(np.maximum(vector, 0.0), power)
    denominator = float(transformed.sum())
    if denominator == 0.0:
        return np.zeros_like(transformed)
    return transformed / denominator
