"""Vector-normalized TOPSIS with explicit treatment of ties."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from ._validation import FloatArray, directions, matrix, weights
from .models import TopsisResult


def _row_norm(values: FloatArray) -> FloatArray:
    scale = np.max(np.abs(values), axis=1)
    scaled = values / np.where(scale > 0, scale, 1)[:, None]
    return scale * np.linalg.norm(scaled, axis=1)


def topsis(
    acm: ArrayLike,
    criterion_weights: ArrayLike,
    types: Sequence[str],
    *,
    tie_method: str = "average",
) -> TopsisResult:
    """Rank alternatives by descending TOPSIS closeness to the positive ideal.

    ``types`` is a sequence of ``'min'`` (cost) and ``'max'`` (benefit).
    Ranks are 1-based, with average ranks for exactly equal scores by default.
    ``tie_method='ordinal'`` explicitly breaks ties in input-row order.
    If both ideal distances vanish, a neutral score of 0.5 is returned.
    """
    if tie_method not in ("average", "ordinal"):
        raise ValueError("tie_method must be 'average' or 'ordinal'")
    acm = matrix(acm)
    w = weights(criterion_weights, n=acm.shape[1])
    benefit = directions(types, acm.shape[1])
    scale = np.max(np.abs(acm), axis=0)
    normalized = acm / np.where(scale > 0, scale, 1)
    norms = np.linalg.norm(normalized, axis=0)
    normalized /= np.where(norms > 0, norms, 1)
    weighted = normalized * w
    maximum, minimum = weighted.max(axis=0), weighted.min(axis=0)
    positive = np.where(benefit, maximum, minimum)
    negative = np.where(benefit, minimum, maximum)
    d_plus = _row_norm(weighted - positive)
    d_minus = _row_norm(weighted - negative)
    total = d_plus + d_minus
    scores = np.full(acm.shape[0], 0.5)
    np.divide(d_minus, total, out=scores, where=total > 0)
    ranks = rankdata(-scores, method=tie_method).astype(float)
    return TopsisResult(scores, ranks, positive, negative, d_plus, d_minus)
