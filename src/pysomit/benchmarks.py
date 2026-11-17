"""CRITIC comparison weights with explicit cost/benefit orientation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from ._validation import FloatArray, degeneracy_policy, directions, matrix, minmax


def critic_weights(
    acm: ArrayLike,
    types: Sequence[str],
    *,
    orientation: str = "preference",
    on_degenerate: str = "raise",
) -> FloatArray:
    """Return CRITIC weights using population standard deviation.

    The default aligns normalized values with preference before computing
    signed correlations. ``orientation='all_increasing'`` reproduces the
    direction convention in the original experiment script. Constant criteria
    are excluded from correlations and receive zero weight. If no conflict
    information remains, explicitly request ``on_degenerate='uniform'`` to
    distribute weight equally across varying criteria (or all if none vary).
    """
    if orientation not in ("preference", "all_increasing"):
        raise ValueError("orientation must be 'preference' or 'all_increasing'")
    degeneracy_policy(on_degenerate)
    acm = matrix(acm)
    benefit = directions(types, acm.shape[1])
    normalized, varying = minmax(acm)
    if orientation == "preference":
        normalized[:, ~benefit] = 1 - normalized[:, ~benefit]
    result = np.zeros(acm.shape[1])
    if varying.sum() == 1:
        result[varying] = 1
        return result
    if varying.any():
        informative = normalized[:, varying]
        centered = informative - informative.mean(axis=0)
        lengths = np.linalg.norm(centered, axis=0)
        standardized = centered / lengths
        correlations = np.clip(standardized.T @ standardized, -1, 1)
        np.fill_diagonal(correlations, 1)
        conflict = 1 - correlations
        conflict[conflict < 32 * np.finfo(float).eps] = 0
        information = informative.std(axis=0, ddof=0) * conflict.sum(axis=1)
        if information.sum() > 0:
            result[varying] = information / information.sum()
            return result
    if on_degenerate == "raise":
        raise ValueError("CRITIC has no conflict information; weights are undefined")
    eligible = varying if varying.any() else np.ones_like(varying)
    result[eligible] = 1 / eligible.sum()
    return result
