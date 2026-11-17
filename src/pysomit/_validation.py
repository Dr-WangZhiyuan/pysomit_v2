"""Shared validation and scale-stable normalization helpers."""

from __future__ import annotations

from collections.abc import Sequence
from operator import index

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def real_array(value: ArrayLike, name: str) -> FloatArray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real numbers")
    try:
        return np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc


def matrix(value: ArrayLike) -> FloatArray:
    result = real_array(value, "acm")
    if result.ndim != 2 or 0 in result.shape or not np.all(np.isfinite(result)):
        raise ValueError("acm must be a nonempty, finite, two-dimensional matrix")
    return result


def weights(value: ArrayLike, name: str = "weights", n: int | None = None) -> FloatArray:
    result = real_array(value, name)
    if result.ndim != 1 or not result.size or (n is not None and result.size != n):
        raise ValueError(f"{name} must be a nonempty vector matching the number of criteria")
    if not np.all(np.isfinite(result)) or np.any(result < 0) or not np.any(result > 0):
        raise ValueError(f"{name} must be finite, nonnegative, and have a positive sum")
    # Scaling before summation also accepts finite weights near float64's limits.
    result /= result.max()
    return result / result.sum()


def weight_matrix(value: ArrayLike) -> FloatArray:
    result = real_array(value, "individual_weights")
    if result.ndim != 2 or 0 in result.shape:
        raise ValueError("individual_weights must be a nonempty two-dimensional matrix")
    return np.stack([weights(row, "individual_weights") for row in result])


def directions(types: Sequence[str], n: int) -> NDArray[np.bool_]:
    if isinstance(types, str):
        raise ValueError("types must contain one 'min' or 'max' per criterion")
    try:
        values = tuple(types)
    except TypeError as exc:
        raise ValueError("types must contain one 'min' or 'max' per criterion") from exc
    if len(values) != n or any(t not in ("min", "max") for t in values):
        raise ValueError("types must contain one 'min' or 'max' per criterion")
    return np.array([t == "max" for t in values])


def degeneracy_policy(policy: str) -> None:
    if policy not in ("raise", "uniform"):
        raise ValueError("on_degenerate must be 'raise' or 'uniform'")


def minmax(acm: FloatArray) -> tuple[FloatArray, NDArray[np.bool_]]:
    # Scaling avoids overflow in max - min, and makes the constant-column test
    # independent of units. No absolute 1e-12 cutoff is applied to raw data.
    scale = np.max(np.abs(acm), axis=0)
    scaled = acm / np.where(scale > 0, scale, 1.0)
    low = scaled.min(axis=0)
    span = scaled.max(axis=0) - low
    varying = span > 0
    normalized = np.zeros_like(scaled)
    np.divide(scaled - low, span, out=normalized, where=varying)
    return normalized, varying
