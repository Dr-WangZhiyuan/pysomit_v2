"""Decision-maker inputs and transparent algorithm results."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType
from typing import Any

import numpy as np

from ._validation import FloatArray, integer


@dataclass(frozen=True)
class DecisionMaker:
    """One expert's elicitation; all criterion indices are zero-based.

    ``comparisons[i]`` is criterion i's importance relative to ``median_idx``.
    Supply every other criterion once. ``direct_hl`` compares ``highest_idx``
    with ``lowest_idx``. Positive ratios, including fraction strings, are accepted.
    Anchor identities are explicit and are not reselected during simulation.
    """

    median_idx: int
    comparisons: Mapping[int, float | str]
    highest_idx: int
    lowest_idx: int
    direct_hl: float | str
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.comparisons, Mapping) or not self.comparisons:
            raise ValueError("comparisons must map all nonmedian indices to positive ratios")
        n = len(self.comparisons) + 1
        for field in ("median_idx", "highest_idx", "lowest_idx"):
            idx = integer(getattr(self, field), field)
            if idx >= n:
                raise ValueError(f"{field} must be in [0, {n - 1}]")
            object.__setattr__(self, field, idx)
        if self.highest_idx == self.lowest_idx:
            raise ValueError("highest_idx and lowest_idx must differ")

        def ratio(value: Any) -> float:
            try:
                result = float(Fraction(value) if isinstance(value, str) else value)
            except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                raise ValueError("comparison ratios must be positive finite numbers") from exc
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(result) or result <= 0:
                raise ValueError("comparison ratios must be positive finite numbers")
            return result

        comparisons = {
            integer(key, "comparison index"): ratio(value)
            for key, value in self.comparisons.items()
        }
        if set(comparisons) != set(range(n)) - {self.median_idx}:
            raise ValueError("comparisons must contain every nonmedian criterion exactly once")
        if self.name is not None and not isinstance(self.name, str):
            raise ValueError("name must be a string or None")
        object.__setattr__(self, "comparisons", MappingProxyType(comparisons))
        object.__setattr__(self, "direct_hl", ratio(self.direct_hl))

    @property
    def n_criteria(self) -> int:
        return len(self.comparisons) + 1


def decision_makers(
    inputs: Sequence[DecisionMaker | Mapping[str, Any]], n: int
) -> tuple[DecisionMaker, ...]:
    result = []
    for value in inputs:
        if isinstance(value, DecisionMaker):
            dm = value
        elif isinstance(value, Mapping):
            try:
                dm = DecisionMaker(**value)
            except TypeError as exc:
                raise ValueError("invalid or missing decision-maker fields") from exc
        else:
            raise ValueError("each decision-maker must be a DecisionMaker or a mapping")
        if dm.n_criteria != n:
            raise ValueError("each decision-maker must compare the same number of criteria as acm")
        result.append(dm)
    if not result:
        raise ValueError("at least one decision-maker is required")
    return tuple(result)


@dataclass(frozen=True)
class SubjectiveResult:
    weights: FloatArray
    objective_value: float
    solver: str
    kkt_violation: float
    iterations: int


@dataclass(frozen=True)
class GroupResult:
    weights: FloatArray
    individual_weights: FloatArray
    consensus_index: float
    solutions: tuple[SubjectiveResult, ...]

    def __iter__(self) -> Iterator[Any]:
        """Support the research script's (group, individual, consensus) unpacking."""
        yield self.weights
        yield self.individual_weights
        yield self.consensus_index


@dataclass(frozen=True)
class GSOMITResult:
    weights: FloatArray
    group: GroupResult
    objective_weights: FloatArray


@dataclass(frozen=True)
class TopsisResult:
    scores: FloatArray
    ranks: FloatArray
    positive_ideal: FloatArray
    negative_ideal: FloatArray
    distance_to_positive: FloatArray
    distance_to_negative: FloatArray

    def __iter__(self) -> Iterator[FloatArray]:
        yield self.scores
        yield self.ranks


@dataclass(frozen=True)
class DistributionSummary:
    mean: FloatArray
    std: FloatArray
    median: FloatArray
    lower: FloatArray
    upper: FloatArray
    percentiles: tuple[float, float]
