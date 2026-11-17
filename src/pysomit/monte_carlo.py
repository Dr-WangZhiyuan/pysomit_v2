"""Seeded stochastic analysis of the full G-SOMIT/TOPSIS pipeline."""

from __future__ import annotations

import platform
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from numpy.typing import ArrayLike

from ._validation import FloatArray, integer, matrix, real_array
from ._version import __version__
from .models import DecisionMaker, DistributionSummary, decision_makers
from .ranking import topsis
from .weighting import SolverError, gsomit


class SimulationError(RuntimeError):
    """A simulation failed; no silently truncated result is returned."""


def _summary(samples: FloatArray, percentiles: tuple[float, float]) -> DistributionSummary:
    limits = real_array(percentiles, "percentiles")
    if (
        limits.shape != (2,)
        or not np.all(np.isfinite(limits))
        or not 0 <= limits[0] < limits[1] <= 100
    ):
        raise ValueError("percentiles must be two increasing values between 0 and 100")
    lower, upper = np.percentile(samples, limits, axis=0, method="linear")
    return DistributionSummary(
        samples.mean(axis=0),
        samples.std(axis=0, ddof=0),
        np.median(samples, axis=0),
        lower,
        upper,
        (float(limits[0]), float(limits[1])),
    )


@dataclass(frozen=True)
class MonteCarloResult:
    ranking_probabilities: FloatArray
    weight_samples: FloatArray
    score_samples: FloatArray
    rank_samples: FloatArray
    consensus_samples: FloatArray
    metadata: dict[str, Any]

    @property
    def n_requested(self) -> int:
        return self.metadata["n_simulations"]

    @property
    def n_completed(self) -> int:
        return self.weight_samples.shape[0]

    def weight_summary(self, percentiles: tuple[float, float] = (5, 95)) -> DistributionSummary:
        """Population moments and simulation uncertainty intervals for weights."""
        return _summary(self.weight_samples, percentiles)

    def score_summary(self, percentiles: tuple[float, float] = (5, 95)) -> DistributionSummary:
        """Population moments and simulation uncertainty intervals for scores."""
        return _summary(self.score_samples, percentiles)

    def __iter__(self) -> Iterator[FloatArray]:
        """Support the research script's (probabilities, weights, scores) unpacking."""
        yield self.ranking_probabilities
        yield self.weight_samples
        yield self.score_samples


class _MatrixSampler:
    def __init__(
        self,
        acm: FloatArray,
        noise: ArrayLike,
        bounds: tuple[ArrayLike, ArrayLike] | None,
        integer_criteria: Sequence[int],
    ):
        self.acm = acm

        def broadcast(value: ArrayLike, name: str) -> FloatArray:
            try:
                return np.broadcast_to(real_array(value, name), acm.shape).copy()
            except ValueError as exc:
                raise ValueError(f"{name} must broadcast to the shape of acm") from exc

        self.noise = broadcast(noise, "acm_noise")
        if not np.all(np.isfinite(self.noise)) or np.any((self.noise < 0) | (self.noise > 1)):
            raise ValueError("acm_noise must be finite and between 0 and 1")
        indices = tuple(integer(i, "integer criterion") for i in integer_criteria)
        if len(set(indices)) != len(indices) or any(i >= acm.shape[1] for i in indices):
            raise ValueError("integer_criteria must contain distinct valid column indices")
        self.integer_criteria = indices
        with np.errstate(over="ignore", invalid="ignore"):
            endpoint1 = acm * (1 - self.noise)
            endpoint2 = acm * (1 + self.noise)
        self.low = np.minimum(endpoint1, endpoint2)
        self.high = np.maximum(endpoint1, endpoint2)
        self.bounds = None
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError("acm_bounds must be a (lower, upper) pair")
            lower = broadcast(bounds[0], "lower bounds")
            upper = broadcast(bounds[1], "upper bounds")
            if np.any(np.isnan(lower) | np.isnan(upper) | (lower > upper)):
                raise ValueError("acm_bounds must be ordered and contain no NaN")
            if np.any((acm < lower) | (acm > upper)):
                raise ValueError("the baseline acm must lie within acm_bounds")
            self.low = np.maximum(self.low, lower)
            self.high = np.minimum(self.high, upper)
            self.bounds = (lower, upper)
        if not np.all(np.isfinite(self.low)) or not np.all(np.isfinite(self.high)):
            raise ValueError("perturbation endpoints exceed the finite numeric range")
        self.discrete_low = np.ceil(self.low[:, indices])
        self.discrete_high = np.floor(self.high[:, indices])
        if indices:
            selected = acm[:, indices]
            if np.any(selected != np.floor(selected)):
                raise ValueError("integer criteria must contain integer baseline values")
            if np.any(self.discrete_low > self.discrete_high):
                raise ValueError("an integer perturbation interval contains no integer")
            if np.any(np.abs(self.low[:, indices]) >= 2**53) or np.any(
                np.abs(self.high[:, indices]) >= 2**53
            ):
                raise ValueError("integer perturbations must remain below 2**53 in magnitude")

    def draw(self, rng: np.random.RandomState | np.random.Generator) -> FloatArray:
        if self.bounds is None and not self.integer_criteria:
            # This call order and arithmetic match the manuscript's random model.
            noise = rng.uniform(-self.noise, self.noise, self.acm.shape)
            return self.acm * (1 + noise)
        uniforms = rng.uniform(0, 1, self.acm.shape)
        result = self.low + (self.high - self.low) * uniforms
        if self.integer_criteria:
            idx = self.integer_criteria
            result[:, idx] = self.discrete_low + np.floor(
                uniforms[:, idx] * (self.discrete_high - self.discrete_low + 1)
            )
        return result


def _ratio_draw(
    ratio: float,
    rng: np.random.RandomState | np.random.Generator,
    sigma: float,
    low: float,
    high: float,
) -> float:
    epsilon = float(rng.normal(0, sigma))
    log_value = np.log(ratio) + epsilon
    if log_value <= np.log(low):
        return low
    if log_value >= np.log(high):
        return high
    # Preserve the legacy arithmetic for ordinary draws; avoid overflow for
    # large log-noise paired with an extremely small baseline ratio.
    value = ratio * np.exp(epsilon) if abs(epsilon) < 700 else np.exp(log_value)
    return float(np.clip(value, low, high))


def _rank_credits(ranks: FloatArray, counts: FloatArray) -> None:
    # A tie occupying slots p..q contributes 1/(q-p+1) to each occupied slot
    # for every tied alternative. This gives unbiased, doubly stochastic
    # rank probabilities and preserves the average ranks stored in samples.
    for rank in np.unique(ranks):
        alternatives = np.flatnonzero(ranks == rank)
        size = alternatives.size
        start = int(round(rank - (size - 1) / 2 - 1))
        counts[alternatives, start : start + size] += 1 / size


def monte_carlo_gsomit(
    dm_inputs: Sequence[DecisionMaker | Mapping[str, Any]],
    acm: ArrayLike,
    types: Sequence[str],
    n_simulations: int = 10000,
    comparison_noise: float = 0.25,
    acm_noise: ArrayLike = 0.10,
    seed: int | None = 42,
    *,
    rng: str = "legacy",
    comparison_bounds: tuple[float, float] = (1 / 9, 9),
    acm_bounds: tuple[ArrayLike, ArrayLike] | None = None,
    integer_criteria: Sequence[int] = (),
    tie_method: str = "average",
    solver: str = "auto",
    on_degenerate: str = "raise",
) -> MonteCarloResult:
    """Perturb inputs and recompute G-SOMIT Parts I-III and TOPSIS every run.

    Default noise matches the manuscript: multiplicative exp(N(0, 0.25^2))
    comparisons clipped to [1/9, 9], and independent uniform +/-10% ACM noise.
    ``rng='legacy'`` uses a local RandomState; ``rng='generator'`` uses a local
    default_rng. Neither changes NumPy's global random state.

    ``acm_noise`` may be a scalar, per-column vector, or matrix. If bounds are
    provided, draw uniformly from the intersection of each relative interval
    with those bounds. Integer columns are sampled uniformly over integers in
    that intersection. Bounds and integer restrictions change the noise model.

    Failed runs raise SimulationError with a 1-based run number. No draws are
    discarded. With average ties, probabilities distribute mass equally over
    tied rank slots; they are probabilities under uniform tie breaking.
    """
    n_simulations = integer(n_simulations, "n_simulations", minimum=1)
    acm = matrix(acm)
    m, n = acm.shape
    dms = decision_makers(dm_inputs, n)
    try:
        comparison_noise = float(comparison_noise)
    except (TypeError, ValueError) as exc:
        raise ValueError("comparison_noise must be finite and nonnegative") from exc
    if not np.isfinite(comparison_noise) or comparison_noise < 0:
        raise ValueError("comparison_noise must be finite and nonnegative")
    limits = real_array(comparison_bounds, "comparison_bounds")
    if limits.shape != (2,) or not np.all(np.isfinite(limits)) or not 0 < limits[0] < limits[1]:
        raise ValueError("comparison_bounds must be two finite positive increasing values")
    low, high = map(float, limits)
    for dm in dms:
        if any(not low <= r <= high for r in (*dm.comparisons.values(), dm.direct_hl)):
            raise ValueError("baseline comparison ratios must lie within comparison_bounds")
    sampler = _MatrixSampler(acm, acm_noise, acm_bounds, integer_criteria)
    if seed is not None:
        seed = integer(seed, "seed")
    if rng == "legacy":
        if seed is not None and seed >= 2**32:
            raise ValueError("legacy seed must be smaller than 2**32")
        random = np.random.RandomState(seed)
    elif rng == "generator":
        random = np.random.default_rng(seed)
    else:
        raise ValueError("rng must be 'legacy' or 'generator'")
    types = tuple(types)
    baseline = gsomit(acm, dms, solver=solver, on_degenerate=on_degenerate)
    topsis(acm, baseline.weights, types, tie_method=tie_method)
    weight_samples = np.empty((n_simulations, n))
    score_samples = np.empty((n_simulations, m))
    rank_samples = np.empty((n_simulations, m))
    consensus_samples = np.empty(n_simulations)
    counts = np.zeros((m, m))
    for simulation in range(n_simulations):
        try:
            perturbed_dms = [
                DecisionMaker(
                    dm.median_idx,
                    {
                        i: _ratio_draw(ratio, random, comparison_noise, low, high)
                        for i, ratio in dm.comparisons.items()
                    },
                    dm.highest_idx,
                    dm.lowest_idx,
                    _ratio_draw(dm.direct_hl, random, comparison_noise, low, high),
                    dm.name,
                )
                for dm in dms
            ]
            perturbed_acm = sampler.draw(random)
            result = gsomit(
                perturbed_acm, perturbed_dms, solver=solver, on_degenerate=on_degenerate
            )
            ranking = topsis(perturbed_acm, result.weights, types, tie_method=tie_method)
        except (SolverError, ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            raise SimulationError(
                f"simulation {simulation + 1} of {n_simulations} failed: {exc}"
            ) from exc
        weight_samples[simulation] = result.weights
        score_samples[simulation] = ranking.scores
        rank_samples[simulation] = ranking.ranks
        consensus_samples[simulation] = result.group.consensus_index
        _rank_credits(ranking.ranks, counts)
    metadata = {
        "n_simulations": n_simulations,
        "seed": seed,
        "rng": rng,
        "comparison_noise": comparison_noise,
        "comparison_bounds": [low, high],
        "acm_noise": sampler.noise.tolist(),
        "acm_bounds": None
        if sampler.bounds is None
        else [np.where(np.isfinite(b), b, None).tolist() for b in sampler.bounds],
        "integer_criteria": list(sampler.integer_criteria),
        "tie_method": tie_method,
        "solver": solver,
        "on_degenerate": on_degenerate,
        "versions": {
            "pysomit": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    return MonteCarloResult(
        counts / n_simulations,
        weight_samples,
        score_samples,
        rank_samples,
        consensus_samples,
        metadata,
    )
