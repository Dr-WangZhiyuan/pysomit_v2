"""SOMIT Parts I-III and aggregation of independently solved expert weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ._validation import (
    FloatArray,
    degeneracy_policy,
    integer,
    matrix,
    minmax,
    weight_matrix,
    weights,
)
from .models import DecisionMaker, GroupResult, GSOMITResult, SubjectiveResult, decision_makers


class SolverError(RuntimeError):
    """The constrained subjective-weight problem did not yield a verified solution."""


def _residual_matrix(dm: DecisionMaker) -> FloatArray:
    n = dm.n_criteria
    residuals = np.zeros((n, n))
    for row, (criterion, ratio) in enumerate(dm.comparisons.items()):
        residuals[row, criterion] = -1
        residuals[row, dm.median_idx] = ratio
    residuals[-1, dm.highest_idx] = -1
    residuals[-1, dm.lowest_idx] = dm.direct_hl
    return residuals


def _checked_solution(candidate: FloatArray, quadratic: FloatArray) -> tuple[FloatArray, float]:
    if (
        not np.all(np.isfinite(candidate))
        or candidate.min() < -1e-9
        or abs(candidate.sum() - 1) > 1e-9
    ):
        raise SolverError("subjective solution violates the simplex constraints")
    result = np.maximum(candidate, 0)
    result /= result.sum()
    gradient = 2 * quadratic @ result
    free = result > 1e-10
    stationarity = gradient - gradient[free].mean()
    violation = float(np.max(np.abs(stationarity[free])))
    if np.any(~free):
        violation = max(violation, float(np.max(np.maximum(-stationarity[~free], 0))))
    if violation > 5e-7:
        raise SolverError(f"subjective solution fails the scaled KKT check ({violation:.3g})")
    return result, violation


def solve_subjective(
    dm: DecisionMaker | Mapping[str, Any], *, solver: str = "auto"
) -> SubjectiveResult:
    """Solve Part I and return weights with objective and optimality diagnostics.

    ``auto`` solves the equality-constrained quadratic system, verifies
    nonnegativity and the Karush-Kuhn-Tucker (KKT) conditions, and falls back
    to SLSQP if necessary. ``slsqp`` uses SciPy with an analytic gradient.
    Neither path returns an unchecked optimizer result.
    """
    if solver not in ("auto", "slsqp"):
        raise ValueError("solver must be 'auto' or 'slsqp'")
    if not isinstance(dm, DecisionMaker):
        if not isinstance(dm, Mapping):
            raise ValueError("dm must be a DecisionMaker or a mapping")
        try:
            dm = DecisionMaker(**dm)
        except TypeError as exc:
            raise ValueError("invalid or missing decision-maker fields") from exc
    residuals = _residual_matrix(dm)
    scaled = residuals / np.max(np.abs(residuals))
    quadratic = scaled.T @ scaled
    n = dm.n_criteria
    result = None
    if solver == "auto":
        kkt = np.zeros((n + 1, n + 1))
        kkt[:n, :n] = 2 * quadratic
        kkt[n, :n] = 1
        kkt[:n, n] = 1
        rhs = np.zeros(n + 1)
        rhs[-1] = 1
        try:
            result, violation = _checked_solution(np.linalg.solve(kkt, rhs)[:n], quadratic)
            used_solver, iterations = "kkt", 1
        except (np.linalg.LinAlgError, SolverError):
            pass
    if result is None:
        optimum = minimize(
            lambda w: float(w @ quadratic @ w),
            np.full(n, 1 / n),
            jac=lambda w: 2 * quadratic @ w,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={
                "type": "eq",
                "fun": lambda w: w.sum() - 1,
                "jac": lambda w: np.ones(n),
            },
            options={"ftol": 1e-14, "maxiter": 2000},
        )
        if not optimum.success:
            raise SolverError(f"SLSQP did not converge: {optimum.message}")
        result, violation = _checked_solution(optimum.x, quadratic)
        used_solver, iterations = "slsqp", int(optimum.nit)
    with np.errstate(over="ignore", invalid="ignore"):
        objective = float(np.sum((residuals @ result) ** 2))
    if not np.isfinite(objective):
        raise SolverError("comparison magnitudes exceed the numerical range of the objective")
    return SubjectiveResult(result, objective, used_solver, violation, iterations)


def somit_part1(
    comparisons: Mapping[int, float | str],
    median_idx: int,
    highest_idx: int,
    lowest_idx: int,
    direct_hl: float | str,
    n: int,
    *,
    solver: str = "auto",
) -> FloatArray:
    """Return one expert's subjective weights using the research-script signature."""
    n = integer(n, "n", minimum=2)
    dm = DecisionMaker(median_idx, comparisons, highest_idx, lowest_idx, direct_hl)
    if dm.n_criteria != n:
        raise ValueError("n must match the number of elicited criteria")
    return solve_subjective(dm, solver=solver).weights


def geometric_mean_weights(individual_weights: ArrayLike) -> FloatArray:
    """Normalize the geometric mean of normalized individual weight vectors.

    A zero is a veto. Disjoint supports raise an error instead of adding epsilon.
    """
    individual = weight_matrix(individual_weights)
    common = np.all(individual > 0, axis=0)
    if not np.any(common):
        raise ValueError("individual weight vectors have no common positive criterion")
    log_mean = np.mean(np.log(individual[:, common]), axis=0)
    result = np.zeros(individual.shape[1])
    result[common] = np.exp(log_mean - log_mean.max())
    return result / result.sum()


def consensus_index(individual_weights: ArrayLike) -> float:
    """Return 1 - mean(population standard deviation / mean).

    This index can be negative. A criterion assigned zero by everyone has zero
    dispersion and contributes zero to the mean coefficient of variation.
    """
    individual = weight_matrix(individual_weights)
    means = individual.mean(axis=0)
    cvs = np.zeros_like(means)
    np.divide(individual.std(axis=0, ddof=0), means, out=cvs, where=means > 0)
    return float(1 - cvs.mean())


def gsomit_part1(
    dm_inputs: Sequence[DecisionMaker | Mapping[str, Any]], n: int, *, solver: str = "auto"
) -> GroupResult:
    """Solve each expert separately, then aggregate by normalized geometric mean."""
    n = integer(n, "n", minimum=2)
    dms = decision_makers(dm_inputs, n)
    solutions = tuple(solve_subjective(dm, solver=solver) for dm in dms)
    individual = np.stack([solution.weights for solution in solutions])
    return GroupResult(
        geometric_mean_weights(individual), individual, consensus_index(individual), solutions
    )


def somit_part2(acm: ArrayLike, *, on_degenerate: str = "raise") -> FloatArray:
    """Objective weights from average absolute deviation from the column median.

    Constant criteria receive zero weight. An entirely constant matrix raises
    by default; ``on_degenerate='uniform'`` explicitly selects equal weights.
    The statistic is an average absolute deviation, not a median absolute deviation.
    """
    degeneracy_policy(on_degenerate)
    normalized, _ = minmax(matrix(acm))
    dispersion = np.mean(np.abs(normalized - np.median(normalized, axis=0)), axis=0)
    total = dispersion.sum()
    if total == 0:
        if on_degenerate == "raise":
            raise ValueError("acm has no varying criteria; objective weights are undefined")
        return np.full(dispersion.size, 1 / dispersion.size)
    return dispersion / total


def somit_part3(ws: ArrayLike, wo: ArrayLike) -> FloatArray:
    """Normalize the elementwise subjective/objective product without underflow."""
    subjective = weights(ws, "subjective weights")
    objective = weights(wo, "objective weights", n=subjective.size)
    common = (subjective > 0) & (objective > 0)
    if not np.any(common):
        raise ValueError("subjective and objective weights have no common positive criterion")
    log_product = np.log(subjective[common]) + np.log(objective[common])
    result = np.zeros_like(subjective)
    result[common] = np.exp(log_product - log_product.max())
    return result / result.sum()


def gsomit(
    acm: ArrayLike,
    dm_inputs: Sequence[DecisionMaker | Mapping[str, Any]],
    *,
    solver: str = "auto",
    on_degenerate: str = "raise",
) -> GSOMITResult:
    """Run all three weighting stages and retain their intermediate results."""
    acm = matrix(acm)
    group = gsomit_part1(dm_inputs, acm.shape[1], solver=solver)
    objective = somit_part2(acm, on_degenerate=on_degenerate)
    return GSOMITResult(somit_part3(group.weights, objective), group, objective)
