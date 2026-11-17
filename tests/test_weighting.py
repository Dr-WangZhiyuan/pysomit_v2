import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pysomit import (
    DecisionMaker,
    SolverError,
    consensus_index,
    geometric_mean_weights,
    gsomit_part1,
    solve_subjective,
    somit_i,
    somit_ii,
    somit_iii,
    somit_part1,
    somit_part2,
    somit_part3,
)


@pytest.mark.parametrize("solver", ["auto", "slsqp"])
def test_consistent_ratios_recover_known_weights(solver):
    dm = DecisionMaker(1, {0: 2, 2: "1/3"}, 0, 2, 6)
    result = solve_subjective(dm, solver=solver)
    assert_allclose(result.weights, [0.6, 0.3, 0.1], atol=2e-7, rtol=0)
    assert result.objective_value < 1e-12
    assert result.kkt_violation < 5e-7
    assert result.iterations >= 1


def test_inconsistent_ratios_have_known_quadratic_optimum():
    # Eliminating w2 from this problem yields 22*w0 + 9*w1 = 14 and
    # 18*w0 + 61*w1 = 22. This oracle does not use the package's solver.
    result = solve_subjective(DecisionMaker(1, {0: 3, 2: 0.5}, 0, 2, 2))
    assert_allclose(result.weights, np.array([164, 58, 73]) / 295, atol=1e-14)
    assert_allclose(result.objective_value, 2360 / 87025, atol=1e-14)
    assert result.solver == "kkt"


def test_median_can_also_be_an_anchor():
    dm = DecisionMaker(0, {1: 0.5, 2: "1/6"}, 0, 2, 6)
    assert_allclose(solve_subjective(dm).weights, [0.6, 0.3, 0.1], atol=1e-14)


def test_two_criteria_and_programmatic_aliases():
    expected = [0.75, 0.25]
    assert_allclose(somit_part1({1: "1/3"}, 0, 0, 1, 3, 2), expected)
    assert somit_i is somit_part1
    assert somit_ii is somit_part2
    assert somit_iii is somit_part3


def test_group_aggregates_solved_weights_and_reduces_to_one_dm():
    dm1 = DecisionMaker(1, {0: 2, 2: "1/3"}, 0, 2, 6)
    dm2 = DecisionMaker(1, {0: "1/3", 2: 2}, 2, 0, 6)
    result = gsomit_part1([dm1, dm2], 3)
    expected = np.array([np.sqrt(0.06), 0.3, np.sqrt(0.06)])
    assert_allclose(result.weights, expected / expected.sum())
    assert_allclose(result.consensus_index, 11 / 21)
    group, individual, ci = result
    assert_array_equal(group, result.weights)
    assert individual.shape == (2, 3)
    assert ci == result.consensus_index
    single = gsomit_part1([dm1], 3)
    assert_allclose(single.weights, [0.6, 0.3, 0.1])
    assert single.consensus_index == 1


def test_consensus_can_be_negative_and_uses_population_standard_deviation():
    cyclic = np.array([[9, 1, 1], [1, 9, 1], [1, 1, 9]]) / 11
    assert_allclose(consensus_index(cyclic), 1 - 8 * np.sqrt(2) / 11)
    assert consensus_index(cyclic) < 0
    assert consensus_index([[1, 0], [1, 0]]) == 1


def test_zero_weight_is_a_veto_and_disjoint_support_is_undefined():
    assert_allclose(geometric_mean_weights([[1, 0, 1], [1, 1, 0]]), [1, 0, 0])
    with pytest.raises(ValueError, match="common positive"):
        geometric_mean_weights([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="common positive"):
        somit_part3([1, 0], [0, 1])


def test_part2_is_average_deviation_from_median_and_excludes_constants():
    acm = [[0, 0, 7], [0, 0, 7], [0, 0.5, 7], [0, 1, 7], [1, 1, 7]]
    assert_allclose(somit_part2(acm), [1 / 3, 2 / 3, 0])


def test_part2_is_invariant_to_units_and_direction():
    acm = np.array([[1, 2, 3], [2, 5, 9], [3, 7, 4], [9, 10, 8]], dtype=float)
    expected = somit_part2(acm)
    assert_allclose(somit_part2(acm * [1e-100, 1e100, -0.01]), expected, atol=1e-14)
    assert_allclose(somit_part2(acm + [100, 500, -1]), expected, atol=1e-14)


@pytest.mark.parametrize("acm", [[[0, 2, 7], [1, 5, 7]], [[0, 2, 7], [0.3, 9, 7], [1, 5, 7]]])
def test_two_and_three_alternatives_have_equal_nonconstant_weights(acm):
    assert_allclose(somit_part2(acm), [0.5, 0.5, 0])


def test_all_constant_requires_an_explicit_policy():
    with pytest.raises(ValueError, match="no varying"):
        somit_part2([[1, 1], [1, 1]])
    assert_allclose(somit_part2([[1, 1]], on_degenerate="uniform"), [0.5, 0.5])
    with pytest.raises(ValueError, match="on_degenerate"):
        somit_part2([[0], [1]], on_degenerate="ignore")


def test_hybrid_is_normalized_product_and_handles_underflow():
    assert_allclose(somit_part3([0.8, 0.2], [0.8, 0.2]), [16 / 17, 1 / 17])
    assert_array_equal(somit_part3([1, 1e-250, 0], [0, 1e-250, 1]), [0, 1, 0])
    assert_allclose(somit_part3([8e307, 2e307], [8e307, 2e307]), [16 / 17, 1 / 17])


@pytest.mark.parametrize("bad", [[], [1, 2], [[np.nan]], [[np.inf]], [[1 + 2j]], [[]]])
def test_invalid_acm_is_rejected(bad):
    with pytest.raises(ValueError):
        somit_part2(bad)


@pytest.mark.parametrize("bad", [[0, 0], [-1, 2], [np.nan, 1], [[1, 1]], [1]])
def test_invalid_hybrid_weights_are_rejected(bad):
    with pytest.raises(ValueError):
        somit_part3([0.5, 0.5], bad)


@pytest.mark.parametrize(
    "updates",
    [
        {"comparisons": {}},
        {"comparisons": {0: 1, 1: 2}},
        {"comparisons": {0: -1, 2: 1}},
        {"comparisons": {0: np.nan, 2: 1}},
        {"comparisons": {0: "1/0", 2: 1}},
        {"comparisons": {0: 1, 2.5: 1}},
        {"median_idx": True},
        {"highest_idx": 3},
        {"highest_idx": 2, "lowest_idx": 2},
        {"direct_hl": 0},
        {"direct_hl": float("inf")},
    ],
)
def test_invalid_elicitation_is_rejected(updates):
    data = dict(
        median_idx=1, comparisons={0: 2, 2: 1 / 3}, highest_idx=0, lowest_idx=2, direct_hl=6
    )
    data.update(updates)
    with pytest.raises(ValueError):
        DecisionMaker(**data)


def test_group_dimensions_and_missing_experts_are_rejected():
    dm = DecisionMaker(1, {0: 2, 2: 1 / 3}, 0, 2, 6)
    with pytest.raises(ValueError, match="same number"):
        gsomit_part1([dm], 4)
    with pytest.raises(ValueError, match="at least one"):
        gsomit_part1([], 3)
    with pytest.raises(ValueError, match="integer"):
        gsomit_part1([dm], 3.5)
    with pytest.raises(ValueError, match="n must match"):
        somit_part1(dm.comparisons, 1, 0, 2, 6, 4)


def test_auto_fallback_still_checks_the_solution(monkeypatch):
    def singular(*args):
        raise np.linalg.LinAlgError("test fallback")

    monkeypatch.setattr("pysomit.weighting.np.linalg.solve", singular)
    result = solve_subjective(DecisionMaker(1, {0: 2, 2: 1 / 3}, 0, 2, 6))
    assert result.solver == "slsqp"
    assert_allclose(result.weights, [0.6, 0.3, 0.1], atol=2e-7)


def test_optimizer_failure_is_not_returned_as_weights(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "pysomit.weighting.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False, message="iteration limit"),
    )
    with pytest.raises(SolverError, match="iteration limit"):
        solve_subjective(DecisionMaker(0, {1: 1}, 0, 1, 1), solver="slsqp")
