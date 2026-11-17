import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pysomit import critic_weights, topsis


def test_topsis_known_mixed_direction_solution_and_ideal_distances():
    result = topsis([[1, 4], [2, 3], [3, 2], [4, 1]], [0.5, 0.5], ["min", "max"])
    assert_allclose(result.scores, [1, 2 / 3, 1 / 3, 0], atol=1e-14)
    assert_array_equal(result.ranks, [1, 2, 3, 4])
    assert result.distance_to_positive[0] == 0
    assert result.distance_to_negative[-1] == 0
    scores, ranks = result
    assert_array_equal(scores, result.scores)
    assert_array_equal(ranks, result.ranks)


def test_vector_normalization_is_stable_and_zero_columns_add_no_distance():
    acm = np.array([[1, 4, 0], [2, 2, 0], [3, 1, 0]], dtype=float)
    expected = topsis(acm[:, :2], [1, 2], ["min", "max"])
    large = topsis(acm * 1e250, [1, 2, 8], ["min", "max", "max"])
    tiny = topsis(acm * 1e-250, [1, 2, 8], ["min", "max", "max"])
    assert_allclose(large.scores, expected.scores)
    assert_allclose(tiny.scores, expected.scores)
    negative_cost = acm.copy()
    negative_cost[:, 0] *= -1
    assert_allclose(topsis(negative_cost, [1, 2, 8], ["max"] * 3).scores, expected.scores)


def test_identical_alternatives_are_neutral_and_tied():
    result = topsis([[2, 0], [2, 0], [2, 0]], [1, 1], ["max", "min"])
    assert_array_equal(result.scores, [0.5, 0.5, 0.5])
    assert_array_equal(result.ranks, [2, 2, 2])
    assert_array_equal(topsis([[0]], [1], ["min"]).ranks, [1])


def test_average_ties_and_explicit_ordinal_ties():
    acm = [[2], [2], [1]]
    assert_array_equal(topsis(acm, [1], ["max"]).ranks, [1.5, 1.5, 3])
    assert_array_equal(topsis(acm, [1], ["max"], tie_method="ordinal").ranks, [1, 2, 3])


def test_small_informative_weights_do_not_underflow_ideal_distances():
    result = topsis([[1, 7], [2, 7]], [1e-250, 1], ["max", "max"])
    assert_array_equal(result.scores, [0, 1])


@pytest.mark.parametrize("types", [["cost", "benefit"], ["MAX", "min"], ["max"], "max"])
def test_unknown_or_missing_directions_are_rejected(types):
    with pytest.raises(ValueError, match="types"):
        topsis([[1, 2], [3, 4]], [1, 1], types)


def test_invalid_tie_method_is_rejected():
    with pytest.raises(ValueError, match="tie_method"):
        topsis([[0], [1]], [1], ["max"], tie_method="random")


def test_critic_aligns_directions_before_signed_correlations():
    acm = [[1, 3, 0], [2, 2, 1], [3, 1, 0]]
    sig_x, sig_z = np.sqrt(1 / 6), np.sqrt(2 / 9)
    aligned = np.array([sig_x, sig_x, 2 * sig_z])
    legacy = np.array([3 * sig_x, 3 * sig_x, 2 * sig_z])
    assert_allclose(critic_weights(acm, ["max", "min", "max"]), aligned / aligned.sum())
    assert_allclose(
        critic_weights(acm, ["max", "min", "max"], orientation="all_increasing"),
        legacy / legacy.sum(),
    )
    reversed_acm = np.array(acm) * [1, -1, 1]
    assert_allclose(critic_weights(reversed_acm, ["max"] * 3), aligned / aligned.sum())


def test_critic_ignores_constants_and_preserves_remaining_weights():
    acm = [[1, 3, 0], [2, 2, 1], [3, 1, 0]]
    expected = critic_weights(acm, ["max", "min", "max"])
    extended = np.column_stack([acm, [7, 7, 7]])
    assert_allclose(critic_weights(extended, ["max", "min", "max", "min"]), [*expected, 0])
    assert_array_equal(critic_weights([[0, 7], [1, 7]], ["min", "max"]), [1, 0])


def test_critic_degenerate_policy_does_not_weight_constants():
    acm = [[0, 0, 9], [1, 1, 9], [2, 2, 9]]
    with pytest.raises(ValueError, match="no conflict"):
        critic_weights(acm, ["max"] * 3)
    assert_allclose(critic_weights(acm, ["max"] * 3, on_degenerate="uniform"), [0.5, 0.5, 0])
    with pytest.raises(ValueError, match="no conflict"):
        critic_weights([[1, 2], [1, 2]], ["min", "max"])
    assert_allclose(critic_weights([[1, 2]], ["min", "max"], on_degenerate="uniform"), [0.5, 0.5])
