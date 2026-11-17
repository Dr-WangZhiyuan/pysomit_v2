import importlib.metadata

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pysomit import __version__, critic_weights, gsomit, solve_subjective, topsis
from pysomit.datasets import load_case_study


@pytest.mark.parametrize(
    "name,scores,ranks,subjective_scores,objective_scores,legacy_scores",
    [
        (
            "production",
            [0.8342, 0.7371, 0.5386, 0.5109, 0.2781],
            [1, 2, 3, 4, 5],
            [0.8265, 0.7367, 0.5664, 0.5033, 0.3180],
            [0.7266, 0.7749, 0.5117, 0.4034, 0.3781],
            [0.7427, 0.7947, 0.5713, 0.4119, 0.4214],
        ),
        (
            "storage",
            [0.4311, 0.3106, 0.4839, 0.6887, 0.4437],
            [4, 5, 2, 1, 3],
            [0.4485, 0.3276, 0.4165, 0.7226, 0.4306],
            [0.4363, 0.2998, 0.4917, 0.6572, 0.4471],
            [0.4345, 0.3185, 0.4196, 0.7018, 0.4219],
        ),
    ],
)
def test_manuscript_tables_5_and_8(
    name, scores, ranks, subjective_scores, objective_scores, legacy_scores
):
    case = load_case_study(name)
    result = gsomit(case.acm, case.decision_makers)
    ranking = topsis(case.acm, result.weights, case.types)
    assert_allclose(ranking.scores, scores, atol=5e-5, rtol=0)
    assert_array_equal(ranking.ranks, ranks)
    assert_allclose(
        topsis(case.acm, result.group.weights, case.types).scores,
        subjective_scores,
        atol=5e-5,
        rtol=0,
    )
    assert_allclose(
        topsis(case.acm, result.objective_weights, case.types).scores,
        objective_scores,
        atol=5e-5,
        rtol=0,
    )
    legacy = critic_weights(case.acm, case.types, orientation="all_increasing")
    assert_allclose(topsis(case.acm, legacy, case.types).scores, legacy_scores, atol=5e-5, rtol=0)


def test_manuscript_table_4_weights():
    case = load_case_study("production")
    result = gsomit(case.acm, case.decision_makers)
    assert_allclose(
        result.group.weights,
        [0.1039, 0.1892, 0.1402, 0.1440, 0.0788, 0.0582, 0.0786, 0.0408, 0.0351, 0.1312],
        atol=5e-5,
        rtol=0,
    )
    assert_allclose(
        result.objective_weights,
        [0.1030, 0.0927, 0.1060, 0.1150, 0.0866, 0.0903, 0.1127, 0.0828, 0.0958, 0.1150],
        atol=5e-5,
        rtol=0,
    )
    assert_allclose(
        result.weights,
        [0.1045, 0.1712, 0.1452, 0.1616, 0.0666, 0.0513, 0.0865, 0.0330, 0.0328, 0.1473],
        atol=5e-5,
        rtol=0,
    )


@pytest.mark.parametrize("name", ["production", "storage"])
def test_case_study_kkt_and_slsqp_solutions_agree(name):
    for dm in load_case_study(name).decision_makers:
        direct = solve_subjective(dm)
        iterative = solve_subjective(dm, solver="slsqp")
        assert_allclose(direct.weights, iterative.weights, atol=2e-6, rtol=0)
        assert_allclose(direct.objective_value, iterative.objective_value, atol=1e-9, rtol=0)


def test_direction_corrected_critic_storage_regression():
    case = load_case_study("storage")
    ranking = topsis(case.acm, critic_weights(case.acm, case.types), case.types)
    assert_allclose(
        ranking.scores,
        [0.4266377570, 0.2923902996, 0.4950890313, 0.6355752411, 0.4486280114],
        atol=1e-9,
        rtol=0,
    )
    assert_array_equal(ranking.ranks, [4, 5, 2, 1, 3])


def test_data_resources_are_independent_and_packaged():
    first, second = load_case_study("production"), load_case_study("production")
    first.acm[0, 0] = -1
    assert second.acm[0, 0] == 850
    assert len(second.criteria) == second.acm.shape[1]
    assert np.all(second.acm >= second.lower_bounds)
    assert np.all(second.acm <= second.upper_bounds)
    assert importlib.metadata.version("pysomit") == __version__
    with pytest.raises(ValueError, match="case study"):
        load_case_study("unknown")
