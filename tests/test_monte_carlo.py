import importlib
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pysomit import DecisionMaker, SimulationError, SolverError, gsomit, monte_carlo_gsomit, topsis
from pysomit.datasets import load_case_study
from pysomit.monte_carlo import MonteCarloResult, _MatrixSampler, _ratio_draw


@pytest.fixture
def problem():
    return np.array([[1, 6], [2, 4], [4, 3], [8, 2]], dtype=float), [
        DecisionMaker(0, {1: 0.5}, 0, 1, 2)
    ]


@pytest.mark.parametrize("rng", ["legacy", "generator"])
def test_seeded_runs_are_reproducible_and_do_not_modify_inputs_or_global_rng(problem, rng):
    acm, dms = problem
    original = acm.copy()
    state_before = np.random.get_state()
    first = monte_carlo_gsomit(dms, acm, ["min", "max"], 16, seed=73, rng=rng)
    second = monte_carlo_gsomit(dms, acm, ["min", "max"], 24, seed=73, rng=rng)
    different = monte_carlo_gsomit(dms, acm, ["min", "max"], 16, seed=74, rng=rng)
    assert_array_equal(first.weight_samples, second.weight_samples[:16])
    assert_array_equal(first.score_samples, second.score_samples[:16])
    assert_array_equal(first.rank_samples, second.rank_samples[:16])
    assert not np.array_equal(first.weight_samples, different.weight_samples)
    state_after = np.random.get_state()
    for before, after in zip(state_before, state_after):
        assert_array_equal(before, after)
    assert_array_equal(acm, original)
    assert dms[0].comparisons[1] == 0.5
    assert first.n_completed == first.n_requested == 16
    json.dumps(first.metadata, allow_nan=False)


def test_zero_noise_recovers_deterministic_pipeline(problem):
    acm, dms = problem
    deterministic = gsomit(acm, dms)
    ranking = topsis(acm, deterministic.weights, ["min", "max"])
    result = monte_carlo_gsomit(dms, acm, ["min", "max"], 5, 0, 0)
    assert_allclose(result.weight_samples, np.tile(deterministic.weights, (5, 1)))
    assert_allclose(result.score_samples, np.tile(ranking.scores, (5, 1)))
    assert_array_equal(result.rank_samples, np.tile(ranking.ranks, (5, 1)))
    assert_allclose(result.weight_summary().std, 0, atol=1e-15)
    probabilities, weights, scores = result
    assert_array_equal(probabilities, result.ranking_probabilities)
    assert_array_equal(weights, result.weight_samples)
    assert_array_equal(scores, result.score_samples)


def test_tied_rank_probabilities_allocate_each_occupied_slot_fairly():
    dms = [DecisionMaker(0, {1: 1}, 0, 1, 1)]
    result = monte_carlo_gsomit(dms, [[2, 2], [2, 2], [1, 1]], ["max", "max"], 5, 0, 0)
    assert_allclose(result.ranking_probabilities, [[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])
    assert_array_equal(result.rank_samples[0], [1.5, 1.5, 3])
    assert_allclose(result.ranking_probabilities.sum(axis=0), 1)
    assert_allclose(result.ranking_probabilities.sum(axis=1), 1)
    ordinal = monte_carlo_gsomit(
        dms, [[2, 2], [2, 2], [1, 1]], ["max", "max"], 5, 0, 0, tie_method="ordinal"
    )
    assert_array_equal(ordinal.ranking_probabilities, np.eye(3))


def test_legacy_sampling_consumes_the_original_numpy_draw_sequence():
    case = load_case_study("production")
    actual_rng, expected_rng = np.random.RandomState(42), np.random.RandomState(42)
    for dm in case.decision_makers:
        for ratio in (*dm.comparisons.values(), dm.direct_hl):
            expected = np.clip(ratio * np.exp(expected_rng.normal(0, 0.25)), 1 / 9, 9)
            assert _ratio_draw(ratio, actual_rng, 0.25, 1 / 9, 9) == expected
    sampler = _MatrixSampler(case.acm, 0.1, None, ())
    expected_acm = case.acm * (1 + expected_rng.uniform(-0.1, 0.1, case.acm.shape))
    assert_array_equal(sampler.draw(actual_rng), expected_acm)


def test_comparison_clipping_and_large_log_noise_stay_finite():
    rng = np.random.default_rng(42)
    draws = np.array([_ratio_draw(1, rng, 1000, 1 / 9, 9) for _ in range(100)])
    assert np.all((draws >= 1 / 9) & (draws <= 9))
    assert np.any(draws == 1 / 9)
    assert np.any(draws == 9)


@pytest.mark.parametrize("name", ["production", "storage"])
def test_bounded_samples_respect_physical_and_integer_scales(name):
    case = load_case_study(name)
    sampler = _MatrixSampler(
        case.acm, 0.1, (case.lower_bounds, case.upper_bounds), case.integer_criteria
    )
    rng = np.random.default_rng(4)
    for _ in range(30):
        perturbed = sampler.draw(rng)
        assert np.all(perturbed >= case.lower_bounds)
        assert np.all(perturbed <= case.upper_bounds)
        assert np.all(perturbed >= case.acm * 0.9)
        assert np.all(perturbed <= case.acm * 1.1)
        integer_values = perturbed[:, case.integer_criteria]
        assert_array_equal(integer_values, np.floor(integer_values))
    result = monte_carlo_gsomit(
        case.decision_makers,
        case.acm,
        case.types,
        8,
        acm_bounds=(case.lower_bounds, case.upper_bounds),
        integer_criteria=case.integer_criteria,
    )
    json.dumps(result.metadata, allow_nan=False)
    assert result.n_completed == 8


def test_bounded_uniform_is_truncated_not_clipped_and_includes_integer_endpoints():
    rng = np.random.RandomState(19)
    sampler = _MatrixSampler(np.array([[100.0, 3.0]]), 1, ([0, 1], [100, 5]), (1,))
    samples = np.concatenate([sampler.draw(rng) for _ in range(2000)])
    assert not np.any(samples[:, 0] == 100)  # No point mass at the upper bound.
    assert abs(samples[:, 0].mean() - 50) < 3
    assert set(samples[:, 1]) == {1, 2, 3, 4, 5}
    assert np.all(np.bincount(samples[:, 1].astype(int))[1:] > 300)


def test_per_criterion_noise_and_negative_data():
    acm = np.array([[2.0, -5.0], [3.0, -9.0]])
    sampler = _MatrixSampler(acm, [0, 0.1], ([0, -10], [5, -4]), ())
    perturbed = sampler.draw(np.random.RandomState(42))
    assert_array_equal(perturbed[:, 0], acm[:, 0])
    assert np.all(perturbed[:, 1] >= acm[:, 1] * 1.1)
    assert np.all(perturbed[:, 1] <= acm[:, 1] * 0.9)


def test_distribution_summary_uses_linear_percentiles_and_population_std():
    samples = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    result = MonteCarloResult(
        np.eye(2), samples, samples, samples, np.ones(3), {"n_simulations": 3}
    )
    for summary in (result.weight_summary(), result.score_summary()):
        assert_allclose(summary.mean, [0.5, 0.5])
        assert_allclose(summary.median, [0.5, 0.5])
        assert_allclose(summary.std, [np.sqrt(1 / 6)] * 2)
        assert_allclose(summary.lower, [0.05, 0.05])
        assert_allclose(summary.upper, [0.95, 0.95])
        assert summary.percentiles == (5, 95)
    with pytest.raises(ValueError, match="percentiles"):
        result.weight_summary((95, 5))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_simulations": 0},
        {"n_simulations": 1.5},
        {"n_simulations": True},
        {"comparison_noise": -1},
        {"comparison_noise": np.nan},
        {"acm_noise": -0.1},
        {"acm_noise": 1.1},
        {"acm_noise": [0.1, 0.2, 0.3]},
        {"seed": -1},
        {"seed": 2**32},
        {"rng": "invalid"},
        {"comparison_bounds": (2, 1)},
        {"comparison_bounds": (1, 2)},
        {"acm_bounds": (10, 20)},
        {"acm_bounds": (np.nan, 20)},
        {"integer_criteria": (2,)},
        {"integer_criteria": (0, 0)},
    ],
)
def test_invalid_simulation_settings_are_rejected(problem, kwargs):
    acm, dms = problem
    settings = {"n_simulations": 1, **kwargs}
    with pytest.raises(ValueError):
        monte_carlo_gsomit(dms, acm, ["min", "max"], **settings)


def test_failed_simulation_is_reported_with_run_number(problem, monkeypatch):
    module = importlib.import_module("pysomit.monte_carlo")
    original, calls = module.gsomit, 0

    def fail_third_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:  # Baseline plus three simulations.
            raise SolverError("test solver failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "gsomit", fail_third_run)
    acm, dms = problem
    with pytest.raises(SimulationError, match="simulation 3 of 5 failed") as caught:
        monte_carlo_gsomit(dms, acm, ["min", "max"], 5)
    assert isinstance(caught.value.__cause__, SolverError)
