"""Subjective-Objective Median-based Importance Technique, single and group."""

from ._version import __version__
from .benchmarks import critic_weights
from .models import (
    DecisionMaker,
    DistributionSummary,
    GroupResult,
    GSOMITResult,
    SubjectiveResult,
    TopsisResult,
)
from .monte_carlo import MonteCarloResult, SimulationError, monte_carlo_gsomit
from .ranking import topsis
from .weighting import (
    SolverError,
    consensus_index,
    geometric_mean_weights,
    gsomit,
    gsomit_part1,
    solve_subjective,
    somit_part1,
    somit_part2,
    somit_part3,
)

# v1 spelling aliases. Part I now takes explicit inputs instead of prompting.
somit_i = somit_part1
somit_ii = somit_part2
somit_iii = somit_part3

__all__ = [
    "__version__",
    "DecisionMaker",
    "DistributionSummary",
    "GroupResult",
    "GSOMITResult",
    "SubjectiveResult",
    "TopsisResult",
    "MonteCarloResult",
    "SolverError",
    "SimulationError",
    "consensus_index",
    "geometric_mean_weights",
    "gsomit",
    "gsomit_part1",
    "solve_subjective",
    "somit_part1",
    "somit_part2",
    "somit_part3",
    "somit_i",
    "somit_ii",
    "somit_iii",
    "topsis",
    "critic_weights",
    "monte_carlo_gsomit",
]
