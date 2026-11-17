# pysomit v2

Python implementation of the **Subjective-Objective Median-based Importance
Technique (SOMIT)** and its group extension, **G-SOMIT**. The package provides
single-expert and group weighting, TOPSIS ranking, a CRITIC benchmark, and seeded
Monte Carlo analysis of uncertainty in expert judgments and alternative data.

[pysomit v1](https://github.com/Dr-WangZhiyuan/pysomit).

## Installation

Requires Python 3.10 or newer. Install this version directly from GitHub:

```bash
python -m pip install "git+https://github.com/Dr-WangZhiyuan/pysomit_v2.git"
```

The distribution and import name are both `pysomit`.

For development and the case-study notebook:

```bash
git clone https://github.com/Dr-WangZhiyuan/pysomit_v2.git
cd pysomit_v2
python -m pip install -e ".[test,examples]"
```

## Quick start

```python
from pysomit import gsomit, topsis, monte_carlo_gsomit
from pysomit.datasets import load_case_study

case = load_case_study("production")  # Also: "storage"
model = gsomit(case.acm, case.decision_makers)
ranking = topsis(case.acm, model.weights, case.types)

print(model.weights)
print(model.group.consensus_index)
print(ranking.scores.round(4))
# [0.8342 0.7371 0.5386 0.5109 0.2781]
print(ranking.ranks)
# [1. 2. 3. 4. 5.]

simulation = monte_carlo_gsomit(
    case.decision_makers,
    case.acm,
    case.types,
    n_simulations=10_000,
    comparison_noise=0.25,
    acm_noise=0.10,
    seed=42,
    rng="legacy",
)
print(simulation.ranking_probabilities)
print(simulation.weight_summary().lower)  # 5th percentile
print(simulation.weight_summary().upper)  # 95th percentile
print(simulation.n_completed)
```

Rows of the alternatives-criteria matrix (ACM) are alternatives and columns are
criteria. Directions are `"min"` for costs and `"max"` for benefits. All criterion
indices are **zero-based**.

## Supply your own expert judgments

```python
from pysomit import DecisionMaker, solve_subjective, gsomit

expert = DecisionMaker(
    median_idx=1,
    comparisons={0: 2, 2: "1/3"},
    highest_idx=0,
    lowest_idx=2,
    direct_hl=6,
    name="Expert 1",
)
solution = solve_subjective(expert)
print(solution.weights)         # [0.6 0.3 0.1]
print(solution.objective_value)
print(solution.kkt_violation)

acm = [[8, 5, 3], [6, 7, 5], [4, 4, 8], [3, 9, 6]]
model = gsomit(acm, [expert])     # Add experts to this list for G-SOMIT.
```

Each expert supplies all `n - 1` comparisons against their chosen median plus
one direct highest/lowest comparison. Fraction strings and positive numeric
ratios are accepted. Plain dictionaries with the same fields are also accepted.
At least two criteria are required for subjective elicitation.

## Bounded uncertainty

For bounded applications, provide explicit bounds and integer criterion indices:

```python
bounded = monte_carlo_gsomit(
    case.decision_makers,
    case.acm,
    case.types,
    n_simulations=2_000,
    seed=42,
    acm_bounds=(case.lower_bounds, case.upper_bounds),
    integer_criteria=case.integer_criteria,
)
```

Continuous draws are uniform over the intersection of the relative uncertainty
interval and the supplied bounds. Integer criteria are uniform over the integers
in that intersection. Small intervals may contain only the baseline rating.
`acm_noise` can also be a per-criterion vector or a matrix, so uncertainty can vary
by variable. These choices define a different simulation model and can change
rank probabilities. Dataset bounds encode the stated measurement scales, not
empirically estimated uncertainty.

## Results and numerical conventions

| Function | Output |
|---|---|
| `solve_subjective(dm)` | Individual weights, objective, solver and KKT diagnostics |
| `somit_part1(comparisons, median_idx, highest_idx, lowest_idx, direct_hl, n)` | Individual weight vector |
| `gsomit_part1(dm_inputs, n)` | Group and individual weights, consensus and diagnostics |
| `somit_part2(acm)` | Objective weights from average absolute deviation from the median |
| `somit_part3(ws, wo)` | Normalized elementwise product of two weight vectors |
| `gsomit(acm, dm_inputs)` | All three weighting stages |
| `topsis(acm, weights, types)` | Scores, ranks, ideals and distances |
| `monte_carlo_gsomit(dm_inputs, acm, types, ...)` | Rank probabilities, samples and settings |

