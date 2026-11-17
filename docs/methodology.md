# Methodology and implementation conventions

## Part I: subjective and group weights

For one expert, build a residual matrix with a row representing
`a_ij * w_j - w_i` for each median comparison and the direct highest/lowest
comparison. Minimize the sum of squared residuals subject to nonnegative weights
that sum to one. The upper bound of one follows from those constraints.

The default solver forms the equality-constrained KKT system, checks feasibility
and first-order optimality, and uses SLSQP with an analytic gradient if the direct
solve fails those checks. The objective is convex, so a feasible KKT solution is
a global optimum. `SubjectiveResult.kkt_violation` measures stationarity and
nonnegativity-dual residuals after scaling the comparison matrix by its largest
absolute entry. The acceptance tolerance is `5e-7` in that scaled system.
Primal constraint violations larger than `1e-9` are rejected. Failed optimizers
raise `SolverError`, even when an iterate is available.

Every expert is solved separately. The group vector is the normalized geometric
mean of the individual vectors, calculated in log space. No epsilon is added to
change a zero weight. With two criteria, the median comparison and direct anchor
can concern the same pair; both elicited residuals enter the objective.
The implementation accepts inconsistent judgments and does not automatically
reselect an expert's anchors. The same identities are retained during simulation.

The consensus index is `1 - mean_j(std_j / mean_j)`, with population standard
deviation (`ddof=0`). It equals one for agreement and can fall below zero. A
criterion everyone weights zero contributes zero dispersion. This index is a
descriptive measure and has no built-in significance or agreement threshold.

## Part II: objective weights

Range-normalize each ACM column, compute its median, and average the absolute
deviations from that median. Normalize these average deviations to sum to one.
Cost/benefit orientation does not change these deviations. Constant columns have
zero dispersion and receive zero weight. The entirely constant case is undefined
unless the caller explicitly requests a uniform fallback.

This is **average absolute deviation from the median (AADM)**, not median
absolute deviation. With two or three alternatives, every nonconstant normalized
column has the same AADM (one-half or one-third, respectively). Objective weights
are therefore uniform across nonconstant criteria in those small problems.

The median center does not make the entire estimator immune to outliers: the
average deviation and min-max normalization both depend on extreme observations.
Outlier behavior should be assessed for the specific dataset and contamination
model. The notebook reports the manuscript's single inflated-emissions scenario
without interpreting it as a universal robustness guarantee.

## Part III: hybrid weights

Normalize the elementwise product of subjective and objective weights. Computation
in log space avoids underflow in small positive products. Inputs may be
unnormalized nonnegative weight vectors; each is normalized first. Disjoint
positive supports make the normalized product undefined and raise an error.
Hybrid weights need not lie between the corresponding subjective and objective
weights, because the product is renormalized across all criteria.

## TOPSIS and CRITIC

TOPSIS vector-normalizes the original ACM, applies normalized weights, selects
cost/benefit ideals and computes Euclidean distances. Closeness is distance from
the negative ideal divided by the sum of both distances. Columns and distance
vectors are scaled before norm calculations to reduce overflow and underflow.
An all-zero column contributes no distance. When both ideal distances vanish,
the score is 0.5 by convention. Ranks are 1-based and descending in score.
Exactly equal scores receive average ranks unless ordinal tie breaking is
explicitly requested. Near-equal floating-point scores are not merged into ties.

CRITIC uses normalized standard deviations multiplied by sums of `1 - r_jk`.
By default, benefit columns increase and cost columns decrease on the normalized
preference scale. The `all_increasing` option preserves the old script's
orientation for comparison. Only varying criteria enter the correlation matrix.
One varying criterion receives weight one. If all conflict information is zero,
the default is an error; an explicit uniform fallback assigns equal weight to
varying criteria, or to all criteria if none vary. Numerical conflict terms below
32 machine epsilons are treated as zero.

## Stochastic analysis

The manuscript mode independently multiplies each median and anchor comparison
by `exp(N(0, comparison_noise**2))` and clips the result to the comparison bounds.
Clipping, rather than the log-normal distribution alone, enforces the Saaty
range. Each ACM element receives independent uniform multiplicative noise. All
three weighting parts and TOPSIS are recalculated in every run.

`rng="legacy"` uses a private NumPy `RandomState`, with draws in expert order,
comparison mapping insertion order, direct-anchor order, then ACM row-major order.
`rng="generator"` uses a private `default_rng`. A seed of `None` requests an
unseeded local generator. No global seed or random state is changed. Identical
inputs, order, parameters and numerical environment reproduce the output;
changing solver or dependency versions can affect the last floating-point digits.
No results are cached and no simulations are silently discarded.

When `acm_bounds` are supplied, each uniform distribution is restricted to the
intersection of its original relative interval and its physical bounds.
This is truncation by rescaling the uniform interval, not clipping draws after
sampling. Integer columns draw uniformly among the feasible integers. Negative
data are allowed, with interval endpoints sorted correctly. Multiplicative noise
cannot move a baseline zero; use a different uncertainty model if zero-valued
measurements require additive variation. The current model assumes independent
perturbations and does not model correlations between measurement errors or
expert judgments.

`acm_noise` and each bound can be scalar, a per-column vector or an array matching
the ACM. Noise must lie between zero and one. Baseline inputs must already satisfy
their bounds. Metadata records expanded arrays, using JSON `null` for unbounded
lower or upper limits, plus the random seed, generator, solver, tie policy and
Python/NumPy/SciPy/package versions. Save the original inputs with this metadata
when archiving a custom study.

`ranking_probabilities[i, r-1]` is the probability of alternative `i` occupying
rank `r`. A tie spanning several slots distributes probability uniformly over
those slots for each tied alternative. Thus the matrix remains doubly stochastic;
these entries are not probabilities of being a co-winner before tie breaking.
`rank_samples` retains average ranks, so expected ranks agree with the probability
matrix. The returned weight and score summaries use population standard deviation
and linearly interpolated percentiles. The default 5th-95th percentile bands are
**simulation uncertainty intervals**, not frequentist confidence intervals.

## Reproduction scope

The package preserves the manuscript's G-SOMIT equations and includes the exact
case-study matrices and elicitation values from the source project. Its default
direct solver uses the written nonnegative constraints, whereas the experiment
script used finite-difference SLSQP with a `1e-6` lower bound and did not check
convergence. Tiny differences in floating-point output are therefore expected.
Published four-decimal baseline scores and seeded top-rank counts are checked
in the examples. CRITIC's default direction treatment and the optional bounded
simulation are reported separately from the legacy comparison/model.
