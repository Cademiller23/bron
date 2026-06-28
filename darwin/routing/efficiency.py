"""The cost/latency-penalized SELECTION fitness — the crux that makes the gene matter.

Two different fitnesses for two different decisions (conflating them is the
central error):

* **TASK fitness** = B1's ``normalized_score`` (Q ∈ [0,1]) — how well the
  supply-chain problem was solved. This, and ONLY this, drives B6's 0.90 gate.
  A team clears the gate by solving the problem, never by being cheap to run.
* **SELECTION fitness** = ``efficiency_adjusted_fitness`` = Q minus a small
  penalty for the *inference* cost/latency of running the team. This drives
  argmax choices (B5's "keep the best candidate", B6's team-growth elitism), so
  that among teams of *similar task quality* the cheaper/faster one is preferred.

Note the two distinct cost concepts: *solution cost* (dollars to ship goods — in
Q, from B1) vs *inference cost* (dollars/latency to run the agents — the penalty,
from B3's ``total_cost_usd`` / ``total_latency_ms``). The penalty is about
inference economics, never the supply-chain solution.

The threshold guard (lexicographic): a team that clears 0.90 always ranks above
one that doesn't, regardless of efficiency — so efficiency can provably never
sacrifice clearing the gate. Below threshold Q dominates (λ small), so the search
climbs while mildly preferring cheaper teams; once teams clear, the efficiency
term trims expensive models that aren't needed — quality held, cost cut.
"""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from darwin.constants import EFFICIENCY_EPSILON, LAMBDA_COST, LAMBDA_LATENCY, SCORE_THRESHOLD
from darwin.team.evaluation import GenomeEvaluation


@dataclass(frozen=True)
class EfficiencyParams:
    lambda_cost: float = LAMBDA_COST
    lambda_latency: float = LAMBDA_LATENCY
    threshold: float = SCORE_THRESHOLD
    epsilon: float = EFFICIENCY_EPSILON


DEFAULT_PARAMS = EfficiencyParams()


@dataclass(frozen=True)
class Bounds:
    cost_min: float = 0.0
    cost_max: float = 0.0
    lat_min: float = 0.0
    lat_max: float = 0.0


def normalize(value: float, lo: float, hi: float) -> float:
    """Map ``value`` into [0,1] against [lo,hi]; a degenerate range → 0.0 (no
    spread means no penalty differentiation this round). Clamped to [0,1]."""
    if not (hi > lo):
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def quality(ev: GenomeEvaluation) -> float:
    """The gate quantity Q = B1's ``normalized_score`` (the cost ratio, ∈ [0,1]).

    Note: ``normalized_score`` is computed independently of feasibility, so for an
    INFEASIBLE solution it can be high while ``ev.fitness`` (final_fitness) is a
    large negative penalty. The 0.90 gate is judged on Q (feasible AND Q ≥ 0.90),
    but SELECTION/adoption rank on ``ev.fitness`` — which equals Q for feasible
    solutions and correctly sinks infeasible ones (scorer.py: feasible →
    final_fitness == normalized_score; infeasible → a negative penalty)."""
    return ev.normalized_score


def cost_of(ev: GenomeEvaluation) -> float:
    return ev.total_cost_usd


def latency_of(ev: GenomeEvaluation) -> float:
    return ev.total_latency_ms


def clears(ev: GenomeEvaluation, threshold: float = SCORE_THRESHOLD) -> bool:
    """Whether the team clears the task gate (feasible AND Q ≥ threshold)."""
    return ev.score_breakdown.feasible and ev.normalized_score >= threshold


def bounds_over(evals: Sequence[GenomeEvaluation]) -> Bounds:
    """Per-round min/max of inference cost and latency (the adaptive penalty
    scale): the cheapest candidate gets C_norm=0, the priciest C_norm=1."""
    if not evals:
        return Bounds()
    costs = [cost_of(e) for e in evals]
    lats = [latency_of(e) for e in evals]
    return Bounds(min(costs), max(costs), min(lats), max(lats))


def efficiency_adjusted_fitness(ev: GenomeEvaluation, *, params: EfficiencyParams, bounds: Bounds) -> float:
    """The SELECTION fitness = raw ``fitness`` − λ_cost·C_norm − λ_latency·L_norm.

    The base is ``ev.fitness`` (final_fitness), NOT ``normalized_score`` — they are
    equal for feasible solutions (where the cost/latency trim is what matters) but
    diverge for infeasible ones, where ``fitness`` is a large negative penalty. Using
    ``fitness`` keeps the SELECTION fitness aligned with the quantity B5's
    non-decreasing trace and never-regress are defined over, so the bounded penalty
    (≤ 0.1) only ever tie-breaks among teams of equal raw fitness — it can never
    promote a more-violated (lower-fitness) infeasible team over a less-violated one.
    """
    c_norm = normalize(cost_of(ev), bounds.cost_min, bounds.cost_max)
    l_norm = normalize(latency_of(ev), bounds.lat_min, bounds.lat_max)
    return ev.fitness - params.lambda_cost * c_norm - params.lambda_latency * l_norm


def selection_key(ev: GenomeEvaluation, *, params: EfficiencyParams, bounds: Bounds) -> Tuple[int, float]:
    """The lexicographic key: (clears-the-gate?, efficiency-adjusted fitness).
    argmax of this is the guarded "best" — clearing always beats non-clearing."""
    return (1 if clears(ev, params.threshold) else 0,
            efficiency_adjusted_fitness(ev, params=params, bounds=bounds))


def compare(a: GenomeEvaluation, b: GenomeEvaluation, *, params: EfficiencyParams = DEFAULT_PARAMS) -> int:
    """Guarded lexicographic comparator over a pair (bounds from the pair).
    Returns +1 if ``a`` is better, −1 if ``b`` is better, 0 on a tie."""
    bounds = bounds_over([a, b])
    ka = selection_key(a, params=params, bounds=bounds)
    kb = selection_key(b, params=params, bounds=bounds)
    if ka > kb:
        return 1
    if ka < kb:
        return -1
    return 0


def best_index(evals: Sequence[GenomeEvaluation], *, params: EfficiencyParams = DEFAULT_PARAMS) -> int:
    """Index of the best evaluation under the guarded comparator, normalizing the
    penalty against THIS round's candidates. Ties resolve to the lowest index."""
    if not evals:
        raise ValueError("best_index requires at least one evaluation")
    bounds = bounds_over(evals)
    best_i = 0
    best_k = selection_key(evals[0], params=params, bounds=bounds)
    for i in range(1, len(evals)):
        k = selection_key(evals[i], params=params, bounds=bounds)
        if k > best_k:  # strict — first max wins (deterministic)
            best_k, best_i = k, i
    return best_i


def improves(candidate: GenomeEvaluation, incumbent: GenomeEvaluation, *,
             params: EfficiencyParams = DEFAULT_PARAMS) -> bool:
    """Adoption rule (hold-quality, cut-cost): adopt ``candidate`` over
    ``incumbent`` iff it strictly improves the efficiency-adjusted fitness AND
    never trades task quality down. So efficiency only ever *trims cost at held
    quality* — which keeps the raw-quality curve non-decreasing (B5's frozen
    guarantee) even while the model search makes the team cheaper.
    """
    cand_clears = clears(candidate, params.threshold)
    inc_clears = clears(incumbent, params.threshold)
    if cand_clears and not inc_clears:
        return True  # climbed past the gate — always an improvement
    if inc_clears and not cand_clears:
        return False  # never drop below the gate
    # both clear, or both don't: never trade raw fitness down AT ALL (this is the
    # quantity B5's non-decreasing trace is defined over; == Q for feasible teams).
    # No epsilon tolerance here, so the curve can never dip — only equal-or-better
    # fitness at lower cost is adopted (the cost trim fires at exactly-held quality).
    if candidate.fitness < incumbent.fitness:
        return False
    bounds = bounds_over([candidate, incumbent])
    eaf_c = efficiency_adjusted_fitness(candidate, params=params, bounds=bounds)
    eaf_i = efficiency_adjusted_fitness(incumbent, params=params, bounds=bounds)
    return eaf_c > eaf_i + params.epsilon


# ---------------------------------------------------------------------------
# Selection strategies — the injectable B5/B6 hook (default reproduces today's
# exact raw-fitness behavior; the efficiency strategy is B7's opt-in).
# ---------------------------------------------------------------------------
class RawFitnessStrategy:
    """The pre-B7 behavior: argmax raw fitness; adopt on strict raw improvement."""

    def __init__(self, epsilon: float) -> None:
        self.epsilon = epsilon

    def best_index(self, evals: Sequence[GenomeEvaluation]) -> int:
        if not evals:
            raise ValueError("best_index requires at least one evaluation")
        best_i = 0
        for i in range(1, len(evals)):
            if evals[i].fitness > evals[best_i].fitness:  # first max wins
                best_i = i
        return best_i

    def improves(self, candidate: GenomeEvaluation, incumbent: GenomeEvaluation) -> bool:
        return candidate.fitness > incumbent.fitness + self.epsilon


class EfficiencyStrategy:
    """B7: argmax / adopt under the guarded efficiency-adjusted comparator."""

    def __init__(self, params: EfficiencyParams = DEFAULT_PARAMS) -> None:
        self.params = params

    def best_index(self, evals: Sequence[GenomeEvaluation]) -> int:
        return best_index(evals, params=self.params)

    def improves(self, candidate: GenomeEvaluation, incumbent: GenomeEvaluation) -> bool:
        return improves(candidate, incumbent, params=self.params)
