"""Efficiency — the penalized SELECTION fitness + the guarded comparator.

The most important tests in B7: the threshold guard must provably never let
efficiency sacrifice clearing the 0.90 gate.
"""

from darwin.problem.schemas import ObjectiveWeights, ScoreBreakdown, Solution
from darwin.problem.scorer import SCORER_VERSION
from darwin.routing import efficiency as E
from darwin.routing.efficiency import (
    Bounds,
    EfficiencyParams,
    EfficiencyStrategy,
    RawFitnessStrategy,
)
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import ArbiterTier

P = EfficiencyParams()  # defaults: λcost 0.05, λlat 0.03, threshold 0.90


def ev(q, cost=0.0, latency=0.0, feasible=True) -> GenomeEvaluation:
    norm = q if feasible else 0.0
    sb = ScoreBreakdown(
        solution_id="s", instance_id="i", feasible=feasible, violations=[], raw_cost=0.0,
        raw_lead_time=0.0, raw_risk=0.0, weighted_objective=0.0, normalized_score=norm,
        total_penalty=0.0, final_fitness=(q if feasible else -1.0),
        objective_weights=ObjectiveWeights.cost_only(), scorer_version=SCORER_VERSION,
        computed_at="t", diagnostics={},
    )
    return GenomeEvaluation(
        genome_id="g", version=1, instance_id="i", completed=feasible,
        final_solution=Solution(solution_id="s", instance_id="i", flows=[]), score_breakdown=sb,
        fitness=(q if feasible else -1.0), normalized_score=norm,
        cleared_threshold=(feasible and norm >= 0.90), arbiter_tier_used=ArbiterTier.PRIMARY,
        total_cost_usd=cost, total_latency_ms=latency,
    )


# -- normalization ----------------------------------------------------------
def test_normalize_maps_min_to_zero_max_to_one():
    assert E.normalize(1.0, 1.0, 3.0) == 0.0
    assert E.normalize(3.0, 1.0, 3.0) == 1.0
    assert E.normalize(2.0, 1.0, 3.0) == 0.5


def test_normalize_degenerate_range_is_zero():
    assert E.normalize(5.0, 2.0, 2.0) == 0.0  # no spread -> no differentiation
    assert E.normalize(5.0, 9.0, 1.0) == 0.0  # inverted -> guarded to 0


def test_bounds_over_round():
    b = E.bounds_over([ev(0.5, 1.0, 100.0), ev(0.5, 3.0, 50.0), ev(0.5, 2.0, 200.0)])
    assert (b.cost_min, b.cost_max) == (1.0, 3.0)
    assert (b.lat_min, b.lat_max) == (50.0, 200.0)


def test_efficiency_adjusted_fitness_exact():
    b = Bounds(cost_min=0.0, cost_max=2.0, lat_min=0.0, lat_max=100.0)
    # Q=0.8, C=2 (norm 1), L=100 (norm 1): 0.8 - 0.05 - 0.03 = 0.72
    assert abs(E.efficiency_adjusted_fitness(ev(0.8, 2.0, 100.0), params=P, bounds=b) - 0.72) < 1e-12


# -- §11 penalty direction & magnitude --------------------------------------
def test_penalty_direction_cheaper_faster_wins_at_equal_quality():
    a = ev(0.8, cost=1.0, latency=100.0)  # cheaper + faster
    b = ev(0.8, cost=2.0, latency=200.0)
    assert E.compare(a, b, params=P) == 1


def test_penalty_never_dominates_task_quality():
    # a clear, small Q advantage (0.08) beats the maximum possible cost penalty
    a = ev(0.88, cost=2.0)  # most expensive
    b = ev(0.80, cost=0.0)  # cheapest
    assert E.compare(a, b, params=P) == 1  # higher Q wins despite higher cost


# -- §11 the threshold guard (critical) -------------------------------------
def test_threshold_guard_clearing_team_always_wins():
    clearing = ev(0.91, cost=100.0, latency=9999.0)  # barely clears, hugely expensive
    cheaper = ev(0.89, cost=0.0, latency=0.0)         # just misses, free
    assert E.compare(clearing, cheaper, params=P) == 1
    assert E.compare(cheaper, clearing, params=P) == -1
    assert E.best_index([cheaper, clearing], params=P) == 1


def test_below_threshold_quality_dominates_with_mild_cost_preference():
    # both below threshold: higher Q wins; among equal Q, cheaper wins
    assert E.compare(ev(0.85, 0.0), ev(0.80, 0.0), params=P) == 1
    assert E.compare(ev(0.80, 0.0), ev(0.80, 5.0), params=P) == 1


# -- §11 among clearing teams: efficiency decides ---------------------------
def test_among_clearing_teams_cheaper_wins_even_at_slightly_lower_quality():
    a = ev(0.92, cost=0.0)   # cheaper, slightly lower Q
    b = ev(0.95, cost=2.0)   # pricier, higher Q
    # post-threshold the efficiency term decides the ORDERING: eaf(a)=0.92 > eaf(b)=0.90
    assert E.compare(a, b, params=P) == 1


def test_infeasible_never_clears():
    assert not E.clears(ev(0.99, feasible=False), P.threshold)
    # an infeasible team (Q=0) loses to any feasible clearing team
    assert E.compare(ev(0.91), ev(0.99, feasible=False), params=P) == 1


# -- determinism ------------------------------------------------------------
def test_comparator_is_antisymmetric_and_deterministic():
    a, b = ev(0.9, 1.0), ev(0.9, 2.0)
    assert E.compare(a, b, params=P) == -E.compare(b, a, params=P)
    assert E.compare(a, b, params=P) == E.compare(a, b, params=P)


def test_best_index_single_candidate_is_zero():
    assert E.best_index([ev(0.7, 5.0, 500.0)], params=P) == 0


def test_best_index_ties_resolve_to_lowest():
    assert E.best_index([ev(0.8, 1.0), ev(0.8, 1.0)], params=P) == 0


# -- the adoption rule (hold quality, cut cost) -----------------------------
def test_improves_adopts_cheaper_at_equal_quality():
    assert E.improves(ev(0.95, cost=0.0), ev(0.95, cost=2.0), params=P) is True


def test_improves_refuses_to_trade_quality_down():
    # both clear, candidate cheaper but lower Q -> NOT adopted (quality held)
    assert E.improves(ev(0.92, cost=0.0), ev(0.95, cost=2.0), params=P) is False


def test_improves_adopts_when_crossing_the_gate():
    assert E.improves(ev(0.91, cost=5.0), ev(0.80, cost=0.0), params=P) is True


def test_improves_never_drops_below_the_gate():
    assert E.improves(ev(0.89, cost=0.0), ev(0.91, cost=5.0), params=P) is False


# -- strategies (the injectable B5/B6 hook) ---------------------------------
def test_raw_strategy_reproduces_argmax_fitness():
    s = RawFitnessStrategy(epsilon=1e-9)
    evals = [ev(0.7), ev(0.9), ev(0.8)]
    assert s.best_index(evals) == 1
    assert s.improves(ev(0.9), ev(0.8)) is True
    assert s.improves(ev(0.8), ev(0.8)) is False  # not strict


def test_efficiency_strategy_delegates_to_module():
    s = EfficiencyStrategy(P)
    assert s.best_index([ev(0.89, 0.0), ev(0.91, 9.0)]) == 1  # guard: clearing wins
    assert s.improves(ev(0.95, 0.0), ev(0.95, 2.0)) is True


# -- regression: infeasible normalized_score diverges from raw fitness -------
def ev_raw(norm, fitness, cost=0.0, latency=0.0, feasible=True) -> GenomeEvaluation:
    """Set normalized_score and fitness INDEPENDENTLY (as B1 does for infeasible:
    normalized is the cost ratio, fitness is a large negative penalty)."""
    sb = ScoreBreakdown(
        solution_id="s", instance_id="i", feasible=feasible, violations=[], raw_cost=0.0, raw_lead_time=0.0,
        raw_risk=0.0, weighted_objective=0.0, normalized_score=norm, total_penalty=(0.0 if feasible else 1.0),
        final_fitness=fitness, objective_weights=ObjectiveWeights.cost_only(), scorer_version=SCORER_VERSION,
        computed_at="t", diagnostics={},
    )
    return GenomeEvaluation(
        genome_id="g", version=1, instance_id="i", completed=feasible,
        final_solution=Solution(solution_id="s", instance_id="i", flows=[]), score_breakdown=sb, fitness=fitness,
        normalized_score=norm, cleared_threshold=(feasible and norm >= 0.90), arbiter_tier_used=ArbiterTier.PRIMARY,
        total_cost_usd=cost, total_latency_ms=latency,
    )


def test_infeasible_cheaper_but_worse_does_not_win_on_normalized_score():
    # the bug: a cheaper-but-more-violated infeasible team has HIGHER normalized_score
    # (0.9) yet MUCH worse fitness (-2000). It must NOT outrank/adopt over the
    # less-violated incumbent (norm 0.3, fitness -500).
    inc = ev_raw(norm=0.3, fitness=-500.0, feasible=False)
    cand = ev_raw(norm=0.9, fitness=-2000.0, cost=0.0, feasible=False)
    assert E.improves(cand, inc, params=P) is False        # never adopt the worse-fitness team
    assert E.best_index([inc, cand], params=P) == 0         # incumbent (higher fitness) is best
    assert E.compare(inc, cand, params=P) == 1


def test_eaf_base_is_raw_fitness_not_normalized():
    b = E.bounds_over([ev_raw(0.9, -2000.0, feasible=False)])
    assert E.efficiency_adjusted_fitness(ev_raw(0.9, -2000.0, feasible=False), params=P, bounds=b) < 0
