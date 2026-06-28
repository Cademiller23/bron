"""B8 Reorganizer tests — heuristic steering from the ScoreBreakdown."""

from darwin.problem.schemas import ObjectiveWeights, ScoreBreakdown, Violation, ViolationType
from darwin.problem.scorer import SCORER_VERSION
from darwin.rearrange.reorganizer import HeuristicReorganizer, default_hints


def _breakdown(*, raw_risk=0.1, feasible=True, normalized=0.95, violations=None):
    return ScoreBreakdown(
        solution_id="s", instance_id="i", feasible=feasible, violations=violations or [],
        raw_cost=10.0, raw_lead_time=1.0, raw_risk=raw_risk, weighted_objective=0.1,
        normalized_score=normalized, total_penalty=0.0, final_fitness=normalized if feasible else -1.0,
        objective_weights=ObjectiveWeights.cost_only(), scorer_version=SCORER_VERSION, computed_at="t",
    )


def test_weak_resilience_biases_toward_elevating_risk_agent():
    hints = default_hints(_breakdown(raw_risk=0.8))
    assert hints.get("swap_arbiter", 1.0) > 1.0
    assert hints.get("redirect_edge", 1.0) > 1.0


def test_capacity_violations_bias_toward_reorder():
    v = Violation(violation_type=ViolationType.OVER_NODE_CAPACITY, location="T", magnitude=3.0)
    hints = default_hints(_breakdown(feasible=False, violations=[v]))
    assert hints.get("reorder_pipeline", 1.0) > 1.0


def test_clean_breakdown_yields_neutral_hints():
    hints = default_hints(_breakdown(raw_risk=0.1, feasible=True, normalized=0.99))
    assert hints == {}  # neutral (uniform sampling)


def test_none_breakdown_is_neutral():
    assert default_hints(None) == {}
    assert HeuristicReorganizer()(None) == {}


def test_heuristic_reorganizer_is_callable():
    r = HeuristicReorganizer()
    assert r(_breakdown(raw_risk=0.9)).get("swap_arbiter", 1.0) > 1.0
