"""diagnose_gap — deterministic routing from a ScoreBreakdown to a gap."""

from darwin.escalation.diagnosis import diagnose_gap
from darwin.escalation.schemas import WeakDimension
from darwin.escalation.fixtures import (
    base_genome,
    capacity_violation,
    demand_unmet_violation,
    evaluation_with,
)


def _ev(**kw):
    return evaluation_with(base_genome(), **kw)


def test_infeasible_demand_routes_to_feasibility_demand():
    gap = diagnose_gap(_ev(feasible=False, fitness=-5.0, violations=[demand_unmet_violation()]),
                       problem_class="transportation")
    assert gap.weak_dimensions[0] == WeakDimension.FEASIBILITY
    assert "demand" in gap.capability_needed.lower()
    assert gap.suggested_role_kind == "proposer"
    assert gap.severity == 1.0
    assert gap.problem_class == "transportation"


def test_infeasible_capacity_routes_to_capacity_checker():
    gap = diagnose_gap(_ev(feasible=False, fitness=-3.0, violations=[capacity_violation()]))
    assert gap.weak_dimensions[0] == WeakDimension.FEASIBILITY
    assert "capacity" in gap.capability_needed.lower()
    assert gap.suggested_role_kind == "checker"


def test_high_risk_routes_to_resilience():
    gap = diagnose_gap(_ev(feasible=True, fitness=0.8, normalized=0.8, raw_risk=0.7))
    assert gap.weak_dimensions[0] == WeakDimension.RESILIENCE
    assert "risk" in gap.capability_needed.lower() or "disruption" in gap.capability_needed.lower()
    assert gap.suggested_role_kind == "specialist"


def test_feasible_below_threshold_routes_to_cost():
    gap = diagnose_gap(_ev(feasible=True, fitness=0.7, normalized=0.7, raw_risk=0.0))
    assert gap.weak_dimensions[0] == WeakDimension.COST
    assert "cost" in gap.capability_needed.lower()
    assert gap.suggested_role_kind == "proposer"


def test_lead_time_weakness_when_norm_lead_high():
    gap = diagnose_gap(_ev(feasible=True, fitness=0.85, normalized=0.85, raw_risk=0.0, norm_lead=0.8))
    assert WeakDimension.LEAD_TIME in gap.weak_dimensions


def test_severity_zero_when_at_threshold():
    gap = diagnose_gap(_ev(feasible=True, fitness=0.95, normalized=0.95, raw_risk=0.0))
    assert gap.severity == 0.0


def test_determinism_same_input_same_gap():
    ev = _ev(feasible=True, fitness=0.7, normalized=0.7, raw_risk=0.5)
    a = diagnose_gap(ev, "transportation")
    b = diagnose_gap(ev, "transportation")
    assert a.model_dump() == b.model_dump()


def test_risk_ranks_above_cost_when_both_weak():
    # risk 0.7 is a bigger weakness than the cost gap (0.90 - 0.85 = 0.05)
    gap = diagnose_gap(_ev(feasible=True, fitness=0.85, normalized=0.85, raw_risk=0.7))
    assert gap.weak_dimensions[0] == WeakDimension.RESILIENCE
    assert WeakDimension.COST in gap.weak_dimensions


def test_infeasibility_dominates_unbounded_lead_time():
    # regression: norm_lead can exceed 1.0 (routing), out-ranking FEASIBILITY's
    # fixed 1.0 — an infeasible solution must STILL diagnose as a feasibility gap
    # and keep the demand/capacity sub-classification.
    gap = diagnose_gap(
        _ev(feasible=False, fitness=-4.0, violations=[demand_unmet_violation()], norm_lead=24.0),
        problem_class="vehicle_routing",
    )
    assert gap.weak_dimensions[0] == WeakDimension.FEASIBILITY
    assert "demand" in gap.capability_needed.lower()
    assert gap.suggested_role_kind == "proposer"
    assert gap.severity == 1.0


def test_infeasible_capacity_with_high_lead_still_routes_capacity():
    gap = diagnose_gap(
        _ev(feasible=False, fitness=-2.0, violations=[capacity_violation()], norm_lead=12.0),
    )
    assert gap.weak_dimensions[0] == WeakDimension.FEASIBILITY
    assert "capacity" in gap.capability_needed.lower()
    assert gap.suggested_role_kind == "checker"
