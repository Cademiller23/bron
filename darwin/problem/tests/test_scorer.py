"""§8.3 Scorer tests — the largest suite (feasibility, objectives, invariants)."""

import itertools
import math
import random
import re
import socket
import time

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from darwin.problem import fixtures as fx
from darwin.problem.scorer import SCORER_VERSION, score
from darwin.problem.schemas import (
    AdditionalConstraint,
    Arc,
    ConstraintType,
    FlowAssignment,
    KnownOptimum,
    Node,
    NodeType,
    ObjectiveWeights,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
    Solution,
    ViolationType,
)
from darwin.problem.tests._helpers import (
    feasible_transportation_solution,
    make_solution,
    random_flow_solution_strategy,
)

COST = ObjectiveWeights.cost_only()


def _types(sb):
    return {v.violation_type for v in sb.violations}


def _mag(sb, vtype):
    return next(v.magnitude for v in sb.violations if v.violation_type == vtype)


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------
def test_known_feasible_solution_has_no_violations():
    sb = score(fx.golden_transportation(), fx.transportation_optimal_solution(), COST)
    assert sb.feasible is True
    assert sb.violations == []


def test_over_arc_capacity_detected_with_magnitude():
    inst = ProblemInstance(
        instance_id="cap", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=8.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0, capacity=5.0)],
    )
    sb = score(inst, make_solution(inst, {"a": 8.0}), COST)
    assert not sb.feasible
    assert ViolationType.OVER_ARC_CAPACITY in _types(sb)
    assert math.isclose(_mag(sb, ViolationType.OVER_ARC_CAPACITY), 3.0)


def test_unmet_demand_detected_with_shortfall():
    inst = fx.golden_transportation()
    sb = score(inst, make_solution(inst, {"S1-D1": 8.0}), COST)  # D2 (demand 7) unmet
    assert not sb.feasible
    assert math.isclose(_mag(sb, ViolationType.DEMAND_UNMET), 7.0)


def test_broken_conservation_at_transshipment():
    inst = ProblemInstance(
        instance_id="ts", source="fixture", problem_class=ProblemClass.TRANSSHIPMENT,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="T", node_type=NodeType.TRANSSHIPMENT),
               Node(node_id="D1", node_type=NodeType.SINK, demand=8.0)],
        arcs=[Arc(arc_id="S-T", from_node="S1", to_node="T", unit_cost=1.0),
              Arc(arc_id="T-D", from_node="T", to_node="D1", unit_cost=1.0)],
    )
    # in=10, out=8 at T (demand still met) => conservation broken by 2, nothing else
    sb = score(inst, make_solution(inst, {"S-T": 10.0, "T-D": 8.0}), COST)
    assert _types(sb) == {ViolationType.CONSERVATION_BROKEN}
    assert math.isclose(_mag(sb, ViolationType.CONSERVATION_BROKEN), 2.0)


def test_supply_exceeded():
    inst = ProblemInstance(
        instance_id="sup", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=8.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0)],
    )
    sb = score(inst, make_solution(inst, {"a": 12.0}), COST)
    assert _types(sb) == {ViolationType.SUPPLY_EXCEEDED}
    assert math.isclose(_mag(sb, ViolationType.SUPPLY_EXCEEDED), 2.0)


def test_flow_into_closed_facility():
    inst = fx.golden_facility_location()
    sol = Solution(
        solution_id="closed", instance_id=inst.instance_id,
        flows=[FlowAssignment(arc_id="F-W1", quantity=10.0), FlowAssignment(arc_id="W1-C1", quantity=10.0),
               FlowAssignment(arc_id="F-W2", quantity=10.0), FlowAssignment(arc_id="W2-C2", quantity=10.0)],
        open_facilities=["W1"],  # W2 left closed but carries flow
    )
    sb = score(inst, sol, COST)
    assert ViolationType.CLOSED_FACILITY_USED in _types(sb)
    assert math.isclose(_mag(sb, ViolationType.CLOSED_FACILITY_USED), 10.0)


def test_lead_time_limit_exceeded():
    inst = ProblemInstance(
        instance_id="lt", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=8.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0, lead_time=5.0)],
        additional_constraints=[AdditionalConstraint(
            constraint_id="lt", constraint_type=ConstraintType.LEAD_TIME_LIMIT, parameters={"limit": 2.0})],
    )
    sb = score(inst, make_solution(inst, {"a": 8.0}), COST)
    assert ViolationType.LEAD_TIME_EXCEEDED in _types(sb)
    assert math.isclose(_mag(sb, ViolationType.LEAD_TIME_EXCEEDED), 3.0)


def test_several_simultaneous_violations_all_detected():
    inst = ProblemInstance(
        instance_id="multi", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=16.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=8.0),
               Node(node_id="D2", node_type=NodeType.SINK, demand=8.0)],
        arcs=[Arc(arc_id="a1", from_node="S1", to_node="D1", unit_cost=1.0, capacity=3.0),
              Arc(arc_id="a2", from_node="S1", to_node="D2", unit_cost=1.0)],
    )
    # a1 carries 20 (> cap 3 and source ships 20 > supply 16); D2 unmet entirely
    sb = score(inst, make_solution(inst, {"a1": 20.0}), COST)
    assert {ViolationType.OVER_ARC_CAPACITY, ViolationType.SUPPLY_EXCEEDED, ViolationType.DEMAND_UNMET} <= _types(sb)


def test_over_node_capacity_detected_with_magnitude():
    inst = ProblemInstance(
        instance_id="ncap", source="fixture", problem_class=ProblemClass.TRANSSHIPMENT,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="T", node_type=NodeType.TRANSSHIPMENT, capacity=5.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=8.0)],
        arcs=[Arc(arc_id="S-T", from_node="S1", to_node="T", unit_cost=1.0),
              Arc(arc_id="T-D", from_node="T", to_node="D1", unit_cost=1.0)],
    )
    sb = score(inst, make_solution(inst, {"S-T": 8.0, "T-D": 8.0}), COST)  # T throughput 8 > cap 5
    assert ViolationType.OVER_NODE_CAPACITY in _types(sb)
    assert math.isclose(_mag(sb, ViolationType.OVER_NODE_CAPACITY), 3.0)


def test_single_source_constraint_violated():
    inst = ProblemInstance(
        instance_id="ss", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
               Node(node_id="S2", node_type=NodeType.SOURCE, supply=10.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=10.0)],
        arcs=[Arc(arc_id="S1-D1", from_node="S1", to_node="D1", unit_cost=1.0),
              Arc(arc_id="S2-D1", from_node="S2", to_node="D1", unit_cost=1.0)],
        additional_constraints=[AdditionalConstraint(
            constraint_id="ss", constraint_type=ConstraintType.SINGLE_SOURCE, parameters={"sink_id": "D1"})],
    )
    # demand met but split across two suppliers -> single-source constraint breached
    sb = score(inst, make_solution(inst, {"S1-D1": 5.0, "S2-D1": 5.0}), COST)
    assert ViolationType.CUSTOM_CONSTRAINT in _types(sb)
    assert not sb.feasible


def test_mutual_exclusion_constraint_violated():
    inst = fx.golden_facility_location().model_copy(
        update={"additional_constraints": [AdditionalConstraint(
            constraint_id="mx", constraint_type=ConstraintType.MUTUAL_EXCLUSION,
            parameters={"node_ids": ["W1", "W2"]})]}
    )
    sol = Solution(
        solution_id="both", instance_id=inst.instance_id,
        flows=[FlowAssignment(arc_id="F-W1", quantity=10.0), FlowAssignment(arc_id="W1-C1", quantity=10.0),
               FlowAssignment(arc_id="F-W2", quantity=10.0), FlowAssignment(arc_id="W2-C2", quantity=10.0)],
        open_facilities=["W1", "W2"],
    )
    sb = score(inst, sol, COST)
    assert ViolationType.CUSTOM_CONSTRAINT in _types(sb)


def test_declared_capacity_constraint_violated():
    inst = ProblemInstance(
        instance_id="dcap", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=8.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0)],
        additional_constraints=[AdditionalConstraint(
            constraint_id="cap", constraint_type=ConstraintType.CAPACITY, parameters={"arc_id": "a", "limit": 5.0})],
    )
    sb = score(inst, make_solution(inst, {"a": 8.0}), COST)
    assert ViolationType.OVER_ARC_CAPACITY in _types(sb)
    assert math.isclose(_mag(sb, ViolationType.OVER_ARC_CAPACITY), 3.0)


# ---------------------------------------------------------------------------
# Objective computation
# ---------------------------------------------------------------------------
def test_cost_computed_to_the_cent():
    sb = score(fx.golden_transportation(), fx.transportation_optimal_solution(), COST)
    assert math.isclose(sb.raw_cost, 23.0)


def test_facility_fixed_costs_included_for_opened_excluded_for_unused():
    sb = score(fx.golden_facility_location(), fx.facility_optimal_solution(), COST)
    # 20*1 + 10*1 + 10*5 + fixed(W1)=50  => 130 ; W2's fixed cost excluded
    assert math.isclose(sb.raw_cost, 130.0)


def test_arc_fixed_cost_only_when_used():
    inst = ProblemInstance(
        instance_id="fc", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=8.0)],
        arcs=[Arc(arc_id="used", from_node="S1", to_node="D1", unit_cost=1.0, fixed_cost=7.0),
              Arc(arc_id="idle", from_node="S1", to_node="D1", unit_cost=1.0, fixed_cost=99.0)],
    )
    sb = score(inst, make_solution(inst, {"used": 8.0}), COST)
    assert math.isclose(sb.raw_cost, 8.0 + 7.0)  # idle arc's fixed cost excluded


def test_raw_lead_time_is_max_over_used_arcs():
    sb = score(fx.golden_transportation(), fx.transportation_optimal_solution(), COST)
    # uses S1-D1 (lead 3) and S2-D2 (lead 4)
    assert math.isclose(sb.raw_lead_time, 4.0)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def test_optimum_solution_scores_one():
    sb = score(fx.golden_transportation(), fx.transportation_optimal_solution(), COST)
    assert math.isclose(sb.normalized_score, 1.0)


def test_twice_optimal_cost_scores_half():
    inst = ProblemInstance(
        instance_id="half", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
               Node(node_id="S2", node_type=NodeType.SOURCE, supply=10.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=10.0)],
        arcs=[Arc(arc_id="cheap", from_node="S1", to_node="D1", unit_cost=2.0),
              Arc(arc_id="dear", from_node="S2", to_node="D1", unit_cost=4.0)],
        known_optimum=KnownOptimum(objective_value=20.0, source=OptimumSource.SOLVER_VERIFIED, verified=True),
    )
    sb = score(inst, make_solution(inst, {"dear": 10.0}), COST)  # cost 40 = 2*optimum
    assert sb.feasible
    assert math.isclose(sb.normalized_score, 0.5)


def test_below_labeled_optimum_is_capped_and_flagged():
    inst = ProblemInstance(
        instance_id="loose", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=10.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=2.0)],
        known_optimum=KnownOptimum(objective_value=30.0, source=OptimumSource.BENCHMARK_LABEL),  # loose label
    )
    sb = score(inst, make_solution(inst, {"a": 10.0}), COST)  # achieved cost 20 < label 30
    assert math.isclose(sb.normalized_score, 1.0)
    assert sb.diagnostics["below_labeled_optimum"] is True


# ---------------------------------------------------------------------------
# The critical ordering invariant
# ---------------------------------------------------------------------------
_INV_INST = fx.golden_transportation()


@settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
@given(random_flow_solution_strategy(_INV_INST))
def test_feasible_iff_fitness_nonnegative(sol):
    sb = score(_INV_INST, sol, COST)
    if sb.feasible:
        assert 0.0 <= sb.final_fitness <= 1.0
    else:
        assert sb.final_fitness < 0.0


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(random_flow_solution_strategy(_INV_INST))
def test_every_infeasible_ranks_below_the_feasible_optimum(sol):
    feasible = score(_INV_INST, fx.transportation_optimal_solution(), COST)
    candidate = score(_INV_INST, sol, COST)
    if not candidate.feasible:
        assert candidate.final_fitness < feasible.final_fitness


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(feasible_transportation_solution())
def test_feasible_branch_always_lands_in_unit_interval(sol):
    """Reliably exercises the FEASIBLE side of the invariant (the random strategy
    almost never produces a feasible solution by chance)."""
    sb = score(_INV_INST, sol, COST)
    assert sb.feasible is True
    assert 0.0 <= sb.final_fitness <= 1.0
    assert sb.final_fitness == sb.normalized_score


def test_more_violations_strictly_lower():
    inst = fx.golden_transportation()
    one = score(inst, make_solution(inst, {"S1-D1": 8.0}), COST)  # D2 unmet (1 violation)
    two = score(inst, make_solution(inst, {}), COST)  # D1 and D2 unmet (2 violations)
    assert len(two.violations) > len(one.violations)
    assert two.final_fitness < one.final_fitness


def test_increasing_violation_magnitude_never_increases_fitness():
    inst = ProblemInstance(
        instance_id="mono", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=10.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0)],
    )
    less = score(inst, make_solution(inst, {"a": 3.0}), COST)  # unmet 7
    more = score(inst, make_solution(inst, {"a": 2.0}), COST)  # unmet 8 (worse)
    # identical count/type, larger magnitude => STRICTLY lower (catches a magnitude-blind scorer)
    assert more.final_fitness < less.final_fitness


def test_malformed_scores_below_any_wellformed_infeasible():
    inst = fx.golden_transportation()
    malformed = score(inst, Solution(solution_id="m", instance_id=inst.instance_id,
                                      flows=[FlowAssignment(arc_id="DOES-NOT-EXIST", quantity=1.0)]), COST)
    infeasible = score(inst, make_solution(inst, {}), COST)
    assert not malformed.feasible
    assert ViolationType.MALFORMED_SOLUTION in _types(malformed)
    assert malformed.final_fitness < infeasible.final_fitness


def test_malformed_below_even_a_saturated_huge_penalty_infeasible():
    """Even when an infeasible penalty is so large the soft clamp saturates,
    the malformed floor stays strictly below it."""
    inst = ProblemInstance(
        instance_id="sat", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S", node_type=NodeType.SOURCE, supply=1e302),
               Node(node_id="D", node_type=NodeType.SINK, demand=1.0)],
        arcs=[Arc(arc_id="a", from_node="S", to_node="D", unit_cost=1.0, capacity=1.0)],
    )
    huge = score(inst, make_solution(inst, {"a": 1e300}), COST)  # enormous over-capacity penalty
    malformed = score(inst, Solution(solution_id="m", instance_id="sat",
                                     flows=[FlowAssignment(arc_id="ghost", quantity=1.0)]), COST)
    assert not huge.feasible
    assert malformed.final_fitness < huge.final_fitness


def test_aggregated_overflow_scores_malformed_not_crash():
    inst = fx.golden_transportation()
    overflow = Solution(
        solution_id="ovf", instance_id=inst.instance_id,
        flows=[FlowAssignment(arc_id="S1-D1", quantity=1e308), FlowAssignment(arc_id="S1-D1", quantity=1e308)],
    )
    sb = score(inst, overflow, COST)  # must not raise
    assert not sb.feasible
    assert ViolationType.MALFORMED_SOLUTION in _types(sb)


def test_deep_network_does_not_overflow_recursion():
    n = 3000
    nodes = [Node(node_id="S", node_type=NodeType.SOURCE, supply=5.0)]
    nodes += [Node(node_id=f"T{i}", node_type=NodeType.TRANSSHIPMENT) for i in range(n)]
    nodes += [Node(node_id="D", node_type=NodeType.SINK, demand=5.0)]
    arcs = [Arc(arc_id="S-T0", from_node="S", to_node="T0", unit_cost=1.0)]
    arcs += [Arc(arc_id=f"T{i}-T{i+1}", from_node=f"T{i}", to_node=f"T{i+1}", unit_cost=1.0) for i in range(n - 1)]
    arcs += [Arc(arc_id=f"T{n-1}-D", from_node=f"T{n-1}", to_node="D", unit_cost=1.0)]
    inst = ProblemInstance(instance_id="deep", source="fixture",
                           problem_class=ProblemClass.TRANSSHIPMENT, nodes=nodes, arcs=arcs)
    flows = ({"S-T0": 5.0, f"T{n-1}-D": 5.0})
    flows.update({f"T{i}-T{i+1}": 5.0 for i in range(n - 1)})
    sb = score(inst, make_solution(inst, flows), COST)  # exercises the resilience max-flow on a deep graph
    assert sb.feasible  # no RecursionError


def test_no_optimum_cost_reference_is_solution_dependent():
    inst = ProblemInstance(
        instance_id="noopt", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=100.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=10.0)],
        arcs=[Arc(arc_id="cheap", from_node="S1", to_node="D1", unit_cost=1.0),
              Arc(arc_id="dear", from_node="S1", to_node="D1", unit_cost=9.0)],
    )  # no known_optimum attached
    cheap = score(inst, make_solution(inst, {"cheap": 10.0}), COST)
    dear = score(inst, make_solution(inst, {"dear": 10.0}), COST)
    # without an optimum, weighted_objective must still reflect cost (not self-normalize to 1.0)
    assert cheap.diagnostics["norm_cost"] != dear.diagnostics["norm_cost"]
    assert cheap.diagnostics["norm_cost"] < dear.diagnostics["norm_cost"]


def test_below_labeled_optimum_flag_in_zero_cost_branch():
    inst = ProblemInstance(
        instance_id="zopt", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=5.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=0.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0)],
        known_optimum=KnownOptimum(objective_value=5.0, source=OptimumSource.BENCHMARK_LABEL),  # positive but loose
    )
    sb = score(inst, Solution(solution_id="z", instance_id="zopt", flows=[]), COST)  # zero achieved cost
    assert math.isclose(sb.normalized_score, 1.0)
    assert sb.diagnostics["below_labeled_optimum"] is True


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------
def test_same_inputs_score_identically_1000x():
    inst, sol = fx.golden_transportation(), fx.transportation_optimal_solution()
    first = score(inst, sol, COST).final_fitness
    for _ in range(1000):
        assert score(inst, sol, COST).final_fitness == first


def test_shuffling_flows_does_not_change_score():
    inst = fx.golden_facility_location()
    base = fx.facility_optimal_solution()
    flows = list(base.flows)
    rng = random.Random(12345)
    results = set()
    for _ in range(50):
        rng.shuffle(flows)
        shuffled = base.model_copy(update={"flows": list(flows)})
        results.add(score(inst, shuffled, COST).final_fitness)
    assert len(results) == 1


def test_order_independence_under_nonassociative_float_sums():
    """The real determinism guard: duplicate same-arc flows with magnitude-disparate
    quantities make naive float accumulation order-dependent. fsum aggregation must
    keep BOTH final_fitness and feasible constant across every permutation."""
    inst = fx.golden_transportation()
    quantities = [1e15, 1e15, 0.1, 0.1]  # float sum is non-associative at these scales
    fitnesses, feasibilities = set(), set()
    for perm in itertools.permutations(range(4)):
        flows = [FlowAssignment(arc_id="S1-D1", quantity=quantities[i]) for i in perm]
        sb = score(inst, Solution(solution_id="dup", instance_id=inst.instance_id, flows=flows), COST)
        fitnesses.add(sb.final_fitness)
        feasibilities.add(sb.feasible)
    assert len(fitnesses) == 1
    assert len(feasibilities) == 1


def test_scoring_uses_no_network_and_is_time_independent(monkeypatch):
    # Defense in depth: the scorer must not open a socket...
    def _boom(*a, **k):
        raise AssertionError("scorer attempted a network/socket call")

    monkeypatch.setattr(socket, "socket", _boom)
    inst, sol = fx.golden_transportation(), fx.transportation_optimal_solution()
    first = score(inst, sol, COST)
    time.sleep(0.001)  # ...and the wall clock must not enter the number
    second = score(inst, sol, COST)
    assert first.feasible
    assert first.final_fitness == second.final_fitness
    # computed_at is metadata only: it may legitimately differ between calls
    assert first.computed_at and second.computed_at


def test_score_is_stamped_with_a_semver():
    sb = score(fx.golden_transportation(), fx.transportation_optimal_solution(), COST)
    assert sb.scorer_version == "1.0.0"  # independently pinned, not just == the import
    assert re.fullmatch(r"\d+\.\d+\.\d+", sb.scorer_version)
    assert sb.scorer_version == SCORER_VERSION
    # ISO-8601 timestamp present (metadata only, never in the fitness number)
    assert "T" in sb.computed_at


# ---------------------------------------------------------------------------
# Weights (B8 hook)
# ---------------------------------------------------------------------------
def test_raising_cost_weight_shifts_weighted_objective():
    inst, sol = fx.golden_transportation(), fx.transportation_optimal_solution()
    cost_heavy = score(inst, sol, ObjectiveWeights(cost_weight=8.0, lead_time_weight=1.0, risk_weight=1.0))
    risk_heavy = score(inst, sol, ObjectiveWeights(cost_weight=1.0, lead_time_weight=1.0, risk_weight=8.0))
    # norm_cost (==1.0 at optimum) exceeds norm_risk, so weighting cost higher raises the blend.
    assert cost_heavy.diagnostics["norm_cost"] > cost_heavy.diagnostics["norm_risk"]
    assert cost_heavy.weighted_objective > risk_heavy.weighted_objective


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
def test_single_score_under_one_millisecond():
    inst, sol = fx.golden_transportation(), fx.transportation_optimal_solution()
    for _ in range(500):  # warm up
        score(inst, sol, COST)
    n = 5000
    start = time.perf_counter()
    for _ in range(n):
        score(inst, sol, COST)
    avg_ms = (time.perf_counter() - start) / n * 1000.0
    assert avg_ms < 1.0, f"average score {avg_ms:.3f} ms exceeds 1 ms budget"


def test_ten_thousand_scorings_within_budget():
    inst, sol = fx.golden_transportation(), fx.transportation_optimal_solution()
    start = time.perf_counter()
    for _ in range(10000):
        score(inst, sol, COST)
    assert time.perf_counter() - start < 5.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_empty_solution_on_positive_demand_is_infeasible_not_crash():
    inst = fx.golden_transportation()
    sb = score(inst, Solution(solution_id="e", instance_id=inst.instance_id, flows=[]), COST)
    assert not sb.feasible and sb.final_fitness < 0.0


def test_nonexistent_arc_is_malformed_graceful():
    inst = fx.golden_transportation()
    sb = score(inst, Solution(solution_id="bad", instance_id=inst.instance_id,
                              flows=[FlowAssignment(arc_id="ghost", quantity=1.0)]), COST)
    assert _types(sb) == {ViolationType.MALFORMED_SOLUTION}


def test_single_node_degenerate_instance():
    inst = ProblemInstance(
        instance_id="solo", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=5.0)], arcs=[],
    )
    sb = score(inst, Solution(solution_id="s", instance_id="solo", flows=[]), COST)
    assert sb.feasible


def test_zero_demand_instance_is_trivially_feasible():
    inst = ProblemInstance(
        instance_id="zero", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=5.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=0.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0)],
        known_optimum=KnownOptimum(objective_value=0.0, source=OptimumSource.SOLVER_VERIFIED, verified=True),
    )
    sb = score(inst, Solution(solution_id="z", instance_id="zero", flows=[]), COST)
    assert sb.feasible
    assert math.isclose(sb.normalized_score, 1.0)


def test_fully_uncapacitated_never_yields_capacity_violation():
    inst = fx.golden_transportation()  # no capacities anywhere
    # ship within supply but heavily; no arc/node capacity can be violated
    sb = score(inst, make_solution(inst, {"S1-D1": 10.0, "S2-D2": 10.0}), COST)
    assert ViolationType.OVER_ARC_CAPACITY not in _types(sb)
    assert ViolationType.OVER_NODE_CAPACITY not in _types(sb)
