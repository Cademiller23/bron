"""§8.2 Resilience tests — the differentiator, validated against hand math."""

import math

from darwin.problem.resilience import ALPHA, BETA, GAMMA, compute_resilience
from darwin.problem.schemas import Arc, Node, NodeType, ProblemClass, ProblemInstance


def _instance(nodes, arcs, pclass=ProblemClass.TRANSPORTATION):
    return ProblemInstance(
        instance_id="res", source="fixture", problem_class=pclass, nodes=nodes, arcs=arcs
    )


def test_all_demand_from_one_source_gives_concentration_one():
    inst = _instance(
        nodes=[
            Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="S2", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="D1", node_type=NodeType.SINK, demand=10.0),
        ],
        arcs=[
            Arc(arc_id="S1-D1", from_node="S1", to_node="D1", unit_cost=1.0),
            Arc(arc_id="S2-D1", from_node="S2", to_node="D1", unit_cost=1.0),
        ],
    )
    res = compute_resilience(inst, {"S1-D1": 10.0})
    assert math.isclose(res.concentration, 1.0)


def test_demand_split_evenly_across_k_sources_gives_one_over_k():
    for k in (2, 3, 4):
        sources = [Node(node_id=f"S{i}", node_type=NodeType.SOURCE, supply=10.0) for i in range(k)]
        sink = Node(node_id="D1", node_type=NodeType.SINK, demand=float(k))
        arcs = [Arc(arc_id=f"S{i}-D1", from_node=f"S{i}", to_node="D1", unit_cost=1.0) for i in range(k)]
        inst = _instance(sources + [sink], arcs)
        flow = {f"S{i}-D1": 1.0 for i in range(k)}
        res = compute_resilience(inst, flow)
        assert math.isclose(res.concentration, 1.0 / k), (k, res.concentration)


def test_exposure_matches_hand_computed_three_arc_example():
    inst = _instance(
        nodes=[
            Node(node_id="S1", node_type=NodeType.SOURCE, supply=40.0),
            Node(node_id="S2", node_type=NodeType.SOURCE, supply=40.0),
            Node(node_id="D1", node_type=NodeType.SINK, demand=10.0),
            Node(node_id="D2", node_type=NodeType.SINK, demand=20.0),
        ],
        arcs=[
            Arc(arc_id="a1", from_node="S1", to_node="D1", unit_cost=1.0, risk_score=0.2),
            Arc(arc_id="a2", from_node="S1", to_node="D2", unit_cost=1.0, risk_score=0.5),
            Arc(arc_id="a3", from_node="S2", to_node="D1", unit_cost=1.0, risk_score=0.9),
        ],
    )
    # used: a1 (10 @ .2), a2 (20 @ .5); a3 unused. E = (10*.2 + 20*.5)/30 = 0.4
    res = compute_resilience(inst, {"a1": 10.0, "a2": 20.0, "a3": 0.0})
    assert math.isclose(res.exposure, 0.4)


def test_worst_case_single_failure_unmet_fraction():
    inst = _instance(
        nodes=[
            Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="S2", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="D1", node_type=NodeType.SINK, demand=15.0),
        ],
        arcs=[
            Arc(arc_id="S1-D1", from_node="S1", to_node="D1", unit_cost=1.0),
            Arc(arc_id="S2-D1", from_node="S2", to_node="D1", unit_cost=1.0),
        ],
    )
    # S1 is the biggest source (outflow 10); removing it leaves S2's 10 vs demand 15.
    res = compute_resilience(inst, {"S1-D1": 10.0, "S2-D1": 5.0})
    assert res.removed_source == "S1"
    assert math.isclose(res.worst_case_unmet, 5.0 / 15.0)
    assert math.isclose(res.deliverable_after_failure, 10.0)


def test_zero_risk_instance_has_zero_exposure():
    inst = _instance(
        nodes=[
            Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="D1", node_type=NodeType.SINK, demand=8.0),
        ],
        arcs=[Arc(arc_id="S1-D1", from_node="S1", to_node="D1", unit_cost=1.0)],
    )
    res = compute_resilience(inst, {"S1-D1": 8.0})
    assert res.exposure == 0.0


def test_raw_risk_matches_hand_derived_blend():
    inst = _instance(
        nodes=[
            Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="S2", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="D1", node_type=NodeType.SINK, demand=15.0),
        ],
        arcs=[
            Arc(arc_id="S1-D1", from_node="S1", to_node="D1", unit_cost=1.0, risk_score=0.5),
            Arc(arc_id="S2-D1", from_node="S2", to_node="D1", unit_cost=1.0, risk_score=0.5),
        ],
    )
    res = compute_resilience(inst, {"S1-D1": 10.0, "S2-D1": 5.0})
    # Hand-derived for this instance (independent of the code's own outputs):
    #   C  = largest supplier share at D1 = 10/15 = 2/3
    #   E  = (10*0.5 + 5*0.5) / 15 = 0.5
    #   W  = remove S1 (biggest, outflow 10); S2 delivers 10 vs demand 15 => 5/15 = 1/3
    assert math.isclose(res.concentration, 2.0 / 3.0)
    assert math.isclose(res.exposure, 0.5)
    assert math.isclose(res.worst_case_unmet, 1.0 / 3.0)
    hand_raw = 0.30 * (2.0 / 3.0) + 0.30 * 0.5 + 0.40 * (1.0 / 3.0)
    assert math.isclose(res.raw_risk, hand_raw)
    # and the module constants are what the hand derivation assumed
    assert (ALPHA, BETA, GAMMA) == (0.30, 0.30, 0.40)


def test_empty_flow_is_safe():
    inst = _instance(
        nodes=[
            Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="D1", node_type=NodeType.SINK, demand=8.0),
        ],
        arcs=[Arc(arc_id="S1-D1", from_node="S1", to_node="D1", unit_cost=1.0)],
    )
    res = compute_resilience(inst, {})
    assert res.concentration == 0.0 and res.exposure == 0.0
