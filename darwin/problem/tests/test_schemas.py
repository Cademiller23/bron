"""§8.1 Schema tests — the contract's guardrails."""

import pytest
from pydantic import ValidationError

from darwin.problem.schemas import (
    Arc,
    Difficulty,
    InstanceMetadata,
    KnownOptimum,
    Node,
    NodeType,
    ObjectiveWeights,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
)


def _mini(**overrides) -> ProblemInstance:
    kwargs = dict(
        instance_id="i1",
        source="fixture",
        problem_class=ProblemClass.TRANSPORTATION,
        nodes=[
            Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
            Node(node_id="D1", node_type=NodeType.SINK, demand=8.0),
        ],
        arcs=[Arc(arc_id="a1", from_node="S1", to_node="D1", unit_cost=2.0)],
        metadata=InstanceMetadata(difficulty=Difficulty.EASY),
    )
    kwargs.update(overrides)
    return ProblemInstance(**kwargs)


def test_valid_instance_constructs_and_derives_metadata():
    inst = _mini()
    assert inst.metadata.num_nodes == 2
    assert inst.metadata.num_arcs == 1
    assert inst.total_supply() == 10.0
    assert inst.total_demand() == 8.0


def test_arc_referencing_unknown_node_raises():
    with pytest.raises(ValidationError):
        _mini(arcs=[Arc(arc_id="a1", from_node="S1", to_node="NOPE", unit_cost=1.0)])


def test_duplicate_node_id_raises():
    with pytest.raises(ValidationError):
        _mini(
            nodes=[
                Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
                Node(node_id="S1", node_type=NodeType.SINK, demand=8.0),
            ]
        )


def test_duplicate_arc_id_raises():
    with pytest.raises(ValidationError):
        _mini(
            arcs=[
                Arc(arc_id="a1", from_node="S1", to_node="D1", unit_cost=1.0),
                Arc(arc_id="a1", from_node="S1", to_node="D1", unit_cost=2.0),
            ]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"capacity": -1.0},
        {"supply": -5.0},
        {"demand": -2.0},
        {"fixed_cost": -3.0},
    ],
)
def test_negative_node_fields_raise(kwargs):
    with pytest.raises(ValidationError):
        Node(node_id="x", node_type=NodeType.SINK, **kwargs)


@pytest.mark.parametrize("kwargs", [{"unit_cost": -1.0}, {"lead_time": -1.0}, {"capacity": -1.0}])
def test_negative_arc_fields_raise(kwargs):
    base = dict(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0)
    base.update(kwargs)
    with pytest.raises(ValidationError):
        Arc(**base)


@pytest.mark.parametrize("value", [-0.01, 1.01, 2.0])
def test_risk_score_out_of_range_raises(value):
    with pytest.raises(ValidationError):
        Node(node_id="x", node_type=NodeType.SINK, risk_score=value)


def test_nan_and_inf_rejected():
    with pytest.raises(ValidationError):
        Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=float("inf"))
    with pytest.raises(ValidationError):
        Node(node_id="x", node_type=NodeType.SINK, demand=float("nan"))


@pytest.mark.parametrize("coords", [(float("inf"), 0.0), (0.0, float("nan")), (float("-inf"), 1.0)])
def test_non_finite_coordinates_rejected(coords):
    with pytest.raises(ValidationError):
        Node(node_id="x", node_type=NodeType.SINK, coordinates=coords)


def test_total_supply_less_than_demand_raises():
    with pytest.raises(ValidationError):
        _mini(
            nodes=[
                Node(node_id="S1", node_type=NodeType.SOURCE, supply=1.0),
                Node(node_id="D1", node_type=NodeType.SINK, demand=9.0),
            ]
        )


def test_immutability():
    inst = _mini()
    with pytest.raises(ValidationError):
        inst.instance_id = "other"
    with pytest.raises(ValidationError):
        inst.nodes[0].supply = 999.0  # nested frozen too


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        Node(node_id="x", node_type=NodeType.SINK, bogus=1)
    with pytest.raises(ValidationError):
        _mini(unexpected="field")


def test_serialization_round_trip():
    inst = _mini(
        known_optimum=KnownOptimum(objective_value=16.0, source=OptimumSource.SOLVER_VERIFIED, verified=True)
    )
    dumped = inst.model_dump()
    restored = ProblemInstance(**dumped)
    assert restored == inst
    assert restored.model_dump() == dumped


def test_objective_weights_normalize_to_one():
    w = ObjectiveWeights(cost_weight=2.0, lead_time_weight=1.0, risk_weight=1.0)
    assert abs((w.cost_weight + w.lead_time_weight + w.risk_weight) - 1.0) < 1e-9
    assert abs(w.cost_weight - 0.5) < 1e-9


def test_objective_weights_negative_raises():
    with pytest.raises(ValidationError):
        ObjectiveWeights(cost_weight=-1.0, lead_time_weight=1.0, risk_weight=1.0)


def test_objective_weights_all_zero_raises():
    with pytest.raises(ValidationError):
        ObjectiveWeights(cost_weight=0.0, lead_time_weight=0.0, risk_weight=0.0)


def test_uncapacitated_fields_default_none():
    n = Node(node_id="x", node_type=NodeType.SOURCE, supply=5.0)
    assert n.capacity is None and n.fixed_cost is None and n.risk_score is None
