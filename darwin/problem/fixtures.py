"""Hand-built tiny "golden" instances whose answers are computed by hand.

These are the **test anchors** for the whole B1 suite: small enough to verify
the optimum with a pencil, rich enough to exercise every scorer branch.

* :func:`golden_transportation` — 2 sources × 2 sinks, hand optimum **23**.
* :func:`golden_facility_location` — 2 optional warehouses, hand optimum **130**.
* :func:`golden_vrp` — 3 customers, capacity 10, optimum **40 + √200 ≈ 54.142**.

Each ``golden_*`` builder has a matching ``*_optimal_solution`` returning a flow
(or route) assignment that achieves the optimum exactly.
"""

import math

from darwin.problem.schemas import (
    AdditionalConstraint,
    Arc,
    ConstraintType,
    Difficulty,
    FlowAssignment,
    InstanceMetadata,
    KnownOptimum,
    Node,
    NodeType,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
    Route,
    Solution,
)

# ---------------------------------------------------------------------------
# Golden #1 — transportation. Hand optimum = 23.
#   S1->D1 = 8 (cost 2), S2->D2 = 7 (cost 1)  =>  8*2 + 7*1 = 23
# ---------------------------------------------------------------------------
def golden_transportation() -> ProblemInstance:
    return ProblemInstance(
        instance_id="golden-transportation",
        source="fixture",
        problem_class=ProblemClass.TRANSPORTATION,
        nodes=[
            Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0, risk_score=0.1),
            Node(node_id="S2", node_type=NodeType.SOURCE, supply=10.0, risk_score=0.4),
            Node(node_id="D1", node_type=NodeType.SINK, demand=8.0),
            Node(node_id="D2", node_type=NodeType.SINK, demand=7.0),
        ],
        arcs=[
            Arc(arc_id="S1-D1", from_node="S1", to_node="D1", unit_cost=2.0, lead_time=3.0, risk_score=0.1),
            Arc(arc_id="S1-D2", from_node="S1", to_node="D2", unit_cost=3.0, lead_time=2.0, risk_score=0.2),
            Arc(arc_id="S2-D1", from_node="S2", to_node="D1", unit_cost=5.0, lead_time=1.0, risk_score=0.5),
            Arc(arc_id="S2-D2", from_node="S2", to_node="D2", unit_cost=1.0, lead_time=4.0, risk_score=0.3),
        ],
        known_optimum=KnownOptimum(
            objective_value=23.0, source=OptimumSource.SOLVER_VERIFIED, verified=True, solver_used="hand"
        ),
        metadata=InstanceMetadata(
            difficulty=Difficulty.EASY, notes="2x2 transportation; hand optimum 23"
        ),
    )


def transportation_optimal_solution() -> Solution:
    return Solution(
        solution_id="golden-transportation-opt",
        instance_id="golden-transportation",
        flows=[
            FlowAssignment(arc_id="S1-D1", quantity=8.0),
            FlowAssignment(arc_id="S2-D2", quantity=7.0),
        ],
        produced_by="hand",
    )


# ---------------------------------------------------------------------------
# Golden #2 — facility location. Hand optimum = 130.
#   Open exactly one warehouse and serve both customers through it.
#   Open W1: F->W1=20 (20) + W1->C1=10 (10) + W1->C2=10 (50) + fixed 50 = 130.
#   (Open both costs 140; open W2 is the symmetric 130.)
# ---------------------------------------------------------------------------
def golden_facility_location() -> ProblemInstance:
    return ProblemInstance(
        instance_id="golden-facility",
        source="fixture",
        problem_class=ProblemClass.FACILITY_LOCATION,
        nodes=[
            Node(node_id="F", node_type=NodeType.SOURCE, supply=100.0),
            Node(node_id="W1", node_type=NodeType.TRANSSHIPMENT, is_optional=True, fixed_cost=50.0, risk_score=0.2),
            Node(node_id="W2", node_type=NodeType.TRANSSHIPMENT, is_optional=True, fixed_cost=50.0, risk_score=0.2),
            Node(node_id="C1", node_type=NodeType.SINK, demand=10.0),
            Node(node_id="C2", node_type=NodeType.SINK, demand=10.0),
        ],
        arcs=[
            Arc(arc_id="F-W1", from_node="F", to_node="W1", unit_cost=1.0),
            Arc(arc_id="F-W2", from_node="F", to_node="W2", unit_cost=1.0),
            Arc(arc_id="W1-C1", from_node="W1", to_node="C1", unit_cost=1.0),
            Arc(arc_id="W1-C2", from_node="W1", to_node="C2", unit_cost=5.0),
            Arc(arc_id="W2-C1", from_node="W2", to_node="C1", unit_cost=5.0),
            Arc(arc_id="W2-C2", from_node="W2", to_node="C2", unit_cost=1.0),
        ],
        known_optimum=KnownOptimum(
            objective_value=130.0, source=OptimumSource.SOLVER_VERIFIED, verified=True, solver_used="hand"
        ),
        metadata=InstanceMetadata(
            difficulty=Difficulty.MEDIUM, notes="2 optional warehouses; hand optimum 130 (open one)"
        ),
    )


def facility_optimal_solution() -> Solution:
    return Solution(
        solution_id="golden-facility-opt",
        instance_id="golden-facility",
        flows=[
            FlowAssignment(arc_id="F-W1", quantity=20.0),
            FlowAssignment(arc_id="W1-C1", quantity=10.0),
            FlowAssignment(arc_id="W1-C2", quantity=10.0),
        ],
        open_facilities=["W1"],
        produced_by="hand",
    )


# ---------------------------------------------------------------------------
# Golden #3 — CVRP. Optimum = 40 + sqrt(200) ≈ 54.142 (capacity forces a split).
# ---------------------------------------------------------------------------
_VRP_OPTIMUM = 40.0 + math.hypot(10.0, 10.0)


def golden_vrp() -> ProblemInstance:
    return ProblemInstance(
        instance_id="golden-vrp",
        source="fixture",
        problem_class=ProblemClass.VEHICLE_ROUTING,
        nodes=[
            Node(node_id="depot", node_type=NodeType.SOURCE, supply=100.0, coordinates=(0.0, 0.0)),
            Node(node_id="c1", node_type=NodeType.SINK, demand=5.0, coordinates=(10.0, 0.0), risk_score=0.1),
            Node(node_id="c2", node_type=NodeType.SINK, demand=5.0, coordinates=(0.0, 10.0), risk_score=0.2),
            Node(node_id="c3", node_type=NodeType.SINK, demand=5.0, coordinates=(10.0, 10.0), risk_score=0.3),
        ],
        # Routing distances are encoded by coordinates; arcs are not required for
        # the VRP branch, but a depot->customer arc set keeps the graph connected
        # for any downstream tooling that expects arcs.
        arcs=[
            Arc(arc_id="depot-c1", from_node="depot", to_node="c1", unit_cost=10.0),
            Arc(arc_id="depot-c2", from_node="depot", to_node="c2", unit_cost=10.0),
            Arc(arc_id="depot-c3", from_node="depot", to_node="c3", unit_cost=14.142135623730951),
        ],
        additional_constraints=[
            AdditionalConstraint(
                constraint_id="veh-cap",
                constraint_type=ConstraintType.CAPACITY,
                parameters={"vehicle_capacity": 10.0},
                description="each vehicle carries at most 10 units",
            )
        ],
        known_optimum=KnownOptimum(
            objective_value=_VRP_OPTIMUM, source=OptimumSource.SOLVER_VERIFIED, verified=True, solver_used="hand"
        ),
        metadata=InstanceMetadata(difficulty=Difficulty.MEDIUM, notes="3-customer CVRP; capacity 10"),
    )


def vrp_optimal_solution() -> Solution:
    return Solution(
        solution_id="golden-vrp-opt",
        instance_id="golden-vrp",
        routes=[
            Route(vehicle_id="v1", node_sequence=["depot", "c1", "c3", "depot"], load=10.0),
            Route(vehicle_id="v2", node_sequence=["depot", "c2", "depot"], load=5.0),
        ],
        produced_by="hand",
    )


VRP_OPTIMUM = _VRP_OPTIMUM
