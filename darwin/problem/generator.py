"""The live fresh-instance generator — the "this isn't memorized" moment.

``generate_instance(seed, problem_class, size_params)`` builds a structurally
identical but numerically fresh problem (new costs/capacities/demands/coords from
a *seeded* RNG), then immediately calls the oracle to compute and attach the
verified optimum — so a generated instance has a real denominator the instant it
is created.

* Same seed → identical instance (reproducible rehearsal).
* Fresh seed on stage → a problem the swarm has provably never seen.
* A guard rejects instances that are numerically identical to a preloaded one.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

from darwin.problem import oracle
from darwin.problem.schemas import (
    AdditionalConstraint,
    Arc,
    ConstraintType,
    Difficulty,
    InstanceMetadata,
    KnownOptimum,
    Node,
    NodeType,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
)

_DEFAULTS: Dict[ProblemClass, Dict[str, Any]] = {
    ProblemClass.TRANSPORTATION: {"num_sources": 3, "num_sinks": 3},
    ProblemClass.TRANSSHIPMENT: {"num_sources": 2, "num_transshipments": 2, "num_sinks": 3},
    ProblemClass.FACILITY_LOCATION: {"num_sources": 1, "num_optional": 3, "num_sinks": 3},
    ProblemClass.VEHICLE_ROUTING: {"num_customers": 4, "capacity": 12.0},
}


class GenerationError(RuntimeError):
    pass


def _signature(instance: ProblemInstance) -> Tuple:
    nodes = tuple(
        (n.node_id, n.node_type.value, n.supply, n.demand, n.fixed_cost, n.capacity, n.coordinates)
        for n in instance.nodes
    )
    arcs = tuple(
        (a.arc_id, a.from_node, a.to_node, round(a.unit_cost, 6), a.capacity, a.fixed_cost)
        for a in instance.arcs
    )
    return (instance.problem_class.value, nodes, arcs)


def generate_instance(
    seed: int,
    problem_class: ProblemClass = ProblemClass.TRANSPORTATION,
    size_params: Optional[Dict[str, Any]] = None,
    existing: Optional[List[ProblemInstance]] = None,
    max_attempts: int = 8,
) -> ProblemInstance:
    """Generate a fresh, feasible, optimum-attached instance."""
    params = dict(_DEFAULTS[problem_class])
    if size_params:
        params.update(size_params)

    existing_sigs = {_signature(i) for i in (existing or [])}

    last_error: Optional[str] = None
    for attempt in range(max_attempts):
        rng = random.Random((seed * 1_000_003) ^ (attempt * 2_654_435_761))
        instance = _build(seed, attempt, problem_class, params, rng)

        if _signature(instance) in existing_sigs:
            last_error = "collision with a preloaded instance"
            continue

        result = oracle.solve_optimum(instance)
        if result.status != oracle.STATUS_OPTIMAL or result.objective_value is None:
            last_error = f"oracle status {result.status}"
            continue

        verified = instance.model_copy(
            update={
                "known_optimum": KnownOptimum(
                    objective_value=round(result.objective_value, 9),
                    source=OptimumSource.SOLVER_VERIFIED,
                    verified=True,
                    solver_used=f"oracle:{result.backend}",
                )
            }
        )
        return verified

    raise GenerationError(
        f"could not generate a usable instance after {max_attempts} attempts: {last_error}"
    )


def _build(seed, attempt, problem_class, params, rng) -> ProblemInstance:
    builders = {
        ProblemClass.TRANSPORTATION: _build_transportation,
        ProblemClass.TRANSSHIPMENT: _build_transshipment,
        ProblemClass.FACILITY_LOCATION: _build_facility,
        ProblemClass.VEHICLE_ROUTING: _build_vrp,
    }
    return builders[problem_class](seed, attempt, params, rng)


def _iid(seed: int, attempt: int, problem_class: ProblemClass, size: str) -> str:
    suffix = f"-a{attempt}" if attempt else ""
    return f"generated-{problem_class.value.lower()}-s{seed}{suffix}-{size}"


def _build_transportation(seed, attempt, params, rng) -> ProblemInstance:
    ns, nd = params["num_sources"], params["num_sinks"]
    demands = [float(rng.randint(3, 12)) for _ in range(nd)]
    total_demand = sum(demands)
    # supplies comfortably cover demand
    base = total_demand / ns
    supplies = [round(base + rng.uniform(1, 6), 2) for _ in range(ns)]

    nodes = [Node(node_id=f"S{i+1}", node_type=NodeType.SOURCE, supply=supplies[i], risk_score=round(rng.uniform(0, 0.6), 3)) for i in range(ns)]
    nodes += [Node(node_id=f"D{j+1}", node_type=NodeType.SINK, demand=demands[j]) for j in range(nd)]
    arcs = [
        Arc(
            arc_id=f"S{i+1}-D{j+1}",
            from_node=f"S{i+1}",
            to_node=f"D{j+1}",
            unit_cost=round(rng.uniform(1, 10), 2),
            lead_time=round(rng.uniform(1, 6), 2),
            risk_score=round(rng.uniform(0, 0.7), 3),
        )
        for i in range(ns)
        for j in range(nd)
    ]
    return ProblemInstance(
        instance_id=_iid(seed, attempt, ProblemClass.TRANSPORTATION, f"{ns}x{nd}"),
        source="generated",
        problem_class=ProblemClass.TRANSPORTATION,
        nodes=nodes,
        arcs=arcs,
        metadata=InstanceMetadata(difficulty=Difficulty.EASY, notes=f"generated seed {seed}"),
    )


def _build_transshipment(seed, attempt, params, rng) -> ProblemInstance:
    ns, nt, nd = params["num_sources"], params["num_transshipments"], params["num_sinks"]
    demands = [float(rng.randint(3, 10)) for _ in range(nd)]
    total_demand = sum(demands)
    base = total_demand / ns
    supplies = [round(base + rng.uniform(2, 8), 2) for _ in range(ns)]

    nodes = [Node(node_id=f"S{i+1}", node_type=NodeType.SOURCE, supply=supplies[i], risk_score=round(rng.uniform(0, 0.5), 3)) for i in range(ns)]
    nodes += [Node(node_id=f"T{k+1}", node_type=NodeType.TRANSSHIPMENT) for k in range(nt)]
    nodes += [Node(node_id=f"D{j+1}", node_type=NodeType.SINK, demand=demands[j]) for j in range(nd)]

    arcs = []
    for i in range(ns):
        for k in range(nt):
            arcs.append(Arc(arc_id=f"S{i+1}-T{k+1}", from_node=f"S{i+1}", to_node=f"T{k+1}", unit_cost=round(rng.uniform(1, 6), 2), lead_time=round(rng.uniform(1, 4), 2), risk_score=round(rng.uniform(0, 0.5), 3)))
    for k in range(nt):
        for j in range(nd):
            arcs.append(Arc(arc_id=f"T{k+1}-D{j+1}", from_node=f"T{k+1}", to_node=f"D{j+1}", unit_cost=round(rng.uniform(1, 6), 2), lead_time=round(rng.uniform(1, 4), 2), risk_score=round(rng.uniform(0, 0.5), 3)))
    return ProblemInstance(
        instance_id=_iid(seed, attempt, ProblemClass.TRANSSHIPMENT, f"{ns}x{nt}x{nd}"),
        source="generated",
        problem_class=ProblemClass.TRANSSHIPMENT,
        nodes=nodes,
        arcs=arcs,
        metadata=InstanceMetadata(difficulty=Difficulty.MEDIUM, notes=f"generated seed {seed}"),
    )


def _build_facility(seed, attempt, params, rng) -> ProblemInstance:
    ns, no, nd = params["num_sources"], params["num_optional"], params["num_sinks"]
    demands = [float(rng.randint(4, 12)) for _ in range(nd)]
    total_demand = sum(demands)

    nodes = [Node(node_id=f"F{i+1}", node_type=NodeType.SOURCE, supply=round(total_demand + rng.uniform(5, 15), 2)) for i in range(ns)]
    nodes += [
        Node(node_id=f"W{k+1}", node_type=NodeType.TRANSSHIPMENT, is_optional=True, fixed_cost=round(rng.uniform(20, 80), 1), risk_score=round(rng.uniform(0, 0.5), 3))
        for k in range(no)
    ]
    nodes += [Node(node_id=f"C{j+1}", node_type=NodeType.SINK, demand=demands[j]) for j in range(nd)]

    arcs = []
    for i in range(ns):
        for k in range(no):
            arcs.append(Arc(arc_id=f"F{i+1}-W{k+1}", from_node=f"F{i+1}", to_node=f"W{k+1}", unit_cost=round(rng.uniform(1, 5), 2)))
    for k in range(no):
        for j in range(nd):
            arcs.append(Arc(arc_id=f"W{k+1}-C{j+1}", from_node=f"W{k+1}", to_node=f"C{j+1}", unit_cost=round(rng.uniform(1, 8), 2), risk_score=round(rng.uniform(0, 0.6), 3)))
    return ProblemInstance(
        instance_id=_iid(seed, attempt, ProblemClass.FACILITY_LOCATION, f"{ns}x{no}x{nd}"),
        source="generated",
        problem_class=ProblemClass.FACILITY_LOCATION,
        nodes=nodes,
        arcs=arcs,
        metadata=InstanceMetadata(difficulty=Difficulty.HARD, notes=f"generated seed {seed}"),
    )


def _build_vrp(seed, attempt, params, rng) -> ProblemInstance:
    nc = params["num_customers"]
    capacity = float(params["capacity"])
    depot_xy = (round(rng.uniform(0, 5), 2), round(rng.uniform(0, 5), 2))
    nodes = [Node(node_id="depot", node_type=NodeType.SOURCE, supply=float(nc * capacity), coordinates=depot_xy)]
    total = 0.0
    for j in range(nc):
        # demand never exceeds capacity so every customer fits in a vehicle
        dem = float(rng.randint(2, max(2, int(capacity))))
        total += dem
        nodes.append(
            Node(
                node_id=f"c{j+1}",
                node_type=NodeType.SINK,
                demand=dem,
                coordinates=(round(rng.uniform(0, 50), 2), round(rng.uniform(0, 50), 2)),
                risk_score=round(rng.uniform(0, 0.5), 3),
            )
        )
    arcs = [
        Arc(arc_id=f"depot-c{j+1}", from_node="depot", to_node=f"c{j+1}",
            unit_cost=round(((nodes[j + 1].coordinates[0] - depot_xy[0]) ** 2 + (nodes[j + 1].coordinates[1] - depot_xy[1]) ** 2) ** 0.5, 4))
        for j in range(nc)
    ]
    constraints = [
        AdditionalConstraint(
            constraint_id="veh-cap",
            constraint_type=ConstraintType.CAPACITY,
            parameters={"vehicle_capacity": capacity},
        )
    ]
    return ProblemInstance(
        instance_id=_iid(seed, attempt, ProblemClass.VEHICLE_ROUTING, f"{nc}c"),
        source="generated",
        problem_class=ProblemClass.VEHICLE_ROUTING,
        nodes=nodes,
        arcs=arcs,
        additional_constraints=constraints,
        metadata=InstanceMetadata(difficulty=Difficulty.MEDIUM, notes=f"generated seed {seed}"),
    )
