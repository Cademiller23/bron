"""Adapter: CVRPLIB / Solomon (TSPLIB-style ``.vrp``) → canonical instance.

Parses the depot, customers (with demands + coordinates) and vehicle capacity
into the ``VEHICLE_ROUTING`` class — the live fresh-instance source for
routes-on-a-map visuals.

Note on metric: CVRPLIB's ``EUC_2D`` weights are *integer-rounded* Euclidean
distances, whereas Darwin's scorer uses continuous Euclidean distance. The
``COMMENT``'s optimal value (if present) is therefore attached as a
``BENCHMARK_LABEL`` only — it is a reference, not a value the continuous scorer
is expected to reproduce exactly.
"""

import re
from typing import Any, Dict, List, Optional

from darwin.problem.adapters.common import read_text
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


def _parse_sections(text: str) -> Dict[str, Any]:
    headers: Dict[str, str] = {}
    coords: Dict[int, tuple] = {}
    demands: Dict[int, float] = {}
    depots: List[int] = []

    section = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "EOF":
            continue
        upper = line.upper()
        if upper.startswith("NODE_COORD_SECTION"):
            section = "coords"
            continue
        if upper.startswith("DEMAND_SECTION"):
            section = "demands"
            continue
        if upper.startswith("DEPOT_SECTION"):
            section = "depots"
            continue
        if ":" in line and section is None:
            key, _, value = line.partition(":")
            headers[key.strip().upper()] = value.strip()
            continue
        if section == "coords":
            parts = line.split()
            coords[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "demands":
            parts = line.split()
            demands[int(parts[0])] = float(parts[1])
        elif section == "depots":
            val = int(line)
            if val >= 0:
                depots.append(val)
            else:
                section = None  # -1 terminates the depot section
    return {"headers": headers, "coords": coords, "demands": demands, "depots": depots}


def _comment_optimum(comment: str) -> Optional[float]:
    match = re.search(r"optimal\s*value\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", comment, re.IGNORECASE)
    return float(match.group(1)) if match else None


def parse(raw: Any) -> ProblemInstance:
    text = read_text(raw)
    parsed = _parse_sections(text)
    headers = parsed["headers"]
    coords = parsed["coords"]
    demands = parsed["demands"]
    depots = set(parsed["depots"])

    name = headers.get("NAME", "cvrplib-instance")
    capacity = float(headers["CAPACITY"]) if "CAPACITY" in headers else None

    nodes: List[Node] = []
    for idx in sorted(coords):
        is_depot = idx in depots
        nodes.append(
            Node(
                node_id=f"n{idx}",
                node_type=NodeType.SOURCE if is_depot else NodeType.SINK,
                supply=None if not is_depot else float(sum(demands.values())),
                demand=None if is_depot else demands.get(idx, 0.0),
                coordinates=coords[idx],
            )
        )

    # Connectivity arcs depot->customer (the VRP scorer works off coordinates;
    # arcs keep the graph well-formed for any flow-oriented tooling).
    depot_ids = [f"n{i}" for i in sorted(depots)]
    depot_id = depot_ids[0] if depot_ids else None
    arcs: List[Arc] = []
    if depot_id is not None:
        for idx in sorted(coords):
            if idx in depots:
                continue
            dx = coords[idx][0] - coords[sorted(depots)[0]][0]
            dy = coords[idx][1] - coords[sorted(depots)[0]][1]
            arcs.append(
                Arc(
                    arc_id=f"{depot_id}-n{idx}",
                    from_node=depot_id,
                    to_node=f"n{idx}",
                    unit_cost=(dx * dx + dy * dy) ** 0.5,
                )
            )

    constraints = []
    if capacity is not None:
        constraints.append(
            AdditionalConstraint(
                constraint_id="vehicle-capacity",
                constraint_type=ConstraintType.CAPACITY,
                parameters={"vehicle_capacity": capacity},
                description="CVRPLIB CAPACITY header",
            )
        )

    known_optimum = None
    opt = _comment_optimum(headers.get("COMMENT", ""))
    if opt is not None:
        known_optimum = KnownOptimum(
            objective_value=opt, source=OptimumSource.BENCHMARK_LABEL, verified=False
        )

    return ProblemInstance(
        instance_id=name,
        source="cvrplib",
        problem_class=ProblemClass.VEHICLE_ROUTING,
        nodes=nodes,
        arcs=arcs,
        additional_constraints=constraints,
        known_optimum=known_optimum,
        metadata=InstanceMetadata(
            difficulty=Difficulty.MEDIUM, notes=f"CVRPLIB {headers.get('TYPE', 'CVRP')}"
        ),
    )
