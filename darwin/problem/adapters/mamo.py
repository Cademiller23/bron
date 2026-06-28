"""Adapter: Mamo (Easy / Complex) → canonical ``ProblemInstance``.

Mamo is the supplementary well of *solver-verified* instances used for warm-up
and additional difficulty. Because Mamo optima are solver-verified upstream, the
attached :class:`KnownOptimum` is marked ``SOLVER_VERIFIED``.
"""

from typing import Any, Dict

from darwin.problem.adapters.common import as_dict
from darwin.problem.schemas import (
    Arc,
    Difficulty,
    InstanceMetadata,
    KnownOptimum,
    Node,
    NodeType,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
)

_CATEGORY_TO_DIFFICULTY = {
    "easy": Difficulty.EASY,
    "complex": Difficulty.HARD,
    "hard": Difficulty.HARD,
    "medium": Difficulty.MEDIUM,
}


def parse(raw: Any) -> ProblemInstance:
    data: Dict[str, Any] = as_dict(raw)
    graph = data["graph"]

    nodes = [
        Node(
            node_id=v["name"],
            node_type=NodeType(v["role"]),
            supply=v.get("available"),
            demand=v.get("required"),
            capacity=v.get("capacity"),
            fixed_cost=v.get("fixed_cost"),
            is_optional=bool(v.get("optional", False)),
            risk_score=v.get("risk"),
        )
        for v in graph["vertices"]
    ]
    arcs = [
        Arc(
            arc_id=e["name"],
            from_node=e["tail"],
            to_node=e["head"],
            unit_cost=e["unit_cost"],
            capacity=e.get("capacity"),
            lead_time=e.get("lead_time", 0.0),
            fixed_cost=e.get("fixed_cost"),
            risk_score=e.get("risk"),
        )
        for e in graph["edges"]
    ]

    known_optimum = None
    if data.get("optimal_value") is not None:
        known_optimum = KnownOptimum(
            objective_value=float(data["optimal_value"]),
            source=OptimumSource.SOLVER_VERIFIED,
            verified=True,
            solver_used="mamo-upstream",
        )

    difficulty = _CATEGORY_TO_DIFFICULTY.get(str(data.get("category", "medium")).lower(), Difficulty.MEDIUM)

    return ProblemInstance(
        instance_id=data["instance_name"],
        source="mamo",
        problem_class=ProblemClass(data["type"]),
        nodes=nodes,
        arcs=arcs,
        known_optimum=known_optimum,
        metadata=InstanceMetadata(difficulty=difficulty, notes=data.get("notes", "")),
    )
