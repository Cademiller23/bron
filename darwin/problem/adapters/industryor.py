"""Adapter: IndustryOR problem representation → canonical ``ProblemInstance``.

IndustryOR ships logistics optimization problems with (often mislabelled)
optima. An upstream extraction step renders each problem as the structured JSON
this adapter consumes; we map its fields onto the canonical schema, attach
provenance (``source="industryor"``) and the *labelled* optimum (which the
oracle must independently verify before it is trusted — see
:func:`darwin.problem.oracle.verify_label`).
"""

from typing import Any, Dict

from darwin.problem.adapters.common import as_dict
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


def parse(raw: Any) -> ProblemInstance:
    data: Dict[str, Any] = as_dict(raw)

    nodes = [
        Node(
            node_id=n["id"],
            node_type=NodeType(n["type"]),
            supply=n.get("supply"),
            demand=n.get("demand"),
            capacity=n.get("capacity"),
            fixed_cost=n.get("fixed_cost"),
            is_optional=bool(n.get("optional", False)),
            coordinates=tuple(n["coordinates"]) if n.get("coordinates") else None,
            risk_score=n.get("risk"),
        )
        for n in data["nodes"]
    ]
    arcs = [
        Arc(
            arc_id=a["id"],
            from_node=a["from"],
            to_node=a["to"],
            unit_cost=a["cost"],
            capacity=a.get("capacity"),
            lead_time=a.get("lead_time", 0.0),
            fixed_cost=a.get("fixed_cost"),
            risk_score=a.get("risk"),
        )
        for a in data["arcs"]
    ]
    constraints = [
        AdditionalConstraint(
            constraint_id=c["id"],
            constraint_type=ConstraintType(c["type"]),
            parameters=c.get("parameters", {}),
            description=c.get("description", ""),
        )
        for c in data.get("constraints", [])
    ]

    known_optimum = None
    if data.get("labeled_optimum") is not None:
        known_optimum = KnownOptimum(
            objective_value=float(data["labeled_optimum"]),
            source=OptimumSource.BENCHMARK_LABEL,
            verified=False,
        )

    return ProblemInstance(
        instance_id=data["id"],
        source="industryor",
        problem_class=ProblemClass(data["problem_class"]),
        nodes=nodes,
        arcs=arcs,
        additional_constraints=constraints,
        known_optimum=known_optimum,
        metadata=InstanceMetadata(
            difficulty=Difficulty(data.get("difficulty", "MEDIUM")),
            notes=data.get("notes", ""),
        ),
    )
