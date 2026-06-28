"""The canonical data model — the contract every Darwin agent depends on.

This module is written *line one* of B1 and is **frozen** as a contract before
B2 begins. It is a node–arc network model covering the dominant
deterministic-scorable supply-chain optimization families (transportation,
transshipment, facility location) with a clean extension point for vehicle
routing.

Enterprise discipline applied throughout:

* every data class is an **immutable** (``frozen=True``) Pydantic model that
  **rejects unknown fields** (``extra="forbid"``);
* every numeric field is range-validated at construction (non-negative costs /
  capacities / demands, risk in ``[0, 1]``, no NaN/inf);
* cross-object invariants (referential integrity, unique ids, supply ≥ demand)
  are checked at construction so a malformed instance fails *loudly and
  immediately*, never producing a silently broken object.
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Numerical tolerance shared across the whole B1 package.
# ---------------------------------------------------------------------------
TOL: float = 1e-6


# ---------------------------------------------------------------------------
# 2.1 Enumerations (defined first; later types reference them).
# ---------------------------------------------------------------------------
class NodeType(str, Enum):
    """The role a location plays in the network."""

    SOURCE = "SOURCE"  # factory / supplier
    TRANSSHIPMENT = "TRANSSHIPMENT"  # warehouse / distribution centre
    SINK = "SINK"  # store / customer / demand point


class ProblemClass(str, Enum):
    """The optimization family an instance belongs to."""

    TRANSPORTATION = "TRANSPORTATION"
    TRANSSHIPMENT = "TRANSSHIPMENT"
    FACILITY_LOCATION = "FACILITY_LOCATION"
    VEHICLE_ROUTING = "VEHICLE_ROUTING"
    F1_CALENDAR = "F1_CALENDAR"  # F1 calendar optimization (scored by f1_scorer, not the flow scorer)


class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"


class ConstraintType(str, Enum):
    """Declared *additional* constraints (structural ones are always enforced)."""

    CAPACITY = "CAPACITY"
    DEMAND_SATISFACTION = "DEMAND_SATISFACTION"
    FLOW_CONSERVATION = "FLOW_CONSERVATION"
    LEAD_TIME_LIMIT = "LEAD_TIME_LIMIT"
    SINGLE_SOURCE = "SINGLE_SOURCE"
    MUTUAL_EXCLUSION = "MUTUAL_EXCLUSION"
    CUSTOM = "CUSTOM"


class OptimumSource(str, Enum):
    BENCHMARK_LABEL = "BENCHMARK_LABEL"  # claimed by the source benchmark
    SOLVER_VERIFIED = "SOLVER_VERIFIED"  # confirmed by our own oracle
    UNKNOWN = "UNKNOWN"


class ViolationType(str, Enum):
    """Mirrors the structural feasibility checks performed by the scorer."""

    OVER_ARC_CAPACITY = "OVER_ARC_CAPACITY"
    OVER_NODE_CAPACITY = "OVER_NODE_CAPACITY"
    SUPPLY_EXCEEDED = "SUPPLY_EXCEEDED"
    DEMAND_UNMET = "DEMAND_UNMET"
    CONSERVATION_BROKEN = "CONSERVATION_BROKEN"
    CLOSED_FACILITY_USED = "CLOSED_FACILITY_USED"
    LEAD_TIME_EXCEEDED = "LEAD_TIME_EXCEEDED"
    MALFORMED_SOLUTION = "MALFORMED_SOLUTION"
    CUSTOM_CONSTRAINT = "CUSTOM_CONSTRAINT"


# ---------------------------------------------------------------------------
# Base model — every B1 type is frozen and rejects unknown fields.
# ---------------------------------------------------------------------------
class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# 2.2 Node
# ---------------------------------------------------------------------------
class Node(_Frozen):
    """A single location in the network."""

    node_id: str = Field(min_length=1)
    node_type: NodeType
    capacity: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    supply: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    demand: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    fixed_cost: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    is_optional: bool = False
    coordinates: Optional[Tuple[float, float]] = None
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)

    @field_validator("coordinates")
    @classmethod
    def _coordinates_finite(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        if v is not None and not all(math.isfinite(c) for c in v):
            raise ValueError("coordinates must be finite (no NaN/inf)")
        return v


# ---------------------------------------------------------------------------
# 2.3 Arc
# ---------------------------------------------------------------------------
class Arc(_Frozen):
    """A directed transport link."""

    arc_id: str = Field(min_length=1)
    from_node: str = Field(min_length=1)
    to_node: str = Field(min_length=1)
    unit_cost: float = Field(ge=0.0, allow_inf_nan=False)
    capacity: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    lead_time: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    fixed_cost: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)
    risk_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)


# ---------------------------------------------------------------------------
# 2.4 AdditionalConstraint
# ---------------------------------------------------------------------------
class AdditionalConstraint(_Frozen):
    """An extra benchmark constraint beyond the always-enforced structural ones."""

    constraint_id: str = Field(min_length=1)
    constraint_type: ConstraintType
    parameters: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# 2.5 KnownOptimum
# ---------------------------------------------------------------------------
class KnownOptimum(_Frozen):
    """The credibility anchor: the true optimal objective value."""

    objective_value: float = Field(ge=0.0, allow_inf_nan=False)
    source: OptimumSource
    verified: bool = False
    solver_used: Optional[str] = None


# ---------------------------------------------------------------------------
# 2.6 InstanceMetadata
# ---------------------------------------------------------------------------
class InstanceMetadata(_Frozen):
    difficulty: Difficulty = Difficulty.MEDIUM
    num_nodes: int = Field(default=0, ge=0)
    num_arcs: int = Field(default=0, ge=0)
    # Demo staging: did we pre-confirm offline whether rearrangement alone clears
    # the threshold, or whether this forces escalation? ``None`` == unknown.
    expected_to_clear_threshold: Optional[bool] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# 2.7 ProblemInstance — the top-level object
# ---------------------------------------------------------------------------
class ProblemInstance(_Frozen):
    """One canonical, validated, in-memory optimization problem."""

    instance_id: str = Field(min_length=1)
    source: str = Field(min_length=1)  # "industryor" | "mamo" | "cvrplib" | "generated"
    problem_class: ProblemClass
    nodes: List[Node]
    arcs: List[Arc]
    additional_constraints: List[AdditionalConstraint] = Field(default_factory=list)
    known_optimum: Optional[KnownOptimum] = None
    metadata: InstanceMetadata = Field(default_factory=InstanceMetadata)

    # -- derived, read-only convenience views -------------------------------
    @property
    def node_index(self) -> Dict[str, Node]:
        return {n.node_id: n for n in self.nodes}

    @property
    def arc_index(self) -> Dict[str, Arc]:
        return {a.arc_id: a for a in self.arcs}

    def sources(self) -> List[Node]:
        return [n for n in self.nodes if n.node_type == NodeType.SOURCE]

    def sinks(self) -> List[Node]:
        return [n for n in self.nodes if n.node_type == NodeType.SINK]

    def transshipments(self) -> List[Node]:
        return [n for n in self.nodes if n.node_type == NodeType.TRANSSHIPMENT]

    def total_supply(self) -> float:
        return sum(n.supply or 0.0 for n in self.nodes)

    def total_demand(self) -> float:
        return sum(n.demand or 0.0 for n in self.nodes)

    # -- construction-time invariants (enterprise robustness lives here) -----
    @model_validator(mode="after")
    def _validate_network(self) -> "ProblemInstance":
        node_ids = [n.node_id for n in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            dupes = sorted({i for i in node_ids if node_ids.count(i) > 1})
            raise ValueError(f"duplicate node_id(s): {dupes}")

        arc_ids = [a.arc_id for a in self.arcs]
        if len(arc_ids) != len(set(arc_ids)):
            dupes = sorted({i for i in arc_ids if arc_ids.count(i) > 1})
            raise ValueError(f"duplicate arc_id(s): {dupes}")

        node_id_set = set(node_ids)
        for arc in self.arcs:
            if arc.from_node not in node_id_set:
                raise ValueError(
                    f"arc {arc.arc_id!r} references unknown from_node {arc.from_node!r}"
                )
            if arc.to_node not in node_id_set:
                raise ValueError(
                    f"arc {arc.arc_id!r} references unknown to_node {arc.to_node!r}"
                )

        # Optional facilities (open/close decisions) only make sense as nodes
        # that declared ``is_optional``; non-optional nodes are always open.
        for c in self.additional_constraints:
            pass  # parameter shapes are validated lazily by the scorer

        # Total supply must be able to cover total demand, otherwise the
        # instance is structurally infeasible and is rejected here — never
        # discovered later inside the scoring loop.
        if self.total_supply() + TOL < self.total_demand():
            raise ValueError(
                "structurally infeasible instance: total supply "
                f"{self.total_supply()} < total demand {self.total_demand()}"
            )

        # Derive metadata counts (immutable model => rebuild via object setattr).
        corrected = self.metadata.model_copy(
            update={"num_nodes": len(self.nodes), "num_arcs": len(self.arcs)}
        )
        object.__setattr__(self, "metadata", corrected)
        return self


# ---------------------------------------------------------------------------
# 2.8 The solution side
# ---------------------------------------------------------------------------
class FlowAssignment(_Frozen):
    """Units shipped along one arc."""

    arc_id: str = Field(min_length=1)
    quantity: float = Field(ge=0.0, allow_inf_nan=False)


class Route(_Frozen):
    """An ordered vehicle tour (vehicle-routing extension).

    ``node_sequence`` is the ordered list of node visits; a well-formed route
    starts and ends at a depot.
    """

    vehicle_id: str = Field(min_length=1)
    node_sequence: List[str]
    load: Optional[float] = Field(default=None, ge=0.0, allow_inf_nan=False)


class Solution(_Frozen):
    """A proposed answer to a :class:`ProblemInstance`."""

    solution_id: str = Field(min_length=1)
    instance_id: str = Field(min_length=1)
    flows: List[FlowAssignment] = Field(default_factory=list)
    # Which optional nodes the solver chose to open (facility location).
    open_facilities: Optional[List[str]] = None
    # Populated only for VEHICLE_ROUTING.
    routes: Optional[List[Route]] = None
    # Provenance: which agent / team genome produced this. Every score traces
    # back to a team structure through this field.
    produced_by: str = "unknown"


# ---------------------------------------------------------------------------
# 2.9 The output side
# ---------------------------------------------------------------------------
class Violation(_Frozen):
    violation_type: ViolationType
    location: str  # which arc / node / constraint
    magnitude: float = Field(ge=0.0, allow_inf_nan=False)  # how badly (units over / short)
    message: str = ""


class ObjectiveWeights(_Frozen):
    """The dial B8 turns when the system improves its own scorer.

    Kept as explicit data (never hard-coded constants) so the second-order
    self-improvement loop is possible. Weights are normalized to sum to 1 on
    construction.
    """

    cost_weight: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    lead_time_weight: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    risk_weight: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _normalize(self) -> "ObjectiveWeights":
        total = self.cost_weight + self.lead_time_weight + self.risk_weight
        if total <= 0.0:
            raise ValueError("objective weights must sum to a positive value")
        object.__setattr__(self, "cost_weight", self.cost_weight / total)
        object.__setattr__(self, "lead_time_weight", self.lead_time_weight / total)
        object.__setattr__(self, "risk_weight", self.risk_weight / total)
        return self

    @classmethod
    def cost_only(cls) -> "ObjectiveWeights":
        return cls(cost_weight=1.0, lead_time_weight=0.0, risk_weight=0.0)

    @classmethod
    def balanced(cls) -> "ObjectiveWeights":
        return cls(cost_weight=1.0, lead_time_weight=1.0, risk_weight=1.0)


class ScoreBreakdown(_Frozen):
    """What the scorer returns — the single source of truth about quality.

    ``final_fitness`` is the one number B5/B6 and the live fitness curve
    consume. ``computed_at`` is *metadata only* and never enters the number.
    """

    solution_id: str
    instance_id: str
    feasible: bool
    violations: List[Violation] = Field(default_factory=list)

    # Three un-blended objectives, kept for the screen and diagnostics.
    raw_cost: float
    raw_lead_time: float
    raw_risk: float

    weighted_objective: float  # blended, normalized objective (lower is better)
    normalized_score: float = Field(ge=0.0, le=1.0)  # fraction of optimum achieved
    total_penalty: float = Field(ge=0.0)
    final_fitness: float  # the single number the rest of Darwin consumes

    objective_weights: ObjectiveWeights
    scorer_version: str
    computed_at: str  # ISO-8601 timestamp; metadata only
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
