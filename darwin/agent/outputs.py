"""The structured output schemas a worker can return.

A worker's legal output is determined by its ``output_contract``. Each model
here is (a) what gets handed to the model as the response JSON Schema, and (b)
what the returned JSON is validated against. They are deliberately **shallow,
enum-rich, and densely described** — deep / union-heavy schemas spike the
model's malformed-output rate (§7, §15).

Reuses B1's ``Solution`` / ``FlowAssignment`` so a worker-produced solution is
exactly what the B1 scorer grades — the worker never grades it.
"""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from darwin.problem.schemas import FlowAssignment, Route, Solution


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# FULL_SOLUTION
# ---------------------------------------------------------------------------
class FullSolutionOutput(_Frozen):
    """A complete candidate solution. ``rationale`` is for the screen / telemetry
    only and is NEVER scored."""

    solution: Solution = Field(description="The complete proposed B1 Solution.")
    rationale: str = Field(default="", description="Short human-readable justification; not scored.")


# ---------------------------------------------------------------------------
# PARTIAL_SOLUTION
# ---------------------------------------------------------------------------
class PartialSolutionOutput(_Frozen):
    """A piece of a solution addressing one named sub-problem, for the team runner
    to stitch together."""

    sub_problem_id: str = Field(description="Which sub-problem this piece addresses.")
    flows: List[FlowAssignment] = Field(
        default_factory=list, description="Flow assignments for this sub-region."
    )
    open_facilities: Optional[List[str]] = Field(
        default=None, description="Optional facilities this piece proposes opening."
    )
    routes: Optional[List[Route]] = Field(
        default=None, description="Vehicle routes for this piece (VRP only)."
    )
    rationale: str = Field(default="", description="Short justification; not scored.")


# ---------------------------------------------------------------------------
# CRITIQUE
# ---------------------------------------------------------------------------
class Issue(_Frozen):
    location: str = Field(description="Which arc / node / route the issue concerns.")
    severity: Severity = Field(description="How serious the issue is.")
    description: str = Field(description="What is wrong.")
    suggested_fix: str = Field(default="", description="How to address it.")


class CritiqueOutput(_Frozen):
    """Structured feedback on another agent's output (a 'checker' / 'critic')."""

    target_agent_id: Optional[str] = Field(
        default=None, description="The agent whose output is being critiqued."
    )
    issues: List[Issue] = Field(default_factory=list, description="The issues found.")


# ---------------------------------------------------------------------------
# CONSTRAINT_REPORT
# ---------------------------------------------------------------------------
class SuspectedViolation(_Frozen):
    constraint_type: str = Field(description="The kind of constraint suspected to be violated.")
    location: str = Field(description="Which arc / node / constraint is implicated.")
    description: str = Field(description="Why the agent suspects a violation.")
    confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="The agent's confidence in [0,1]."
    )


class ConstraintReportOutput(_Frozen):
    """The agent's *guess* at which constraints a candidate violates.

    NOT ground truth — only B1's deterministic scorer is the authority on
    feasibility. The team uses this to reason; the scorer decides.
    """

    suspected_violations: List[SuspectedViolation] = Field(
        default_factory=list, description="Suspected (not confirmed) violations."
    )


# ---------------------------------------------------------------------------
# ARBITRATION
# ---------------------------------------------------------------------------
class ArbitrationOutput(_Frozen):
    """A final merged/selected solution synthesized from multiple sibling inputs."""

    solution: Solution = Field(description="The synthesized final Solution.")
    drawn_from: List[str] = Field(
        default_factory=list, description="agent_ids of the inputs this draws from."
    )
    rationale: str = Field(default="", description="Why this synthesis; not scored.")


# ---------------------------------------------------------------------------
# DECOMPOSITION
# ---------------------------------------------------------------------------
class SubProblem(_Frozen):
    sub_problem_id: str = Field(description="Stable id for this sub-problem.")
    description: str = Field(description="What this sub-problem covers.")
    node_ids: List[str] = Field(default_factory=list, description="Nodes in this sub-problem's boundary.")
    arc_ids: List[str] = Field(default_factory=list, description="Arcs in this sub-problem's boundary.")


class DecompositionOutput(_Frozen):
    """A proposed split of the problem into named sub-problems."""

    sub_problems: List[SubProblem] = Field(
        default_factory=list, description="The proposed sub-problems."
    )


# ---------------------------------------------------------------------------
# Registry: OutputKind -> model class. The single place worker/tests resolve it.
# ---------------------------------------------------------------------------
def output_model_for(output_kind) -> type:
    """Resolve an ``OutputKind`` to its Pydantic output model."""
    from darwin.agent.spec import OutputKind

    mapping = {
        OutputKind.FULL_SOLUTION: FullSolutionOutput,
        OutputKind.PARTIAL_SOLUTION: PartialSolutionOutput,
        OutputKind.CRITIQUE: CritiqueOutput,
        OutputKind.CONSTRAINT_REPORT: ConstraintReportOutput,
        OutputKind.ARBITRATION: ArbitrationOutput,
        OutputKind.DECOMPOSITION: DecompositionOutput,
    }
    return mapping[output_kind]


OUTPUT_MODELS = (
    FullSolutionOutput,
    PartialSolutionOutput,
    CritiqueOutput,
    ConstraintReportOutput,
    ArbitrationOutput,
    DecompositionOutput,
)

# The union of every legal worker output (used by AgentResult.output).
OutputUnion = Union[
    FullSolutionOutput,
    PartialSolutionOutput,
    CritiqueOutput,
    ConstraintReportOutput,
    ArbitrationOutput,
    DecompositionOutput,
]
