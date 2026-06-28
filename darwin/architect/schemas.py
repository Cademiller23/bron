"""The Architect's own output schemas — what the frontier model emits.

The Architect is a **meta-agent**: it never emits a Solution (flows/routes); it
emits a *team design* — agent specifications + wiring — as structured data that
``assembly.py`` turns into a runnable B3 ``TeamGenome``.

These are exactly the shapes the frontier model returns (via B2's
structured-output machinery) and exactly what assembly consumes.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field

from darwin.agent.spec import InputKind, OutputKind, ThinkingLevel
from darwin.team.genome import EdgeType


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ProblemAnalysis(_Frozen):
    """The Architect's structured reading of the problem (great screen/voice narration)."""

    problem_class: str = Field(description="The optimization family (echoed from the instance).")
    dominant_objectives: List[str] = Field(
        default_factory=list, description="Which of cost / lead_time / risk matter most here."
    )
    binding_constraints: List[str] = Field(
        default_factory=list, description="Constraints that look tight (capacity-limited? demand-heavy? long lead times?)."
    )
    difficulty_estimate: str = Field(default="MEDIUM", description="EASY / MEDIUM / HARD.")
    suggested_part_count: int = Field(default=3, ge=1, description="How many parts to decompose into (dynamic).")
    rationale: str = Field(default="", description="Why this reading.")


class AgentSpecDraft(_Frozen):
    """An agent authored by the Architect (agent_id is assigned later by assembly)."""

    role_name: str = Field(min_length=1, description="A coined role, e.g. 'cost minimizer' (free text).")
    role_description: str = Field(min_length=1, description="The job — what this agent does, authored.")
    input_contract: InputKind = Field(description="What this agent receives.")
    output_contract: OutputKind = Field(description="What structured shape it must produce.")
    model_id: str = Field(min_length=1, description="The model chosen for this agent (from the catalog).")
    thinking_level: ThinkingLevel = Field(default=ThinkingLevel.MEDIUM, description="Per-agent reasoning dial.")
    responsibility: str = Field(default="", description="One line: what this agent owns.")
    why_this_model: str = Field(default="", description="Justification for the model choice (seeds B7 + narration).")


class EdgeDraft(_Frozen):
    """Wiring expressed by role names (assembly resolves to agent_ids)."""

    from_role_name: str = Field(min_length=1)
    to_role_name: str = Field(min_length=1)
    edge_type: EdgeType


class ArchitectTeamDesign(_Frozen):
    """The top-level emission: the authored team + wiring + arbiter designation."""

    analysis: ProblemAnalysis
    agents: List[AgentSpecDraft] = Field(min_length=1, description="The authored agents.")
    edges: List[EdgeDraft] = Field(default_factory=list, description="The wiring.")
    arbiter_role_name: str = Field(min_length=1, description="Which authored agent is the final arbitrator.")
    design_rationale: str = Field(default="", description="Why this team / decomposition / wiring.")


class CuratedAgentDraft(_Frozen):
    """The escalation emission: exactly ONE new agent for a diagnosed gap, plus how
    to wire it into the existing team (B6 commits it via ADD_CURATED_AGENT)."""

    agent: AgentSpecDraft = Field(description="The single new agent supplying the missing capability.")
    edges: List[EdgeDraft] = Field(
        default_factory=list, description="Edges connecting the new agent to existing agents (by role name)."
    )
    rationale: str = Field(default="", description="Why this agent closes the gap to 90%.")
