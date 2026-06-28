"""B6 data model — the gap diagnosis, the corpus entry, and the solve result.

Frozen Pydantic, ``extra="forbid"`` throughout. ``GapDescription`` is the single
diagnosis that drives BOTH corpus search and curation. ``SolveResult`` is the
whole brain's output — sealed if cleared, best-so-far if the budget runs out.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from darwin.agent.spec import AgentSpec
from darwin.constants import (
    MAX_ESCALATIONS,
    MAX_TEAM_SIZE,
    MAX_TOTAL_COST_USD,
    MAX_WALL_CLOCK_SECONDS,
)
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import TeamGenome


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class WeakDimension(str, Enum):
    COST = "COST"
    LEAD_TIME = "LEAD_TIME"
    RESILIENCE = "RESILIENCE"
    FEASIBILITY = "FEASIBILITY"


class EscalationMethod(str, Enum):
    CORPUS = "CORPUS"  # reused a proven agent from the corpus
    CURATED = "CURATED"  # the Architect authored a brand-new agent
    NONE_AVAILABLE = "NONE_AVAILABLE"  # neither could supply a usable agent


class SolveStatus(str, Enum):
    SEALED = "SEALED"  # cleared the 0.90 threshold
    EXHAUSTED = "EXHAUSTED"  # budget ran out; returning the best-so-far


# ---------------------------------------------------------------------------
# Gap diagnosis (the unifying signal, deterministic from the ScoreBreakdown)
# ---------------------------------------------------------------------------
class GapDescription(_Frozen):
    capability_needed: str = Field(
        description="Natural-language missing capability — the corpus query AND the curation seed."
    )
    weak_dimensions: List[WeakDimension] = Field(default_factory=list, description="Ranked weak dimensions.")
    dominant_violations: List[str] = Field(default_factory=list, description="Most common ViolationTypes.")
    problem_class: str = Field(default="", description="Echoed from the instance (for corpus filtering).")
    suggested_role_kind: str = Field(default="proposer", description="proposer / checker / specialist.")
    severity: float = Field(default=0.0, ge=0.0, description="How far below threshold (0 == at/above).")


# ---------------------------------------------------------------------------
# The corpus
# ---------------------------------------------------------------------------
class CorpusEntry(_Frozen):
    entry_id: str = Field(description="Stable id; maps to Mongo _id.")
    agent_spec: AgentSpec  # the full B2 spec, ready to instantiate
    role_name: str
    role_description: str
    role_description_embedding: List[float] = Field(default_factory=list)
    helped_problem_classes: List[str] = Field(default_factory=list)
    avg_fitness_contribution: float = 0.0  # running average of the score delta it produced
    times_reused: int = 0
    success_count: int = 0
    failure_count: int = 0
    created_at: str = ""
    last_used_at: str = ""
    origin_instance_id: str = ""


class ScoredCorpusEntry(_Frozen):
    entry: CorpusEntry
    similarity: float  # cosine similarity to the gap query
    combined_score: float  # relevance x track-record (the ranking key)


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------
class EscalationResult(_Frozen):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    method: EscalationMethod
    genome: Optional[TeamGenome] = None  # the mutated genome (None for NONE_AVAILABLE)
    gap: GapDescription
    added_spec: Optional[AgentSpec] = None
    added_agent_id: Optional[str] = None
    corpus_entry_id: Optional[str] = None  # set for CORPUS (for update_stats)
    corpus_candidates_considered: int = 0
    description: str = ""


# ---------------------------------------------------------------------------
# The conductor's budget + result
# ---------------------------------------------------------------------------
class SolveBudget(_Frozen):
    max_escalations: int = Field(default=MAX_ESCALATIONS, ge=0)
    max_team_size: int = Field(default=MAX_TEAM_SIZE, ge=1)
    max_wall_clock_seconds: float = Field(default=MAX_WALL_CLOCK_SECONDS, gt=0)
    max_total_cost_usd: float = Field(default=MAX_TOTAL_COST_USD, gt=0)


class SolveResult(_Frozen):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    instance_id: str
    final_genome: TeamGenome
    final_evaluation: GenomeEvaluation
    cleared_threshold: bool
    status: SolveStatus
    escalation_rounds: int = 0
    agents_added: List[Dict[str, Any]] = Field(default_factory=list)  # {agent_id, role, method}
    corpus_hits: int = 0
    corpus_promotions: int = 0
    full_fitness_trace: List[float] = Field(default_factory=list)  # the continuous climbing curve
    trace_markers: List[Dict[str, Any]] = Field(default_factory=list)  # {index, label} escalation markers
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    per_round_summary: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
