"""The GenomeEvaluation result type — the runner's output.

``TeamRunner.evaluate`` always returns one of these with a **real** ``fitness``
number (floored on failure, never absent) and a full B1 ``ScoreBreakdown`` — it
never raises. ``completed`` reflects whether the arbiter produced a real answer
(Tier 1), *not* whether evaluation crashed (evaluation never crashes).
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from darwin.agent.worker import AgentResult
from darwin.problem.schemas import ScoreBreakdown, Solution
from darwin.team.genome import ArbiterTier


class GenomeEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    genome_id: str
    version: int
    instance_id: str

    # Did the arbiter produce a real answer (Tier 1), vs a fallback being used?
    completed: bool

    final_solution: Solution  # the B1 Solution that was scored (or the sentinel)
    score_breakdown: ScoreBreakdown
    fitness: float  # final_fitness B5/B6 consume — floored on failure, never absent
    normalized_score: float  # for the 0.90 comparison
    cleared_threshold: bool  # normalized_score >= SCORE_THRESHOLD

    arbiter_tier_used: ArbiterTier
    agent_results: List[AgentResult] = Field(default_factory=list)
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    error: Optional[str] = None  # populated only if the catch-all boundary fired
