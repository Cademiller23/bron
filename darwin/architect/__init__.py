"""Darwin Phase B4 — the Architect (the curator / meta-agent).

The Architect designs teams; it never solves the problem. Two frozen
capabilities are the handoff to B5/B6:
* ``design_initial_team(instance, weights) -> TeamGenome`` — authors and
  assembles the initial curated team (always valid, persisted, never raises).
* ``curate_agent_for_gap(genome, instance, evaluation) -> (AgentSpec, edges)`` —
  authors one targeted agent for a scorer-diagnosed gap.
"""

from darwin.architect.architect import Architect
from darwin.architect.assembly import AssemblyError, assemble
from darwin.architect.schemas import (
    AgentSpecDraft,
    ArchitectTeamDesign,
    CuratedAgentDraft,
    EdgeDraft,
    ProblemAnalysis,
)

__all__ = [
    "Architect",
    "assemble",
    "AssemblyError",
    "ProblemAnalysis",
    "AgentSpecDraft",
    "EdgeDraft",
    "ArchitectTeamDesign",
    "CuratedAgentDraft",
]
