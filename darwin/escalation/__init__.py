"""Darwin Phase B6 — the Threshold Gate, Escalation, and the Top-Level Conductor.

Frozen handoff: ``Conductor.solve(instance, weights, budget) -> SolveResult``.

The conductor is the whole brain in one call: B4 designs a team, B5 always
rearranges it, the 0.90 gate is checked, and — only when reshaping can't clear
it — B6 escalates by GROWING the team. Escalation is two ordered steps:
reuse a proven agent from the corpus (cheap), else curate a brand-new one.
Team-growth elitism keeps an added agent only if it strictly improved the score
(otherwise the team is rolled back); useful curated agents are promoted to the
corpus, which is Darwin's genuine "gets better across problems" mechanism.

Every boundary degrades rather than crashes: ``solve`` always returns a
``SolveResult`` with a real fitness — sealed if cleared, best-so-far otherwise.
"""

from darwin.escalation.conductor import Conductor
from darwin.escalation.corpus import AgentCorpus
from darwin.escalation.diagnosis import diagnose_gap
from darwin.escalation.embedding import (
    Embedder,
    KeywordEmbedder,
    VoyageEmbedder,
    cosine_similarity,
)
from darwin.escalation.escalator import Escalator
from darwin.escalation.schemas import (
    CorpusEntry,
    EscalationMethod,
    EscalationResult,
    GapDescription,
    ScoredCorpusEntry,
    SolveBudget,
    SolveResult,
    SolveStatus,
    WeakDimension,
)

__all__ = [
    "Conductor",
    "Escalator",
    "AgentCorpus",
    "diagnose_gap",
    "Embedder",
    "KeywordEmbedder",
    "VoyageEmbedder",
    "cosine_similarity",
    "GapDescription",
    "CorpusEntry",
    "ScoredCorpusEntry",
    "EscalationResult",
    "EscalationMethod",
    "WeakDimension",
    "SolveBudget",
    "SolveResult",
    "SolveStatus",
]
