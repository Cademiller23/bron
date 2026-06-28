"""Darwin Phase B3 — the Team Genome & the Team Runner (the recipe card & the kitchen).

Four frozen surfaces are the handoff to B4 (the Architect) and B5 (rearrangement):
* ``TeamGenome`` — a self-contained, validated, version-stamped team graph.
* ``GenomeStore`` — an atomic, conflict-safe ``mutate``/``retry_mutate`` API.
* ``TeamRunner.evaluate(genome, instance, weights) -> GenomeEvaluation`` — always
  returns a real number, never raises.
* ``InferenceGate`` — a shared semaphore bounding global inference concurrency.
"""

from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import (
    AgentNode,
    ArbiterTier,
    Edge,
    EdgeType,
    GenomeStatus,
    MutationActor,
    MutationRecord,
    MutationType,
    TeamGenome,
)
from darwin.team.inference_gate import InferenceGate
from darwin.team.runner import TeamRunner
from darwin.team.store import GenomeStore, OptimisticLockError
from darwin.team.validation import ValidationResult, validate

__all__ = [
    "TeamGenome",
    "AgentNode",
    "Edge",
    "EdgeType",
    "GenomeStatus",
    "MutationType",
    "MutationActor",
    "MutationRecord",
    "ArbiterTier",
    "GenomeEvaluation",
    "GenomeStore",
    "OptimisticLockError",
    "InferenceGate",
    "TeamRunner",
    "ValidationResult",
    "validate",
]
