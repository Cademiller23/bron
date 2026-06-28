"""Darwin Phase B5 — the Rearrangement Loop (the always-on inner loop).

Frozen handoff: ``RearrangementLoop.run(genome, instance, weights) ->
RearrangementResult``. It always runs at least one rearrangement pass, never
regresses (elitism), never crashes (candidates that fail score at the floor and
are not adopted), and produces a non-decreasing fitness trace.
"""

from darwin.rearrange.generator import generate_candidates
from darwin.rearrange.loop import RearrangementLoop, RearrangementResult
from darwin.rearrange.operators import (
    ALL_OPERATORS,
    CandidateRearrangement,
    reassign_model,
    redirect_edge,
    reorder_pipeline,
    swap_arbiter,
)
from darwin.rearrange.reorganizer import HeuristicReorganizer, default_hints

__all__ = [
    "RearrangementLoop",
    "RearrangementResult",
    "generate_candidates",
    "CandidateRearrangement",
    "ALL_OPERATORS",
    "reassign_model",
    "redirect_edge",
    "reorder_pipeline",
    "swap_arbiter",
    "HeuristicReorganizer",
    "default_hints",
]
