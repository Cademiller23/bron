"""Darwin Phase B1 — the Problem Loader & Scorer.

This package defines:

* The **canonical data model** (:mod:`darwin.problem.schemas`) — the frozen,
  validated contract every Darwin agent speaks. ``ProblemInstance`` /
  ``Solution`` / ``ScoreBreakdown`` are the three nouns the whole system
  shares.
* The **deterministic resilience metric** (:mod:`darwin.problem.resilience`).
* The **deterministic fitness scorer** (:mod:`darwin.problem.scorer`) — pure
  arithmetic, never an LLM, sub-millisecond, byte-for-byte reproducible.
* The **solver oracle** (:mod:`darwin.problem.oracle`) — establishes and
  verifies ground-truth optima (OR-Tools when available, exact pure-Python
  fallback otherwise).
* The **loader** (:mod:`darwin.problem.loader`) + adapters, and the live
  **instance generator** (:mod:`darwin.problem.generator`).

The single most important rule: *the scorer is arithmetic, not judgment.*
No model call ever touches the fitness number.
"""

from darwin.problem.schemas import (
    AdditionalConstraint,
    Arc,
    ConstraintType,
    Difficulty,
    FlowAssignment,
    InstanceMetadata,
    KnownOptimum,
    Node,
    NodeType,
    ObjectiveWeights,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
    Route,
    ScoreBreakdown,
    Solution,
    Violation,
    ViolationType,
)
from darwin.problem.scorer import SCORER_VERSION, score

__all__ = [
    "AdditionalConstraint",
    "Arc",
    "ConstraintType",
    "Difficulty",
    "FlowAssignment",
    "InstanceMetadata",
    "KnownOptimum",
    "Node",
    "NodeType",
    "ObjectiveWeights",
    "OptimumSource",
    "ProblemClass",
    "ProblemInstance",
    "Route",
    "ScoreBreakdown",
    "Solution",
    "Violation",
    "ViolationType",
    "SCORER_VERSION",
    "score",
]
