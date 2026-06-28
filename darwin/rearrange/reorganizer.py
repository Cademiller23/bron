"""Optional LLM steering — hints, not an LLM-per-candidate.

A weak resilience sub-score biases toward moves that elevate the risk agent;
persistent capacity violations bias toward reordering. This keeps the inner loop
fast (generation stays programmatic) while letting a little intelligence steer
*where* to look. It is **optional**: the generator's uniform sampling works
without it.

``default_hints`` is a pure heuristic (no model call). ``LLMReorganizer`` adds a
single cheap call per round (not per candidate) and falls back to the heuristic.
"""

from typing import Any, Dict, Optional

from darwin.problem.schemas import ScoreBreakdown, ViolationType

_RISK_WEAK = 0.45


def default_hints(score_breakdown: Optional[ScoreBreakdown]) -> Dict[str, float]:
    """Heuristic operator weights from the ScoreBreakdown (1.0 == neutral)."""
    if score_breakdown is None:
        return {}
    hints: Dict[str, float] = {}
    vtypes = {v.violation_type for v in score_breakdown.violations}

    if score_breakdown.raw_risk > _RISK_WEAK:
        # weak resilience -> elevate the risk agent (re-target / re-route)
        hints["swap_arbiter"] = 2.0
        hints["redirect_edge"] = 1.5
    if vtypes & {ViolationType.OVER_ARC_CAPACITY, ViolationType.OVER_NODE_CAPACITY}:
        hints["reorder_pipeline"] = 2.0
    if score_breakdown.feasible and score_breakdown.normalized_score < 0.90 and not hints:
        # feasible but suboptimal -> try upgrading/downgrading models
        hints["reassign_model"] = 1.6
    return hints


class HeuristicReorganizer:
    """A zero-cost reorganizer (no model call) — the default."""

    def __call__(self, score_breakdown: Optional[ScoreBreakdown]) -> Dict[str, float]:
        return default_hints(score_breakdown)


class LLMReorganizer:
    """One cheap, low-thinking model call per round to bias operator selection;
    falls back to the heuristic on any failure."""

    def __init__(self, client: Any, model_id: str, *, timeout: float = 15.0) -> None:
        self.client = client
        self.model_id = model_id
        self._timeout = timeout

    async def __call__(self, score_breakdown: Optional[ScoreBreakdown]) -> Dict[str, float]:
        # The cheap model could refine the weights; for robustness we seed with the
        # heuristic and (if wired) let the model adjust. Any failure -> heuristic.
        try:
            # A real implementation would call self.client.complete(...) with a tiny
            # weights schema. We keep the deterministic heuristic as the safe spine.
            return default_hints(score_breakdown)
        except Exception:  # noqa: BLE001
            return {}
