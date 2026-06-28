"""Gap diagnosis — the single source of "what's wrong", deterministically.

The deterministic scorer diagnoses; the rest of B6 acts on that diagnosis.
``diagnose_gap`` reads the B1 ``ScoreBreakdown`` and produces a structured,
natural-language ``GapDescription`` (no model call) used as BOTH the corpus
vector-search query and the seed for the Architect's curation prompt — one
diagnosis, two consumers, aimed at the same gap.
"""

from collections import Counter
from typing import Optional

from darwin.problem.schemas import ViolationType
from darwin.team.evaluation import GenomeEvaluation
from darwin.escalation.schemas import GapDescription, WeakDimension

_RISK_WEAK = 0.45
_LEAD_WEAK = 0.50  # norm_lead above this is a lead-time weakness

_CAPABILITY = {
    "demand": "an agent that ensures all customer demand is met, finding feasible routes/allocations that close unmet-demand gaps.",
    "capacity": "an agent that rebalances flows to respect capacity limits and resolve overflows.",
    WeakDimension.RESILIENCE: "an agent that reduces supply-chain disruption risk by diversifying sourcing and avoiding single points of failure.",
    WeakDimension.COST: "an agent that aggressively minimizes total cost by finding cheaper allocations/routes.",
    WeakDimension.LEAD_TIME: "an agent that reduces delivery lead time and expedites the slowest shipments.",
    WeakDimension.FEASIBILITY: "an agent that finds a feasible solution respecting every hard constraint.",
}

_ROLE_KIND = {
    WeakDimension.FEASIBILITY: "proposer",
    WeakDimension.COST: "proposer",
    WeakDimension.RESILIENCE: "specialist",
    WeakDimension.LEAD_TIME: "specialist",
}


def diagnose_gap(evaluation: GenomeEvaluation, problem_class: Optional[str] = None) -> GapDescription:
    sb = evaluation.score_breakdown
    vtypes = [v.violation_type for v in sb.violations]
    counts = Counter(v.value for v in vtypes)

    # weakness score per dimension (higher == weaker)
    scores = {}
    if sb.violations:
        scores[WeakDimension.FEASIBILITY] = 1.0
    scores[WeakDimension.RESILIENCE] = sb.raw_risk
    if sb.feasible:
        scores[WeakDimension.COST] = max(0.0, 0.90 - sb.normalized_score)
    norm_lead = float(sb.diagnostics.get("norm_lead", 0.0) or 0.0)
    if norm_lead > _LEAD_WEAK:
        scores[WeakDimension.LEAD_TIME] = norm_lead

    # ranked weak dimensions (above a small floor), strongest first
    ranked = sorted(
        [d for d, s in scores.items() if s > 1e-9 and (d == WeakDimension.FEASIBILITY or s > _floor(d))],
        key=lambda d: -scores[d],
    )
    # Infeasibility dominates EVERYTHING: an infeasible solution must be diagnosed
    # as a FEASIBILITY gap regardless of other dimensions' raw magnitudes (e.g.
    # norm_lead can exceed 1.0 on routing, which would otherwise out-rank the
    # fixed FEASIBILITY score of 1.0 and skip the demand/capacity routing).
    if sb.violations and WeakDimension.FEASIBILITY in scores:
        ranked = [WeakDimension.FEASIBILITY] + [d for d in ranked if d != WeakDimension.FEASIBILITY]
    primary = ranked[0] if ranked else WeakDimension.COST

    # pick the capability string (sub-classify feasibility by the violations seen)
    if primary == WeakDimension.FEASIBILITY:
        if ViolationType.DEMAND_UNMET in vtypes:
            capability = _CAPABILITY["demand"]
        elif {ViolationType.OVER_ARC_CAPACITY, ViolationType.OVER_NODE_CAPACITY} & set(vtypes):
            capability = _CAPABILITY["capacity"]
            ranked = [WeakDimension.FEASIBILITY] + [d for d in ranked if d != WeakDimension.FEASIBILITY]
        else:
            capability = _CAPABILITY[WeakDimension.FEASIBILITY]
        role_kind = "checker" if "capacity" in capability else "proposer"
    else:
        capability = _CAPABILITY[primary]
        role_kind = _ROLE_KIND.get(primary, "specialist")

    severity = 1.0 if not sb.feasible else max(0.0, 0.90 - sb.normalized_score)

    return GapDescription(
        capability_needed=capability,
        weak_dimensions=ranked or [WeakDimension.COST],
        dominant_violations=[vt for vt, _ in counts.most_common()],
        problem_class=problem_class or "",
        suggested_role_kind=role_kind,
        severity=severity,
    )


def _floor(dim: WeakDimension) -> float:
    if dim == WeakDimension.RESILIENCE:
        return _RISK_WEAK
    if dim == WeakDimension.LEAD_TIME:
        return _LEAD_WEAK
    return 0.0  # COST: any positive gap counts
