"""The deterministic resilience / risk metric — Darwin's differentiator.

Cost and lead time are obvious; resilience is what makes the supply-chain
story land and what justifies the risk agent the Architect curates. It must be
deterministic, sub-millisecond, and genuinely meaningful.

``raw_risk`` blends three components, each in ``[0, 1]`` (lower is better):

* **C — single-source dependency (concentration).** For each sink, the fraction
  of its met demand that arrives from its single largest-contributing immediate
  supplier; demand-weighted-averaged across sinks. ``C = 1`` means every
  customer depends entirely on one supplier (maximally fragile). Equivalent to
  the dominant term of a Herfindahl index over sourcing shares.
* **E — expected disruption exposure.** ``Σ(flow × arc.risk) / Σ flow`` over
  used arcs: how much volume rides on individually risky links.
* **W — worst-case single-failure unmet demand.** Remove the single
  highest-flow source ("our biggest supplier goes down"), then recompute how
  much total demand can still be met given the remaining capacity. The fraction
  left unmet is the most legible resilience number there is, and for the small
  instances Darwin runs it is still sub-millisecond.

``raw_risk = α·C + β·E + γ·W`` with fixed constants ``α, β, γ`` that sum to 1,
so ``raw_risk`` is itself normalized to ``[0, 1]``.

The pitch line this earns: *"the swarm learns to de-risk by diversifying
sourcing, not just chase the cheapest route."*
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from darwin.problem.flow import max_deliverable_demand
from darwin.problem.schemas import TOL, NodeType, ProblemInstance

# Fixed blend constants (sum to 1.0 so raw_risk stays in [0, 1]).
ALPHA: float = 0.30  # concentration
BETA: float = 0.30  # exposure
GAMMA: float = 0.40  # worst-case single failure (the headline term)


@dataclass(frozen=True)
class ResilienceBreakdown:
    raw_risk: float
    concentration: float
    exposure: float
    worst_case_unmet: float
    herfindahl: float
    removed_source: Optional[str]
    deliverable_after_failure: float
    components: Dict[str, float] = field(default_factory=dict)


def _concentration(instance: ProblemInstance, arc_flow: Dict[str, float]) -> "tuple[float, float]":
    """Demand-weighted single-largest-supplier share, plus average Herfindahl."""
    arc_index = instance.arc_index
    # supplier -> flow, per sink
    per_sink_inbound: Dict[str, Dict[str, float]] = {}
    for arc_id, qty in arc_flow.items():
        if qty <= TOL:
            continue
        arc = arc_index[arc_id]
        to_node = instance.node_index[arc.to_node]
        if to_node.node_type != NodeType.SINK:
            continue
        per_sink_inbound.setdefault(arc.to_node, {})
        per_sink_inbound[arc.to_node][arc.from_node] = (
            per_sink_inbound[arc.to_node].get(arc.from_node, 0.0) + qty
        )

    weighted_share = 0.0
    weighted_herfindahl = 0.0
    weight_total = 0.0
    # iterate sinks in deterministic id order
    for sink_id in sorted(per_sink_inbound):
        inbound = per_sink_inbound[sink_id]
        total_in = sum(inbound.values())
        if total_in <= TOL:
            continue
        largest_share = max(inbound.values()) / total_in
        herfindahl = sum((v / total_in) ** 2 for v in inbound.values())
        demand = instance.node_index[sink_id].demand
        weight = demand if demand else total_in
        weighted_share += weight * largest_share
        weighted_herfindahl += weight * herfindahl
        weight_total += weight

    if weight_total <= TOL:
        return 0.0, 0.0
    return weighted_share / weight_total, weighted_herfindahl / weight_total


def _exposure(instance: ProblemInstance, arc_flow: Dict[str, float]) -> float:
    arc_index = instance.arc_index
    risky_volume = 0.0
    total_flow = 0.0
    for arc_id, qty in arc_flow.items():
        if qty <= TOL:
            continue
        risk = arc_index[arc_id].risk_score or 0.0
        risky_volume += qty * risk
        total_flow += qty
    if total_flow <= TOL:
        return 0.0
    return risky_volume / total_flow


def _biggest_source(instance: ProblemInstance, arc_flow: Dict[str, float]) -> Optional[str]:
    """The single source carrying the most flow (deterministic tie-break by id)."""
    arc_index = instance.arc_index
    outflow: Dict[str, float] = {}
    for arc_id, qty in arc_flow.items():
        if qty <= TOL:
            continue
        from_node = instance.node_index[arc_index[arc_id].from_node]
        if from_node.node_type == NodeType.SOURCE:
            outflow[from_node.node_id] = outflow.get(from_node.node_id, 0.0) + qty

    if outflow:
        # max flow, tie-break by smallest node_id
        return min(sorted(outflow), key=lambda nid: (-outflow[nid], nid))

    sources = instance.sources()
    if not sources:
        return None
    # fall back to the largest-supply source, tie-break by id
    return min(
        sorted(s.node_id for s in sources),
        key=lambda nid: (-(instance.node_index[nid].supply or 0.0), nid),
    )


def _worst_case_unmet(
    instance: ProblemInstance, arc_flow: Dict[str, float]
) -> "tuple[float, Optional[str], float]":
    total_demand = instance.total_demand()
    removed = _biggest_source(instance, arc_flow)
    if total_demand <= TOL or removed is None:
        return 0.0, removed, max_deliverable_demand(instance)

    deliverable = max_deliverable_demand(instance, excluded_sources=frozenset({removed}))
    unmet = max(0.0, total_demand - deliverable) / total_demand
    return min(1.0, unmet), removed, deliverable


def compute_resilience(
    instance: ProblemInstance, arc_flow: Dict[str, float]
) -> ResilienceBreakdown:
    """Compute the full deterministic resilience breakdown for a flow.

    ``arc_flow`` maps ``arc_id`` → aggregated quantity shipped on that arc.
    """
    concentration, herfindahl = _concentration(instance, arc_flow)
    exposure = _exposure(instance, arc_flow)
    worst_case, removed, deliverable = _worst_case_unmet(instance, arc_flow)

    raw_risk = ALPHA * concentration + BETA * exposure + GAMMA * worst_case
    # Guard against tiny floating drift outside [0, 1].
    raw_risk = min(1.0, max(0.0, raw_risk))

    return ResilienceBreakdown(
        raw_risk=raw_risk,
        concentration=concentration,
        exposure=exposure,
        worst_case_unmet=worst_case,
        herfindahl=herfindahl,
        removed_source=removed,
        deliverable_after_failure=deliverable,
        components={
            "alpha": ALPHA,
            "beta": BETA,
            "gamma": GAMMA,
            "concentration": concentration,
            "exposure": exposure,
            "worst_case_unmet": worst_case,
        },
    )
