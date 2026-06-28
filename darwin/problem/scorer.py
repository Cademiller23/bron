"""The deterministic fitness scorer — the single source of truth on quality.

``score(instance, solution, weights) -> ScoreBreakdown`` executes the §4 steps
in this exact order, every time, with **zero randomness, zero wall-clock in the
number, and zero model calls**:

1. Structural validation of the solution (malformed ⇒ score dead-last, never crash).
2. Feasibility checking (the hard constraints) — one ``Violation`` per breach.
3. Compute the three raw objectives (cost, lead time, risk) — even if infeasible.
4. Blend into a scale-normalized weighted objective.
5. Normalize against the known optimum (``optimum / achieved``, capped at 1.0).
6. Apply penalties so an infeasible solution can never outrank a feasible one.
7. Assemble and stamp (``scorer_version`` + ``computed_at`` metadata).

The single most important rule: *the scorer is arithmetic, not judgment.* No
model call ever touches the fitness number — that is what makes the recursion
story airtight.
"""

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from darwin.problem.resilience import compute_resilience
from darwin.problem.schemas import (
    TOL,
    AdditionalConstraint,
    ConstraintType,
    NodeType,
    ObjectiveWeights,
    ProblemClass,
    ProblemInstance,
    ScoreBreakdown,
    Solution,
    Violation,
    ViolationType,
)

# Bumped by B8 whenever the scorer logic changes, so every score ever computed
# is traceable to the exact arithmetic that produced it.
SCORER_VERSION: str = "1.0.0"

# Per-type severity used to scale infeasible penalties. All strictly positive so
# that adding any violation strictly lowers fitness (monotonicity invariant).
_SEVERITY: Dict[ViolationType, float] = {
    ViolationType.DEMAND_UNMET: 10.0,
    ViolationType.SUPPLY_EXCEEDED: 8.0,
    ViolationType.CONSERVATION_BROKEN: 8.0,
    ViolationType.CLOSED_FACILITY_USED: 7.0,
    ViolationType.OVER_NODE_CAPACITY: 6.0,
    ViolationType.OVER_ARC_CAPACITY: 6.0,
    ViolationType.LEAD_TIME_EXCEEDED: 4.0,
    ViolationType.CUSTOM_CONSTRAINT: 3.0,
    ViolationType.MALFORMED_SOLUTION: 1.0,
}

# A base charge per violation guarantees that *more violations* always means a
# strictly lower fitness, even when individual magnitudes are zero.
_BASE_PENALTY_PER_VIOLATION: float = 1.0

# Infeasible solutions are mapped, via a strictly-decreasing soft clamp, into the
# interval (-_INFEASIBLE_FLOOR, 0): for realistic penalties the mapping is
# essentially -penalty (full gradient for the rearrangement loop to climb), and
# it asymptotes to -_INFEASIBLE_FLOOR for pathologically huge penalties. The floor
# is chosen large (1e18) so the float-cancellation point (where penalty + floor ==
# penalty) sits past any realistic penalty (~1e33), keeping the mapping strictly
# monotonic across every plausible instance.
_INFEASIBLE_FLOOR: float = 1.0e18

# A MALFORMED solution uses a *strictly lower* floor (10x), so it is provably below
# EVERY well-formed (even wildly/saturated-infeasible) solution — whose fitness is
# > -_INFEASIBLE_FLOOR — regardless of magnitude or float saturation.
_MALFORMED_FITNESS: float = -_INFEASIBLE_FLOOR * 10.0
_MALFORMED_PENALTY: float = _INFEASIBLE_FLOOR * 10.0


def _infeasible_fitness(total_penalty: float) -> float:
    """Strictly-decreasing soft clamp: penalty>0 -> (-_INFEASIBLE_FLOOR, 0)."""
    if not math.isfinite(total_penalty):
        return -_INFEASIBLE_FLOOR
    fraction = total_penalty / (_INFEASIBLE_FLOOR + total_penalty)
    return -_INFEASIBLE_FLOOR * fraction


def _safe_fsum(values) -> float:
    """``math.fsum`` that returns ``inf`` instead of raising on overflow, so a
    pathological solution scores dead-last gracefully rather than crashing."""
    try:
        return math.fsum(values)
    except OverflowError:
        return math.inf


# ===========================================================================
# Public entry point
# ===========================================================================
def score(
    instance: ProblemInstance,
    solution: Solution,
    weights: Optional[ObjectiveWeights] = None,
) -> ScoreBreakdown:
    """Score ``solution`` against ``instance`` — pure deterministic arithmetic."""
    if weights is None:
        weights = ObjectiveWeights.cost_only()

    if instance.problem_class == ProblemClass.VEHICLE_ROUTING:
        return _score_routing(instance, solution, weights)
    return _score_flow(instance, solution, weights)


# ===========================================================================
# Network-flow scorer (transportation / transshipment / facility location)
# ===========================================================================
def _aggregate_flows(
    instance: ProblemInstance, solution: Solution
) -> Tuple[Optional[Dict[str, float]], Optional[Violation]]:
    """Sum quantities per arc; return ``(arc_flow, None)`` or ``(None, malformed)``.

    Aggregation is **order-independent**: per-arc quantities are summed with
    ``math.fsum`` (associative regardless of list order), so shuffling
    ``solution.flows`` — including duplicate entries on the same arc — can never
    change the result. The aggregated total is re-checked for finiteness so a
    same-arc overflow (e.g. ``1e308 + 1e308``) scores dead-last instead of
    crashing the scorer.
    """
    arc_index = instance.arc_index
    per_arc: Dict[str, List[float]] = {}
    for fa in solution.flows:
        if fa.arc_id not in arc_index:
            return None, Violation(
                violation_type=ViolationType.MALFORMED_SOLUTION,
                location=fa.arc_id,
                magnitude=0.0,
                message=f"flow references unknown arc_id {fa.arc_id!r}",
            )
        if not math.isfinite(fa.quantity) or fa.quantity < 0:
            return None, Violation(
                violation_type=ViolationType.MALFORMED_SOLUTION,
                location=fa.arc_id,
                magnitude=0.0,
                message=f"flow quantity on {fa.arc_id!r} is not a finite non-negative number",
            )
        per_arc.setdefault(fa.arc_id, []).append(fa.quantity)

    arc_flow: Dict[str, float] = {}
    for arc_id in sorted(per_arc):
        total = _safe_fsum(per_arc[arc_id])
        if not math.isfinite(total):
            return None, Violation(
                violation_type=ViolationType.MALFORMED_SOLUTION,
                location=arc_id,
                magnitude=0.0,
                message=f"aggregated flow on {arc_id!r} overflowed to a non-finite value",
            )
        arc_flow[arc_id] = total
    # A finite grand total guarantees every downstream node/objective sum is
    # finite too (each is a subset sum bounded by this), so feasibility checks
    # never see an inf magnitude.
    if not math.isfinite(_safe_fsum(arc_flow.values())):
        return None, Violation(
            violation_type=ViolationType.MALFORMED_SOLUTION,
            location=instance.instance_id,
            magnitude=0.0,
            message="aggregate flow across arcs overflowed to a non-finite value",
        )
    return arc_flow, None


def _validate_open_facilities(
    instance: ProblemInstance, solution: Solution
) -> Optional[Violation]:
    if not solution.open_facilities:
        return None
    node_index = instance.node_index
    for nid in solution.open_facilities:
        node = node_index.get(nid)
        if node is None:
            return Violation(
                violation_type=ViolationType.MALFORMED_SOLUTION,
                location=nid,
                magnitude=0.0,
                message=f"open_facilities references unknown node {nid!r}",
            )
        if not node.is_optional:
            return Violation(
                violation_type=ViolationType.MALFORMED_SOLUTION,
                location=nid,
                magnitude=0.0,
                message=f"open_facilities references non-optional node {nid!r}",
            )
    return None


def _node_flows(
    instance: ProblemInstance, arc_flow: Dict[str, float]
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Per-node inflow/outflow, accumulated order-independently with ``fsum``."""
    arc_index = instance.arc_index
    in_terms: Dict[str, List[float]] = {n.node_id: [] for n in instance.nodes}
    out_terms: Dict[str, List[float]] = {n.node_id: [] for n in instance.nodes}
    for arc_id in sorted(arc_flow):
        qty = arc_flow[arc_id]
        arc = arc_index[arc_id]
        out_terms[arc.from_node].append(qty)
        in_terms[arc.to_node].append(qty)
    inflow = {nid: _safe_fsum(terms) for nid, terms in in_terms.items()}
    outflow = {nid: _safe_fsum(terms) for nid, terms in out_terms.items()}
    return inflow, outflow


def _check_feasibility(
    instance: ProblemInstance,
    solution: Solution,
    arc_flow: Dict[str, float],
    inflow: Dict[str, float],
    outflow: Dict[str, float],
) -> List[Violation]:
    violations: List[Violation] = []
    node_index = instance.node_index
    open_set = set(solution.open_facilities or [])

    # (a) Arc capacity
    for arc in sorted(instance.arcs, key=lambda a: a.arc_id):
        if arc.capacity is None:
            continue
        qty = arc_flow.get(arc.arc_id, 0.0)
        if qty > arc.capacity + TOL:
            violations.append(
                Violation(
                    violation_type=ViolationType.OVER_ARC_CAPACITY,
                    location=arc.arc_id,
                    magnitude=qty - arc.capacity,
                    message=f"arc {arc.arc_id} carries {qty} > capacity {arc.capacity}",
                )
            )

    # (b) Node throughput capacity
    for node in sorted(instance.nodes, key=lambda n: n.node_id):
        if node.capacity is None:
            continue
        throughput = max(inflow[node.node_id], outflow[node.node_id])
        if throughput > node.capacity + TOL:
            violations.append(
                Violation(
                    violation_type=ViolationType.OVER_NODE_CAPACITY,
                    location=node.node_id,
                    magnitude=throughput - node.capacity,
                    message=f"node {node.node_id} throughput {throughput} > capacity {node.capacity}",
                )
            )

    # (c) Supply limits (net injection at a source)
    for node in sorted(instance.nodes, key=lambda n: n.node_id):
        if node.node_type != NodeType.SOURCE or node.supply is None:
            continue
        net_out = outflow[node.node_id] - inflow[node.node_id]
        if net_out > node.supply + TOL:
            violations.append(
                Violation(
                    violation_type=ViolationType.SUPPLY_EXCEEDED,
                    location=node.node_id,
                    magnitude=net_out - node.supply,
                    message=f"source {node.node_id} ships {net_out} > supply {node.supply}",
                )
            )

    # (d) Demand satisfaction (net delivery at a sink)
    for node in sorted(instance.nodes, key=lambda n: n.node_id):
        if node.node_type != NodeType.SINK or not node.demand:
            continue
        net_in = inflow[node.node_id] - outflow[node.node_id]
        if net_in < node.demand - TOL:
            violations.append(
                Violation(
                    violation_type=ViolationType.DEMAND_UNMET,
                    location=node.node_id,
                    magnitude=node.demand - net_in,
                    message=f"sink {node.node_id} receives {net_in} < demand {node.demand}",
                )
            )

    # (e) Flow conservation at transshipment nodes
    for node in sorted(instance.nodes, key=lambda n: n.node_id):
        if node.node_type != NodeType.TRANSSHIPMENT:
            continue
        imbalance = inflow[node.node_id] - outflow[node.node_id]
        if abs(imbalance) > TOL:
            violations.append(
                Violation(
                    violation_type=ViolationType.CONSERVATION_BROKEN,
                    location=node.node_id,
                    magnitude=abs(imbalance),
                    message=f"node {node.node_id} in {inflow[node.node_id]} != out {outflow[node.node_id]}",
                )
            )

    # (f) Facility-open consistency: no flow may touch a closed optional facility.
    for node in sorted(instance.nodes, key=lambda n: n.node_id):
        if not node.is_optional or node.node_id in open_set:
            continue
        throughput = max(inflow[node.node_id], outflow[node.node_id])
        if throughput > TOL:
            violations.append(
                Violation(
                    violation_type=ViolationType.CLOSED_FACILITY_USED,
                    location=node.node_id,
                    magnitude=throughput,
                    message=f"closed optional facility {node.node_id} carries flow {throughput}",
                )
            )

    # (g) + (h) Declared additional constraints (incl. LEAD_TIME_LIMIT).
    for constraint in sorted(instance.additional_constraints, key=lambda c: c.constraint_id):
        violations.extend(
            _check_additional_constraint(instance, solution, arc_flow, inflow, outflow, open_set, constraint)
        )

    return violations


def _check_additional_constraint(
    instance: ProblemInstance,
    solution: Solution,
    arc_flow: Dict[str, float],
    inflow: Dict[str, float],
    outflow: Dict[str, float],
    open_set: set,
    constraint: AdditionalConstraint,
) -> List[Violation]:
    out: List[Violation] = []
    p = constraint.parameters
    ctype = constraint.constraint_type
    arc_index = instance.arc_index

    if ctype == ConstraintType.LEAD_TIME_LIMIT:
        limit = float(p.get("limit", math.inf))
        target_arc = p.get("arc_id")
        for arc in sorted(instance.arcs, key=lambda a: a.arc_id):
            if target_arc is not None and arc.arc_id != target_arc:
                continue
            if arc_flow.get(arc.arc_id, 0.0) <= TOL:
                continue
            if arc.lead_time > limit + TOL:
                out.append(
                    Violation(
                        violation_type=ViolationType.LEAD_TIME_EXCEEDED,
                        location=arc.arc_id,
                        magnitude=arc.lead_time - limit,
                        message=f"used arc {arc.arc_id} lead_time {arc.lead_time} > limit {limit}",
                    )
                )

    elif ctype == ConstraintType.CAPACITY:
        limit = float(p.get("limit", math.inf))
        if "arc_id" in p:
            qty = arc_flow.get(p["arc_id"], 0.0)
            if qty > limit + TOL:
                out.append(
                    Violation(
                        violation_type=ViolationType.OVER_ARC_CAPACITY,
                        location=str(p["arc_id"]),
                        magnitude=qty - limit,
                        message=f"declared arc-capacity constraint {constraint.constraint_id} breached",
                    )
                )
        elif "node_id" in p:
            nid = p["node_id"]
            throughput = max(inflow.get(nid, 0.0), outflow.get(nid, 0.0))
            if throughput > limit + TOL:
                out.append(
                    Violation(
                        violation_type=ViolationType.OVER_NODE_CAPACITY,
                        location=str(nid),
                        magnitude=throughput - limit,
                        message=f"declared node-capacity constraint {constraint.constraint_id} breached",
                    )
                )

    elif ctype == ConstraintType.SINGLE_SOURCE:
        sink_id = p.get("sink_id") or p.get("node_id")
        if sink_id is not None:
            suppliers = {
                arc_index[aid].from_node
                for aid, qty in arc_flow.items()
                if qty > TOL and arc_index[aid].to_node == sink_id
            }
            if len(suppliers) > 1:
                out.append(
                    Violation(
                        violation_type=ViolationType.CUSTOM_CONSTRAINT,
                        location=str(sink_id),
                        magnitude=float(len(suppliers) - 1),
                        message=f"single-source constraint: sink {sink_id} fed by {len(suppliers)} suppliers",
                    )
                )

    elif ctype == ConstraintType.MUTUAL_EXCLUSION:
        ids = list(p.get("node_ids", []))
        used = [
            nid
            for nid in ids
            if nid in open_set or max(inflow.get(nid, 0.0), outflow.get(nid, 0.0)) > TOL
        ]
        if len(used) > 1:
            out.append(
                Violation(
                    violation_type=ViolationType.CUSTOM_CONSTRAINT,
                    location=",".join(sorted(used)),
                    magnitude=float(len(used) - 1),
                    message=f"mutual-exclusion constraint {constraint.constraint_id}: {used} all used",
                )
            )

    # DEMAND_SATISFACTION / FLOW_CONSERVATION are already enforced structurally;
    # CUSTOM with unknown semantics is skipped gracefully (cannot be evaluated).
    return out


def _raw_objectives(
    instance: ProblemInstance,
    solution: Solution,
    arc_flow: Dict[str, float],
) -> Tuple[float, float, float, float, object]:
    """Return ``(raw_cost, raw_lead_time, weighted_lead_time, total_flow, resilience)``."""
    open_set = set(solution.open_facilities or [])

    cost_terms: List[float] = []
    lead_terms: List[float] = []
    flow_terms: List[float] = []
    max_lead = 0.0

    for arc in instance.arcs:
        qty = arc_flow.get(arc.arc_id, 0.0)
        cost_terms.append(arc.unit_cost * qty)
        if qty > TOL:
            if arc.fixed_cost is not None:
                cost_terms.append(arc.fixed_cost)
            max_lead = max(max_lead, arc.lead_time)
            lead_terms.append(arc.lead_time * qty)
            flow_terms.append(qty)

    # Facility fixed costs: optional nodes count only if opened; non-optional
    # nodes with a fixed_cost are always open and always charged.
    for node in instance.nodes:
        if node.fixed_cost is None:
            continue
        if node.is_optional:
            if node.node_id in open_set:
                cost_terms.append(node.fixed_cost)
        else:
            cost_terms.append(node.fixed_cost)

    raw_cost = _safe_fsum(cost_terms)
    total_flow = _safe_fsum(flow_terms)
    weighted_lead = (_safe_fsum(lead_terms) / total_flow) if total_flow > TOL else 0.0
    resilience = compute_resilience(instance, arc_flow)
    return raw_cost, max_lead, weighted_lead, total_flow, resilience


def _score_flow(
    instance: ProblemInstance, solution: Solution, weights: ObjectiveWeights
) -> ScoreBreakdown:
    # --- Step 1: structural validation ------------------------------------
    arc_flow, malformed = _aggregate_flows(instance, solution)
    if malformed is None:
        malformed = _validate_open_facilities(instance, solution)
    if malformed is not None or arc_flow is None:
        return _malformed_breakdown(instance, solution, weights, malformed)

    # --- Step 2: feasibility ----------------------------------------------
    inflow, outflow = _node_flows(instance, arc_flow)
    violations = _check_feasibility(instance, solution, arc_flow, inflow, outflow)
    feasible = len(violations) == 0

    # --- Step 3: raw objectives -------------------------------------------
    raw_cost, raw_lead, weighted_lead, total_flow, resilience = _raw_objectives(
        instance, solution, arc_flow
    )
    if not math.isfinite(raw_cost):  # cost overflow on a pathological solution
        return _malformed_breakdown(
            instance, solution, weights,
            Violation(
                violation_type=ViolationType.MALFORMED_SOLUTION,
                location=instance.instance_id,
                magnitude=0.0,
                message="raw cost overflowed to a non-finite value",
            ),
        )
    raw_risk = resilience.raw_risk

    diagnostics: Dict[str, object] = {
        "total_flow": total_flow,
        "weighted_lead_time": weighted_lead,
        "resilience": dict(resilience.components),
        "removed_source": resilience.removed_source,
        "deliverable_after_failure": resilience.deliverable_after_failure,
    }

    return _assemble(
        instance, solution, weights, feasible, violations, raw_cost, raw_lead, raw_risk, diagnostics
    )


# ===========================================================================
# Vehicle-routing scorer (CVRP) — a parallel path producing the same breakdown.
# ===========================================================================
def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _vehicle_capacity(instance: ProblemInstance) -> Optional[float]:
    for c in instance.additional_constraints:
        if c.constraint_type == ConstraintType.CAPACITY and "vehicle_capacity" in c.parameters:
            return float(c.parameters["vehicle_capacity"])
    # fall back to the depot's capacity
    for n in instance.sources():
        if n.capacity is not None:
            return n.capacity
    return None


def _score_routing(
    instance: ProblemInstance, solution: Solution, weights: ObjectiveWeights
) -> ScoreBreakdown:
    node_index = instance.node_index
    routes = solution.routes or []
    depots = [n.node_id for n in instance.sources()]
    depot = depots[0] if depots else None

    # --- Step 1: structural validation ------------------------------------
    for route in routes:
        for nid in route.node_sequence:
            if nid not in node_index:
                return _malformed_breakdown(
                    instance, solution, weights,
                    Violation(
                        violation_type=ViolationType.MALFORMED_SOLUTION,
                        location=route.vehicle_id,
                        magnitude=0.0,
                        message=f"route {route.vehicle_id} visits unknown node {nid!r}",
                    ),
                )
        if depot is not None and route.node_sequence and (
            route.node_sequence[0] != depot or route.node_sequence[-1] != depot
        ):
            return _malformed_breakdown(
                instance, solution, weights,
                Violation(
                    violation_type=ViolationType.MALFORMED_SOLUTION,
                    location=route.vehicle_id,
                    magnitude=0.0,
                    message=f"route {route.vehicle_id} must start and end at depot {depot}",
                ),
            )
    # coordinates required to compute distances
    for n in instance.nodes:
        if n.coordinates is None:
            return _malformed_breakdown(
                instance, solution, weights,
                Violation(
                    violation_type=ViolationType.MALFORMED_SOLUTION,
                    location=n.node_id,
                    magnitude=0.0,
                    message=f"VRP node {n.node_id} is missing coordinates",
                ),
            )

    # --- Step 2: feasibility ----------------------------------------------
    violations: List[Violation] = []
    capacity = _vehicle_capacity(instance)
    customers = {n.node_id for n in instance.sinks()}

    visit_count: Dict[str, int] = {c: 0 for c in customers}
    for route in routes:
        load = 0.0
        for nid in route.node_sequence:
            if nid in customers:
                visit_count[nid] += 1
                load += node_index[nid].demand or 0.0
        if capacity is not None and load > capacity + TOL:
            violations.append(
                Violation(
                    violation_type=ViolationType.OVER_NODE_CAPACITY,
                    location=route.vehicle_id,
                    magnitude=load - capacity,
                    message=f"vehicle {route.vehicle_id} load {load} > capacity {capacity}",
                )
            )

    for cid in sorted(customers):
        if visit_count[cid] == 0:
            violations.append(
                Violation(
                    violation_type=ViolationType.DEMAND_UNMET,
                    location=cid,
                    magnitude=node_index[cid].demand or 1.0,
                    message=f"customer {cid} is never visited",
                )
            )
        elif visit_count[cid] > 1:
            violations.append(
                Violation(
                    violation_type=ViolationType.CUSTOM_CONSTRAINT,
                    location=cid,
                    magnitude=float(visit_count[cid] - 1),
                    message=f"customer {cid} visited {visit_count[cid]} times (must be once)",
                )
            )

    feasible = len(violations) == 0

    # --- Step 3: raw objectives -------------------------------------------
    total_distance = 0.0
    max_route_distance = 0.0
    for route in routes:
        d = 0.0
        seq = route.node_sequence
        for i in range(len(seq) - 1):
            d += _euclid(node_index[seq[i]].coordinates, node_index[seq[i + 1]].coordinates)
        total_distance += d
        max_route_distance = max(max_route_distance, d)

    raw_cost = total_distance
    raw_lead = max_route_distance
    raw_risk, risk_components = _routing_risk(instance, routes)

    diagnostics: Dict[str, object] = {
        "num_routes": len(routes),
        "max_route_distance": max_route_distance,
        "vehicle_capacity": capacity,
        "resilience": risk_components,
    }

    return _assemble(
        instance, solution, weights, feasible, violations, raw_cost, raw_lead, raw_risk, diagnostics
    )


def _routing_risk(instance, routes) -> Tuple[float, Dict[str, float]]:
    from darwin.problem.resilience import ALPHA, BETA, GAMMA

    node_index = instance.node_index
    customers = {n.node_id for n in instance.sinks()}
    total_demand = sum(node_index[c].demand or 0.0 for c in customers)
    if total_demand <= TOL:
        return 0.0, {"concentration": 0.0, "exposure": 0.0, "worst_case_unmet": 0.0}

    route_demands = []
    exposure_numer = 0.0
    for route in routes:
        rd = 0.0
        for nid in route.node_sequence:
            if nid in customers:
                dem = node_index[nid].demand or 0.0
                rd += dem
                exposure_numer += dem * (node_index[nid].risk_score or 0.0)
        route_demands.append(rd)

    biggest_route = max(route_demands) if route_demands else 0.0
    concentration = biggest_route / total_demand
    exposure = exposure_numer / total_demand
    worst_case = biggest_route / total_demand  # losing the biggest vehicle strands its customers

    raw_risk = ALPHA * concentration + BETA * exposure + GAMMA * worst_case
    raw_risk = min(1.0, max(0.0, raw_risk))
    return raw_risk, {
        "concentration": concentration,
        "exposure": exposure,
        "worst_case_unmet": worst_case,
    }


# ===========================================================================
# Steps 4–7 (shared): blend, normalize, penalize, assemble & stamp.
# ===========================================================================
def _instance_cost_bound(instance: ProblemInstance) -> float:
    """A solution-independent upper-bound-ish reference for cost normalization,
    used only when no known optimum is attached. Depends only on the instance, so
    the weighted objective still reflects how expensive a solution is."""
    max_unit = max((a.unit_cost for a in instance.arcs), default=0.0)
    fixed = math.fsum(
        [a.fixed_cost for a in instance.arcs if a.fixed_cost is not None]
        + [n.fixed_cost for n in instance.nodes if n.fixed_cost is not None]
    )
    return max(max_unit * instance.total_demand() + fixed, 1.0)


def _normalization_refs(instance: ProblemInstance, raw_cost: float, raw_lead: float):
    opt = instance.known_optimum.objective_value if instance.known_optimum else None
    cost_ref = opt if (opt is not None and opt > TOL) else _instance_cost_bound(instance)
    instance_max_lead = max((a.lead_time for a in instance.arcs), default=0.0)
    lead_ref = instance_max_lead if instance_max_lead > TOL else max(raw_lead, 1.0)
    return opt, cost_ref, lead_ref


def _assemble(
    instance: ProblemInstance,
    solution: Solution,
    weights: ObjectiveWeights,
    feasible: bool,
    violations: List[Violation],
    raw_cost: float,
    raw_lead: float,
    raw_risk: float,
    diagnostics: Dict[str, object],
) -> ScoreBreakdown:
    opt, cost_ref, lead_ref = _normalization_refs(instance, raw_cost, raw_lead)

    # --- Step 4: blend into a scale-normalized weighted objective ----------
    norm_cost = raw_cost / cost_ref if cost_ref > TOL else 0.0
    norm_lead = raw_lead / lead_ref if lead_ref > TOL else 0.0
    norm_risk = raw_risk  # already in [0, 1]
    weighted_objective = (
        weights.cost_weight * norm_cost
        + weights.lead_time_weight * norm_lead
        + weights.risk_weight * norm_risk
    )

    # --- Step 5: normalize against the known optimum (cost-normalized) ------
    below_optimum = False
    if opt is not None:
        if raw_cost <= TOL:
            normalized_score = 1.0
            below_optimum = opt > TOL  # achieved ~zero cost vs a positive optimum
        else:
            ratio = opt / raw_cost
            normalized_score = min(1.0, ratio)
            below_optimum = ratio > 1.0 + TOL  # achieved cheaper than the labelled optimum
    else:
        normalized_score = 1.0 / (1.0 + raw_cost)  # fallback when no optimum is attached

    # --- Step 6: penalties so infeasible can never beat feasible -----------
    if feasible:
        total_penalty = 0.0
        final_fitness = normalized_score
    else:
        total_penalty = _BASE_PENALTY_PER_VIOLATION * len(violations) + math.fsum(
            _SEVERITY[v.violation_type] * v.magnitude for v in violations
        )
        final_fitness = _infeasible_fitness(total_penalty)
        if not math.isfinite(total_penalty):  # keep the stored field serializable
            total_penalty = _INFEASIBLE_FLOOR

    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "norm_cost": norm_cost,
            "norm_lead": norm_lead,
            "norm_risk": norm_risk,
            "cost_ref": cost_ref,
            "lead_ref": lead_ref,
            "optimum": opt,
            "below_labeled_optimum": below_optimum,
            "no_optimum_reference": opt is None,
        }
    )

    # --- Step 7: assemble and stamp ---------------------------------------
    return ScoreBreakdown(
        solution_id=solution.solution_id,
        instance_id=instance.instance_id,
        feasible=feasible,
        violations=violations,
        raw_cost=raw_cost,
        raw_lead_time=raw_lead,
        raw_risk=raw_risk,
        weighted_objective=weighted_objective,
        normalized_score=normalized_score,
        total_penalty=total_penalty,
        final_fitness=final_fitness,
        objective_weights=weights,
        scorer_version=SCORER_VERSION,
        computed_at=_now_iso(),
        diagnostics=diagnostics,
    )


def _malformed_breakdown(
    instance: ProblemInstance,
    solution: Solution,
    weights: ObjectiveWeights,
    violation: Optional[Violation],
) -> ScoreBreakdown:
    """A garbage solution from a misbehaving agent scores dead-last gracefully."""
    if violation is None:
        violation = Violation(
            violation_type=ViolationType.MALFORMED_SOLUTION,
            location=solution.solution_id,
            magnitude=0.0,
            message="malformed solution",
        )
    return ScoreBreakdown(
        solution_id=solution.solution_id,
        instance_id=instance.instance_id,
        feasible=False,
        violations=[violation],
        raw_cost=0.0,
        raw_lead_time=0.0,
        raw_risk=0.0,
        weighted_objective=0.0,
        normalized_score=0.0,
        total_penalty=_MALFORMED_PENALTY,
        # Strictly below every well-formed solution, whose fitness is always
        # > -_INFEASIBLE_FLOOR even under float saturation.
        final_fitness=_MALFORMED_FITNESS,
        objective_weights=weights,
        scorer_version=SCORER_VERSION,
        computed_at=_now_iso(),
        diagnostics={"malformed": True},
    )


def _now_iso() -> str:
    """Wall-clock timestamp — *metadata only*, never part of the fitness number."""
    return datetime.now(timezone.utc).isoformat()
