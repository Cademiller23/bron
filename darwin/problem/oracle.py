"""The solver oracle — establishing and verifying ground truth.

Darwin's *agents* never call a solver (that would make them code-generators,
not reasoners). B1's *tooling* uses one, offline, for two jobs:

1. **Verify** that benchmark-labelled optima are actually correct (IndustryOR is
   known to mislabel a large fraction of its optima), and
2. **Compute** the true optimum of instances generated live.

Backends (selected automatically):

* **OR-Tools** (``pywraplp``) — an LP for pure transportation/transshipment and
  a MILP (CBC) for facility location / fixed charges. Floats are handled
  natively, so the oracle's objective matches the scorer's ``raw_cost`` exactly.
* **Pure-Python fallback** — exact successive-shortest-path min-cost flow
  (:mod:`darwin.problem.flow`) for the pure-flow case, and guarded subset
  enumeration for facility location. Used when OR-Tools is unavailable so the
  oracle never hard-depends on a heavy binary.
* **Exact brute-force CVRP** — partition + per-route TSP enumeration for the
  small vehicle-routing instances Darwin runs.

``solve_optimum(instance)`` returns an :class:`OracleResult`; on infeasible
instances it returns a clear ``INFEASIBLE`` status rather than a wrong number.
"""

import itertools
import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from darwin.problem import flow as flowmod
from darwin.problem.schemas import (
    TOL,
    ConstraintType,
    FlowAssignment,
    NodeType,
    ProblemClass,
    ProblemInstance,
    Route,
    Solution,
)

# ---------------------------------------------------------------------------
# OR-Tools is optional. The oracle degrades gracefully to pure Python.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised by whichever environment runs the tests
    from ortools.linear_solver import pywraplp

    ORTOOLS_AVAILABLE = True
except Exception:  # pragma: no cover
    pywraplp = None  # type: ignore
    ORTOOLS_AVAILABLE = False

STATUS_OPTIMAL = "OPTIMAL"
STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_UNSUPPORTED = "UNSUPPORTED"

# Guard so the pure-Python enumeration fallback can never blow up.
_MAX_ENUM_BITS = 18
_MAX_VRP_CUSTOMERS = 9


@dataclass(frozen=True)
class OracleResult:
    status: str
    objective_value: Optional[float]
    solution: Optional[Solution]
    backend: str


# ===========================================================================
# Public entry points
# ===========================================================================
def solve_optimum(instance: ProblemInstance, backend: str = "auto") -> OracleResult:
    """Solve ``instance`` to optimality and return the verified optimum."""
    if instance.problem_class == ProblemClass.VEHICLE_ROUTING:
        return _solve_vrp_bruteforce(instance)

    use_ortools = ORTOOLS_AVAILABLE if backend == "auto" else (backend == "ortools")
    if use_ortools and not ORTOOLS_AVAILABLE:
        raise RuntimeError("OR-Tools backend requested but ortools is not installed")
    if use_ortools:
        return _solve_flow_ortools(instance)
    return _solve_flow_pure(instance)


def verify_label(
    instance: ProblemInstance, rel_tol: float = 1e-4, abs_tol: float = 1e-4
) -> Tuple[bool, Optional[float], Optional[float], str]:
    """Compare a labelled optimum against the oracle's own computed optimum.

    Returns ``(agrees, labeled_value, solver_value, status)``. ``agrees`` is
    ``False`` when there is no label, the instance is infeasible, or the values
    disagree beyond tolerance — the IndustryOR mislabelling guard.
    """
    labeled = instance.known_optimum.objective_value if instance.known_optimum else None
    result = solve_optimum(instance)
    solver_value = result.objective_value
    if result.status != STATUS_OPTIMAL or labeled is None or solver_value is None:
        return False, labeled, solver_value, result.status
    agrees = math.isclose(labeled, solver_value, rel_tol=rel_tol, abs_tol=abs_tol)
    return agrees, labeled, solver_value, result.status


# ===========================================================================
# OR-Tools backend (LP / MILP)
# ===========================================================================
def _solve_flow_ortools(instance: ProblemInstance) -> OracleResult:
    needs_mip = any(n.is_optional for n in instance.nodes) or any(
        a.fixed_cost is not None for a in instance.arcs
    )
    solver = pywraplp.Solver.CreateSolver("CBC")
    if solver is None:  # pragma: no cover
        solver = pywraplp.Solver.CreateSolver("SCIP")
    if solver is None:  # pragma: no cover
        return _solve_flow_pure(instance)

    inf = solver.infinity()
    total_demand = instance.total_demand()
    big_m = total_demand + 1.0

    flow_vars = {}
    for arc in instance.arcs:
        ub = inf if arc.capacity is None else arc.capacity
        flow_vars[arc.arc_id] = solver.NumVar(0.0, ub, f"f_{arc.arc_id}")

    open_vars = {}
    for node in instance.nodes:
        if node.is_optional:
            open_vars[node.node_id] = solver.IntVar(0, 1, f"open_{node.node_id}")

    used_vars = {}
    for arc in instance.arcs:
        if arc.fixed_cost is not None:
            used_vars[arc.arc_id] = solver.IntVar(0, 1, f"used_{arc.arc_id}")

    inflow = {n.node_id: [] for n in instance.nodes}
    outflow = {n.node_id: [] for n in instance.nodes}
    for arc in instance.arcs:
        outflow[arc.from_node].append(flow_vars[arc.arc_id])
        inflow[arc.to_node].append(flow_vars[arc.arc_id])

    def _sum(terms):
        return solver.Sum(terms) if terms else 0.0

    for node in instance.nodes:
        net_in = _sum(inflow[node.node_id]) - _sum(outflow[node.node_id])
        if node.node_type == NodeType.SOURCE:
            if node.supply is not None:
                solver.Add(-net_in <= node.supply)  # net_out <= supply
        elif node.node_type == NodeType.SINK:
            if node.demand:
                solver.Add(net_in == node.demand)
        else:  # TRANSSHIPMENT
            solver.Add(net_in == 0.0)

        if node.capacity is not None:
            solver.Add(_sum(inflow[node.node_id]) <= node.capacity)
            solver.Add(_sum(outflow[node.node_id]) <= node.capacity)

    # Facility open gating: no flow may touch a closed optional node.
    for arc in instance.arcs:
        for nid in (arc.from_node, arc.to_node):
            if nid in open_vars:
                solver.Add(flow_vars[arc.arc_id] <= big_m * open_vars[nid])
        if arc.arc_id in used_vars:
            solver.Add(flow_vars[arc.arc_id] <= big_m * used_vars[arc.arc_id])

    nonoptional_constant = sum(
        n.fixed_cost for n in instance.nodes if n.fixed_cost is not None and not n.is_optional
    )

    objective_terms = [arc.unit_cost * flow_vars[arc.arc_id] for arc in instance.arcs]
    objective_terms += [
        arc.fixed_cost * used_vars[arc.arc_id] for arc in instance.arcs if arc.arc_id in used_vars
    ]
    objective_terms += [
        node.fixed_cost * open_vars[node.node_id]
        for node in instance.nodes
        if node.is_optional and node.fixed_cost is not None
    ]
    solver.Minimize(_sum(objective_terms))

    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return OracleResult(STATUS_INFEASIBLE, None, None, "ortools")

    flows = [
        FlowAssignment(arc_id=arc.arc_id, quantity=round(flow_vars[arc.arc_id].solution_value(), 9))
        for arc in instance.arcs
        if flow_vars[arc.arc_id].solution_value() > TOL
    ]
    open_facilities = sorted(
        nid for nid, var in open_vars.items() if var.solution_value() > 0.5
    )
    objective_value = solver.Objective().Value() + nonoptional_constant
    solution = Solution(
        solution_id=f"{instance.instance_id}-oracle",
        instance_id=instance.instance_id,
        flows=flows,
        open_facilities=open_facilities or None,
        produced_by="oracle:ortools",
    )
    return OracleResult(STATUS_OPTIMAL, objective_value, solution, "ortools")


# ===========================================================================
# Pure-Python backend (exact, dependency-free)
# ===========================================================================
def _solve_flow_pure(instance: ProblemInstance) -> OracleResult:
    optional_nodes = [n.node_id for n in instance.nodes if n.is_optional]
    fixed_arcs = [a.arc_id for a in instance.arcs if a.fixed_cost is not None]
    nonoptional_constant = sum(
        n.fixed_cost for n in instance.nodes if n.fixed_cost is not None and not n.is_optional
    )

    # Simple case: a pure min-cost flow, no open/close or fixed-charge decisions.
    if not optional_nodes and not fixed_arcs:
        status, cost, flow_by_arc = flowmod.min_cost_flow_meet_demand(instance)
        if status != flowmod.STATUS_OPTIMAL:
            return OracleResult(STATUS_INFEASIBLE, None, None, "pure")
        sol = _flow_solution(instance, flow_by_arc, [])
        return OracleResult(STATUS_OPTIMAL, cost + nonoptional_constant, sol, "pure")

    n_bits = len(optional_nodes) + len(fixed_arcs)
    if n_bits > _MAX_ENUM_BITS:
        return OracleResult(STATUS_UNSUPPORTED, None, None, "pure")

    arc_fixed = {a.arc_id: a.fixed_cost for a in instance.arcs if a.fixed_cost is not None}
    node_fixed = {n.node_id: n.fixed_cost for n in instance.nodes if n.is_optional and n.fixed_cost is not None}
    all_arc_ids = {a.arc_id for a in instance.arcs}

    best_total = math.inf
    best_flows: Dict[str, float] = {}
    best_open: List[str] = []

    for node_bits in itertools.product([0, 1], repeat=len(optional_nodes)):
        open_nodes = {optional_nodes[i] for i, b in enumerate(node_bits) if b}
        closed_nodes = frozenset(set(optional_nodes) - open_nodes)
        for arc_bits in itertools.product([0, 1], repeat=len(fixed_arcs)):
            on_fixed = {fixed_arcs[i] for i, b in enumerate(arc_bits) if b}
            # allowed arcs = all non-fixed-charge arcs + the "on" fixed-charge arcs
            allowed = frozenset((all_arc_ids - set(fixed_arcs)) | on_fixed)
            status, cost, flow_by_arc = flowmod.min_cost_flow_meet_demand(
                instance, closed_nodes=closed_nodes, allowed_arcs=allowed
            )
            if status != flowmod.STATUS_OPTIMAL:
                continue
            total = cost
            total += sum(node_fixed[nid] for nid in open_nodes if nid in node_fixed)
            total += sum(
                arc_fixed[aid] for aid in on_fixed if flow_by_arc.get(aid, 0.0) > TOL
            )
            if total < best_total - TOL:
                best_total = total
                best_flows = flow_by_arc
                best_open = sorted(nid for nid in open_nodes if flow_by_arc_touches(instance, flow_by_arc, nid))

    if best_total is math.inf or best_total == math.inf:
        return OracleResult(STATUS_INFEASIBLE, None, None, "pure")

    sol = _flow_solution(instance, best_flows, best_open)
    return OracleResult(STATUS_OPTIMAL, best_total + nonoptional_constant, sol, "pure")


def flow_by_arc_touches(instance: ProblemInstance, flow_by_arc: Dict[str, float], node_id: str) -> bool:
    arc_index = instance.arc_index
    for aid, qty in flow_by_arc.items():
        if qty <= TOL:
            continue
        arc = arc_index[aid]
        if arc.from_node == node_id or arc.to_node == node_id:
            return True
    return False


def _flow_solution(
    instance: ProblemInstance, flow_by_arc: Dict[str, float], open_facilities: List[str]
) -> Solution:
    flows = [
        FlowAssignment(arc_id=aid, quantity=round(qty, 9))
        for aid, qty in sorted(flow_by_arc.items())
        if qty > TOL
    ]
    return Solution(
        solution_id=f"{instance.instance_id}-oracle",
        instance_id=instance.instance_id,
        flows=flows,
        open_facilities=open_facilities or None,
        produced_by="oracle:pure",
    )


# ===========================================================================
# Exact brute-force CVRP
# ===========================================================================
def _vehicle_capacity(instance: ProblemInstance) -> Optional[float]:
    for c in instance.additional_constraints:
        if c.constraint_type == ConstraintType.CAPACITY and "vehicle_capacity" in c.parameters:
            return float(c.parameters["vehicle_capacity"])
    for n in instance.sources():
        if n.capacity is not None:
            return n.capacity
    return None


def _set_partitions(items: List[str]):
    """Yield every partition of ``items`` into non-empty blocks."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in _set_partitions(rest):
        for i in range(len(partition)):
            yield partition[:i] + [[first] + partition[i]] + partition[i + 1 :]
        yield [[first]] + partition


def _best_route_distance(depot_xy, block, coords) -> float:
    best = math.inf
    for perm in itertools.permutations(block):
        d = flowmod_dist(depot_xy, coords[perm[0]])
        for i in range(len(perm) - 1):
            d += flowmod_dist(coords[perm[i]], coords[perm[i + 1]])
        d += flowmod_dist(coords[perm[-1]], depot_xy)
        if d < best:
            best = d
            best_perm = perm
    return best, list(best_perm)


def flowmod_dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _solve_vrp_bruteforce(instance: ProblemInstance) -> OracleResult:
    depots = instance.sources()
    if not depots:
        return OracleResult(STATUS_UNSUPPORTED, None, None, "vrp-bruteforce")
    depot = depots[0]
    if depot.coordinates is None:
        return OracleResult(STATUS_UNSUPPORTED, None, None, "vrp-bruteforce")

    customers = instance.sinks()
    if len(customers) > _MAX_VRP_CUSTOMERS:
        return OracleResult(STATUS_UNSUPPORTED, None, None, "vrp-bruteforce")
    for c in customers:
        if c.coordinates is None:
            return OracleResult(STATUS_UNSUPPORTED, None, None, "vrp-bruteforce")

    capacity = _vehicle_capacity(instance)
    coords = {n.node_id: n.coordinates for n in instance.nodes}
    demand = {c.node_id: (c.demand or 0.0) for c in customers}
    cust_ids = [c.node_id for c in customers]

    best_total = math.inf
    best_routes: List[List[str]] = []
    for partition in _set_partitions(cust_ids):
        feasible = True
        total = 0.0
        routes_here = []
        for block in partition:
            if capacity is not None and sum(demand[c] for c in block) > capacity + TOL:
                feasible = False
                break
            dist, order = _best_route_distance(depot.coordinates, block, coords)
            total += dist
            routes_here.append(order)
        if not feasible:
            continue
        if total < best_total - TOL:
            best_total = total
            best_routes = routes_here

    if best_total is math.inf or best_total == math.inf:
        return OracleResult(STATUS_INFEASIBLE, None, None, "vrp-bruteforce")

    routes = [
        Route(
            vehicle_id=f"v{i+1}",
            node_sequence=[depot.node_id] + order + [depot.node_id],
            load=sum(demand[c] for c in order),
        )
        for i, order in enumerate(best_routes)
    ]
    sol = Solution(
        solution_id=f"{instance.instance_id}-oracle",
        instance_id=instance.instance_id,
        routes=routes,
        produced_by="oracle:vrp-bruteforce",
    )
    return OracleResult(STATUS_OPTIMAL, best_total, sol, "vrp-bruteforce")
