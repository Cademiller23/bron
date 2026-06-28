"""Exact pure-Python network-flow primitives.

Two deterministic, dependency-free algorithms used elsewhere in B1:

* :class:`MaxFlow` — Dinic's algorithm, used by the resilience metric to ask
  *"if our biggest supplier goes down, how much demand can still be met?"*
* :class:`MinCostFlow` — successive-shortest-path (SPFA / Bellman-Ford
  potentials), used by the oracle's pure-Python fallback to compute the exact
  minimum-cost flow that meets all demand.

Both operate on floating-point capacities/costs (no integer scaling) and are
fully deterministic: edges are processed in insertion order and ties are broken
deterministically, so the same graph always yields the same result.

These also power a **node-splitting** construction (every node becomes an
``in``→``out`` pair joined by a throughput-capacity edge) so node-capacity
limits are honoured exactly.
"""

from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

from darwin.problem.schemas import NodeType, ProblemInstance

INF: float = 1e18
_EPS: float = 1e-9

# Solver status strings returned by the min-cost-flow helper.
STATUS_OPTIMAL = "OPTIMAL"
STATUS_INFEASIBLE = "INFEASIBLE"


class MaxFlow:
    """Dinic max-flow on float capacities."""

    def __init__(self, n: int) -> None:
        self.n = n
        # graph[u] = list of [to, residual_cap, index_of_reverse_edge]
        self.graph: List[List[list]] = [[] for _ in range(n)]
        self._level: List[int] = []
        self._it: List[int] = []

    def add_edge(self, u: int, v: int, cap: float) -> None:
        self.graph[u].append([v, cap, len(self.graph[v])])
        self.graph[v].append([u, 0.0, len(self.graph[u]) - 1])

    def _bfs(self, s: int, t: int) -> bool:
        level = [-1] * self.n
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for to, cap, _ in self.graph[u]:
                if cap > _EPS and level[to] < 0:
                    level[to] = level[u] + 1
                    q.append(to)
        self._level = level
        return level[t] >= 0

    def _augment(self, s: int, t: int) -> float:
        """Find one augmenting path in the current level graph and push along it.

        Iterative (explicit-stack) blocking-flow DFS — recursion depth no longer
        scales with path length, so deep networks can't overflow the stack.
        """
        stack = [s]
        while stack:
            node = stack[-1]
            if node == t:
                break
            advanced = False
            g = self.graph[node]
            while self._it[node] < len(g):
                to, cap, _rev = g[self._it[node]]
                if cap > _EPS and self._level[to] == self._level[node] + 1:
                    stack.append(to)
                    advanced = True
                    break
                self._it[node] += 1
            if not advanced:
                # dead end for this phase: prune the node and back up
                self._level[node] = -1
                stack.pop()
        if not stack:  # never reached the sink
            return 0.0
        # bottleneck along the discovered path
        bottleneck = INF
        for i in range(len(stack) - 1):
            u = stack[i]
            bottleneck = min(bottleneck, self.graph[u][self._it[u]][1])
        # augment
        for i in range(len(stack) - 1):
            u = stack[i]
            edge = self.graph[u][self._it[u]]
            edge[1] -= bottleneck
            self.graph[edge[0]][edge[2]][1] += bottleneck
        return bottleneck

    def max_flow(self, s: int, t: int) -> float:
        flow = 0.0
        while self._bfs(s, t):
            self._it = [0] * self.n
            while True:
                f = self._augment(s, t)
                if f <= _EPS:
                    break
                flow += f
        return flow


class MinCostFlow:
    """Successive-shortest-path min-cost flow on float capacities/costs.

    Costs are assumed non-negative (true for all B1 instances), so SPFA finds
    shortest augmenting paths without negative-cycle concerns.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        # graph[u] = list of [to, residual_cap, cost, index_of_reverse_edge]
        self.graph: List[List[list]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: float, cost: float) -> Tuple[int, int, float]:
        """Add a directed edge; return a handle ``(u, pos, init_cap)`` so the
        caller can later recover the flow pushed along it."""
        pos = len(self.graph[u])
        self.graph[u].append([v, cap, cost, len(self.graph[v])])
        self.graph[v].append([u, 0.0, -cost, pos])
        return (u, pos, cap)

    def flow(self, s: int, t: int, max_flow: float = INF) -> Tuple[float, float]:
        """Push up to ``max_flow`` units s→t at minimum cost.

        Returns ``(units_sent, total_cost)``.
        """
        sent = 0.0
        cost = 0.0
        while sent + _EPS < max_flow:
            dist = [INF] * self.n
            in_queue = [False] * self.n
            prev_v = [-1] * self.n
            prev_e = [-1] * self.n
            dist[s] = 0.0
            q = deque([s])
            in_queue[s] = True
            while q:
                u = q.popleft()
                in_queue[u] = False
                du = dist[u]
                for i, edge in enumerate(self.graph[u]):
                    to, cap, ecost, _ = edge
                    if cap > _EPS and du + ecost < dist[to] - 1e-12:
                        dist[to] = du + ecost
                        prev_v[to] = u
                        prev_e[to] = i
                        if not in_queue[to]:
                            q.append(to)
                            in_queue[to] = True
            if dist[t] >= INF:
                break  # no more augmenting paths
            # bottleneck along the path
            d = max_flow - sent
            v = t
            while v != s:
                d = min(d, self.graph[prev_v[v]][prev_e[v]][1])
                v = prev_v[v]
            v = t
            while v != s:
                edge = self.graph[prev_v[v]][prev_e[v]]
                edge[1] -= d
                self.graph[v][edge[3]][1] += d
                v = prev_v[v]
            sent += d
            cost += d * dist[t]
        return sent, cost

    def edge_flow(self, handle: Tuple[int, int, float]) -> float:
        u, pos, init_cap = handle
        return init_cap - self.graph[u][pos][1]


# ---------------------------------------------------------------------------
# Node-split index helpers shared by both constructions.
# ---------------------------------------------------------------------------
def _node_split_index(instance: ProblemInstance) -> Tuple[Dict[str, int], Dict[str, int], int]:
    """Assign every node an ``in`` and ``out`` vertex; return ``(in_idx, out_idx,
    n_vertices)`` with two extra vertices reserved for the super-source/sink."""
    in_idx: Dict[str, int] = {}
    out_idx: Dict[str, int] = {}
    counter = 0
    for node in instance.nodes:
        in_idx[node.node_id] = counter
        out_idx[node.node_id] = counter + 1
        counter += 2
    # +2 for super source (counter) and super sink (counter+1)
    return in_idx, out_idx, counter + 2


def _big_bound(instance: ProblemInstance) -> float:
    """A finite stand-in for "uncapacitated".

    Using a *finite* instance-sized bound (rather than ``1e18``) is essential:
    no useful arc/node ever carries more than the total flow in the network, and
    a giant ``1e18`` capacity destroys float precision when recovering per-arc
    flow (``1e18 - (1e18 - 20) == 0`` in float64). This bound is always
    non-binding while keeping every residual computation exact.
    """
    return instance.total_supply() + instance.total_demand() + 1.0


def _node_throughput_cap(instance: ProblemInstance, node_id: str, big: float) -> float:
    cap = instance.node_index[node_id].capacity
    return big if cap is None else cap


def max_deliverable_demand(
    instance: ProblemInstance, excluded_sources: FrozenSet[str] = frozenset()
) -> float:
    """Maximum total demand the network can satisfy, optionally with some
    sources removed (used for worst-case single-failure resilience).

    Builds a node-split flow network: super-source → each (non-excluded) source
    capped at its supply; each sink → super-sink capped at its demand; the
    max-flow value is the maximum deliverable demand.
    """
    big = _big_bound(instance)
    in_idx, out_idx, n = _node_split_index(instance)
    ss = n - 2
    tt = n - 1
    mf = MaxFlow(n)

    for node in instance.nodes:
        mf.add_edge(in_idx[node.node_id], out_idx[node.node_id], _node_throughput_cap(instance, node.node_id, big))
        if node.node_type == NodeType.SOURCE and node.node_id not in excluded_sources:
            supply = big if node.supply is None else node.supply
            mf.add_edge(ss, in_idx[node.node_id], supply)
        if node.node_type == NodeType.SINK and node.demand:
            mf.add_edge(out_idx[node.node_id], tt, node.demand)

    for arc in instance.arcs:
        cap = big if arc.capacity is None else arc.capacity
        mf.add_edge(out_idx[arc.from_node], in_idx[arc.to_node], cap)

    return mf.max_flow(ss, tt)


def min_cost_flow_meet_demand(
    instance: ProblemInstance,
    closed_nodes: FrozenSet[str] = frozenset(),
    allowed_arcs: Optional[FrozenSet[str]] = None,
) -> Tuple[str, float, Dict[str, float]]:
    """Exact minimum-cost flow that satisfies *all* demand.

    ``closed_nodes`` are removed entirely (no flow may pass through them — used
    by the facility-location fallback). ``allowed_arcs`` (if given) restricts
    which arcs may carry flow.

    Returns ``(status, transport_cost, flow_by_arc_id)``. ``status`` is
    ``OPTIMAL`` or ``INFEASIBLE`` (demand cannot be met).
    """
    total_demand = instance.total_demand()
    big = _big_bound(instance)
    in_idx, out_idx, n = _node_split_index(instance)
    ss = n - 2
    tt = n - 1
    mcf = MinCostFlow(n)

    for node in instance.nodes:
        if node.node_id in closed_nodes:
            # throughput capacity 0 => no flow can traverse this node
            mcf.add_edge(in_idx[node.node_id], out_idx[node.node_id], 0.0, 0.0)
            continue
        mcf.add_edge(in_idx[node.node_id], out_idx[node.node_id], _node_throughput_cap(instance, node.node_id, big), 0.0)
        if node.node_type == NodeType.SOURCE:
            supply = big if node.supply is None else node.supply
            mcf.add_edge(ss, in_idx[node.node_id], supply, 0.0)
        if node.node_type == NodeType.SINK and node.demand:
            mcf.add_edge(out_idx[node.node_id], tt, node.demand, 0.0)

    arc_handles: Dict[str, Tuple[int, int, float]] = {}
    for arc in instance.arcs:
        if allowed_arcs is not None and arc.arc_id not in allowed_arcs:
            continue
        if arc.from_node in closed_nodes or arc.to_node in closed_nodes:
            continue
        cap = big if arc.capacity is None else arc.capacity
        arc_handles[arc.arc_id] = mcf.add_edge(
            out_idx[arc.from_node], in_idx[arc.to_node], cap, arc.unit_cost
        )

    sent, cost = mcf.flow(ss, tt, max_flow=total_demand)
    if sent + 1e-6 < total_demand:
        return STATUS_INFEASIBLE, float("nan"), {}

    flow_by_arc = {aid: mcf.edge_flow(h) for aid, h in arc_handles.items()}
    return STATUS_OPTIMAL, cost, flow_by_arc
