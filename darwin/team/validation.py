"""Genome structural validation — reject malformed teams before they run.

A malformed team would crash the runner or run nonsensically, so the Architect
(B4) / escalation (B6) validates a genome before it executes. ``validate`` records
*every* problem so the caller can re-curate.
"""

from collections import deque
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

from darwin.agent.registry import ModelRegistry, default_registry
from darwin.agent.spec import InputKind
from darwin.team.genome import EdgeType, TeamGenome


class ValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    valid: bool
    errors: List[str] = []

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ValueError("invalid genome: " + "; ".join(self.errors))


def _has_cycle(genome: TeamGenome) -> bool:
    """Kahn's algorithm — True if the edge graph contains a cycle."""
    ids = [a.agent_id for a in genome.agents]
    indeg = {i: 0 for i in ids}
    adj = {i: [] for i in ids}
    for e in genome.edges:
        if e.from_agent_id == e.to_agent_id:
            continue  # self-edges are flagged separately; ignore here
        adj[e.from_agent_id].append(e.to_agent_id)
        indeg[e.to_agent_id] += 1
    queue = deque(sorted(i for i in ids if indeg[i] == 0))
    seen = 0
    while queue:
        u = queue.popleft()
        seen += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    return seen != len(ids)


def _reaches_arbiter(genome: TeamGenome) -> set:
    """Set of agent_ids with a directed path to the arbiter (arbiter included)."""
    rev = {a.agent_id: [] for a in genome.agents}
    for e in genome.edges:
        rev[e.to_agent_id].append(e.from_agent_id)
    seen = {genome.arbiter_id}
    queue = deque([genome.arbiter_id])
    while queue:
        node = queue.popleft()
        for pred in rev[node]:
            if pred not in seen:
                seen.add(pred)
                queue.append(pred)
    return seen


def validate(genome: TeamGenome, registry: Optional[ModelRegistry] = None) -> ValidationResult:
    registry = registry or default_registry()
    errors: List[str] = []
    id_set = {a.agent_id for a in genome.agents}

    if not genome.agents:
        errors.append("genome has no agents")

    # arbiter exists (constructor enforces, but defend) and is terminal
    if genome.arbiter_id not in id_set:
        errors.append(f"arbiter_id {genome.arbiter_id!r} is not a node")
    elif genome.downstream_edges(genome.arbiter_id):
        outs = [e.to_agent_id for e in genome.downstream_edges(genome.arbiter_id)]
        errors.append(f"arbiter {genome.arbiter_id!r} has downstream edges to {outs} (must be terminal)")

    # edge legality
    seen_edges = set()
    for e in genome.edges:
        if e.from_agent_id == e.to_agent_id:
            errors.append(f"self-edge on {e.from_agent_id!r}")
        if e.edge_type == EdgeType.FEEDS_ARBITER and e.to_agent_id != genome.arbiter_id:
            errors.append(
                f"FEEDS_ARBITER edge {e.from_agent_id}->{e.to_agent_id} does not terminate at the arbiter"
            )
        key = (e.from_agent_id, e.to_agent_id, e.edge_type.value)
        if key in seen_edges:
            errors.append(f"duplicate edge {e.from_agent_id}->{e.to_agent_id} ({e.edge_type.value})")
        seen_edges.add(key)

    # acyclicity
    if id_set and _has_cycle(genome):
        errors.append("genome edge graph contains a cycle (must be a DAG)")

    # reachability — every agent must reach the arbiter (no orphans)
    if genome.arbiter_id in id_set and not _has_cycle(genome):
        reachable = _reaches_arbiter(genome)
        orphans = sorted(id_set - reachable)
        if orphans:
            errors.append(f"orphan agents (no path to the arbiter): {orphans}")

    # model legality + contract compatibility
    for node in genome.agents:
        if not registry.contains(node.spec.model_id):
            errors.append(f"agent {node.agent_id!r} uses model_id {node.spec.model_id!r} not in the registry")

        upstream = genome.upstream_edges(node.agent_id)
        ic = node.spec.input_contract
        if ic == InputKind.SIBLING_OUTPUTS and not upstream:
            errors.append(
                f"agent {node.agent_id!r} expects SIBLING_OUTPUTS but has no upstream edges feeding it"
            )
        if ic == InputKind.PROBLEM_PLUS_DRAFT and not any(
            e.edge_type == EdgeType.PASSES_PROPOSAL for e in upstream
        ):
            errors.append(
                f"agent {node.agent_id!r} expects PROBLEM_PLUS_DRAFT but no upstream PASSES_PROPOSAL edge feeds it"
            )

    return ValidationResult(valid=not errors, errors=errors)
