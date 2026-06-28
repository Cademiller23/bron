"""Typed mutation operations — the genome-edit vocabulary B5/B6 call.

Each operation returns a ``derive_fn`` ``(current_genome) -> (set_ops,
MutationRecord)`` suitable for ``store.retry_mutate``. The derive_fn rebuilds a
*validated* candidate genome (so referential integrity is enforced and an
illegal mutation — e.g. a bad model_id or a dangling edge — is rejected before it
is written), then emits the minimal ``$set`` ops plus the lineage record.

These are the atomic, conflict-safe edits behind dynamic curation: adding a
corpus/curated agent, rearranging edges, swapping a model, retargeting the
arbiter, removing an agent.
"""

from typing import Any, Dict, List, Optional, Tuple

from darwin.agent.spec import AgentSpec
from darwin.team.genome import (
    AgentNode,
    Edge,
    MutationActor,
    MutationRecord,
    MutationType,
    TeamGenome,
)

DeriveResult = Tuple[Dict[str, Any], MutationRecord]


def _rebuild(
    g: TeamGenome,
    *,
    agents: Optional[List[AgentNode]] = None,
    edges: Optional[List[Edge]] = None,
    arbiter_id: Optional[str] = None,
    registry: Any = None,
) -> TeamGenome:
    """Re-validate a candidate genome with the given overrides (runs all the
    referential-integrity validators). ``registry`` (if given) is threaded as
    validation context so embedded specs re-validate against the intended fleet."""
    data = g.model_dump()
    if agents is not None:
        data["agents"] = [a.model_dump() for a in agents]
    if edges is not None:
        data["edges"] = [e.model_dump() for e in edges]
    if arbiter_id is not None:
        data["arbiter_id"] = arbiter_id
    return TeamGenome.model_validate(data, context={"registry": registry})


def _record(g: TeamGenome, mtype: MutationType, actor: MutationActor, description: str) -> MutationRecord:
    return MutationRecord(
        mutation_type=mtype,
        actor=actor,
        description=description,
        from_version=g.version,
        to_version=g.version + 1,
        fitness_before=g.current_fitness,
    )


def _agents_json(candidate: TeamGenome) -> List[Dict[str, Any]]:
    return candidate.model_dump(mode="json")["agents"]


def _edges_json(candidate: TeamGenome) -> List[Dict[str, Any]]:
    return candidate.model_dump(mode="json")["edges"]


# ---------------------------------------------------------------------------
# Edge operations
# ---------------------------------------------------------------------------
def add_edge(edge: Edge, *, actor: MutationActor = MutationActor.REARRANGER, description: str = "", registry: Any = None):
    def derive(g: TeamGenome) -> DeriveResult:
        candidate = _rebuild(g, edges=list(g.edges) + [edge], registry=registry)
        rec = _record(g, MutationType.REARRANGE_EDGE, actor,
                      description or f"add {edge.edge_type.value} edge {edge.from_agent_id}->{edge.to_agent_id}")
        return {"edges": _edges_json(candidate)}, rec

    return derive


def rearrange_edge(
    old_edge: Edge, new_edge: Edge, *, actor: MutationActor = MutationActor.REARRANGER, description: str = "", registry: Any = None
):
    def derive(g: TeamGenome) -> DeriveResult:
        remaining = [e for e in g.edges if e != old_edge]
        candidate = _rebuild(g, edges=remaining + [new_edge], registry=registry)
        rec = _record(g, MutationType.REARRANGE_EDGE, actor,
                      description or f"reroute {old_edge.from_agent_id}->{old_edge.to_agent_id} "
                                    f"to {new_edge.from_agent_id}->{new_edge.to_agent_id}")
        return {"edges": _edges_json(candidate)}, rec

    return derive


# ---------------------------------------------------------------------------
# Model swap
# ---------------------------------------------------------------------------
def swap_model(agent_id: str, new_model_id: str, *, actor: MutationActor = MutationActor.REARRANGER, description: str = "", registry: Any = None):
    def derive(g: TeamGenome) -> DeriveResult:
        new_agents: List[AgentNode] = []
        found = False
        for node in g.agents:
            if node.agent_id == agent_id:
                found = True
                # rebuild the spec WITH validation so a bad model_id is rejected
                new_spec = AgentSpec.model_validate(
                    {**node.spec.model_dump(), "model_id": new_model_id}, context={"registry": registry}
                )
                new_agents.append(node.model_copy(update={"spec": new_spec}))
            else:
                new_agents.append(node)
        if not found:
            raise KeyError(f"agent {agent_id!r} not in genome")
        candidate = _rebuild(g, agents=new_agents, registry=registry)
        rec = _record(g, MutationType.SWAP_MODEL, actor,
                      description or f"swap {agent_id} model -> {new_model_id}")
        return {"agents": _agents_json(candidate)}, rec

    return derive


# ---------------------------------------------------------------------------
# Arbiter retarget
# ---------------------------------------------------------------------------
def retarget_arbiter(new_arbiter_id: str, *, actor: MutationActor = MutationActor.REARRANGER, description: str = "", registry: Any = None):
    def derive(g: TeamGenome) -> DeriveResult:
        candidate = _rebuild(g, arbiter_id=new_arbiter_id, registry=registry)
        rec = _record(g, MutationType.RETARGET_ARBITER, actor,
                      description or f"retarget arbiter -> {new_arbiter_id}")
        return {"arbiter_id": new_arbiter_id}, rec

    return derive


# ---------------------------------------------------------------------------
# Add / remove agent (the dynamic-curation substrate)
# ---------------------------------------------------------------------------
def add_agent(
    node: AgentNode,
    new_edges: List[Edge],
    *,
    mutation_type: MutationType = MutationType.ADD_CURATED_AGENT,
    actor: MutationActor = MutationActor.ESCALATION,
    description: str = "",
    registry: Any = None,
):
    if mutation_type not in (MutationType.ADD_CURATED_AGENT, MutationType.ADD_AGENT_FROM_CORPUS):
        raise ValueError("add_agent mutation_type must be ADD_CURATED_AGENT or ADD_AGENT_FROM_CORPUS")

    def derive(g: TeamGenome) -> DeriveResult:
        candidate = _rebuild(g, agents=list(g.agents) + [node], edges=list(g.edges) + list(new_edges), registry=registry)
        rec = _record(g, mutation_type, actor,
                      description or f"add agent {node.agent_id} ({node.spec.role_name})")
        return {"agents": _agents_json(candidate), "edges": _edges_json(candidate)}, rec

    return derive


def remove_agent(agent_id: str, *, actor: MutationActor = MutationActor.REARRANGER, description: str = "", registry: Any = None):
    def derive(g: TeamGenome) -> DeriveResult:
        if agent_id == g.arbiter_id:
            raise ValueError("cannot remove the arbiter; retarget it first")
        new_agents = [a for a in g.agents if a.agent_id != agent_id]
        if len(new_agents) == len(g.agents):
            raise KeyError(f"agent {agent_id!r} not in genome")
        new_edges = [e for e in g.edges if agent_id not in (e.from_agent_id, e.to_agent_id)]
        candidate = _rebuild(g, agents=new_agents, edges=new_edges, registry=registry)
        rec = _record(g, MutationType.REMOVE_AGENT, actor, description or f"remove agent {agent_id}")
        return {"agents": _agents_json(candidate), "edges": _edges_json(candidate)}, rec

    return derive
