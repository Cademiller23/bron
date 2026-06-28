"""Rearrangement operators — reshape only, never add/remove an agent.

Each operator is a pure function ``op(genome, rng, registry) ->
Optional[CandidateRearrangement]``. The **agent set is invariant** — every
operator preserves exactly the same agents; only the wiring / model assignment
changes. Growing the team is strictly B6's job.

A candidate carries an in-memory mutated copy of the genome (for evaluation) and
a B3 ``derive_fn`` that would commit it (for adoption). The commit derives are
**relative** (recompute on the freshly-loaded genome), so an adopted
rearrangement is conflict-safe under B3's optimistic lock.
"""

import json
import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from pydantic import ValidationError

from darwin.team import mutations as M
from darwin.team.genome import (
    Edge,
    EdgeType,
    MutationActor,
    MutationRecord,
    MutationType,
    TeamGenome,
)
from darwin.team.validation import validate


@dataclass(frozen=True)
class CandidateRearrangement:
    genome: TeamGenome  # the in-memory mutated copy (for evaluation)
    derive_fn: Callable  # the B3 derive for committing the adopted rearrangement
    mutation_type: MutationType
    description: str
    signature: str  # for de-duplication


def signature(genome: TeamGenome) -> str:
    """A stable structural fingerprint (wiring + arbiter + model assignment).
    The agent set is invariant, so this captures everything B5 can change."""
    edges = sorted((e.from_agent_id, e.to_agent_id, e.edge_type.value) for e in genome.edges)
    models = sorted((a.agent_id, a.spec.model_id) for a in genome.agents)
    return json.dumps({"edges": edges, "arbiter": genome.arbiter_id, "models": models}, sort_keys=True)


def _candidate_from_derive(
    genome: TeamGenome, derive_fn: Callable, mutation_type: MutationType, description: str, registry: Any
) -> Optional[CandidateRearrangement]:
    """Apply a derive to build + validate the in-memory candidate. Returns None
    if the rearrangement is invalid (the generator resamples)."""
    try:
        set_ops, _record = derive_fn(genome)
        candidate = TeamGenome.model_validate({**genome.model_dump(), **set_ops}, context={"registry": registry})
    except (ValidationError, ValueError, KeyError):
        return None
    if not validate(candidate, registry).valid:
        return None
    # the agent set MUST be invariant
    if {a.agent_id for a in candidate.agents} != {a.agent_id for a in genome.agents}:
        return None
    return CandidateRearrangement(
        genome=candidate, derive_fn=derive_fn, mutation_type=mutation_type,
        description=description, signature=signature(candidate),
    )


# ---------------------------------------------------------------------------
# reassign_model — change one agent's model (the B5-level model search)
# ---------------------------------------------------------------------------
def reassign_model(genome: TeamGenome, rng: random.Random, registry: Any) -> Optional[CandidateRearrangement]:
    others = [m for m in registry.all_ids()]
    node = rng.choice(genome.agents)
    alt = [m for m in others if m != node.spec.model_id]
    if not alt:
        return None
    new_model = rng.choice(alt)
    derive = M.swap_model(node.agent_id, new_model, actor=MutationActor.REARRANGER,
                          description=f"swap {node.agent_id} model -> {new_model}", registry=registry)
    return _candidate_from_derive(genome, derive, MutationType.SWAP_MODEL,
                                  f"reassigned {node.agent_id} to {new_model}", registry)


# ---------------------------------------------------------------------------
# redirect_edge — change WHO an agent reports to (the edge's target)
# ---------------------------------------------------------------------------
def redirect_edge(genome: TeamGenome, rng: random.Random, registry: Any) -> Optional[CandidateRearrangement]:
    # FEEDS_ARBITER edges must terminate at the arbiter, so only redirect others.
    movable = [e for e in genome.edges if e.edge_type != EdgeType.FEEDS_ARBITER]
    if not movable:
        return None
    old = rng.choice(movable)
    targets = [a.agent_id for a in genome.agents if a.agent_id not in (old.from_agent_id, old.to_agent_id)]
    if not targets:
        return None
    new_to = rng.choice(targets)
    new_edge = Edge(from_agent_id=old.from_agent_id, to_agent_id=new_to, edge_type=old.edge_type)
    derive = M.rearrange_edge(old, new_edge, actor=MutationActor.REARRANGER,
                             description=f"redirect {old.from_agent_id} -> {new_to}", registry=registry)
    return _candidate_from_derive(genome, derive, MutationType.REARRANGE_EDGE,
                                  f"redirected {old.from_agent_id}->{old.to_agent_id} to ->{new_to}", registry)


# ---------------------------------------------------------------------------
# reorder_pipeline — change WHO feeds a given consumer (the edge's source)
# ---------------------------------------------------------------------------
def reorder_pipeline(genome: TeamGenome, rng: random.Random, registry: Any) -> Optional[CandidateRearrangement]:
    if not genome.edges:
        return None
    old = rng.choice(genome.edges)
    # the arbiter is terminal, so it can never be an edge source
    sources = [
        a.agent_id for a in genome.agents
        if a.agent_id not in (old.from_agent_id, old.to_agent_id) and a.agent_id != genome.arbiter_id
    ]
    if not sources:
        return None
    new_from = rng.choice(sources)
    new_edge = Edge(from_agent_id=new_from, to_agent_id=old.to_agent_id, edge_type=old.edge_type)
    derive = M.rearrange_edge(old, new_edge, actor=MutationActor.REARRANGER,
                             description=f"reorder: {old.to_agent_id} now fed by {new_from}", registry=registry)
    return _candidate_from_derive(genome, derive, MutationType.REARRANGE_EDGE,
                                  f"reordered: {old.to_agent_id} now fed by {new_from}", registry)


# ---------------------------------------------------------------------------
# swap_arbiter — designate a different existing agent as the arbitrator
# ---------------------------------------------------------------------------
def swap_arbiter(genome: TeamGenome, rng: random.Random, registry: Any) -> Optional[CandidateRearrangement]:
    candidates = [a.agent_id for a in genome.agents if a.agent_id != genome.arbiter_id]
    if not candidates:
        return None
    new_arb = rng.choice(candidates)
    old_arb = genome.arbiter_id

    def derive(current: TeamGenome):
        # recompute relative to `current` (agent set is invariant under B5):
        # the new arbiter becomes terminal (drop its outgoing edges); the old
        # arbiter becomes a mid-pipeline aggregator (its FEEDS_ARBITER feeders are
        # retyped to PASSES_PROPOSAL since they no longer feed THE arbiter) and
        # feeds the new arbiter, so reachability and the contract rules hold.
        new_edges: List[Edge] = []
        for e in current.edges:
            if e.from_agent_id == new_arb:
                continue  # new arbiter must be terminal -> drop its outgoing edges
            if e.edge_type == EdgeType.FEEDS_ARBITER and e.to_agent_id == old_arb:
                new_edges.append(Edge(from_agent_id=e.from_agent_id, to_agent_id=old_arb, edge_type=EdgeType.PASSES_PROPOSAL))
            else:
                new_edges.append(e)
        if not any(e.from_agent_id == old_arb and e.to_agent_id == new_arb for e in new_edges):
            new_edges.append(Edge(from_agent_id=old_arb, to_agent_id=new_arb, edge_type=EdgeType.FEEDS_ARBITER))
        candidate = TeamGenome.model_validate(
            {**current.model_dump(), "edges": [e.model_dump() for e in new_edges], "arbiter_id": new_arb},
            context={"registry": registry},
        )
        record = MutationRecord(
            mutation_type=MutationType.RETARGET_ARBITER, actor=MutationActor.REARRANGER,
            description=f"swap arbiter {old_arb} -> {new_arb}", from_version=current.version,
            to_version=current.version + 1, fitness_before=current.current_fitness,
        )
        return {"edges": candidate.model_dump(mode="json")["edges"], "arbiter_id": new_arb}, record

    return _candidate_from_derive(genome, derive, MutationType.RETARGET_ARBITER,
                                  f"swapped arbiter {old_arb} -> {new_arb}", registry)


ALL_OPERATORS: List[Callable] = [reassign_model, redirect_edge, reorder_pipeline, swap_arbiter]
