"""The TeamGenome data model — the recipe card as a graph.

A genome is a small directed acyclic graph: nodes are agents (each carrying an
embedded, self-contained B2 ``AgentSpec``), edges are "passes work to". It is the
contract B4 (the Architect) authors, B5 (rearrangement) mutates, and B3 (the
runner) executes.

In memory the genome is a **frozen** Pydantic model (immutable, ``extra="forbid"``).
On disk it is **mutated in place** via atomic optimistic-locking updates (see
``store.py``); the in-memory object is rebuilt from the returned document after
each mutation, so in-memory immutability and on-disk mutability coexist cleanly.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from darwin.agent.spec import AgentSpec, OutputKind


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class EdgeType(str, Enum):
    PASSES_PROPOSAL = "PASSES_PROPOSAL"  # a proposer's candidate flows downstream
    CHECKS = "CHECKS"  # a checker audits another agent's output
    FEEDS_ARBITER = "FEEDS_ARBITER"  # output flows into the final arbitrator
    SENDS_FEEDBACK = "SENDS_FEEDBACK"  # a critique flows back to a proposer to revise


class GenomeStatus(str, Enum):
    DRAFT = "DRAFT"  # curated, not yet evaluated
    EVALUATED = "EVALUATED"
    CLEARED_THRESHOLD = "CLEARED_THRESHOLD"  # normalized_score >= 0.90
    ESCALATING = "ESCALATING"  # below threshold, in corpus/curation escalation
    SEALED = "SEALED"  # final answer accepted


class MutationType(str, Enum):
    INITIAL_CURATION = "INITIAL_CURATION"
    REARRANGE_EDGE = "REARRANGE_EDGE"
    SWAP_MODEL = "SWAP_MODEL"
    RETARGET_ARBITER = "RETARGET_ARBITER"
    ADD_AGENT_FROM_CORPUS = "ADD_AGENT_FROM_CORPUS"
    ADD_CURATED_AGENT = "ADD_CURATED_AGENT"
    REMOVE_AGENT = "REMOVE_AGENT"
    ARBITER_FALLBACK_USED = "ARBITER_FALLBACK_USED"
    EVAL_ERROR = "EVAL_ERROR"
    EVALUATED = "EVALUATED"  # plain outcome write (fitness/status update)


class MutationActor(str, Enum):
    ARCHITECT = "ARCHITECT"  # B4
    REARRANGER = "REARRANGER"  # B5
    ESCALATION = "ESCALATION"  # B6
    RUNNER = "RUNNER"  # B3 itself (fallback / error / outcome records)


class ArbiterTier(str, Enum):
    PRIMARY = "PRIMARY"  # first arbiter attempt succeeded
    RETRY = "RETRY"  # a retry of the arbiter succeeded (still Tier 1)
    BEST_PROPOSAL_FALLBACK = "BEST_PROPOSAL_FALLBACK"  # Tier 2
    INFEASIBLE_SENTINEL = "INFEASIBLE_SENTINEL"  # Tier 3


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Sub-types
# ---------------------------------------------------------------------------
class AgentNode(_Frozen):
    """One agent in the team. The B2 ``AgentSpec`` is *embedded*, not referenced,
    so the genome is fully self-contained and runnable with zero extra input."""

    agent_id: str = Field(min_length=1)
    spec: AgentSpec
    layout: Optional[Dict[str, float]] = None  # {x, y} for the on-screen org chart

    @model_validator(mode="after")
    def _id_matches_spec(self) -> "AgentNode":
        if self.spec.agent_id != self.agent_id:
            raise ValueError(
                f"AgentNode.agent_id {self.agent_id!r} != embedded spec.agent_id {self.spec.agent_id!r}"
            )
        return self


class Edge(_Frozen):
    """A directed connection: who passes/checks/feeds whom."""

    from_agent_id: str = Field(min_length=1)
    to_agent_id: str = Field(min_length=1)
    edge_type: EdgeType


class MutationRecord(_Frozen):
    """One lineage entry, appended to ``history`` on every edit."""

    mutation_id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_now_iso)
    mutation_type: MutationType
    actor: MutationActor
    description: str = ""
    from_version: int = Field(ge=0)
    to_version: int = Field(ge=0)
    fitness_before: Optional[float] = None
    fitness_after: Optional[float] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# TeamGenome — the top-level MongoDB document
# ---------------------------------------------------------------------------
class TeamGenome(_Frozen):
    genome_id: str = Field(default_factory=_new_id)  # stable; maps to Mongo "_id"
    version: int = Field(default=1, ge=1)  # optimistic-lock token; $inc on every mutation
    instance_id: str = Field(min_length=1)
    agents: List[AgentNode]
    edges: List[Edge] = Field(default_factory=list)
    arbiter_id: str = Field(min_length=1)
    generation: int = Field(default=0, ge=0)
    parent_genome_id: Optional[str] = None  # set only for forks (rare under in-place model)
    current_fitness: Optional[float] = None
    current_normalized_score: Optional[float] = None
    status: GenomeStatus = GenomeStatus.DRAFT
    history: List[MutationRecord] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now_iso)
    updated_at: str = Field(default_factory=_now_iso)

    # -- referential integrity at construction (graph validation lives in
    #    validation.py; this is the basic "the document is well-formed" check) --
    @model_validator(mode="after")
    def _check_references(self) -> "TeamGenome":
        ids = [a.agent_id for a in self.agents]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate agent_id(s): {dupes}")
        id_set = set(ids)
        if self.arbiter_id not in id_set:
            raise ValueError(f"arbiter_id {self.arbiter_id!r} is not an agent in the genome")
        for edge in self.edges:
            if edge.from_agent_id not in id_set:
                raise ValueError(f"edge from unknown agent {edge.from_agent_id!r}")
            if edge.to_agent_id not in id_set:
                raise ValueError(f"edge to unknown agent {edge.to_agent_id!r}")
        return self

    # -- convenience views ---------------------------------------------------
    @property
    def node_index(self) -> Dict[str, AgentNode]:
        return {a.agent_id: a for a in self.agents}

    def upstream_edges(self, agent_id: str) -> List[Edge]:
        """Edges pointing INTO ``agent_id`` (its dependencies)."""
        return [e for e in self.edges if e.to_agent_id == agent_id]

    def downstream_edges(self, agent_id: str) -> List[Edge]:
        return [e for e in self.edges if e.from_agent_id == agent_id]

    def proposer_ids(self) -> List[str]:
        """Agents that emit a full, scorable Solution (used by the Tier-2 fallback)."""
        return [
            a.agent_id
            for a in self.agents
            if a.spec.output_contract in (OutputKind.FULL_SOLUTION, OutputKind.ARBITRATION)
            and a.agent_id != self.arbiter_id
        ]

    def arbiter_feeder_ids(self) -> List[str]:
        return [e.from_agent_id for e in self.upstream_edges(self.arbiter_id)]

    # -- factory: a fresh genome starts at version 1 with one INITIAL_CURATION
    #    record, so the lineage is non-empty from birth -----------------------
    @classmethod
    def create(
        cls,
        *,
        instance_id: str,
        agents: List[AgentNode],
        edges: List[Edge],
        arbiter_id: str,
        actor: MutationActor = MutationActor.ARCHITECT,
        description: str = "Initial team curated by the Architect",
        genome_id: Optional[str] = None,
        generation: int = 0,
    ) -> "TeamGenome":
        gid = genome_id or _new_id()
        initial = MutationRecord(
            mutation_type=MutationType.INITIAL_CURATION,
            actor=actor,
            description=description,
            from_version=0,
            to_version=1,
        )
        return cls(
            genome_id=gid,
            version=1,
            instance_id=instance_id,
            agents=agents,
            edges=edges,
            arbiter_id=arbiter_id,
            generation=generation,
            status=GenomeStatus.DRAFT,
            history=[initial],
        )
