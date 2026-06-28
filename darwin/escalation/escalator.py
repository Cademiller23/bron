"""The escalator — the two ordered escalation steps.

When a team can't clear 0.90 by rearrangement alone, grow it:
1. **Corpus first** — semantically search the corpus for a proven agent that
   addresses the diagnosed gap and reuse it (cheap, fast).
2. **Curate new** — only if the corpus has nothing usable, ask the Architect to
   author a brand-new targeted agent.

Every addition is an atomic, optimistic-locked B3 mutation
(``ADD_AGENT_FROM_CORPUS`` / ``ADD_CURATED_AGENT``) with a lineage record
explaining why. Every step degrades rather than crashes.
"""

import logging
from typing import Any, List, Optional

from darwin.agent.spec import AgentSpec, InputKind, OutputKind
from darwin.constants import CORPUS_SEARCH_K, CORPUS_SIM_THRESHOLD
from darwin.escalation.diagnosis import diagnose_gap
from darwin.escalation.schemas import EscalationMethod, EscalationResult, GapDescription
from darwin.team import mutations as M
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import (
    AgentNode,
    Edge,
    EdgeType,
    MutationActor,
    MutationType,
    TeamGenome,
)
from darwin.team.validation import validate

logger = logging.getLogger("darwin.escalation.escalator")

_PROPOSAL_OUTPUTS = {OutputKind.FULL_SOLUTION, OutputKind.PARTIAL_SOLUTION, OutputKind.ARBITRATION}
_CHECK_OUTPUTS = {OutputKind.CRITIQUE, OutputKind.CONSTRAINT_REPORT}


class Escalator:
    def __init__(self, corpus: Any, architect: Any, store: Any = None, registry: Any = None,
                 *, k: int = CORPUS_SEARCH_K, sim_threshold: float = CORPUS_SIM_THRESHOLD):
        self.corpus = corpus
        self.architect = architect
        self.store = store
        self.registry = registry
        self._k = k
        self._sim_threshold = sim_threshold

    async def escalate(
        self, genome: TeamGenome, instance: Any, weights: Any, evaluation: GenomeEvaluation
    ) -> EscalationResult:
        gap = diagnose_gap(evaluation, problem_class=instance.problem_class.value)
        existing_roles = {n.spec.role_name for n in genome.agents}

        # -- Step 1: corpus reuse --------------------------------------------
        candidates = await self.corpus.search(gap, k=self._k, problem_class=instance.problem_class.value)
        for cand in candidates:  # ranked best-first
            if cand.similarity < self._sim_threshold:
                continue
            if cand.entry.role_name in existing_roles:
                continue  # no duplicate role
            spec = self._uniquify(cand.entry.agent_spec, genome)
            edges = self._wire(spec, genome)
            new_genome = await self._commit(
                genome, spec, edges, MutationType.ADD_AGENT_FROM_CORPUS,
                f"corpus reuse: added {spec.role_name} to address {gap.weak_dimensions[0].value.lower()} gap",
            )
            if new_genome is not None:
                return EscalationResult(
                    method=EscalationMethod.CORPUS, genome=new_genome, gap=gap, added_spec=spec,
                    added_agent_id=spec.agent_id, corpus_entry_id=cand.entry.entry_id,
                    corpus_candidates_considered=len(candidates),
                    description=f"reused '{cand.entry.role_name}' (sim {cand.similarity:.2f}, avg {cand.entry.avg_fitness_contribution:.2f})",
                )

        # -- Step 2: curate a new agent --------------------------------------
        try:
            spec, edges = await self.architect.curate_agent_for_gap(genome, instance, evaluation)
        except Exception as exc:  # noqa: BLE001 - curation failure degrades to NONE_AVAILABLE
            logger.warning("curation failed during escalation: %s", exc)
            return EscalationResult(method=EscalationMethod.NONE_AVAILABLE, gap=gap,
                                    corpus_candidates_considered=len(candidates), description=f"curation failed: {exc}")
        if spec is None or spec.role_name in existing_roles:
            return EscalationResult(method=EscalationMethod.NONE_AVAILABLE, gap=gap,
                                    corpus_candidates_considered=len(candidates),
                                    description="no usable corpus or curated agent")

        new_genome = await self._commit(
            genome, spec, list(edges), MutationType.ADD_CURATED_AGENT,
            f"curated new {spec.role_name} — no corpus match for {gap.weak_dimensions[0].value.lower()} gap",
        )
        if new_genome is None:
            return EscalationResult(method=EscalationMethod.NONE_AVAILABLE, gap=gap,
                                    corpus_candidates_considered=len(candidates),
                                    description="curated agent could not be wired validly")
        return EscalationResult(
            method=EscalationMethod.CURATED, genome=new_genome, gap=gap, added_spec=spec,
            added_agent_id=spec.agent_id, corpus_candidates_considered=len(candidates),
            description=f"curated '{spec.role_name}'",
        )

    # -- helpers ------------------------------------------------------------
    def _uniquify(self, spec: AgentSpec, genome: TeamGenome) -> AgentSpec:
        taken = {a.agent_id for a in genome.agents} | {a.spec.role_name for a in genome.agents}
        if spec.agent_id not in taken and spec.role_name not in taken:
            return spec
        base = spec.agent_id
        i = 2
        new_id = f"{base}-{i}"
        while new_id in taken:
            i += 1
            new_id = f"{base}-{i}"
        return AgentSpec.model_validate(
            {**spec.model_dump(), "agent_id": new_id, "role_name": new_id}, context={"registry": self.registry}
        )

    def _wire(self, spec: AgentSpec, genome: TeamGenome) -> List[Edge]:
        """Wire an added agent validly (B5 will optimize it afterward) per §4.1.

        The upstream feeder edge type is dictated by the agent's INPUT contract,
        not its output (the validator keys its rule off the input):
          * PROBLEM_PLUS_DRAFT *requires* a PASSES_PROPOSAL feeder.
          * SIBLING_OUTPUTS accepts any feeder; a check-shaped output reads best
            as a CHECKS edge, anything else as PASSES_PROPOSAL.
        """
        aid = spec.agent_id
        edges: List[Edge] = []
        if spec.input_contract in (InputKind.SIBLING_OUTPUTS, InputKind.PROBLEM_PLUS_DRAFT):
            feeders = genome.proposer_ids() or [
                a.agent_id for a in genome.agents if a.agent_id != genome.arbiter_id
            ]
            if feeders:
                if spec.input_contract == InputKind.PROBLEM_PLUS_DRAFT:
                    etype = EdgeType.PASSES_PROPOSAL  # validator demands PASSES_PROPOSAL upstream
                elif spec.output_contract in _CHECK_OUTPUTS:
                    etype = EdgeType.CHECKS
                else:
                    etype = EdgeType.PASSES_PROPOSAL
                edges.append(Edge(from_agent_id=feeders[0], to_agent_id=aid, edge_type=etype))
        # output flows to the arbiter (an added agent is never the arbiter)
        edges.append(Edge(from_agent_id=aid, to_agent_id=genome.arbiter_id, edge_type=EdgeType.FEEDS_ARBITER))
        return edges

    async def _commit(
        self, genome: TeamGenome, spec: AgentSpec, edges: List[Edge], mutation_type: MutationType, description: str
    ) -> Optional[TeamGenome]:
        node = AgentNode(agent_id=spec.agent_id, spec=spec)
        # validate the candidate first (invalid wiring => reject this addition)
        data = genome.model_dump()
        data["agents"] = data["agents"] + [node.model_dump()]
        data["edges"] = data["edges"] + [e.model_dump() for e in edges]
        try:
            candidate = TeamGenome.model_validate(data, context={"registry": self.registry})
        except Exception:  # noqa: BLE001
            return None
        if not validate(candidate, self.registry).valid:
            return None

        if self.store is None:
            return candidate  # in-memory (tests)
        derive = M.add_agent(node, edges, mutation_type=mutation_type, actor=MutationActor.ESCALATION,
                             description=description, registry=self.registry)
        try:
            return await self.store.retry_mutate(genome.genome_id, derive)
        except Exception as exc:  # noqa: BLE001 - commit failure degrades (no addition)
            logger.warning("escalation commit failed for %s: %s", genome.genome_id, exc)
            return None
