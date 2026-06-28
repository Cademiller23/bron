"""B6 test doubles — fake corpus collection, eval builders, and B4/B5 mocks."""

import asyncio
import copy
from typing import Any, Dict, List, Optional

from darwin.agent.spec import AgentSpec, InputKind, OutputKind
from darwin.escalation.corpus import AgentCorpus
from darwin.escalation.embedding import KeywordEmbedder
from darwin.problem.schemas import (
    ObjectiveWeights,
    ScoreBreakdown,
    Solution,
    Violation,
    ViolationType,
)
from darwin.problem.scorer import SCORER_VERSION
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import ArbiterTier, TeamGenome


# ===========================================================================
# Fake async corpus collection (no $vectorSearch -> corpus falls to brute-force)
# ===========================================================================
class _Cursor:
    def __init__(self, docs):
        self._docs = docs

    async def to_list(self, length=None):
        await asyncio.sleep(0)
        return [copy.deepcopy(d) for d in (self._docs if length is None else self._docs[:length])]


class CorpusFakeCollection:
    def __init__(self):
        self._docs: Dict[Any, dict] = {}

    async def insert_one(self, doc):
        await asyncio.sleep(0)
        self._docs[doc["_id"]] = copy.deepcopy(doc)
        return type("R", (), {"inserted_id": doc["_id"]})()

    async def find_one(self, filt):
        await asyncio.sleep(0)
        for doc in self._docs.values():
            if all(doc.get(k) == v for k, v in filt.items()):
                return copy.deepcopy(doc)
        return None

    def find(self, filt=None):
        docs = [d for d in self._docs.values() if all(d.get(k) == v for k, v in (filt or {}).items())]
        return _Cursor(docs)

    def aggregate(self, pipeline):  # no vector search in the fake -> corpus brute-forces
        raise NotImplementedError("$vectorSearch not supported in the fake collection")

    async def update_one(self, filt, update, upsert=False):
        await asyncio.sleep(0)
        target = None
        for _id, doc in self._docs.items():
            if all(doc.get(k) == v for k, v in filt.items()):
                target = _id
                break
        if target is None:
            if upsert:
                new = dict(filt)
                self._apply(new, update)
                new.setdefault("_id", filt.get("_id") or f"up-{len(self._docs)}")
                self._docs[new["_id"]] = new
            return
        self._apply(self._docs[target], update)

    @staticmethod
    def _apply(doc, update):
        for k, v in update.get("$set", {}).items():
            doc[k] = copy.deepcopy(v)
        for k, n in update.get("$inc", {}).items():
            doc[k] = doc.get(k, 0) + n
        for k, v in update.get("$addToSet", {}).items():
            doc.setdefault(k, [])
            if v not in doc[k]:
                doc[k].append(v)


def make_corpus(embedder=None) -> AgentCorpus:
    return AgentCorpus(CorpusFakeCollection(), embedder or KeywordEmbedder())


# ===========================================================================
# Spec / eval builders
# ===========================================================================
def spec(role: str, desc: str, oc=OutputKind.FULL_SOLUTION, ic=InputKind.FULL_PROBLEM, model="gemini-3.5-flash") -> AgentSpec:
    return AgentSpec(agent_id=role, role_name=role, role_description=desc, input_contract=ic,
                     output_contract=oc, model_id=model)


def risk_specialist_spec() -> AgentSpec:
    return spec("disruption_risk_modeler",
                "Reduce disruption risk by diversifying sourcing and avoiding single points of failure. "
                "Reason directly; never call a solver.")


def cost_specialist_spec() -> AgentSpec:
    return spec("cost_reduction_specialist",
                "Aggressively minimize total cost by finding cheaper allocations and routes. "
                "Reason directly; never call a solver.")


def evaluation_with(
    genome: TeamGenome, instance_id: str = "golden-transportation", *,
    fitness: float = 0.7, raw_risk: float = 0.1, raw_lead_time: float = 0.0,
    feasible: bool = True, normalized: Optional[float] = None, violations: Optional[List[Violation]] = None,
    norm_lead: float = 0.0,
) -> GenomeEvaluation:
    normalized = normalized if normalized is not None else (max(0.0, min(1.0, fitness)) if feasible else 0.0)
    sb = ScoreBreakdown(
        solution_id="s", instance_id=instance_id, feasible=feasible, violations=violations or [],
        raw_cost=10.0, raw_lead_time=raw_lead_time, raw_risk=raw_risk, weighted_objective=0.1,
        normalized_score=normalized, total_penalty=0.0 if feasible else abs(fitness),
        final_fitness=fitness, objective_weights=ObjectiveWeights.cost_only(), scorer_version=SCORER_VERSION,
        computed_at="t", diagnostics={"norm_lead": norm_lead},
    )
    return GenomeEvaluation(
        genome_id=genome.genome_id, version=genome.version, instance_id=instance_id,
        completed=feasible, final_solution=Solution(solution_id="s", instance_id=instance_id, flows=[]),
        score_breakdown=sb, fitness=fitness, normalized_score=normalized,
        cleared_threshold=(feasible and normalized >= 0.90), arbiter_tier_used=ArbiterTier.PRIMARY,
    )


def demand_unmet_violation() -> Violation:
    return Violation(violation_type=ViolationType.DEMAND_UNMET, location="D1", magnitude=5.0)


def capacity_violation() -> Violation:
    return Violation(violation_type=ViolationType.OVER_NODE_CAPACITY, location="T", magnitude=3.0)


# ===========================================================================
# Mock B4 (Architect) + B5 (RearrangementLoop)
# ===========================================================================
class MockArchitect:
    """Returns a fixed curated (spec, edges) for the gap; records calls."""

    def __init__(self, curated_spec: AgentSpec, fail: bool = False):
        self._spec = curated_spec
        self._fail = fail
        self.calls = 0

    async def curate_agent_for_gap(self, genome, instance, evaluation):
        self.calls += 1
        if self._fail:
            raise RuntimeError("curation failed")
        from darwin.team.genome import Edge, EdgeType
        # wire the new agent to the arbiter
        agent_id = self._spec.agent_id
        edges = [Edge(from_agent_id=agent_id, to_agent_id=genome.arbiter_id, edge_type=EdgeType.FEEDS_ARBITER)]
        return self._spec, edges


class MockRearrangementLoop:
    """Returns a RearrangementResult whose fitness is fitness_fn(genome).

    ``cost_per_run`` reports a per-run cumulative cost (the whole-rearrange spend
    the conductor accumulates against the cost budget).
    """

    def __init__(self, fitness_fn, *, cost_per_run: float = 0.0):
        self.fitness_fn = fitness_fn
        self.cost_per_run = cost_per_run
        self.calls: List[TeamGenome] = []

    async def run(self, genome, instance, weights=None):
        from darwin.rearrange.loop import RearrangementResult
        self.calls.append(genome)
        f = self.fitness_fn(genome)
        ev = evaluation_with(genome, getattr(instance, "instance_id", "i"), fitness=f)
        return RearrangementResult(
            best_genome=genome, best_evaluation=ev, fitness_trace=[f], normalized_trace=[ev.normalized_score],
            adopted_count=0, iterations=1, cleared_threshold=ev.cleared_threshold,
            total_cost_usd=self.cost_per_run,
        )


# ===========================================================================
# Conductor-level doubles (whole-brain wiring)
# ===========================================================================
def base_genome(instance_id: str = "golden-transportation") -> TeamGenome:
    """A valid proposer/checker/arbiter starting team."""
    from darwin.team.fixtures import proposer_checker_arbiter_genome

    return proposer_checker_arbiter_genome(instance_id)


def simple_gap():
    from darwin.escalation.schemas import GapDescription, WeakDimension

    return GapDescription(
        capability_needed="reduce total cost", weak_dimensions=[WeakDimension.COST],
        dominant_violations=[], problem_class="transportation", suggested_role_kind="cost_specialist",
        severity=0.2,
    )


class MockConductorArchitect:
    """B4 stand-in for the Conductor: designs an initial team and safe-default."""

    def __init__(self, genome: Optional[TeamGenome] = None, *, design_fail: bool = False,
                 safe_default_fail: bool = False):
        self._genome = genome
        self._design_fail = design_fail
        self._safe_default_fail = safe_default_fail
        self.design_calls = 0
        self.curate_calls = 0

    async def design_initial_team(self, instance, weights=None):
        self.design_calls += 1
        if self._design_fail:
            raise RuntimeError("design exploded")
        return self._genome or base_genome(getattr(instance, "instance_id", "i"))

    def _safe_default(self, instance) -> TeamGenome:
        if self._safe_default_fail:
            raise RuntimeError("registry is empty; cannot build a fallback team")
        return self._genome or base_genome(getattr(instance, "instance_id", "i"))

    async def curate_agent_for_gap(self, genome, instance, evaluation):
        from darwin.team.genome import Edge, EdgeType
        self.curate_calls += 1
        s = spec(f"curated_{len(genome.agents)}", "curated cost agent")
        edges = [Edge(from_agent_id=s.agent_id, to_agent_id=genome.arbiter_id, edge_type=EdgeType.FEEDS_ARBITER)]
        return s, edges


class MockEscalator:
    """Adds a fresh agent each call (so team_size grows); method/none controllable."""

    def __init__(self, *, method=None, none_after: Optional[int] = None,
                 corpus_entry_id: str = "ce-1", fail: bool = False):
        from darwin.escalation.schemas import EscalationMethod
        self.method = method or EscalationMethod.CORPUS
        self.none_after = none_after
        self.corpus_entry_id = corpus_entry_id
        self.fail = fail
        self.calls = 0

    async def escalate(self, genome, instance, weights, evaluation):
        from darwin.escalation.schemas import EscalationMethod, EscalationResult
        from darwin.team.genome import AgentNode, Edge, EdgeType
        self.calls += 1
        if self.fail:
            raise RuntimeError("escalator boom")
        if self.none_after is not None and self.calls > self.none_after:
            return EscalationResult(method=EscalationMethod.NONE_AVAILABLE, gap=simple_gap(), description="exhausted")
        new_spec = spec(f"added_{self.calls}", f"added agent {self.calls}")
        node = AgentNode(agent_id=new_spec.agent_id, spec=new_spec)
        data = genome.model_dump()
        data["agents"].append(node.model_dump())
        data["edges"].append(
            Edge(from_agent_id=new_spec.agent_id, to_agent_id=genome.arbiter_id,
                 edge_type=EdgeType.FEEDS_ARBITER).model_dump()
        )
        new_genome = TeamGenome.model_validate(data)
        return EscalationResult(
            method=self.method, genome=new_genome, gap=simple_gap(), added_spec=new_spec,
            added_agent_id=new_spec.agent_id,
            corpus_entry_id=self.corpus_entry_id if self.method == EscalationMethod.CORPUS else None,
            description="mock escalation",
        )


class RecordingCorpus:
    """Counts promote / update_stats calls for conductor elitism assertions."""

    def __init__(self):
        self.promotions: List[tuple] = []
        self.stats: List[tuple] = []

    async def promote(self, agent_spec, fitness_contribution, problem_class, origin_instance_id):
        self.promotions.append((agent_spec.role_name, fitness_contribution, problem_class))
        return True

    async def update_stats(self, entry_id, fitness_contribution, succeeded):
        self.stats.append((entry_id, fitness_contribution, succeeded))
        return True

    async def search(self, gap, k=5, problem_class=None):
        return []
