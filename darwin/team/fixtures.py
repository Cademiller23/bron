"""Test doubles & fixtures for B3 — no network, no real Mongo, no real models.

* ``FakeMongoCollection`` — an in-memory async collection whose
  ``find_one_and_update`` is atomic under asyncio (no internal await), so it
  faithfully models optimistic locking.
* Fixture genomes (one-agent; proposer→checker→arbiter).
* Canned ``AgentResult`` builders and a scripted-worker factory the runner uses.
"""

import asyncio
import copy
from typing import Any, Callable, Dict, List, Optional, Union

from darwin.agent.client import Usage
from darwin.agent.outputs import ArbitrationOutput, CritiqueOutput, FullSolutionOutput, Issue, Severity
from darwin.agent.spec import AgentSpec, InputKind, OutputKind
from darwin.agent.telemetry import InMemoryTelemetrySink
from darwin.agent.worker import AgentResult
from darwin.problem.schemas import FlowAssignment, Solution
from darwin.team.genome import AgentNode, Edge, EdgeType, TeamGenome


# ===========================================================================
# In-memory async Mongo collection (atomic optimistic locking)
# ===========================================================================
class FakeMongoCollection:
    def __init__(self) -> None:
        self._docs: Dict[Any, Dict[str, Any]] = {}
        self.last_update: Optional[Dict[str, Any]] = None  # spy for shape assertions

    async def insert_one(self, doc: Dict[str, Any]):
        await asyncio.sleep(0)  # model a network round-trip / yield point
        _id = doc["_id"]
        if _id in self._docs:
            raise ValueError(f"duplicate _id {_id!r}")
        self._docs[_id] = copy.deepcopy(doc)
        return type("R", (), {"inserted_id": _id})()

    async def find_one(self, filt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        await asyncio.sleep(0)  # yield, so a load and its later mutate can interleave
        for doc in self._docs.values():
            if self._matches(doc, filt):
                return copy.deepcopy(doc)
        return None

    async def find_one_and_update(self, filt, update, return_document=True):
        # The yield models latency; the check-and-set AFTER it has no await, so it
        # is atomic under asyncio — faithful to Mongo's atomic findOneAndUpdate.
        await asyncio.sleep(0)
        self.last_update = copy.deepcopy(update)
        target = None
        for _id, doc in self._docs.items():
            if self._matches(doc, filt):
                target = _id
                break
        if target is None:
            return None
        doc = self._docs[target]
        before = copy.deepcopy(doc)
        self._apply(doc, update)
        # honor return_document (pymongo ReturnDocument.AFTER is truthy; BEFORE falsy)
        return copy.deepcopy(doc) if return_document else before

    async def create_index(self, *args, **kwargs):
        return None

    @staticmethod
    def _matches(doc: Dict[str, Any], filt: Dict[str, Any]) -> bool:
        return all(doc.get(k) == v for k, v in filt.items())

    @staticmethod
    def _apply(doc: Dict[str, Any], update: Dict[str, Any]) -> None:
        for k, v in update.get("$set", {}).items():
            doc[k] = copy.deepcopy(v)
        for k, n in update.get("$inc", {}).items():
            doc[k] = doc.get(k, 0) + n
        for k, v in update.get("$push", {}).items():
            doc.setdefault(k, []).append(copy.deepcopy(v))


# ===========================================================================
# Spec / genome fixtures
# ===========================================================================
def _spec(agent_id: str, role: str, ic: InputKind, oc: OutputKind, **kw) -> AgentSpec:
    return AgentSpec(
        agent_id=agent_id, role_name=role, role_description=f"{role.replace('_', ' ')} job",
        input_contract=ic, output_contract=oc, **kw,
    )


def one_agent_genome(instance_id: str = "golden-transportation") -> TeamGenome:
    solver = AgentNode(
        agent_id="solver",
        spec=_spec("solver", "cost_minimizer", InputKind.FULL_PROBLEM, OutputKind.ARBITRATION),
    )
    return TeamGenome.create(instance_id=instance_id, agents=[solver], edges=[], arbiter_id="solver")


def proposer_checker_arbiter_genome(instance_id: str = "golden-transportation") -> TeamGenome:
    p1 = AgentNode(agent_id="p1", spec=_spec("p1", "cost_minimizer", InputKind.FULL_PROBLEM, OutputKind.FULL_SOLUTION))
    p2 = AgentNode(agent_id="p2", spec=_spec("p2", "risk_minimizer", InputKind.FULL_PROBLEM, OutputKind.FULL_SOLUTION))
    chk = AgentNode(agent_id="chk", spec=_spec("chk", "capacity_auditor", InputKind.SIBLING_OUTPUTS, OutputKind.CRITIQUE))
    arb = AgentNode(agent_id="arb", spec=_spec("arb", "arbitrator", InputKind.SIBLING_OUTPUTS, OutputKind.ARBITRATION))
    edges = [
        Edge(from_agent_id="p1", to_agent_id="chk", edge_type=EdgeType.CHECKS),
        Edge(from_agent_id="p1", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER),
        Edge(from_agent_id="p2", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER),
        Edge(from_agent_id="chk", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER),
    ]
    return TeamGenome.create(instance_id=instance_id, agents=[p1, p2, chk, arb], edges=edges, arbiter_id="arb")


# ===========================================================================
# Canned solutions / outputs / AgentResults
# ===========================================================================
def optimal_solution(instance_id: str = "golden-transportation") -> Solution:
    return Solution(
        solution_id="opt", instance_id=instance_id,
        flows=[FlowAssignment(arc_id="S1-D1", quantity=8.0), FlowAssignment(arc_id="S2-D2", quantity=7.0)],
        produced_by="fixture",
    )


def suboptimal_solution(instance_id: str = "golden-transportation") -> Solution:
    # feasible but pricier: D1 from S2 (cost 5), D2 from S1 (cost 3) => 40+21 = 61
    return Solution(
        solution_id="sub", instance_id=instance_id,
        flows=[FlowAssignment(arc_id="S2-D1", quantity=8.0), FlowAssignment(arc_id="S1-D2", quantity=7.0)],
        produced_by="fixture",
    )


def full_solution_output(sol: Optional[Solution] = None) -> FullSolutionOutput:
    return FullSolutionOutput(solution=sol or optimal_solution(), rationale="fixture")


def arbitration_output(sol: Optional[Solution] = None, drawn_from: Optional[List[str]] = None) -> ArbitrationOutput:
    return ArbitrationOutput(solution=sol or optimal_solution(), drawn_from=drawn_from or ["p1"], rationale="merge")


def critique_output() -> CritiqueOutput:
    return CritiqueOutput(issues=[Issue(location="S1-D1", severity=Severity.LOW, description="fine")])


def agent_result(
    agent_id: str,
    output=None,
    *,
    success: bool = True,
    role: str = "role",
    model_id: str = "gemini-3.5-flash",
    est_cost: float = 0.001,
    latency_ms: float = 10.0,
    error: Optional[str] = None,
) -> AgentResult:
    return AgentResult(
        agent_id=agent_id, role_name=role, model_id=model_id, success=success, output=output,
        raw_text="", num_repairs=0, latency_ms=latency_ms, usage=Usage(tokens_in=40, tokens_out=20),
        est_cost=est_cost, error=error if not success else None, produced_at="2026-06-27T00:00:00+00:00",
    )


# ===========================================================================
# Scripted worker + factory (the runner instantiates one worker per node)
# ===========================================================================
class ScriptedWorker:
    """Returns canned AgentResults in sequence (supports retry scenarios)."""

    def __init__(self, results: Union[AgentResult, List[AgentResult]], *, delay: float = 0.0) -> None:
        self._results = results if isinstance(results, list) else [results]
        self._i = 0
        self._delay = delay
        self.calls = 0

    async def run(self, agent_input) -> AgentResult:
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        result = self._results[min(self._i, len(self._results) - 1)]
        self._i += 1
        return result


def scripted_worker_factory(
    scripts: Dict[str, Union[AgentResult, List[AgentResult]]], delays: Optional[Dict[str, float]] = None
) -> Callable:
    delays = delays or {}
    workers = {aid: ScriptedWorker(res, delay=delays.get(aid, 0.0)) for aid, res in scripts.items()}

    def factory(spec, client, telemetry):
        return workers[spec.agent_id]

    factory.workers = workers  # exposed for call-count assertions
    return factory


def success_output_for(output_contract, instance_id: str = "golden-transportation"):
    """A successful output appropriate to an agent's output_contract."""
    from darwin.agent.spec import OutputKind

    if output_contract == OutputKind.FULL_SOLUTION:
        return full_solution_output(optimal_solution(instance_id))
    if output_contract == OutputKind.ARBITRATION:
        return arbitration_output(optimal_solution(instance_id))
    if output_contract == OutputKind.CRITIQUE:
        return critique_output()
    if output_contract == OutputKind.CONSTRAINT_REPORT:
        from darwin.agent.outputs import ConstraintReportOutput

        return ConstraintReportOutput(suspected_violations=[])
    if output_contract == OutputKind.PARTIAL_SOLUTION:
        from darwin.agent.outputs import PartialSolutionOutput

        return PartialSolutionOutput(sub_problem_id="all")
    from darwin.agent.outputs import DecompositionOutput

    return DecompositionOutput(sub_problems=[])


def saturating_worker_factory(instance_id: str = "golden-transportation", delay: float = 0.01) -> Callable:
    """A factory that returns a FRESH always-succeeding worker per call (so many
    independent genomes can be evaluated concurrently). The delay makes calls
    overlap, exercising the inference gate."""

    def factory(spec, client, telemetry):
        return ScriptedWorker(
            agent_result(spec.agent_id, success_output_for(spec.output_contract, instance_id), role=spec.role_name),
            delay=delay,
        )

    return factory


def new_store():
    from darwin.team.store import GenomeStore

    return GenomeStore(FakeMongoCollection())


def telemetry_sink() -> InMemoryTelemetrySink:
    return InMemoryTelemetrySink()
