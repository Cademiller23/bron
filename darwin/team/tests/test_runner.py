"""§12.6 Runner tests — mocked client, topology, resilience, persistence."""

import asyncio
import math

import pytest

from darwin.team import fixtures as F
from darwin.team.genome import GenomeStatus, MutationActor, MutationRecord, MutationType
from darwin.team.inference_gate import InferenceGate
from darwin.team.runner import TeamRunner
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import (
    Arc,
    FlowAssignment,
    KnownOptimum,
    Node,
    NodeType,
    ObjectiveWeights,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
    Solution,
)

INSTANCE = golden_transportation()
WEIGHTS = ObjectiveWeights.cost_only()


async def _run(scripts, *, genome=None, store=None, scorer=None, gate=None):
    genome = genome or F.proposer_checker_arbiter_genome()
    store = store or F.new_store()
    if await store.load(genome.genome_id) is None:
        await store.save_new(genome)
    runner = TeamRunner(
        scorer=scorer, model_client=None, telemetry=F.telemetry_sink(),
        inference_gate=gate or InferenceGate(8), store=store,
        worker_factory=F.scripted_worker_factory(scripts),
    )
    ev = await runner.evaluate(genome, INSTANCE, WEIGHTS)
    return ev, store


def _happy_scripts():
    return {
        "p1": F.agent_result("p1", F.full_solution_output(F.optimal_solution())),
        "p2": F.agent_result("p2", F.full_solution_output(F.suboptimal_solution())),
        "chk": F.agent_result("chk", F.critique_output()),
        "arb": F.agent_result("arb", F.arbitration_output(F.optimal_solution())),
    }


# ---------------------------------------------------------------------------
# Happy path + topology
# ---------------------------------------------------------------------------
async def test_happy_path_produces_real_scored_evaluation():
    ev, store = await _run(_happy_scripts())
    assert ev.error is None
    assert math.isclose(ev.fitness, 1.0)
    assert ev.cleared_threshold is True
    assert ev.arbiter_tier_used.value == "PRIMARY"
    assert len(ev.agent_results) == 4  # p1, p2, chk, arb


class _Rec:
    def __init__(self):
        self.order = []
        self.in_flight = 0
        self.peak = 0


class _InstrumentedWorker:
    def __init__(self, rec, agent_id, result, delay):
        self.rec, self.agent_id, self.result, self.delay = rec, agent_id, result, delay

    async def run(self, agent_input):
        self.rec.order.append(self.agent_id)
        self.rec.in_flight += 1
        self.rec.peak = max(self.rec.peak, self.rec.in_flight)
        await asyncio.sleep(self.delay)
        self.rec.in_flight -= 1
        return self.result


def _instrumented_factory(rec, scripts, delay=0.01):
    def factory(spec, client, telemetry):
        return _InstrumentedWorker(rec, spec.agent_id, scripts[spec.agent_id], delay)

    return factory


async def test_topological_order_and_intra_level_concurrency():
    rec = _Rec()
    genome = F.proposer_checker_arbiter_genome()
    store = F.new_store()
    await store.save_new(genome)
    runner = TeamRunner(
        model_client=None, telemetry=F.telemetry_sink(), inference_gate=InferenceGate(8),
        store=store, worker_factory=_instrumented_factory(rec, _happy_scripts()),
    )
    await runner.evaluate(genome, INSTANCE, WEIGHTS)

    pos = {aid: i for i, aid in enumerate(rec.order)}
    # chk depends on p1; arbiter depends on everything -> topological order respected
    assert pos["p1"] < pos["chk"]
    assert pos["chk"] < pos["arb"]
    assert pos["p1"] < pos["arb"] and pos["p2"] < pos["arb"]
    # p1 and p2 are in the same level -> they ran concurrently (overlapped)
    assert rec.peak >= 2


# ---------------------------------------------------------------------------
# Agent-level resilience
# ---------------------------------------------------------------------------
async def test_agent_retry_once_then_success():
    scripts = _happy_scripts()
    scripts["p1"] = [F.agent_result("p1", None, success=False, error="x"),
                     F.agent_result("p1", F.full_solution_output(F.optimal_solution()))]
    factory = F.scripted_worker_factory(scripts)
    genome = F.proposer_checker_arbiter_genome()
    store = F.new_store(); await store.save_new(genome)
    runner = TeamRunner(model_client=None, telemetry=F.telemetry_sink(), inference_gate=InferenceGate(8),
                        store=store, worker_factory=factory)
    ev = await runner.evaluate(genome, INSTANCE, WEIGHTS)
    assert factory.workers["p1"].calls == 2  # exactly one retry
    assert ev.error is None


async def test_agent_route_around_when_it_keeps_failing():
    scripts = _happy_scripts()
    scripts["chk"] = F.agent_result("chk", None, success=False, error="checker dead")  # always fails
    factory = F.scripted_worker_factory(scripts)
    genome = F.proposer_checker_arbiter_genome()
    store = F.new_store(); await store.save_new(genome)
    runner = TeamRunner(model_client=None, telemetry=F.telemetry_sink(), inference_gate=InferenceGate(8),
                        store=store, worker_factory=factory)
    ev = await runner.evaluate(genome, INSTANCE, WEIGHTS)
    assert factory.workers["chk"].calls == 2  # tried twice (1 + 1 retry), then routed around
    assert ev.error is None
    assert math.isclose(ev.fitness, 1.0)  # arbiter still got the proposals
    chk_result = next(r for r in ev.agent_results if r.agent_id == "chk")
    assert chk_result.success is False  # recorded as failed


# ---------------------------------------------------------------------------
# Evaluation boundary
# ---------------------------------------------------------------------------
async def test_evaluation_boundary_floors_and_logs_eval_error():
    def boom(*a, **k):
        raise RuntimeError("unexpected scorer explosion")

    ev, store = await _run(_happy_scripts(), scorer=boom)
    assert ev.error is not None and "RuntimeError" in ev.error
    assert ev.fitness < 0  # floored
    assert ev.cleared_threshold is False
    after = await store.load(ev.genome_id)
    assert any(h.mutation_type == MutationType.EVAL_ERROR for h in after.history)


class _RaisingWorker:
    def __init__(self, exc):
        self._exc = exc

    async def run(self, agent_input):
        raise self._exc


def _factory_with_raiser(scripts, raiser_id, exc):
    base = F.scripted_worker_factory(scripts)

    def factory(spec, client, telemetry):
        if spec.agent_id == raiser_id:
            return _RaisingWorker(exc)
        return base(spec, client, telemetry)

    return factory


async def test_raising_non_arbiter_worker_is_routed_around_not_floored():
    # A contract-violating worker that RAISES (not just returns success=False) must
    # not floor the whole genome — it is routed around like any failed agent.
    genome = F.proposer_checker_arbiter_genome()
    store = F.new_store(); await store.save_new(genome)
    factory = _factory_with_raiser(_happy_scripts(), "chk", RuntimeError("checker exploded"))
    runner = TeamRunner(model_client=None, telemetry=F.telemetry_sink(), inference_gate=InferenceGate(8),
                        store=store, worker_factory=factory)
    ev = await runner.evaluate(genome, INSTANCE, WEIGHTS)
    assert ev.error is None  # genome NOT floored
    assert math.isclose(ev.fitness, 1.0)  # arbiter still got the proposals
    chk = next(r for r in ev.agent_results if r.agent_id == "chk")
    assert chk.success is False and "raised" in chk.error


async def test_worker_raising_cancelled_error_floors_not_escapes():
    # CancelledError is a BaseException; per spec §10 an asyncio cancellation is
    # floored at the boundary, and evaluate() must NOT propagate it.
    genome = F.proposer_checker_arbiter_genome()
    store = F.new_store(); await store.save_new(genome)
    factory = _factory_with_raiser(_happy_scripts(), "p1", asyncio.CancelledError())
    runner = TeamRunner(model_client=None, telemetry=F.telemetry_sink(), inference_gate=InferenceGate(8),
                        store=store, worker_factory=factory)
    ev = await runner.evaluate(genome, INSTANCE, WEIGHTS)  # must NOT raise
    assert ev.fitness < 0 and ev.cleared_threshold is False
    assert "cancel" in (ev.error or "").lower()


async def test_invalid_genome_is_floored_not_run():
    from darwin.team.genome import TeamGenome

    g = F.proposer_checker_arbiter_genome()
    # remove all edges -> arbiter has no feeders + orphans -> invalid
    invalid = TeamGenome.model_validate({**g.model_dump(), "edges": []})
    ev, store = await _run({}, genome=invalid)
    assert ev.error is not None and "invalid genome" in ev.error
    assert ev.fitness < 0


# ---------------------------------------------------------------------------
# Outcome persistence (optimistic-locked, conflict-safe)
# ---------------------------------------------------------------------------
async def test_outcome_persisted_and_status_updated():
    ev, store = await _run(_happy_scripts())
    after = await store.load(ev.genome_id)
    assert after.current_fitness == ev.fitness
    assert after.status == GenomeStatus.CLEARED_THRESHOLD
    assert after.version > ev.version  # the outcome write bumped the version


async def test_persistence_survives_a_concurrent_version_advance():
    genome = F.proposer_checker_arbiter_genome()
    store = F.new_store()
    await store.save_new(genome)
    # advance the stored version out from under the (v1) genome being evaluated
    await store.mutate(genome.genome_id, 1, {"generation": 1},
                       MutationRecord(mutation_type=MutationType.REARRANGE_EDGE, actor=MutationActor.REARRANGER,
                                      description="concurrent", from_version=1, to_version=2))
    runner = TeamRunner(model_client=None, telemetry=F.telemetry_sink(), inference_gate=InferenceGate(8),
                        store=store, worker_factory=F.scripted_worker_factory(_happy_scripts()))
    ev = await runner.evaluate(genome, INSTANCE, WEIGHTS)  # given v1, but store is at v2
    assert ev.version == 1  # records the version it evaluated
    after = await store.load(genome.genome_id)
    assert after.current_fitness == ev.fitness  # persisted via retry_mutate reload
    assert after.version == 3  # v2 (concurrent) -> v3 (outcome)


# ---------------------------------------------------------------------------
# Threshold flag (the 90% gate that drives B6)
# ---------------------------------------------------------------------------
def _threshold_instance():
    return ProblemInstance(
        instance_id="thr", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=1000.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=100.0)],
        arcs=[Arc(arc_id="a1", from_node="S1", to_node="D1", unit_cost=0.93),
              Arc(arc_id="a2", from_node="S1", to_node="D1", unit_cost=1.0),
              Arc(arc_id="a3", from_node="S1", to_node="D1", unit_cost=1.2)],
        known_optimum=KnownOptimum(objective_value=93.0, source=OptimumSource.SOLVER_VERIFIED, verified=True),
    )


@pytest.mark.parametrize("arc_id,expected_cleared", [("a2", True), ("a3", False)])
async def test_threshold_gate(arc_id, expected_cleared):
    from darwin.agent.outputs import ArbitrationOutput

    inst = _threshold_instance()
    genome = F.one_agent_genome(instance_id="thr")  # single arbiter reads the problem
    store = F.new_store(); await store.save_new(genome)
    sol = Solution(solution_id="s", instance_id="thr", flows=[FlowAssignment(arc_id=arc_id, quantity=100.0)])
    scripts = {"solver": F.agent_result("solver", ArbitrationOutput(solution=sol, drawn_from=[]))}
    runner = TeamRunner(model_client=None, telemetry=F.telemetry_sink(), inference_gate=InferenceGate(8),
                        store=store, worker_factory=F.scripted_worker_factory(scripts))
    ev = await runner.evaluate(genome, inst, WEIGHTS)
    # a2: 100 cost vs optimum 93 -> normalized 0.93 -> cleared; a3: 120 -> 0.775 -> not
    assert ev.cleared_threshold is expected_cleared
