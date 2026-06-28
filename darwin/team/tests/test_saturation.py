"""§12.3 [MANDATORY] Pre-flight test #3 — saturation load at real swarm width.

Gates every run: the shared semaphore bounds global inference concurrency, so a
wide swarm degrades gracefully instead of saturating the endpoint and producing
timeouts that masquerade as bad genomes (which would corrupt B5's signal).
"""

import asyncio

import pytest

from darwin.team import fixtures as F
from darwin.team.inference_gate import InferenceGate
from darwin.team.runner import TeamRunner
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import ObjectiveWeights

INSTANCE = golden_transportation()
WEIGHTS = ObjectiveWeights.cost_only()

# "Real swarm width": B5 evaluates several candidate genomes concurrently, each
# with several agents.
SWARM_GENOMES = 6
GATE_LIMIT = 4


class _ConcurrencyMeter:
    """Counts ACTUAL concurrent worker.run() bodies, independently of the gate."""

    def __init__(self):
        self.live = 0
        self.peak = 0


class _MeteredWorker:
    def __init__(self, meter, agent_id, output, delay):
        self.meter, self.agent_id, self.output, self.delay = meter, agent_id, output, delay

    async def run(self, agent_input):
        self.meter.live += 1
        self.meter.peak = max(self.meter.peak, self.meter.live)
        try:
            await asyncio.sleep(self.delay)
            return F.agent_result(self.agent_id, self.output)
        finally:
            self.meter.live -= 1


def _metered_factory(meter, delay=0.01):
    def factory(spec, client, telemetry):
        return _MeteredWorker(meter, spec.agent_id, F.success_output_for(spec.output_contract), delay)

    return factory


async def _evaluate_swarm(gate: InferenceGate, factory):
    runner = TeamRunner(
        model_client=None, telemetry=F.telemetry_sink(), inference_gate=gate, store=None,
        worker_factory=factory,
    )
    genomes = [F.proposer_checker_arbiter_genome() for _ in range(SWARM_GENOMES)]
    return await asyncio.gather(*[runner.evaluate(g, INSTANCE, WEIGHTS) for g in genomes])


async def test_semaphore_bounds_peak_concurrency_and_no_spurious_failures():
    gate = InferenceGate(max_concurrent=GATE_LIMIT)
    meter = _ConcurrencyMeter()
    evaluations = await _evaluate_swarm(gate, _metered_factory(meter))

    # REAL in-flight model calls (measured independently of the gate) never
    # exceeded the ceiling — a runner that bypassed the gate would fail this.
    assert meter.peak <= GATE_LIMIT
    assert gate.peak_concurrency <= GATE_LIMIT

    # no genome was spuriously floored — every eval produced a real cleared answer
    for ev in evaluations:
        assert ev.error is None
        assert ev.cleared_threshold is True  # the scripted arbiter returns the optimum
        assert ev.fitness == 1.0


async def test_without_the_gate_peak_concurrency_is_unbounded():
    # A huge gate ~ "no gate": peak concurrency now reflects the raw swarm width,
    # demonstrating the bounded gate above was actually doing its job.
    huge = InferenceGate(max_concurrent=10_000)
    meter = _ConcurrencyMeter()
    await _evaluate_swarm(huge, _metered_factory(meter))
    # level 0 alone has 2 proposers per genome x 6 genomes = 12 concurrent calls,
    # far above the GATE_LIMIT of 4.
    assert meter.peak > GATE_LIMIT
    assert meter.peak >= 2 * SWARM_GENOMES


def test_gate_reused_across_event_loops_under_contention():
    # The gate is a long-lived singleton; reusing it across asyncio.run() boundaries
    # (different loops) must not raise "got Future attached to a different loop".
    gate = InferenceGate(max_concurrent=2)

    async def contend():
        async def hold():
            async with gate.acquire():
                await asyncio.sleep(0.01)
        await asyncio.gather(*[hold() for _ in range(8)])  # >2 -> forces the semaphore slow path
        return gate.peak_concurrency

    peak1 = asyncio.run(contend())  # loop #1
    peak2 = asyncio.run(contend())  # loop #2 — would raise without the rebind fix
    assert peak1 <= 2 and peak2 <= 2
