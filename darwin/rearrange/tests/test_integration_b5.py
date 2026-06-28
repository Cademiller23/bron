"""B8 Integration test (gated) — the real loop rearranges a real B4 genome.

Proves B1+B2+B3+B4+B5 connect. Skipped in CI without a key.
"""

import os

import pytest

from darwin.agent.client import ModelClient
from darwin.agent.registry import reset_default_registry
from darwin.agent.telemetry import InMemoryTelemetrySink
from darwin.architect.architect import Architect
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import ObjectiveWeights
from darwin.rearrange.loop import RearrangementLoop
from darwin.team import fixtures as TF
from darwin.team.inference_gate import InferenceGate
from darwin.team.runner import TeamRunner

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="requires GEMINI_API_KEY (real-network integration test)",
)


async def test_real_loop_rearranges_a_real_genome():
    reset_default_registry()
    instance = golden_transportation()
    weights = ObjectiveWeights.balanced()
    store = TF.new_store()
    gate = InferenceGate(4)

    genome = await Architect(ModelClient(), store=store).design_initial_team(instance, weights)
    runner = TeamRunner(model_client=ModelClient(), telemetry=InMemoryTelemetrySink(),
                        inference_gate=gate, store=store)
    loop = RearrangementLoop(runner, store=store, registry=None, k=4)

    result = await loop.run(genome, instance, weights)

    # the climbing curve is non-decreasing and at least one rearrangement was evaluated
    trace = result.fitness_trace
    assert all(trace[i] <= trace[i + 1] + 1e-9 for i in range(len(trace) - 1))
    assert result.iterations >= 1
