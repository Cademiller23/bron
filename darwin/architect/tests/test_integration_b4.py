"""A9 Integration test (gated) — the real frontier model designs a runnable team.

Proves B1+B2+B3+B4 connect. Skipped in CI without a key.
"""

import os

import pytest

from darwin.agent.client import ModelClient
from darwin.agent.registry import reset_default_registry
from darwin.agent.telemetry import InMemoryTelemetrySink
from darwin.architect.architect import Architect
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import ObjectiveWeights
from darwin.team import fixtures as TF
from darwin.team.inference_gate import InferenceGate
from darwin.team.runner import TeamRunner
from darwin.team.validation import validate

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="requires GEMINI_API_KEY (real-network integration test)",
)


async def test_real_architect_designs_runnable_team():
    reset_default_registry()
    instance = golden_transportation()
    weights = ObjectiveWeights.balanced()
    store = TF.new_store()

    architect = Architect(ModelClient(), store=store)
    genome = await architect.design_initial_team(instance, weights)

    assert validate(genome).valid
    assert (await store.load(genome.genome_id)) is not None

    # the designed team runs end-to-end through B3 to a scored evaluation
    runner = TeamRunner(
        model_client=ModelClient(), telemetry=InMemoryTelemetrySink(),
        inference_gate=InferenceGate(4), store=store,
    )
    evaluation = await runner.evaluate(genome, instance, weights)
    assert evaluation.score_breakdown.scorer_version
    assert isinstance(evaluation.fitness, float)
