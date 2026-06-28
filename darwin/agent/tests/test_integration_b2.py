"""§12.7 Integration tests — gated behind GEMINI_API_KEY (the only network tests).

Skipped in CI without a key. Proves the atom connects to a real model and that a
worker-produced Solution flows into the B1 scorer.
"""

import os

import pytest

from darwin.agent.client import ModelClient
from darwin.agent.registry import reset_default_registry
from darwin.agent.spec import AgentSpec, InputKind, OutputKind, ThinkingLevel
from darwin.agent.telemetry import InMemoryTelemetrySink
from darwin.agent.worker import AgentInput, WorkerAgent
from darwin.problem import score
from darwin.problem.fixtures import golden_transportation

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="requires GEMINI_API_KEY (real-network integration test)",
)


async def test_real_gemini_call_returns_scorable_solution():
    reset_default_registry()
    spec = AgentSpec(
        agent_id="real-1",
        role_name="cost_minimizer",
        role_description=(
            "You solve a small transportation problem. Ship from sources to sinks to meet all "
            "demand at minimum total cost, respecting supplies. Return a FullSolutionOutput whose "
            "solution.flows reference real arc_ids from the problem."
        ),
        input_contract=InputKind.FULL_PROBLEM,
        output_contract=OutputKind.FULL_SOLUTION,
        thinking_level=ThinkingLevel.LOW,
        max_output_tokens=2048,
    )
    worker = WorkerAgent(spec, ModelClient(), InMemoryTelemetrySink())
    instance = golden_transportation()

    result = await worker.run(AgentInput(instance=instance))

    assert result.success, f"worker failed: {result.error}\nraw: {result.raw_text[:500]}"
    # End-to-end atom -> scorer: the worker's Solution is graded by B1.
    breakdown = score(instance, result.output.solution)
    assert breakdown.scorer_version  # a real ScoreBreakdown came back
    assert 0.0 <= breakdown.normalized_score <= 1.0
