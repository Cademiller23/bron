"""§12.7 Integration test (gated) — B1 + B2 + B3 connect end-to-end.

A small real genome of gemini-3.5-flash agents runs over a real B1 instance and
produces a scored GenomeEvaluation. Skipped in CI without a key.
"""

import os

import pytest

from darwin.agent.client import ModelClient
from darwin.agent.registry import reset_default_registry
from darwin.agent.spec import AgentSpec, InputKind, OutputKind, ThinkingLevel
from darwin.agent.telemetry import InMemoryTelemetrySink
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import ObjectiveWeights
from darwin.team import fixtures as F
from darwin.team.genome import AgentNode, Edge, EdgeType, TeamGenome
from darwin.team.inference_gate import InferenceGate
from darwin.team.runner import TeamRunner

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="requires GEMINI_API_KEY (real-network integration test)",
)


async def test_real_genome_evaluates_end_to_end():
    reset_default_registry()
    instance = golden_transportation()

    def spec(aid, role, desc, ic, oc):
        return AgentSpec(agent_id=aid, role_name=role, role_description=desc,
                         input_contract=ic, output_contract=oc, thinking_level=ThinkingLevel.LOW)

    proposer = AgentNode(agent_id="prop", spec=spec(
        "prop", "cost_minimizer",
        "Solve the transportation problem: meet all demand at minimum cost, respecting supplies. "
        "Return a FullSolutionOutput whose solution.flows reference real arc_ids.",
        InputKind.FULL_PROBLEM, OutputKind.FULL_SOLUTION))
    arbiter = AgentNode(agent_id="arb", spec=spec(
        "arb", "arbitrator",
        "You are given sibling proposed solutions. Choose/merge the best feasible one and return an "
        "ArbitrationOutput whose solution.flows reference real arc_ids.",
        InputKind.SIBLING_OUTPUTS, OutputKind.ARBITRATION))
    genome = TeamGenome.create(
        instance_id=instance.instance_id, agents=[proposer, arbiter],
        edges=[Edge(from_agent_id="prop", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER)],
        arbiter_id="arb")

    store = F.new_store()
    await store.save_new(genome)
    runner = TeamRunner(
        model_client=ModelClient(), telemetry=InMemoryTelemetrySink(),
        inference_gate=InferenceGate(4), store=store,
    )
    evaluation = await runner.evaluate(genome, instance, ObjectiveWeights.cost_only())

    # B1+B2+B3 connect: a real, scored evaluation with a real fitness number.
    assert evaluation.score_breakdown.scorer_version
    assert isinstance(evaluation.fitness, float)
    assert evaluation.version == 1
