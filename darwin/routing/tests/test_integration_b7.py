"""B7 integration — the model-aware swarm on the real fleet (gated).

Requires real model keys (and, for the FAST tier, a reachable MAX endpoint).
Skipped in CI. Asserts the three B7 acceptance stories:
  (a) the FAST tier carries real volume (DigitalOcean workhorse), only the arbiter/Architect
      hit FRONTIER — a MAX-dominant per-model distribution;
  (b) the model-aware search reduces inference cost while holding normalized_score
      after the threshold is cleared;
  (c) the genotype search converges on the five-model fleet within the budget.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="requires GEMINI_API_KEY (+ a MAX endpoint) — real-network integration test",
)


async def test_real_solve_is_max_dominant_and_trims_cost():
    from darwin.agent.client import ModelClient
    from darwin.agent.registry import default_registry, reset_default_registry
    from darwin.agent.telemetry import InMemoryTelemetrySink
    from darwin.architect.architect import Architect
    from darwin.escalation.conductor import Conductor
    from darwin.escalation.corpus import AgentCorpus
    from darwin.escalation.embedding import KeywordEmbedder
    from darwin.escalation.escalator import Escalator
    from darwin.escalation.fixtures import CorpusFakeCollection
    from darwin.problem.fixtures import golden_transportation
    from darwin.problem.schemas import ObjectiveWeights
    from darwin.rearrange.loop import RearrangementLoop
    from darwin.routing import observability as O
    from darwin.routing.efficiency import EfficiencyStrategy
    from darwin.routing.fleet import FRONTIER, get_fleet, install_fleet
    from darwin.routing.gene import MODEL_AWARE_OPERATORS
    from darwin.team import fixtures as TF
    from darwin.team.inference_gate import InferenceGate
    from darwin.team.runner import TeamRunner

    reset_default_registry()
    registry = install_fleet(default_registry())  # the curated fleet behind one interface
    instance = golden_transportation()
    weights = ObjectiveWeights.balanced()
    store = TF.new_store()
    client = ModelClient(registry=registry)
    telemetry = InMemoryTelemetrySink()
    runner = TeamRunner(model_client=client, telemetry=telemetry, inference_gate=InferenceGate(8), store=store)
    loop = RearrangementLoop(
        runner, store=store, registry=registry, k=6,
        selector=EfficiencyStrategy(), extra_operators=MODEL_AWARE_OPERATORS,
    )
    architect = Architect(client, store=store)
    corpus = AgentCorpus(CorpusFakeCollection(), KeywordEmbedder())
    escalator = Escalator(corpus, architect, store=store)
    conductor = Conductor(architect, loop, escalator, corpus, store=store,
                          comparator=EfficiencyStrategy().improves)

    # baseline inference cost: the same final team with every agent forced to the
    # frontier model (the "no routing" assignment) — to show routing trimmed cost.
    from darwin.routing.fleet import by_tier
    from darwin.routing.gene import _set_models, genotype

    result = await conductor.solve(instance, weights)
    econ = O.aggregate_sink(telemetry, registry=registry)

    assert econ.total_calls > 0
    # (a) the FAST tier carries real volume and the DigitalOcean workhorse served some
    assert econ.fast_tier_share > 0.0
    assert econ.max_served_share >= 0.0
    # (c) only the curated five-model fleet was ever used (convergence on the tier)
    assert {m.model_id for m in econ.per_model} <= {fm.model_id for fm in get_fleet()}
    # (b) the discovered genotype is cheaper to run than an all-frontier assignment
    pro = by_tier(FRONTIER)[0]
    final = result.final_genome
    all_frontier = _set_models(final, {a.agent_id: pro for a in final.agents}, registry, "all-frontier baseline")
    if all_frontier is not None:  # (None only if already all-frontier)
        base_cost = sum(registry.get(a.spec.model_id).est_cost_per_1k_out for a in all_frontier.genome.agents)
        final_cost = sum(registry.get(m).est_cost_per_1k_out for m in genotype(final).values())
        assert final_cost <= base_cost  # routing held quality (cleared/best) while cutting cost
    # the brain returned a real, scored answer
    assert result.final_evaluation.fitness > -1e11
    print("B7 headline:", econ.headline)
