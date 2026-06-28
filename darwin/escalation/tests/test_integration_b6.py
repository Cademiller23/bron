"""B6 integration — the whole brain end-to-end.

The offline test is the honest "gets better across problems" demonstration:
solving the FIRST problem curates a new agent and promotes it to the corpus;
solving a SECOND, similar problem REUSES that proven agent from the corpus
instead of inventing a new one. That compounding is the MongoDB story.

The gated test wires the real B4 Architect over the network.
"""

import os

import pytest

from darwin.escalation.conductor import Conductor
from darwin.escalation.corpus import AgentCorpus
from darwin.escalation.embedding import KeywordEmbedder
from darwin.escalation.escalator import Escalator
from darwin.escalation.schemas import GapDescription, WeakDimension
from darwin.escalation.fixtures import (
    CorpusFakeCollection,
    MockConductorArchitect,
    MockRearrangementLoop,
    base_genome,
)


class FakeInstance:
    def __init__(self, pc="transportation", iid="golden-transportation"):
        self.instance_id = iid
        self.problem_class = type("PC", (), {"value": pc})()


def _fitness_by_size(mapping, default):
    return lambda g: mapping.get(len(g.agents), default)


async def test_corpus_compounds_across_problems():
    corpus = AgentCorpus(CorpusFakeCollection(), KeywordEmbedder())
    architect = MockConductorArchitect(base_genome())
    escalator = Escalator(corpus, architect, store=None)
    # base team scores 0.7 (a cost gap); adding one agent clears 0.90
    loop = MockRearrangementLoop(_fitness_by_size({4: 0.7}, 0.95))
    conductor = Conductor(architect, loop, escalator, corpus)

    # --- Problem 1: cold corpus -> curate a new agent and promote it ---------
    r1 = await conductor.solve(FakeInstance(iid="p1"))
    assert r1.cleared_threshold is True
    assert len(r1.agents_added) == 1
    assert r1.agents_added[0]["method"] == "CURATED"
    assert r1.corpus_promotions == 1
    assert r1.corpus_hits == 0

    # the corpus now holds the proven agent
    cost_gap = GapDescription(
        capability_needed="an agent that aggressively minimizes total cost by finding cheaper allocations/routes.",
        weak_dimensions=[WeakDimension.COST], problem_class="transportation",
    )
    found = await corpus.search(cost_gap)
    assert found, "the curated agent should be searchable after promotion"

    # --- Problem 2: warm corpus -> REUSE the proven agent (no new curation) ---
    architect2 = MockConductorArchitect(base_genome())  # fresh team, same corpus
    escalator2 = Escalator(corpus, architect2, store=None)
    conductor2 = Conductor(architect2, MockRearrangementLoop(_fitness_by_size({4: 0.7}, 0.95)),
                           escalator2, corpus)
    r2 = await conductor2.solve(FakeInstance(iid="p2"))
    assert r2.cleared_threshold is True
    assert r2.agents_added[0]["method"] == "CORPUS"  # reused, not re-curated
    assert r2.corpus_hits == 1
    assert architect2.curate_calls == 0  # never had to invent a new agent


# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="requires GEMINI_API_KEY (real-network integration test)",
)
async def test_real_conductor_solves_golden_transportation():
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

    reset_default_registry()
    instance = golden_transportation()
    weights = ObjectiveWeights.balanced()
    store = TF.new_store()
    client = ModelClient()
    architect = Architect(client, store=store)
    runner = TeamRunner(model_client=client, telemetry=InMemoryTelemetrySink(),
                        inference_gate=InferenceGate(4), store=store)
    loop = RearrangementLoop(runner, store=store, registry=None, k=4)
    corpus = AgentCorpus(CorpusFakeCollection(), KeywordEmbedder())
    escalator = Escalator(corpus, architect, store=store)
    conductor = Conductor(architect, loop, escalator, corpus, store=store)

    result = await conductor.solve(instance, weights)
    assert result.final_evaluation.fitness > -1e11  # a real, non-floored score
    assert result.full_fitness_trace
