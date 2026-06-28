"""B8 integration — a solve narrates a durable, ordered stream that replays exactly.

The offline test ties the whole pipeline together (solve → emitter → store →
replay → a fresh bus), proving full traceability and the replay backup. The gated
test wires the real Conductor over the network.
"""

import os

import pytest

from darwin.escalation.conductor import Conductor
from darwin.escalation.schemas import EscalationMethod
from darwin.escalation.fixtures import (
    MockConductorArchitect,
    MockEscalator,
    MockRearrangementLoop,
    RecordingCorpus,
    base_genome,
)
from darwin.observability.bus import EventBus
from darwin.observability.emitter import EventEmitter
from darwin.observability.replay import replay_run
from darwin.observability.store import EventStore
from darwin.observability.fixtures import FakeEventCollection


class FakeInstance:
    def __init__(self, pc="transportation", iid="golden-transportation"):
        self.instance_id = iid
        self.problem_class = type("PC", (), {"value": pc})()


async def test_solve_durably_logs_and_replays_exactly():
    store = EventStore(FakeEventCollection(), FakeEventCollection())
    emitter = EventEmitter("run-int", store, EventBus())
    cond = Conductor(
        MockConductorArchitect(base_genome()),
        MockRearrangementLoop(lambda g: 0.95 if len(g.agents) > 4 else 0.70),
        MockEscalator(method=EscalationMethod.CORPUS), RecordingCorpus(), emitter=emitter,
    )

    res = await cond.solve(FakeInstance())
    assert res.cleared_threshold is True

    # the run is durably logged in order
    live = await store.load_run("run-int")
    assert [e.sequence_number for e in live] == list(range(len(live)))
    original_types = [e.event_type for e in live]

    # replay reconstructs the SAME ordered stream onto a fresh bus (the demo backup)
    replay_bus = EventBus()
    sub = replay_bus.subscribe(maxsize=10000)
    n = await replay_run(store, "run-int", bus=replay_bus, speed=None)
    assert n == len(live)
    replayed = []
    while not sub._queue.empty():
        replayed.append((await sub.get()).event_type)
    assert replayed == original_types  # byte-for-byte the same narrative, in order


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="requires GEMINI_API_KEY (real-network integration test)",
)
async def test_real_solve_emits_complete_ordered_stream():
    from darwin.agent.client import ModelClient
    from darwin.agent.registry import reset_default_registry
    from darwin.agent.telemetry import InMemoryTelemetrySink
    from darwin.architect.architect import Architect
    from darwin.escalation.corpus import AgentCorpus
    from darwin.escalation.embedding import KeywordEmbedder
    from darwin.escalation.escalator import Escalator
    from darwin.escalation.fixtures import CorpusFakeCollection
    from darwin.observability.events import RunEventType
    from darwin.problem.fixtures import golden_transportation
    from darwin.problem.schemas import ObjectiveWeights
    from darwin.rearrange.loop import RearrangementLoop
    from darwin.team import fixtures as TF
    from darwin.team.inference_gate import InferenceGate
    from darwin.team.runner import TeamRunner

    reset_default_registry()
    instance = golden_transportation()
    store = EventStore(FakeEventCollection(), FakeEventCollection())
    emitter = EventEmitter("run-real", store, EventBus())
    client = ModelClient()
    gstore = TF.new_store()
    runner = TeamRunner(model_client=client, telemetry=InMemoryTelemetrySink(),
                        inference_gate=InferenceGate(4), store=gstore)
    architect = Architect(client, store=gstore)
    corpus = AgentCorpus(CorpusFakeCollection(), KeywordEmbedder())
    conductor = Conductor(architect, RearrangementLoop(runner, store=gstore, registry=None, k=4),
                          Escalator(corpus, architect, store=gstore), corpus, store=gstore, emitter=emitter)

    await conductor.solve(instance, ObjectiveWeights.balanced())

    events = await store.load_run("run-real")
    assert events[0].event_type == RunEventType.RUN_STARTED
    assert events[-1].event_type in (RunEventType.RUN_SEALED, RunEventType.RUN_EXHAUSTED)
    assert [e.sequence_number for e in events] == list(range(len(events)))
