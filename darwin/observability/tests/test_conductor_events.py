"""Conductor threading — a solve narrates a complete, ordered event stream."""

import asyncio

from darwin.escalation.conductor import Conductor
from darwin.escalation.schemas import EscalationMethod, SolveBudget
from darwin.escalation.fixtures import (
    MockConductorArchitect,
    MockEscalator,
    MockRearrangementLoop,
    RecordingCorpus,
    base_genome,
)
from darwin.observability.bus import EventBus
from darwin.observability.emitter import EventEmitter
from darwin.observability.events import RunEventType
from darwin.observability.fixtures import FakeEventCollection
from darwin.observability.store import EventStore


class FakeInstance:
    def __init__(self, pc="transportation", iid="golden-transportation"):
        self.instance_id = iid
        self.problem_class = type("PC", (), {"value": pc})()


def _conductor(fitness_fn, *, escalator=None, store_genome=None, emitter=None):
    return Conductor(
        MockConductorArchitect(base_genome()),
        MockRearrangementLoop(fitness_fn),
        escalator or MockEscalator(method=EscalationMethod.CORPUS),
        RecordingCorpus(), emitter=emitter,
    )


def _make_emitter():
    store = EventStore(FakeEventCollection(), FakeEventCollection())
    bus = EventBus()
    return EventEmitter("run-x", store, bus), store, bus


async def _types(store):
    return [e.event_type for e in await store.load_run("run-x")]


async def test_no_emitter_is_byte_identical():
    cond = _conductor(lambda g: 0.95)  # no emitter
    res = await cond.solve(FakeInstance())
    assert res.cleared_threshold is True  # behaves exactly as before


async def test_clear_without_escalation_narrates():
    emitter, store, bus = _make_emitter()
    sub = bus.subscribe()
    cond = _conductor(lambda g: 0.95, emitter=emitter)
    await cond.solve(FakeInstance())
    types = await _types(store)
    assert types[0] == RunEventType.RUN_STARTED
    assert RunEventType.TEAM_DESIGNED in types
    assert RunEventType.GENOME_EVALUATED in types
    assert RunEventType.MODEL_PANEL_UPDATE in types
    assert RunEventType.THRESHOLD_CHECK in types
    assert types[-1] == RunEventType.RUN_SEALED
    # the bus saw them live in the same order
    live = []
    while not sub._queue.empty():
        live.append((await sub.get()).event_type)
    assert live == types


async def test_sequence_numbers_are_monotonic_and_gap_free():
    emitter, store, _ = _make_emitter()
    cond = _conductor(lambda g: 0.95, emitter=emitter)
    await cond.solve(FakeInstance())
    events = await store.load_run("run-x")
    assert [e.sequence_number for e in events] == list(range(len(events)))


async def test_escalation_that_helps_narrates_corpus_hit_and_seal():
    emitter, store, _ = _make_emitter()
    cond = _conductor(lambda g: 0.95 if len(g.agents) > 4 else 0.70,
                      escalator=MockEscalator(method=EscalationMethod.CORPUS), emitter=emitter)
    res = await cond.solve(FakeInstance())
    assert res.cleared_threshold is True
    types = await _types(store)
    assert RunEventType.ESCALATION_CORPUS_HIT in types
    assert RunEventType.REARRANGE_ADOPTED in types
    assert types[-1] == RunEventType.RUN_SEALED


async def test_curated_escalation_emits_curated_event():
    emitter, store, _ = _make_emitter()
    cond = _conductor(lambda g: 0.95 if len(g.agents) > 4 else 0.70,
                      escalator=MockEscalator(method=EscalationMethod.CURATED), emitter=emitter)
    await cond.solve(FakeInstance())
    assert RunEventType.ESCALATION_CURATED in await _types(store)


async def test_rollback_narrates_and_exhausts():
    emitter, store, _ = _make_emitter()
    cond = _conductor(lambda g: 0.70,  # never improves -> rollback every round
                      escalator=MockEscalator(method=EscalationMethod.CORPUS), emitter=emitter)
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_escalations=2))
    types = await _types(store)
    assert RunEventType.AGENT_ROLLED_BACK in types
    assert types[-1] == RunEventType.RUN_EXHAUSTED
    assert res.agents_added == []


async def test_emit_failure_does_not_break_solve():
    class BrokenEmitter:
        async def emit(self, *a, **k):
            raise RuntimeError("emit down")

    cond = _conductor(lambda g: 0.95, emitter=BrokenEmitter())
    res = await cond.solve(FakeInstance())  # must still return a result
    assert res.cleared_threshold is True


async def test_boundary_crash_still_emits_a_terminal_event():
    # regression: a solve that crashes after RUN_STARTED must NOT dangle without a
    # terminal — the boundary emits RUN_EXHAUSTED so the stream stays complete.
    from darwin.escalation.conductor import Conductor

    emitter, store, _ = _make_emitter()
    cond = Conductor(
        MockConductorArchitect(base_genome(), design_fail=True),  # _solve raises early
        MockRearrangementLoop(lambda g: 0.95), MockEscalator(), RecordingCorpus(), emitter=emitter,
    )
    res = await cond.solve(FakeInstance())  # floors, never raises
    assert res.cleared_threshold is False
    types = await _types(store)
    assert types[0] == RunEventType.RUN_STARTED
    assert types[-1] == RunEventType.RUN_EXHAUSTED  # terminal present, no dangling run


async def test_exactly_one_terminal_event_on_the_happy_path():
    # regression: no double terminal — the result is built before the terminal emit
    emitter, store, _ = _make_emitter()
    cond = _conductor(lambda g: 0.95, emitter=emitter)
    await cond.solve(FakeInstance())
    types = await _types(store)
    terminals = [t for t in types if t in (RunEventType.RUN_SEALED, RunEventType.RUN_EXHAUSTED)]
    assert terminals == [RunEventType.RUN_SEALED]  # exactly one, and it's last


async def test_cancellation_during_terminal_emit_still_returns_floor():
    # regression: _solve raises an ordinary Exception (boundary commits to
    # flooring); a CancelledError then landing on the boundary's terminal-emit
    # await must NOT escape solve() and lose the floor result.
    from darwin.escalation.conductor import Conductor

    class CancelOnTerminal:
        async def emit(self, event_type, *a, **k):
            if str(getattr(event_type, "value", event_type)) == "RUN_EXHAUSTED":
                raise asyncio.CancelledError()  # cancellation lands on the terminal emit
            return None  # earlier emits (RUN_STARTED) succeed

    cond = Conductor(
        MockConductorArchitect(base_genome(), design_fail=True),  # _solve raises -> boundary floors
        MockRearrangementLoop(lambda g: 0.95), MockEscalator(), RecordingCorpus(), emitter=CancelOnTerminal(),
    )
    res = await cond.solve(FakeInstance())  # must return the floor, not raise CancelledError
    assert res.cleared_threshold is False and res.error is not None
