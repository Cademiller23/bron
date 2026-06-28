"""EventEmitter — append + publish, monotonic sequence, failure-safe, coercion."""

import asyncio

from darwin.observability.bus import EventBus
from darwin.observability.emitter import EventEmitter
from darwin.observability.events import RunEventType
from darwin.observability.fixtures import FakeEventCollection
from darwin.observability.store import EventStore


def _wired():
    store = EventStore(FakeEventCollection(), FakeEventCollection())
    bus = EventBus()
    return EventEmitter("run-1", store, bus), store, bus


async def test_emit_appends_and_publishes():
    emitter, store, bus = _wired()
    sub = bus.subscribe()
    ev = await emitter.emit(RunEventType.RUN_STARTED, {"instance_id": "i"}, description="started")
    assert ev is not None and ev.sequence_number == 0
    # published to the bus
    assert (await sub.get(timeout=1.0)).event_type == RunEventType.RUN_STARTED
    # durably appended
    loaded = await store.load_run("run-1")
    assert len(loaded) == 1 and loaded[0].description == "started"


async def test_sequence_is_monotonic():
    emitter, store, _ = _wired()
    for i in range(4):
        await emitter.emit(RunEventType.GENOME_EVALUATED, {"i": i})
    seqs = [e.sequence_number for e in await store.load_run("run-1")]
    assert seqs == [0, 1, 2, 3]


async def test_sequence_gap_free_under_concurrency():
    emitter, _, _ = _wired()
    events = await asyncio.gather(*[emitter.emit(RunEventType.GENOME_EVALUATED) for _ in range(50)])
    seqs = sorted(e.sequence_number for e in events)
    assert seqs == list(range(50))  # no duplicates, no gaps


async def test_emit_without_store_or_bus_still_returns_event():
    emitter = EventEmitter("r")
    ev = await emitter.emit(RunEventType.RUN_SEALED)
    assert ev is not None and ev.sequence_number == 0


async def test_emit_is_failure_safe_against_broken_store_and_bus():
    class BrokenStore:
        async def append(self, event):
            raise RuntimeError("mongo down")

    class BrokenBus:
        def publish(self, event):
            raise RuntimeError("bus down")

    emitter = EventEmitter("r", BrokenStore(), BrokenBus())
    ev = await emitter.emit(RunEventType.THRESHOLD_CHECK)  # must not raise
    assert ev is not None and ev.sequence_number == 0


async def test_payload_is_coerced_to_jsonable():
    class Weird:
        def __str__(self):
            return "weird-obj"

    emitter, store, _ = _wired()
    await emitter.emit(RunEventType.MODEL_PANEL_UPDATE, {"obj": Weird(), "nested": {"k": Weird()}, "n": 3})
    ev = (await store.load_run("r"))[0] if False else (await store.load_run("run-1"))[0]
    assert ev.payload["obj"] == "weird-obj" and ev.payload["nested"]["k"] == "weird-obj" and ev.payload["n"] == 3
