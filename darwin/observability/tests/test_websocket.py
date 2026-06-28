"""WebSocket bridge — live forwarding, resume-from-sequence, disconnect-safety."""

import asyncio

from darwin.observability.bus import EventBus
from darwin.observability.events import RunEvent, RunEventType
from darwin.observability.fixtures import FakeEventCollection
from darwin.observability.store import EventStore
from darwin.observability.websocket_server import WebSocketServer


def _ev(seq, run="r"):
    return RunEvent(run_id=run, sequence_number=seq, event_type=RunEventType.GENOME_EVALUATED)


async def _wait_subscribed(bus):
    while bus.subscriber_count == 0:
        await asyncio.sleep(0)


async def test_session_forwards_live_events_in_order():
    bus = EventBus()
    received = []

    async def send(doc):
        received.append(doc)

    server = WebSocketServer(bus)
    task = asyncio.create_task(server.handle(send))
    await _wait_subscribed(bus)
    for i in range(3):
        bus.publish(_ev(i))
    bus.close()  # sentinel queued after the 3 events
    await asyncio.wait_for(task, timeout=1.0)
    assert [d["sequence_number"] for d in received] == [0, 1, 2]
    assert all(isinstance(d, dict) for d in received)  # sent as JSON dicts


async def test_reconnecting_client_resumes_from_last_sequence():
    store = EventStore(FakeEventCollection())
    for seq in range(5):
        await store.append(_ev(seq))
    bus = EventBus()
    received = []

    async def send(doc):
        received.append(doc["sequence_number"])

    server = WebSocketServer(bus, store)
    task = asyncio.create_task(server.handle(send, run_id="r", last_sequence=2))
    await _wait_subscribed(bus)  # subscribed first (catch-up may still be in flight)
    bus.publish(_ev(5))  # a live event
    bus.close()
    await asyncio.wait_for(task, timeout=1.0)
    assert received == [3, 4, 5]  # missed range replayed, then the live event — no gap/dup


async def test_resume_dedups_overlap_no_gap_no_dup():
    # event 5 is in BOTH the store (catch-up range) AND the live stream; it must
    # appear exactly once, and the live-only 6 must not be missed.
    store = EventStore(FakeEventCollection())
    for seq in range(6):  # store has 0..5
        await store.append(_ev(seq))
    bus = EventBus()
    received = []

    async def send(doc):
        received.append(doc["sequence_number"])

    server = WebSocketServer(bus, store)
    task = asyncio.create_task(server.handle(send, run_id="r", last_sequence=2))
    await _wait_subscribed(bus)
    bus.publish(_ev(5))  # overlaps the catch-up range
    bus.publish(_ev(6))  # live-only
    bus.close()
    await asyncio.wait_for(task, timeout=1.0)
    assert received == [3, 4, 5, 6]  # 5 not duplicated, 6 not missed


async def test_disconnect_closes_session_cleanly():
    bus = EventBus()

    async def dead_send(doc):
        raise RuntimeError("client gone")

    server = WebSocketServer(bus)
    task = asyncio.create_task(server.handle(dead_send))
    await _wait_subscribed(bus)
    bus.publish(_ev(0))  # first send fails -> session ends itself
    await asyncio.wait_for(task, timeout=1.0)
    assert bus.subscriber_count == 0  # unsubscribed in finally; other subscribers unaffected


async def test_one_dead_client_does_not_affect_another():
    bus = EventBus()
    alive = []

    async def good(doc):
        alive.append(doc["sequence_number"])

    async def dead(doc):
        raise RuntimeError("gone")

    server = WebSocketServer(bus)
    t_dead = asyncio.create_task(server.handle(dead))
    t_good = asyncio.create_task(server.handle(good))
    while bus.subscriber_count < 2:
        await asyncio.sleep(0)
    bus.publish(_ev(0))
    bus.publish(_ev(1))
    bus.close()
    await asyncio.wait_for(asyncio.gather(t_dead, t_good), timeout=1.0)
    assert alive == [0, 1]  # the healthy client got everything despite the dead one
