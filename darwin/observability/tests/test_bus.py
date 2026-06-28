"""EventBus — fan-out, slow-subscriber drop, close, broken-subscriber isolation."""

import asyncio

from darwin.observability.bus import EventBus
from darwin.observability.events import RunEvent, RunEventType


def _ev(seq):
    return RunEvent(run_id="r", sequence_number=seq, event_type=RunEventType.GENOME_EVALUATED)


async def test_subscriber_receives_published_event():
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish(_ev(0))
    got = await sub.get(timeout=1.0)
    assert got.sequence_number == 0


async def test_all_subscribers_receive():
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    assert bus.subscriber_count == 2
    bus.publish(_ev(1))
    assert (await a.get(timeout=1.0)).sequence_number == 1
    assert (await b.get(timeout=1.0)).sequence_number == 1


async def test_slow_subscriber_drops_oldest_and_does_not_block_others():
    bus = EventBus()
    slow = bus.subscribe(maxsize=2)   # tiny queue, never drained
    fast = bus.subscribe(maxsize=100)
    for i in range(5):
        bus.publish(_ev(i))  # must not block despite `slow` being full
    # the fast subscriber got everything in order
    received = [(await fast.get(timeout=1.0)).sequence_number for _ in range(5)]
    assert received == [0, 1, 2, 3, 4]
    # the slow subscriber dropped the overflow but kept the most recent
    assert slow.dropped == 3
    drained = []
    while not slow._queue.empty():
        drained.append((await slow.get()).sequence_number)
    assert drained[-1] == 4  # newest retained


async def test_close_ends_iteration():
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish(_ev(0))
    bus.publish(_ev(1))

    async def collect():
        out = []
        async for e in sub:
            out.append(e.sequence_number)
        return out

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)
    sub.close()
    out = await asyncio.wait_for(task, timeout=1.0)
    assert out == [0, 1]
    assert bus.subscriber_count == 0  # closing unsubscribes


async def test_broken_subscriber_does_not_break_publish():
    bus = EventBus()
    good = bus.subscribe()

    class _Broken:
        dropped = 0

        def _put(self, event):
            raise RuntimeError("boom")

    bus._subs.append(_Broken())  # inject a hostile subscriber
    bus.publish(_ev(7))  # must not raise
    assert (await good.get(timeout=1.0)).sequence_number == 7


async def test_bus_close_closes_all():
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    bus.close()
    assert bus.subscriber_count == 0


async def test_subscribe_after_close_does_not_hang():
    # regression: a late subscriber (e.g. a connect during shutdown) must get a
    # pre-closed subscription whose async-for terminates at once, never hangs.
    bus = EventBus()
    bus.close()
    sub = bus.subscribe()

    async def drain():
        out = []
        async for e in sub:
            out.append(e)
        return out

    assert await asyncio.wait_for(drain(), timeout=1.0) == []
