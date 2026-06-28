"""Replay — re-emit a stored run in exact sequence order, original/accelerated pace."""

from darwin.observability.bus import EventBus
from darwin.observability.events import RunEvent, RunEventType
from darwin.observability.fixtures import FakeEventCollection
from darwin.observability.replay import replay_run
from darwin.observability.store import EventStore


def _ev(seq, run="r"):
    return RunEvent(run_id=run, sequence_number=seq, event_type=RunEventType.GENOME_EVALUATED,
                    timestamp=f"2026-06-27T00:00:{seq:02d}+00:00")


async def _seed(seqs, run="r"):
    s = EventStore(FakeEventCollection())
    for seq in seqs:
        await s.append(_ev(seq, run))
    return s


async def test_replay_publishes_in_sequence_order():
    s = await _seed([2, 0, 1, 3])  # stored out of order
    bus = EventBus()
    sub = bus.subscribe()
    n = await replay_run(s, "r", bus=bus, speed=None)
    assert n == 4
    got = []
    while not sub._queue.empty():
        got.append((await sub.get()).sequence_number)
    assert got == [0, 1, 2, 3]


async def test_replay_to_emit_callback_preserves_order():
    s = await _seed([0, 1, 2])
    seen = []

    async def emit(ev):
        seen.append(ev.sequence_number)

    await replay_run(s, "r", emit=emit, speed=None)
    assert seen == [0, 1, 2]


async def test_accelerated_replay_paces_from_timestamps():
    s = await _seed([0, 1, 2])  # 1s apart in recorded time
    slept = []

    async def fake_sleep(d):
        slept.append(d)

    await replay_run(s, "r", speed=2.0, sleep=fake_sleep)  # 2x faster -> ~0.5s gaps
    assert len(slept) == 2 and all(abs(d - 0.5) < 1e-6 for d in slept)


async def test_empty_run_replays_zero():
    s = await _seed([])
    assert await replay_run(s, "missing", bus=EventBus()) == 0


async def test_replay_continues_past_a_broken_subscriber():
    s = await _seed([0, 1])
    bus = EventBus()
    good = bus.subscribe()

    class _Broken:
        dropped = 0

        def _put(self, ev):
            raise RuntimeError("boom")

    bus._subs.append(_Broken())
    n = await replay_run(s, "r", bus=bus)  # must not raise
    assert n == 2
    assert (await good.get(timeout=1.0)).sequence_number == 0
