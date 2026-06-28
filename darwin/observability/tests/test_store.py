"""EventStore — ordered persistence, resume catch-up, scorer versions, failure-safe."""

from darwin.observability.events import RunEvent, RunEventType
from darwin.observability.fixtures import FakeEventCollection
from darwin.observability.store import EventStore


def _ev(seq, run="r1", etype=RunEventType.GENOME_EVALUATED):
    return RunEvent(run_id=run, sequence_number=seq, event_type=etype)


def _store():
    return EventStore(FakeEventCollection(), FakeEventCollection())


async def test_append_and_load_run_sorted():
    s = _store()
    for seq in (2, 0, 1):  # inserted out of order
        await s.append(_ev(seq))
    loaded = await s.load_run("r1")
    assert [e.sequence_number for e in loaded] == [0, 1, 2]


async def test_load_run_filters_by_run_id():
    s = _store()
    await s.append(_ev(0, run="A"))
    await s.append(_ev(0, run="B"))
    assert len(await s.load_run("A")) == 1


async def test_load_since_resume():
    s = _store()
    for seq in range(5):
        await s.append(_ev(seq))
    since = await s.load_since("r1", after_sequence=2)
    assert [e.sequence_number for e in since] == [3, 4]


async def test_max_sequence():
    s = _store()
    assert await s.max_sequence("r1") == -1
    for seq in range(3):
        await s.append(_ev(seq))
    assert await s.max_sequence("r1") == 2


async def test_scorer_versions_roundtrip_sorted():
    s = _store()
    await s.save_scorer_version({"scorer_version": "1.0.1", "correlation": 0.7, "timestamp": "t2"})
    await s.save_scorer_version({"scorer_version": "1.0.0", "correlation": 0.5, "timestamp": "t1"})
    versions = await s.load_scorer_versions()
    assert [v["timestamp"] for v in versions] == ["t1", "t2"]


async def test_append_failure_safe():
    class Broken:
        async def insert_one(self, doc):
            raise RuntimeError("mongo down")

    s = EventStore(Broken())
    assert await s.append(_ev(0)) is False  # degrades, never raises


async def test_load_failure_safe():
    class Broken:
        def find(self, *a, **k):
            raise RuntimeError("mongo down")

    s = EventStore(Broken())
    assert await s.load_run("r1") == []
    assert await s.max_sequence("r1") == -1


async def test_corrupt_row_skipped_in_load():
    col = FakeEventCollection()
    await col.insert_one({"_id": "bad", "run_id": "r1", "sequence_number": "not-an-int",
                          "event_type": "GENOME_EVALUATED"})  # bad sequence_number
    s = EventStore(col)
    await s.append(_ev(0))  # also a good one
    loaded = await s.load_run("r1")
    assert [e.sequence_number for e in loaded] == [0]  # corrupt row skipped, good kept


async def test_scorer_versions_without_collection_degrades():
    s = EventStore(FakeEventCollection())  # no scorer_versions collection
    assert await s.save_scorer_version({"x": 1}) is False
    assert await s.load_scorer_versions() == []


async def test_load_since_skips_corrupt_row_without_collapsing():
    # regression: a corrupt non-int sequence_number must not crash the $gt query
    # (type-safe _match) nor collapse the whole catch-up to [] — the good rows
    # in the resume range must still be returned. (Breaks "no gap" on reconnect.)
    col = FakeEventCollection()
    await col.insert_one({"_id": "bad", "run_id": "r1", "sequence_number": "not-an-int",
                          "event_type": "GENOME_EVALUATED"})
    s = EventStore(col)
    for seq in range(5):
        await s.append(_ev(seq))
    since = await s.load_since("r1", after_sequence=2)
    assert [e.sequence_number for e in since] == [3, 4]  # corrupt row excluded, good kept


async def test_load_since_is_consistent_with_load_run_on_coercible_seq():
    # regression: a coercible numeric-string sequence_number (e.g. "3") must be
    # included by BOTH paths — load_since is load_run filtered on the VALIDATED int,
    # so the resume catch-up can never disagree with the full replay (no gap).
    col = FakeEventCollection()
    for seq in (0, 1, 2, 4):
        await col.insert_one(_ev(seq).to_doc())
    await col.insert_one({"_id": "s3", "run_id": "r1", "sequence_number": "3",
                          "event_type": "GENOME_EVALUATED"})  # coercible string
    s = EventStore(col)
    full = [e.sequence_number for e in await s.load_run("r1")]
    since = [e.sequence_number for e in await s.load_since("r1", after_sequence=1)]
    assert full == [0, 1, 2, 3, 4]
    assert since == [2, 3, 4]  # 3 NOT dropped — load_since is a subset of load_run
