"""§12.1 [MANDATORY] Pre-flight test #1 — Optimistic-lock contention.

Gates every run: concurrent writers can never clobber each other.
"""

import asyncio

import pytest

from darwin.team import fixtures as F
from darwin.team.genome import MutationActor, MutationRecord, MutationType
from darwin.team.store import GenomeStore


def _record(from_v: int, desc: str) -> MutationRecord:
    return MutationRecord(
        mutation_type=MutationType.REARRANGE_EDGE,
        actor=MutationActor.REARRANGER,
        description=desc,
        from_version=from_v,
        to_version=from_v + 1,
    )


async def _fresh_store():
    store = GenomeStore(F.FakeMongoCollection())
    genome = F.proposer_checker_arbiter_genome()
    await store.save_new(genome)
    return store, genome


async def test_two_writers_same_version_exactly_one_wins():
    store, g = await _fresh_store()
    assert g.version == 1
    r1, r2 = await asyncio.gather(
        store.mutate(g.genome_id, 1, {"generation": 10}, _record(1, "writer-A")),
        store.mutate(g.genome_id, 1, {"generation": 20}, _record(1, "writer-B")),
    )
    winners = [r for r in (r1, r2) if r is not None]
    losers = [r for r in (r1, r2) if r is None]
    assert len(winners) == 1 and len(losers) == 1  # exactly one wins
    assert winners[0].version == 2


async def test_loser_retry_mutate_resolves_no_lost_update():
    store, g = await _fresh_store()
    # a winner advances the version out from under a stale writer
    await store.mutate(g.genome_id, 1, {"generation": 1}, _record(1, "winner"))
    # the stale writer's direct mutate at v=1 now fails
    assert await store.mutate(g.genome_id, 1, {"generation": 2}, _record(1, "stale")) is None

    # but retry_mutate reloads (to v=2) and succeeds at v=3
    def derive(current):
        return {"generation": current.generation + 1}, _record(current.version, "loser-retried")

    result = await store.retry_mutate(g.genome_id, derive, sleep=lambda s: asyncio.sleep(0))
    assert result.version == 3
    descs = [h.description for h in result.history]
    assert "winner" in descs and "loser-retried" in descs  # both mutations survive, in order


async def test_burst_of_concurrent_mutators_no_clobbering():
    store, g = await _fresh_store()
    n = 12

    def make_derive(i):
        def derive(current):
            return {"generation": current.generation + 1}, _record(current.version, f"mutator-{i}")

        return derive

    # generous retry budget: under a wide burst on ONE genome, optimistic locking
    # needs enough reloads to let every writer eventually land (the default-5
    # give-up behaviour is covered in test_store.py).
    await asyncio.gather(
        *[
            store.retry_mutate(g.genome_id, make_derive(i), max_attempts=100, sleep=lambda s: asyncio.sleep(0))
            for i in range(n)
        ]
    )
    final = await store.load(g.genome_id)
    assert final.version == 1 + n  # every mutation applied exactly once
    descs = [h.description for h in final.history]
    for i in range(n):
        assert f"mutator-{i}" in descs  # no torn writes, no clobbering
    assert final.generation == n
