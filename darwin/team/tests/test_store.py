"""§12.5 Store tests — save_new / load / the exact mutate shape / retry_mutate."""

import asyncio

import pytest

from darwin.team import fixtures as F
from darwin.team.genome import MutationActor, MutationRecord, MutationType
from darwin.team.store import GenomeStore, OptimisticLockError


def _record(from_v):
    return MutationRecord(mutation_type=MutationType.SWAP_MODEL, actor=MutationActor.REARRANGER,
                          description="m", from_version=from_v, to_version=from_v + 1)


async def _store_with_genome():
    col = F.FakeMongoCollection()
    store = GenomeStore(col)
    g = F.proposer_checker_arbiter_genome()
    await store.save_new(g)
    return store, col, g


async def test_save_new_writes_version_1_with_initial_history():
    store, col, g = await _store_with_genome()
    loaded = await store.load(g.genome_id)
    assert loaded.version == 1
    assert len(loaded.history) == 1
    assert loaded.history[0].mutation_type == MutationType.INITIAL_CURATION
    assert loaded == g  # full round-trip


async def test_load_unknown_returns_none():
    store, col, g = await _store_with_genome()
    assert await store.load("does-not-exist") is None


async def test_mutate_uses_the_exact_findoneandupdate_shape():
    store, col, g = await _store_with_genome()
    result = await store.mutate(g.genome_id, 1, {"generation": 5}, _record(1))
    assert result.version == 2 and result.generation == 5

    update = col.last_update
    # filter shape is checked implicitly (only matching {_id, version} updates);
    # assert the update document shape:
    assert update["$inc"] == {"version": 1}
    assert "history" in update["$push"]
    assert update["$set"]["generation"] == 5
    assert "updated_at" in update["$set"]
    # the pushed record was appended to history
    assert result.history[-1].description == "m"


async def test_fake_collection_honors_return_document():
    # guards against a store regression that requests ReturnDocument.BEFORE
    from pymongo import ReturnDocument

    store, col, g = await _store_with_genome()
    after = await col.find_one_and_update({"_id": g.genome_id, "version": 1},
                                          {"$set": {"generation": 9}, "$inc": {"version": 1}},
                                          return_document=ReturnDocument.AFTER)
    assert after["generation"] == 9 and after["version"] == 2
    before = await col.find_one_and_update({"_id": g.genome_id, "version": 2},
                                           {"$set": {"generation": 99}, "$inc": {"version": 1}},
                                           return_document=ReturnDocument.BEFORE)
    assert before["generation"] == 9 and before["version"] == 2  # the pre-update doc


async def test_mutate_returns_none_on_version_mismatch():
    store, col, g = await _store_with_genome()
    assert await store.mutate(g.genome_id, 999, {"generation": 1}, _record(999)) is None


async def test_mutate_without_record_skips_history_push():
    store, col, g = await _store_with_genome()
    result = await store.mutate(g.genome_id, 1, {"generation": 7}, None)
    assert result.version == 2
    assert "$push" not in col.last_update
    assert len(result.history) == 1  # unchanged


async def test_retry_mutate_gives_up_cleanly_after_max_attempts():
    store, col, g = await _store_with_genome()

    # a derive_fn that always targets a stale version -> mutate always returns None.
    # We simulate perpetual contention by advancing the version on every load.
    async def advancing_load(_id):
        # bump version behind the caller's back, then return the (now stale) doc
        doc = col._docs[g.genome_id]
        real = GenomeStore._from_doc(doc)
        await store.mutate(g.genome_id, real.version, {}, None)  # advance it
        return real  # caller holds a stale version -> its mutate will miss

    store.load = advancing_load  # type: ignore[assignment]

    def derive(current):
        return {"generation": current.generation + 1}, _record(current.version)

    with pytest.raises(OptimisticLockError):
        await store.retry_mutate(g.genome_id, derive, max_attempts=3, sleep=lambda s: asyncio.sleep(0))


async def test_registry_aware_store_reloads_fleet_models():
    # A genome using a fleet model not in the DEFAULT registry must reload
    # against the store's registry (threaded as validation context to the
    # embedded AgentSpec), not always the default.
    from darwin.agent.registry import ModelEntry, ModelRegistry, Provider
    from darwin.team.genome import TeamGenome

    reg = ModelRegistry({
        "gemini-3.5-flash": ModelEntry(model_id="gemini-3.5-flash", provider=Provider.GEMINI),
        "fleet-x": ModelEntry(model_id="fleet-x", provider=Provider.OPENAI_COMPAT, endpoint="https://x/v1"),
    })
    # build a genome using a fleet model via validation context (the reload path)
    doc = F.one_agent_genome(instance_id="i").model_dump()
    doc["agents"][0]["spec"]["model_id"] = "fleet-x"
    g = TeamGenome.model_validate(doc, context={"registry": reg})

    col = F.FakeMongoCollection()
    reg_store = GenomeStore(col, registry=reg)
    await reg_store.save_new(g)
    reloaded = await reg_store.load(g.genome_id)  # validates against reg -> OK
    assert reloaded.agents[0].spec.model_id == "fleet-x"

    default_store = GenomeStore(col)  # no registry -> default (no fleet-x)
    with pytest.raises(Exception):
        await default_store.load(g.genome_id)


async def test_retry_mutate_succeeds_first_try_when_uncontended():
    store, col, g = await _store_with_genome()

    def derive(current):
        return {"generation": current.generation + 1}, _record(current.version)

    result = await store.retry_mutate(g.genome_id, derive, sleep=lambda s: asyncio.sleep(0))
    assert result.version == 2 and result.generation == 1
