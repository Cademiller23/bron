"""AgentCorpus — search ranking, promote/compounding, and degrade-on-failure."""

import pytest

from darwin.escalation.corpus import AgentCorpus
from darwin.escalation.embedding import KeywordEmbedder, cosine_similarity
from darwin.escalation.schemas import GapDescription, WeakDimension
from darwin.escalation.fixtures import (
    CorpusFakeCollection,
    cost_specialist_spec,
    make_corpus,
    risk_specialist_spec,
)


def _gap(text, dim, pc="transportation"):
    return GapDescription(capability_needed=text, weak_dimensions=[dim], problem_class=pc)


async def test_empty_corpus_search_returns_empty():
    corpus = make_corpus()
    out = await corpus.search(_gap("reduce cost", WeakDimension.COST))
    assert out == []


async def test_promote_then_search_finds_it():
    corpus = make_corpus()
    ok = await corpus.promote(risk_specialist_spec(), 0.2, "transportation", "inst-1")
    assert ok
    out = await corpus.search(_gap("reduce disruption risk and diversify sourcing", WeakDimension.RESILIENCE))
    assert out and out[0].entry.role_name == "disruption_risk_modeler"
    assert out[0].similarity > 0.4


async def test_search_ranks_relevant_specialist_first():
    corpus = make_corpus()
    await corpus.promote(risk_specialist_spec(), 0.2, "transportation", "i1")
    await corpus.promote(cost_specialist_spec(), 0.2, "transportation", "i2")
    risk_hit = await corpus.search(_gap("reduce disruption risk by diversifying sourcing", WeakDimension.RESILIENCE))
    cost_hit = await corpus.search(_gap("aggressively minimize total cost cheaper routes", WeakDimension.COST))
    assert risk_hit[0].entry.role_name == "disruption_risk_modeler"
    assert cost_hit[0].entry.role_name == "cost_reduction_specialist"


async def test_promote_is_idempotent_by_role_and_compounds_average():
    corpus = make_corpus()
    await corpus.promote(cost_specialist_spec(), 0.10, "transportation", "i1")
    await corpus.promote(cost_specialist_spec(), 0.20, "transportation", "i2")
    out = await corpus.search(_gap("minimize total cost", WeakDimension.COST))
    assert len(out) == 1  # one row, not two
    e = out[0].entry
    assert e.times_reused == 2
    assert abs(e.avg_fitness_contribution - 0.15) < 1e-9  # running average of 0.10, 0.20


async def test_update_stats_failure_records_failure_and_lowers_average():
    corpus = make_corpus()
    await corpus.promote(cost_specialist_spec(), 0.20, "transportation", "i1")
    out = await corpus.search(_gap("minimize cost", WeakDimension.COST))
    entry_id = out[0].entry.entry_id
    assert await corpus.update_stats(entry_id, -0.1, succeeded=False)
    refetched = await corpus.search(_gap("minimize cost", WeakDimension.COST))
    assert refetched[0].entry.failure_count == 1
    assert refetched[0].entry.avg_fitness_contribution < 0.20


async def test_update_stats_unknown_id_returns_false():
    corpus = make_corpus()
    assert await corpus.update_stats("does-not-exist", 0.1, True) is False


async def test_class_match_boosts_ranking():
    corpus = make_corpus()
    await corpus.promote(cost_specialist_spec(), 0.2, "transportation", "i1")
    same = await corpus.search(_gap("minimize cost", WeakDimension.COST, pc="transportation"))
    other = await corpus.search(_gap("minimize cost", WeakDimension.COST, pc="vehicle_routing"))
    assert same[0].combined_score > other[0].combined_score  # 1.25x class boost


async def test_broken_collection_degrades_to_empty():
    class Broken:
        def find(self, *a, **k):
            raise RuntimeError("mongo down")

        def aggregate(self, *a, **k):
            raise RuntimeError("mongo down")

    corpus = AgentCorpus(Broken(), KeywordEmbedder())
    assert await corpus.search(_gap("cost", WeakDimension.COST)) == []


async def test_promote_failure_degrades_to_false():
    class Broken:
        async def find_one(self, *a, **k):
            raise RuntimeError("down")

    corpus = AgentCorpus(Broken(), KeywordEmbedder())
    assert await corpus.promote(cost_specialist_spec(), 0.1, "t", "i") is False


async def test_corrupt_row_is_skipped_not_fatal():
    col = CorpusFakeCollection()
    await col.insert_one({"_id": "bad", "role_description_embedding": KeywordEmbedder().embed("minimize cost"),
                          "avg_fitness_contribution": 0.1})  # missing required CorpusEntry fields
    corpus = AgentCorpus(col, KeywordEmbedder())
    # should not raise; the corrupt row is skipped
    out = await corpus.search(_gap("minimize cost", WeakDimension.COST))
    assert out == []


def test_cosine_similarity_edges():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9


# --- regression: a numerically-corrupt persisted row must not poison search ----
async def _seed_one_good(col):
    """Insert one valid cost-specialist row; return its embedding for crafting siblings."""
    corpus = AgentCorpus(col, KeywordEmbedder())
    await corpus.promote(cost_specialist_spec(), 0.2, "transportation", "i-good")
    emb = KeywordEmbedder().embed("aggressively minimize total cost cheaper routes")
    return corpus, emb


async def _good_search(corpus):
    return await corpus.search(_gap("aggressively minimize total cost cheaper routes", WeakDimension.COST))


@pytest.mark.parametrize("corrupt", [
    {"role_description_embedding": [None] * 9, "avg_fitness_contribution": 0.1, "times_reused": 1},  # bad embedding
    {"avg_fitness_contribution": "high", "times_reused": 1},                                          # non-numeric avg
    {"times_reused": "many", "avg_fitness_contribution": 0.1},                                        # non-numeric count
    {"times_reused": -2, "avg_fitness_contribution": 0.1},                                            # math.log domain
])
async def test_corrupt_numeric_row_is_skipped_good_rows_survive(corrupt):
    col = CorpusFakeCollection()
    corpus, emb = await _seed_one_good(col)
    row = {"_id": "corrupt", "role_name": "corrupt", "role_description": "minimize total cost",
           "role_description_embedding": emb, "helped_problem_classes": ["transportation"]}
    row.update(corrupt)
    await col.insert_one(row)
    out = await _good_search(corpus)  # must not raise
    assert any(s.entry.role_name == "cost_reduction_specialist" for s in out)


async def test_infinite_avg_row_does_not_outrank_legitimate_agents():
    # regression: float('inf') is a valid float (doesn't raise), but an inf combined
    # score must not sort to rank 0 and starve real agents.
    col = CorpusFakeCollection()
    corpus, emb = await _seed_one_good(col)
    await col.insert_one({"_id": "poison", "role_name": "poison", "role_description": "minimize total cost",
                          "role_description_embedding": emb, "helped_problem_classes": ["transportation"],
                          "avg_fitness_contribution": float("inf"), "times_reused": 1})
    out = await _good_search(corpus)
    # the poisoned row is dropped from ranking; the legitimate agent is returned and ranks first
    assert out and out[0].entry.role_name == "cost_reduction_specialist"
    assert all(s.entry.role_name != "poison" for s in out)


async def test_search_returns_empty_not_crash_when_only_row_is_corrupt():
    col = CorpusFakeCollection()
    await col.insert_one({"_id": "c", "role_name": "c", "role_description": "minimize cost",
                          "role_description_embedding": KeywordEmbedder().embed("minimize total cost"),
                          "times_reused": -5, "avg_fitness_contribution": "nope"})
    corpus = AgentCorpus(col, KeywordEmbedder())
    assert await corpus.search(_gap("minimize total cost", WeakDimension.COST)) == []
