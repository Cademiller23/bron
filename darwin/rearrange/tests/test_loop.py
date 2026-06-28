"""B8 Loop tests — climbs / never-regresses / stops / commits / never-crashes."""

import random

import pytest

from darwin.agent.registry import default_registry, reset_default_registry
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import ObjectiveWeights
from darwin.rearrange import fixtures as RF
from darwin.rearrange.loop import RearrangementLoop
from darwin.rearrange.operators import signature
from darwin.team import fixtures as TF
from darwin.team.genome import MutationActor, MutationRecord, MutationType

INSTANCE = golden_transportation()
WEIGHTS = ObjectiveWeights.cost_only()


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


def _is_non_decreasing(seq):
    return all(seq[i] <= seq[i + 1] + 1e-12 for i in range(len(seq) - 1))


async def _saved_genome():
    g = TF.proposer_checker_arbiter_genome()
    store = TF.new_store()
    await store.save_new(g)
    return g, store


async def test_climbs_and_adopts_improvements():
    g, store = await _saved_genome()
    loop = RearrangementLoop(RF.MockRunner(RF.climbing_fitness), store=store, registry=default_registry(),
                             k=6, rng=random.Random(1))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    assert res.adopted_count >= 2
    assert _is_non_decreasing(res.fitness_trace)
    assert res.fitness_trace[-1] > res.fitness_trace[0]  # the curve climbed


class _CostRunner:
    """A MockRunner that stamps a fixed cost on every evaluation."""

    def __init__(self, fitness_fn, cost):
        self.fitness_fn = fitness_fn
        self.cost = cost
        self.calls = []

    async def evaluate(self, genome, instance, weights=None, *, persist_outcome=True):
        self.calls.append(genome)
        ev = RF.synthetic_evaluation(genome, getattr(instance, "instance_id", "i"), self.fitness_fn(genome))
        return ev.model_copy(update={"total_cost_usd": self.cost})


async def test_total_cost_sums_every_evaluation_not_just_the_winner():
    g, _store = await _saved_genome()
    runner = _CostRunner(RF.regressive_fitness(signature(g)), cost=0.01)
    loop = RearrangementLoop(runner, store=None, registry=default_registry(),
                             k=5, patience=2, max_iters=99, rng=random.Random(7))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    n_evals = len(runner.calls)  # baseline + K per iteration
    assert n_evals > 1
    assert abs(res.total_cost_usd - 0.01 * n_evals) < 1e-9
    # the cumulative spend exceeds any single winning genome's cost (the old undercount)
    assert res.total_cost_usd > res.best_evaluation.total_cost_usd


async def test_never_regresses_when_all_candidates_worse():
    g, _store = await _saved_genome()
    base_sig = signature(g)
    loop = RearrangementLoop(RF.MockRunner(RF.regressive_fitness(base_sig)), store=None,
                             registry=default_registry(), k=5, rng=random.Random(2))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    assert res.adopted_count == 0
    assert set(res.fitness_trace) == {res.fitness_trace[0]}  # perfectly flat — never worse


async def test_stops_on_plateau_via_patience():
    g, _store = await _saved_genome()
    base_sig = signature(g)
    loop = RearrangementLoop(RF.MockRunner(RF.regressive_fitness(base_sig)), store=None,
                             registry=default_registry(), k=5, patience=2, max_iters=99, rng=random.Random(3))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    assert res.iterations == 2  # stopped after PATIENCE no-improve rounds


async def test_bounded_by_max_iters():
    g, store = await _saved_genome()
    loop = RearrangementLoop(RF.MockRunner(RF.climbing_fitness), store=store, registry=default_registry(),
                             k=6, patience=99, max_iters=2, rng=random.Random(4))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    assert res.iterations <= 2


async def test_stops_at_ceiling():
    g, store = await _saved_genome()
    loop = RearrangementLoop(RF.MockRunner(RF.climbing_fitness), store=store, registry=default_registry(),
                             k=6, ceiling=0.45, patience=99, max_iters=99, rng=random.Random(1))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    assert res.best_evaluation.normalized_score >= 0.45  # reached the ceiling
    assert res.iterations <= 3


async def test_always_runs_at_least_one_round_even_when_strong():
    g, _store = await _saved_genome()
    loop = RearrangementLoop(RF.MockRunner(lambda gg: 1.0), store=None, registry=default_registry(),
                             k=5, threshold_stop=True, rng=random.Random(5))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    assert res.iterations >= 1  # unconditional: even a perfect baseline gets a pass
    assert res.cleared_threshold is True


async def test_commits_adopted_mutations_with_lineage():
    g, store = await _saved_genome()
    loop = RearrangementLoop(RF.MockRunner(RF.climbing_fitness), store=store, registry=default_registry(),
                             k=6, rng=random.Random(1))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    final = await store.load(g.genome_id)
    # each adoption is an atomic store mutation with a REARRANGER record
    rearranger_records = [h for h in final.history if h.actor == MutationActor.REARRANGER]
    assert len(rearranger_records) == res.adopted_count
    assert final.version == 1 + res.adopted_count  # version bumped per adoption
    # every record carries BOTH the pre- and post-adoption fitness
    assert all(h.fitness_after is not None for h in rearranger_records)
    assert all(h.fitness_before is not None for h in rearranger_records)
    # the first adoption's fitness_before is the real baseline fitness (0.40), not None
    assert abs(rearranger_records[0].fitness_before - 0.40) < 1e-9
    # records chain: each fitness_after == the next's fitness_before
    for a, b in zip(rearranger_records, rearranger_records[1:]):
        assert abs(a.fitness_after - b.fitness_before) < 1e-9


async def test_commit_survives_a_real_in_flight_version_conflict():
    g, store = await _saved_genome()
    orig_mutate = store.mutate
    state = {"injected": False}

    async def conflicting_mutate(genome_id, expected_version, set_ops, record=None):
        # on the loop's FIRST commit, a concurrent writer wins in-flight: advance
        # the version and report failure (None), forcing retry_mutate to reload.
        if not state["injected"]:
            state["injected"] = True
            await orig_mutate(genome_id, expected_version, {"generation": 777}, None)
            return None
        return await orig_mutate(genome_id, expected_version, set_ops, record)

    store.mutate = conflicting_mutate
    loop = RearrangementLoop(RF.MockRunner(RF.climbing_fitness), store=store, registry=default_registry(),
                             k=6, rng=random.Random(1))
    res = await loop.run(g, INSTANCE, WEIGHTS)  # the first commit hits a conflict and must reload-retry
    assert state["injected"] is True  # the conflict path was actually exercised
    final = await store.load(g.genome_id)
    assert res.adopted_count >= 1
    assert len([h for h in final.history if h.actor == MutationActor.REARRANGER]) == res.adopted_count


async def test_threshold_stop_fires_in_isolation():
    g, _store = await _saved_genome()
    # cleared baseline (0.95), but NOT at the ceiling; only threshold_stop can fire here
    loop = RearrangementLoop(RF.MockRunner(lambda gg: 0.95), store=None, registry=default_registry(),
                             k=5, ceiling=1.0, patience=99, max_iters=99, threshold_stop=True, rng=random.Random(7))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    assert res.iterations == 1  # one mandatory pass, then the threshold stop
    assert res.cleared_threshold is True


def test_normalized_trace_does_not_dip_across_feasibility():
    from darwin.problem.schemas import Solution
    from darwin.rearrange.fixtures import synthetic_breakdown
    from darwin.rearrange.loop import _norm
    from darwin.team.evaluation import GenomeEvaluation
    from darwin.team.genome import ArbiterTier

    # an infeasible solution can score normalized_score=1.0 in B1 (raw_cost 0);
    # _norm must clamp it to 0 so the normalized_trace can't dip when a feasible
    # candidate at 0.8 is adopted.
    sb = synthetic_breakdown("i", 1.0).model_copy(update={"feasible": False, "normalized_score": 1.0})
    ev = GenomeEvaluation(genome_id="g", version=1, instance_id="i", completed=False,
                          final_solution=Solution(solution_id="s", instance_id="i", flows=[]),
                          score_breakdown=sb, fitness=-1.0, normalized_score=1.0, cleared_threshold=False,
                          arbiter_tier_used=ArbiterTier.INFEASIBLE_SENTINEL)
    assert _norm(ev) == 0.0  # infeasible -> 0, not a misleading 1.0


async def test_candidates_are_unpersisted_only_winners_committed():
    runner = RF.MockRunner(RF.climbing_fitness)
    g, store = await _saved_genome()
    loop = RearrangementLoop(runner, store=store, registry=default_registry(), k=4, rng=random.Random(1))
    res = await loop.run(g, INSTANCE, WEIGHTS)
    # every candidate evaluation used persist_outcome=False
    assert all(persist is False for _genome, persist in runner.calls)
    # only the adopted winners landed in the store's history
    final = await store.load(g.genome_id)
    assert len([h for h in final.history if h.actor == MutationActor.REARRANGER]) == res.adopted_count


async def test_never_crashes_on_floor_scored_candidates():
    g, _store = await _saved_genome()
    # baseline feasible at 0.5; every candidate floor-scores (simulated failure)
    fitness = lambda gg: 0.5 if signature(gg) == signature(g) else -1.0e12
    loop = RearrangementLoop(RF.MockRunner(fitness), store=None, registry=default_registry(),
                             k=5, rng=random.Random(6))
    res = await loop.run(g, INSTANCE, WEIGHTS)  # must not raise
    assert res.adopted_count == 0
    assert _is_non_decreasing(res.fitness_trace)
    assert res.fitness_trace[0] == 0.5


async def test_emits_events_each_round():
    g, store = await _saved_genome()
    events = []
    loop = RearrangementLoop(RF.MockRunner(RF.climbing_fitness), store=store, registry=default_registry(),
                             k=6, rng=random.Random(1), event_sink=events.append)
    res = await loop.run(g, INSTANCE, WEIGHTS)
    assert len(events) == res.iterations
    for e in events:
        assert {"iteration", "best_fitness", "normalized_score", "adopted", "mutation_description", "genome_version"} <= set(e)
    assert all(e["mutation_description"] for e in events if e["adopted"])  # adoptions carry a description
