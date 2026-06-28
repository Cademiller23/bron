"""Conductor — the whole brain: gate, team-growth elitism, rollback, budgets,
promotion, the solve boundary, and live events."""

from darwin.escalation.conductor import Conductor
from darwin.escalation.schemas import EscalationMethod, SolveBudget, SolveStatus
from darwin.escalation.fixtures import (
    MockConductorArchitect,
    MockEscalator,
    MockRearrangementLoop,
    RecordingCorpus,
    base_genome,
)
from darwin.team.fixtures import new_store


class FakeInstance:
    def __init__(self, pc="transportation", iid="golden-transportation"):
        self.instance_id = iid
        self.problem_class = type("PC", (), {"value": pc})()


def fitness_by_size(mapping, default):
    return lambda g: mapping.get(len(g.agents), default)


def _conductor(fitness_fn, *, escalator=None, corpus=None, store=None, event_sink=None,
               architect=None):
    return Conductor(
        architect or MockConductorArchitect(base_genome()),
        MockRearrangementLoop(fitness_fn),
        escalator or MockEscalator(method=EscalationMethod.CORPUS),
        corpus or RecordingCorpus(),
        store=store, event_sink=event_sink,
    )


# ---------------------------------------------------------------------------
async def test_clears_without_escalation():
    cond = _conductor(lambda g: 0.95)
    res = await cond.solve(FakeInstance())
    assert res.status == SolveStatus.SEALED
    assert res.cleared_threshold is True
    assert res.escalation_rounds == 0
    assert res.agents_added == []


async def test_single_escalation_clears_and_is_kept():
    corpus = RecordingCorpus()
    cond = _conductor(fitness_by_size({4: 0.7}, 0.95), corpus=corpus)
    res = await cond.solve(FakeInstance())
    assert res.status == SolveStatus.SEALED
    assert res.escalation_rounds == 1
    assert len(res.agents_added) == 1
    assert res.corpus_hits == 1
    # a kept corpus reuse records a SUCCESS stat
    assert corpus.stats and corpus.stats[-1][2] is True
    # the final team grew by one
    assert len(res.final_genome.agents) == 5


async def test_unhelpful_escalation_is_rolled_back():
    corpus = RecordingCorpus()
    cond = _conductor(lambda g: 0.7, corpus=corpus,  # never improves
                      escalator=MockEscalator(method=EscalationMethod.CORPUS))
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_escalations=2))
    assert res.status == SolveStatus.EXHAUSTED
    assert res.escalation_rounds == 2
    assert res.agents_added == []  # nothing kept
    assert len(res.final_genome.agents) == 4  # back to the original team
    # each rolled-back corpus reuse records a FAILURE stat
    assert all(s[2] is False for s in corpus.stats) and len(corpus.stats) == 2
    assert all(r["kept"] is False for r in res.per_round_summary)


async def test_curated_useful_agent_is_promoted():
    corpus = RecordingCorpus()
    cond = _conductor(fitness_by_size({4: 0.7}, 0.95), corpus=corpus,
                      escalator=MockEscalator(method=EscalationMethod.CURATED))
    res = await cond.solve(FakeInstance())
    assert res.status == SolveStatus.SEALED
    assert res.corpus_promotions == 1
    assert corpus.promotions and corpus.promotions[0][2] == "transportation"


async def test_curated_unhelpful_agent_is_not_promoted():
    corpus = RecordingCorpus()
    cond = _conductor(lambda g: 0.7, corpus=corpus,
                      escalator=MockEscalator(method=EscalationMethod.CURATED))
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_escalations=1))
    assert res.corpus_promotions == 0
    assert corpus.promotions == []


async def test_max_escalations_budget_caps_rounds():
    cond = _conductor(lambda g: 0.7)
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_escalations=3))
    assert res.escalation_rounds == 3
    assert res.status == SolveStatus.EXHAUSTED


async def test_max_team_size_budget_halts_growth():
    cond = _conductor(fitness_by_size({4: 0.7, 5: 0.75}, 0.8))
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_team_size=5, max_escalations=10))
    assert len(res.final_genome.agents) == 5  # stopped at the cap
    assert res.escalation_rounds == 1


async def test_none_available_breaks_loop():
    cond = _conductor(lambda g: 0.7, escalator=MockEscalator(none_after=0))
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_escalations=5))
    assert res.escalation_rounds == 0
    assert res.agents_added == []
    assert res.status == SolveStatus.EXHAUSTED


async def test_escalator_crash_is_contained():
    cond = _conductor(lambda g: 0.7, escalator=MockEscalator(fail=True))
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_escalations=3))
    # one bad escalation stops growth but still returns a best-so-far result
    assert res.status == SolveStatus.EXHAUSTED
    assert res.escalation_rounds == 0
    assert res.final_evaluation.fitness == 0.7


async def test_solve_boundary_returns_floor_on_design_crash():
    cond = _conductor(lambda g: 0.95, architect=MockConductorArchitect(base_genome(), design_fail=True))
    res = await cond.solve(FakeInstance())
    assert res.status == SolveStatus.EXHAUSTED
    assert res.cleared_threshold is False
    assert res.error is not None
    assert res.final_genome is not None  # safe-default team


async def test_floor_result_never_raises_even_when_safe_default_fails():
    # regression: design crashes AND _safe_default raises (empty registry) — solve
    # must STILL return a floor SolveResult, not propagate the exception.
    arch = MockConductorArchitect(base_genome(), design_fail=True, safe_default_fail=True)
    cond = _conductor(lambda g: 0.95, architect=arch)
    res = await cond.solve(FakeInstance())  # must not raise
    assert res.status == SolveStatus.EXHAUSTED
    assert res.cleared_threshold is False
    assert res.error is not None
    assert res.final_genome is not None  # minimal inline floor team
    assert res.final_evaluation.fitness <= -1e11  # floored


async def test_floor_result_tolerates_empty_instance_id():
    # regression: an empty instance_id must not make Solution(min_length=1) raise
    # inside the solve boundary. solve() must still return a floor SolveResult.
    arch = MockConductorArchitect(base_genome(), design_fail=True, safe_default_fail=True)
    cond = _conductor(lambda g: 0.95, architect=arch)
    res = await cond.solve(FakeInstance(iid=""))  # empty id; must not raise
    assert res.status == SolveStatus.EXHAUSTED
    assert res.instance_id == "unknown"
    assert res.final_genome is not None


async def test_cost_budget_halts_growth():
    # regression: the cost guard sees the WHOLE rearrange spend (cost_per_run),
    # not just the winning candidate, so it actually halts.
    cond = Conductor(
        MockConductorArchitect(base_genome()),
        MockRearrangementLoop(lambda g: 0.7, cost_per_run=3.0),
        MockEscalator(method=EscalationMethod.CORPUS), RecordingCorpus(),
    )
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_total_cost_usd=5.0, max_escalations=10))
    # initial rearrange = 3.0; one escalation round adds 3.0 -> 6.0 >= 5.0 -> stop
    assert res.escalation_rounds == 1
    assert res.total_cost_usd >= 5.0
    assert res.status == SolveStatus.EXHAUSTED


async def test_trace_and_markers_track_escalations():
    cond = _conductor(fitness_by_size({4: 0.7}, 0.95))
    res = await cond.solve(FakeInstance())
    assert len(res.full_fitness_trace) >= 2  # initial + post-escalation
    assert res.trace_markers and "agent added" in res.trace_markers[0]["label"]
    assert res.per_round_summary[0]["kept"] is True


async def test_events_emitted():
    events = []
    cond = _conductor(fitness_by_size({4: 0.7}, 0.95), event_sink=events.append)
    await cond.solve(FakeInstance())
    types = {e["event_type"] for e in events}
    assert "THRESHOLD_CHECK" in types
    assert "ESCALATION" in types
    assert any(e.get("cleared") for e in events if e["event_type"] == "THRESHOLD_CHECK")


async def test_rollback_persists_remove_agent_record():
    store = new_store()
    genome = base_genome()
    await store.save_new(genome)
    cond = _conductor(lambda g: 0.7, store=store,
                      architect=MockConductorArchitect(genome),
                      escalator=MockEscalator(method=EscalationMethod.CORPUS))
    res = await cond.solve(FakeInstance(), budget=SolveBudget(max_escalations=1))
    assert res.escalation_rounds == 1 and res.agents_added == []
    reloaded = await store.load(genome.genome_id)
    assert reloaded.history[-1].mutation_type.value == "REMOVE_AGENT"
    assert reloaded.version == genome.version + 1


async def test_total_latency_is_populated():
    ticks = iter([100.0, 100.5])
    cond = Conductor(
        MockConductorArchitect(base_genome()), MockRearrangementLoop(lambda g: 0.95),
        MockEscalator(), RecordingCorpus(), clock=lambda: next(ticks),
    )
    res = await cond.solve(FakeInstance())
    assert res.total_latency_ms == 500.0
