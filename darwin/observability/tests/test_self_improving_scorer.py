"""The self-improving scorer — calibration, re-tune recovery, persistence, firewall."""

import inspect

from darwin.observability import self_improving_scorer as SIS
from darwin.observability.bus import EventBus
from darwin.observability.emitter import EventEmitter
from darwin.observability.events import RunEventType
from darwin.observability.fixtures import FakeEventCollection
from darwin.observability.self_improving_scorer import (
    CalibrationSample,
    SelfImprovingScorer,
    bump_version,
    correlation,
    retune,
    spearman,
)
from darwin.observability.store import EventStore
from darwin.problem.schemas import ObjectiveWeights

# Truth is governed by RISK (true_quality = -risk); cost is ANTI-correlated with
# risk. So a cost-weighted scorer mis-ranks (negative correlation) while a
# risk-weighted scorer is perfect — the oracle anchor the loop must recover.
SAMPLES = [
    CalibrationSample(cost=0.1, lead_time=0.5, risk=0.9, true_quality=-0.9),
    CalibrationSample(cost=0.2, lead_time=0.4, risk=0.8, true_quality=-0.8),
    CalibrationSample(cost=0.3, lead_time=0.5, risk=0.7, true_quality=-0.7),
    CalibrationSample(cost=0.4, lead_time=0.6, risk=0.6, true_quality=-0.6),
    CalibrationSample(cost=0.6, lead_time=0.4, risk=0.4, true_quality=-0.4),
    CalibrationSample(cost=0.7, lead_time=0.5, risk=0.3, true_quality=-0.3),
    CalibrationSample(cost=0.8, lead_time=0.6, risk=0.2, true_quality=-0.2),
    CalibrationSample(cost=0.9, lead_time=0.5, risk=0.1, true_quality=-0.1),
]


# -- spearman ---------------------------------------------------------------
def test_spearman_perfect_positive_and_negative():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9


def test_spearman_with_ties():
    assert abs(spearman([1, 1, 2, 2], [1, 1, 2, 2]) - 1.0) < 1e-9


def test_spearman_degenerate_is_zero():
    assert spearman([1], [1]) == 0.0
    assert spearman([5, 5, 5], [1, 2, 3]) == 0.0  # no spread in xs


def test_bump_version():
    assert bump_version("1.0.0") == "1.0.1"
    assert bump_version("2.3.9") == "2.3.10"
    assert bump_version("weird") == "weird.1"


# -- calibration ------------------------------------------------------------
def test_calibration_detects_misweighted_scorer():
    cost_corr = correlation(SAMPLES, ObjectiveWeights.cost_only())
    risk_corr = correlation(SAMPLES, ObjectiveWeights(cost_weight=0, lead_time_weight=0, risk_weight=1))
    assert cost_corr < 0.0          # cost ranks against the truth
    assert risk_corr > 0.99         # risk ranks with the truth


# -- re-tune ----------------------------------------------------------------
def test_retune_recovers_risk_heavy_weights():
    best_w, best_c = retune(SAMPLES, grid_steps=8)
    assert best_c > 0.99
    assert best_w.risk_weight > best_w.cost_weight
    assert best_w.risk_weight > best_w.lead_time_weight


async def test_maybe_retune_fixes_a_degraded_scorer_and_persists():
    store = EventStore(FakeEventCollection(), FakeEventCollection())
    bus = EventBus()
    sub = bus.subscribe()
    emitter = EventEmitter("run-sis", store, bus)
    sis = SelfImprovingScorer(ObjectiveWeights.cost_only(), store=store, emitter=emitter, scorer_version="1.0.0")

    result = await sis.maybe_retune(SAMPLES)

    assert result.retuned is True
    assert result.correlation_after > result.correlation_before  # the predictive-validity curve rose
    assert result.correlation_after > 0.99
    assert sis.scorer_version == "1.0.1"          # version bumped (B1 stamps every score with it)
    assert sis.weights.risk_weight > sis.weights.cost_weight
    # persisted to scorer_versions
    versions = await store.load_scorer_versions()
    assert len(versions) == 1 and versions[0]["scorer_version"] == "1.0.1"
    # emitted SCORER_RETUNED
    assert (await sub.get(timeout=1.0)).event_type == RunEventType.SCORER_RETUNED


async def test_maybe_retune_skips_a_well_calibrated_scorer():
    sis = SelfImprovingScorer(ObjectiveWeights(cost_weight=0, lead_time_weight=0, risk_weight=1))
    result = await sis.maybe_retune(SAMPLES)
    assert result.retuned is False and result.scorer_version == "1.0.0"


async def test_maybe_retune_skips_with_too_few_samples():
    sis = SelfImprovingScorer(ObjectiveWeights.cost_only(), min_samples=8)
    result = await sis.maybe_retune(SAMPLES[:3])
    assert result.retuned is False and "insufficient" in result.reason


async def test_maybe_retune_is_failure_safe_against_a_broken_store():
    # regression: a Mongo hiccup in scorer_versions persistence must degrade, not crash
    class BrokenStore:
        async def save_scorer_version(self, record):
            raise RuntimeError("mongo down")

    sis = SelfImprovingScorer(ObjectiveWeights.cost_only(), store=BrokenStore())
    result = await sis.maybe_retune(SAMPLES)  # must not raise
    assert result.retuned is True and sis.weights.risk_weight > sis.weights.cost_weight


async def test_retune_frequency_is_bounded():
    sis = SelfImprovingScorer(ObjectiveWeights.cost_only(), store=None, max_retunes=1)
    first = await sis.maybe_retune(SAMPLES)
    assert first.retuned is True
    # force a degraded state again; the budget is exhausted so it must not re-tune
    sis.weights = ObjectiveWeights.cost_only()
    second = await sis.maybe_retune(SAMPLES)
    assert second.retuned is False and "budget" in second.reason


# -- the firewall (anchored to the oracle, never an LLM) --------------------
def test_meta_loop_contains_no_model_call():
    # Structural firewall: the meta-loop must depend on NO model/LLM layer — it is
    # anchored to the oracle truth only. Check actual imports (AST) + call sites,
    # not docstring prose (which legitimately discusses "never an LLM").
    import ast

    src = inspect.getsource(SIS)
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, ast.Import):
            imported.update(n.name for n in node.names)
    assert not any("agent" in m for m in imported), f"must not import the model layer: {imported}"
    assert not any("provider" in m for m in imported)
    assert not any("routing" in m for m in imported)
    # no model client construction or completion call anywhere in the code
    assert "ModelClient" not in src and ".complete(" not in src


def test_primary_scorer_stays_deterministic_under_a_retuned_weight():
    # the meta-loop only returns weights; B1's score() with any weight is pure math
    from darwin.problem import oracle
    from darwin.problem.fixtures import golden_transportation
    from darwin.problem.scorer import score

    inst = golden_transportation()
    sol = oracle.solve_optimum(inst).solution
    w = ObjectiveWeights(cost_weight=0.2, lead_time_weight=0.3, risk_weight=0.5)
    a = score(inst, sol, w)
    b = score(inst, sol, w)
    assert a.final_fitness == b.final_fitness and a.normalized_score == b.normalized_score
