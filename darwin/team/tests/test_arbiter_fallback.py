"""§12.2 [MANDATORY] Pre-flight test #2 — the three-tier arbiter fallback.

Gates every run: a dead arbiter never kills the team's answer, and evaluation
always yields a real scored fitness without raising.
"""

import math

import pytest

from darwin.team import fixtures as F
from darwin.team.genome import ArbiterTier, MutationType
from darwin.team.inference_gate import InferenceGate
from darwin.team.runner import TeamRunner
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import ObjectiveWeights

INSTANCE = golden_transportation()
WEIGHTS = ObjectiveWeights.cost_only()


async def _evaluate(scripts):
    genome = F.proposer_checker_arbiter_genome()
    store = F.new_store()
    await store.save_new(genome)
    factory = F.scripted_worker_factory(scripts)
    runner = TeamRunner(
        model_client=None, telemetry=F.telemetry_sink(), inference_gate=InferenceGate(8),
        store=store, worker_factory=factory,
    )
    evaluation = await runner.evaluate(genome, INSTANCE, WEIGHTS)
    after = await store.load(genome.genome_id)
    return evaluation, after, factory


def _base_scripts():
    return {
        "p1": F.agent_result("p1", F.full_solution_output(F.optimal_solution())),  # feasible, optimal
        "p2": F.agent_result("p2", F.full_solution_output(F.suboptimal_solution())),  # feasible, pricier
        "chk": F.agent_result("chk", F.critique_output()),
    }


async def test_tier1_retry_then_success_is_RETRY():
    scripts = _base_scripts()
    # arbiter fails once, then succeeds on retry
    scripts["arb"] = [
        F.agent_result("arb", None, success=False, error="hiccup"),
        F.agent_result("arb", F.arbitration_output(F.optimal_solution())),
    ]
    ev, after, factory = await _evaluate(scripts)
    assert ev.arbiter_tier_used == ArbiterTier.RETRY
    assert ev.completed is True
    assert factory.workers["arb"].calls == 2  # one retry
    assert math.isclose(ev.normalized_score, 1.0)


async def test_tier1_exhausts_retries_at_three_attempts():
    scripts = _base_scripts()
    scripts["arb"] = F.agent_result("arb", None, success=False, error="always down")  # never succeeds
    ev, after, factory = await _evaluate(scripts)
    assert factory.workers["arb"].calls == 3  # 1 + ARBITER_MAX_RETRIES(2)
    # feasible proposers present -> deterministically falls through to Tier 2
    assert ev.arbiter_tier_used == ArbiterTier.BEST_PROPOSAL_FALLBACK


async def test_tier2_best_feasible_proposal():
    scripts = _base_scripts()
    scripts["arb"] = F.agent_result("arb", None, success=False, error="dead arbiter")
    ev, after, factory = await _evaluate(scripts)
    assert ev.arbiter_tier_used == ArbiterTier.BEST_PROPOSAL_FALLBACK
    assert ev.completed is False
    # the optimal proposal (p1) must be chosen over the pricier p2
    assert math.isclose(ev.normalized_score, 1.0)
    assert ev.final_solution.flows  # a real proposal, not the sentinel
    assert any(h.mutation_type == MutationType.ARBITER_FALLBACK_USED for h in after.history)


async def test_tier3_infeasible_sentinel_when_no_usable_proposal():
    scripts = {
        "p1": F.agent_result("p1", None, success=False, error="x"),
        "p2": F.agent_result("p2", None, success=False, error="x"),
        "chk": F.agent_result("chk", F.critique_output()),
        "arb": F.agent_result("arb", None, success=False, error="dead arbiter"),
    }
    ev, after, factory = await _evaluate(scripts)
    assert ev.arbiter_tier_used == ArbiterTier.INFEASIBLE_SENTINEL
    assert ev.score_breakdown.feasible is False
    assert ev.fitness < 0  # floored / infeasible
    assert ev.cleared_threshold is False
    assert any(h.mutation_type == MutationType.ARBITER_FALLBACK_USED for h in after.history)


async def test_no_tier_ever_raises_and_fitness_is_always_real():
    # All four scenarios return a GenomeEvaluation with a finite, real fitness.
    scenarios = [
        {**_base_scripts(), "arb": F.agent_result("arb", F.arbitration_output(F.optimal_solution()))},  # PRIMARY
        {**_base_scripts(), "arb": F.agent_result("arb", None, success=False, error="x")},  # Tier 2
        {  # Tier 3
            "p1": F.agent_result("p1", None, success=False, error="x"),
            "p2": F.agent_result("p2", None, success=False, error="x"),
            "chk": F.agent_result("chk", F.critique_output()),
            "arb": F.agent_result("arb", None, success=False, error="x"),
        },
    ]
    for scripts in scenarios:
        ev, _after, _f = await _evaluate(scripts)  # must not raise
        assert math.isfinite(ev.fitness)
        assert ev.score_breakdown is not None
