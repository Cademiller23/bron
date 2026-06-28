"""B5/B6 wiring — the efficiency selector + model-aware operators discover the
routing rule, and the penalty is what makes the model gene matter."""

import random

import pytest

from darwin.agent.registry import CapabilityTier, default_registry, reset_default_registry
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import ObjectiveWeights, ScoreBreakdown, Solution
from darwin.problem.scorer import SCORER_VERSION
from darwin.rearrange.loop import RearrangementLoop
from darwin.routing.efficiency import EfficiencyStrategy
from darwin.routing.fleet import FAST, FRONTIER, MID, by_tier, install_fleet
from darwin.routing.gene import MODEL_AWARE_OPERATORS, genotype, model_of
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.fixtures import proposer_checker_arbiter_genome
from darwin.team.genome import ArbiterTier
from darwin.agent.spec import OutputKind

INSTANCE = golden_transportation()
WEIGHTS = ObjectiveWeights.cost_only()


@pytest.fixture(autouse=True)
def _fleet():
    reset_default_registry()
    install_fleet()
    yield
    reset_default_registry()


class QCostRunner:
    """Q depends on a sensible assignment (arbiter FRONTIER, proposers ≥ MID);
    inference cost/latency sum the per-model profile. So the cheapest assignment
    that still clears is: arbiter=FRONTIER, proposers=MID, mechanical=FAST."""

    def __init__(self, registry):
        self.reg = registry

    def _q(self, genome) -> float:
        arb_tier = self.reg.get(model_of(genome, genome.arbiter_id)).capability_tier
        if arb_tier != FRONTIER:
            return 0.70  # the arbiter needs a frontier model to resolve well
        for a in genome.agents:
            if a.agent_id == genome.arbiter_id:
                continue
            if a.spec.output_contract in (OutputKind.FULL_SOLUTION, OutputKind.PARTIAL_SOLUTION):
                if self.reg.get(a.spec.model_id).capability_tier == CapabilityTier.CHEAP:
                    return 0.88  # a too-cheap proposer degrades quality below the gate
        return 0.95

    async def evaluate(self, genome, instance, weights=None, *, persist_outcome=True):
        q = self._q(genome)
        cost = sum(self.reg.get(a.spec.model_id).est_cost_per_1k_out for a in genome.agents)
        latency = sum(self.reg.get(a.spec.model_id).est_latency_ms for a in genome.agents)
        sb = ScoreBreakdown(
            solution_id="s", instance_id=instance.instance_id, feasible=True, violations=[], raw_cost=0.0,
            raw_lead_time=0.0, raw_risk=0.0, weighted_objective=0.0, normalized_score=q, total_penalty=0.0,
            final_fitness=q, objective_weights=WEIGHTS, scorer_version=SCORER_VERSION, computed_at="t",
            diagnostics={},
        )
        return GenomeEvaluation(
            genome_id=genome.genome_id, version=genome.version, instance_id=instance.instance_id,
            completed=True, final_solution=Solution(solution_id="s", instance_id=instance.instance_id, flows=[]),
            score_breakdown=sb, fitness=q, normalized_score=q, cleared_threshold=(q >= 0.90),
            arbiter_tier_used=ArbiterTier.PRIMARY, total_cost_usd=cost, total_latency_ms=latency,
        )


def _checker_id(g):
    return next(a.agent_id for a in g.agents
               if a.spec.output_contract in (OutputKind.CRITIQUE, OutputKind.CONSTRAINT_REPORT))


async def test_efficiency_search_discovers_the_routing_rule():
    reg = default_registry()
    g = proposer_checker_arbiter_genome()  # all on flash (MID) -> arbiter not frontier -> Q 0.70
    loop = RearrangementLoop(
        QCostRunner(reg), store=None, registry=reg, k=8, patience=5, max_iters=30,
        rng=random.Random(7), selector=EfficiencyStrategy(), extra_operators=MODEL_AWARE_OPERATORS,
    )
    res = await loop.run(g, INSTANCE, WEIGHTS)
    final = res.best_genome
    # cleared the gate, and discovered: frontier center, cheap mechanical periphery
    assert res.best_evaluation.normalized_score >= 0.90
    assert model_of(final, final.arbiter_id) in by_tier(FRONTIER)
    chk = _checker_id(final)
    assert model_of(final, chk) in by_tier(FAST)
    # proposers held at >= MID (the quality guard refuses to make them too cheap)
    for a in final.agents:
        if a.spec.output_contract == OutputKind.FULL_SOLUTION:
            assert reg.get(a.spec.model_id).capability_tier in (CapabilityTier.MID, CapabilityTier.FRONTIER)


async def test_penalty_is_what_makes_the_gene_matter():
    reg = default_registry()
    g = proposer_checker_arbiter_genome()
    # RAW mode (no selector): same operators, but no cost penalty.
    raw = RearrangementLoop(
        QCostRunner(reg), store=None, registry=reg, k=8, patience=5, max_iters=30,
        rng=random.Random(7), extra_operators=MODEL_AWARE_OPERATORS,
    )
    raw_res = await raw.run(g, INSTANCE, WEIGHTS)
    rf = raw_res.best_genome
    # raw still upgrades the arbiter (to clear the gate) ...
    assert raw_res.best_evaluation.normalized_score >= 0.90
    assert model_of(rf, rf.arbiter_id) in by_tier(FRONTIER)
    # ... but never trims the mechanical checker (equal Q is no raw improvement):
    assert model_of(rf, _checker_id(rf)) == "gemini-3.5-flash"  # unchanged (MID)


async def test_efficiency_mode_reduces_inference_cost_at_held_quality():
    reg = default_registry()
    g = proposer_checker_arbiter_genome()
    runner = QCostRunner(reg)
    loop = RearrangementLoop(
        runner, store=None, registry=reg, k=8, patience=5, max_iters=30,
        rng=random.Random(3), selector=EfficiencyStrategy(), extra_operators=MODEL_AWARE_OPERATORS,
    )
    res = await loop.run(g, INSTANCE, WEIGHTS)
    # the checker's model got cheaper than the MID flash it started on
    chk = _checker_id(res.best_genome)
    assert reg.get(model_of(res.best_genome, chk)).est_cost_per_1k_out < reg.get("gemini-3.5-flash").est_cost_per_1k_out
    assert res.best_evaluation.normalized_score >= 0.90  # quality held at/above the gate


def test_default_loop_is_unchanged_without_a_selector():
    # sanity: the wiring is opt-in — no selector/extra_operators => the field defaults
    loop = RearrangementLoop(QCostRunner(default_registry()))
    assert loop._selector is None and loop._extra_operators == []


class _InfeasibleRunner:
    """Baseline scores fitness -50 (norm 0.75); every rearranged candidate is CHEAPER
    with a HIGHER normalized_score (0.80) but a WORSE fitness (-120) — the infeasible
    divergence. The efficiency selector must NOT adopt the worse-fitness candidate."""

    def __init__(self, start_sig):
        self.start_sig = start_sig

    def _ev(self, genome, norm, fitness, cost):
        sb = ScoreBreakdown(
            solution_id="s", instance_id=INSTANCE.instance_id, feasible=False, violations=[], raw_cost=0.0,
            raw_lead_time=0.0, raw_risk=0.0, weighted_objective=0.0, normalized_score=norm, total_penalty=1.0,
            final_fitness=fitness, objective_weights=WEIGHTS, scorer_version=SCORER_VERSION, computed_at="t",
            diagnostics={},
        )
        return GenomeEvaluation(
            genome_id=genome.genome_id, version=genome.version, instance_id=INSTANCE.instance_id, completed=False,
            final_solution=Solution(solution_id="s", instance_id=INSTANCE.instance_id, flows=[]), score_breakdown=sb,
            fitness=fitness, normalized_score=norm, cleared_threshold=False, arbiter_tier_used=ArbiterTier.PRIMARY,
            total_cost_usd=cost, total_latency_ms=0.0,
        )

    async def evaluate(self, genome, instance, weights=None, *, persist_outcome=True):
        from darwin.rearrange.operators import signature
        if signature(genome) == self.start_sig:
            return self._ev(genome, norm=0.75, fitness=-50.0, cost=5.0)
        return self._ev(genome, norm=0.80, fitness=-120.0, cost=0.0)  # cheaper, higher norm, worse fitness


async def test_efficiency_loop_keeps_fitness_non_decreasing_on_infeasible_rounds():
    from darwin.rearrange.operators import signature
    reg = default_registry()
    g = proposer_checker_arbiter_genome()
    loop = RearrangementLoop(
        _InfeasibleRunner(signature(g)), store=None, registry=reg, k=6, patience=3, max_iters=12,
        rng=random.Random(1), selector=EfficiencyStrategy(), extra_operators=MODEL_AWARE_OPERATORS,
    )
    res = await loop.run(g, INSTANCE, WEIGHTS)
    # the worse-fitness (but cheaper, higher-normalized) candidates are never adopted
    assert res.adopted_count == 0
    assert res.best_evaluation.fitness == -50.0
    assert all(res.fitness_trace[i] <= res.fitness_trace[i + 1] + 1e-12 for i in range(len(res.fitness_trace) - 1))


# -- B6 conductor: the guarded comparator wired into team-growth elitism -----
class _FakeInstance:
    def __init__(self, pc="transportation", iid="golden-transportation"):
        self.instance_id = iid
        self.problem_class = type("PC", (), {"value": pc})()


async def test_conductor_comparator_keeps_helping_escalation():
    from darwin.escalation.conductor import Conductor
    from darwin.escalation.schemas import EscalationMethod, SolveStatus
    from darwin.escalation.fixtures import (
        MockConductorArchitect, MockEscalator, MockRearrangementLoop, RecordingCorpus, base_genome,
    )

    cond = Conductor(
        MockConductorArchitect(base_genome()),
        MockRearrangementLoop(lambda g: 0.95 if len(g.agents) > 4 else 0.70),
        MockEscalator(method=EscalationMethod.CORPUS), RecordingCorpus(),
        comparator=EfficiencyStrategy().improves,
    )
    res = await cond.solve(_FakeInstance())
    assert res.status == SolveStatus.SEALED and res.escalation_rounds == 1
    assert len(res.agents_added) == 1


async def test_conductor_comparator_rolls_back_unhelpful_escalation():
    from darwin.escalation.conductor import Conductor
    from darwin.escalation.schemas import EscalationMethod, SolveBudget, SolveStatus
    from darwin.escalation.fixtures import (
        MockConductorArchitect, MockEscalator, MockRearrangementLoop, RecordingCorpus, base_genome,
    )

    cond = Conductor(
        MockConductorArchitect(base_genome()),
        MockRearrangementLoop(lambda g: 0.70),  # never clears, never improves
        MockEscalator(method=EscalationMethod.CORPUS), RecordingCorpus(),
        comparator=EfficiencyStrategy().improves,
    )
    res = await cond.solve(_FakeInstance(), budget=SolveBudget(max_escalations=2))
    assert res.status == SolveStatus.EXHAUSTED
    assert res.agents_added == []  # comparator rejected the non-improving growth
    assert len(res.final_genome.agents) == 4
