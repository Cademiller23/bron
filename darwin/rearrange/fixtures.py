"""Deterministic B5 test doubles — a mock runner whose fitness depends on the
genome's structure, so a known rearrangement is known to improve the score."""

from datetime import datetime, timezone
from typing import Callable, List, Optional

from darwin.problem.schemas import ObjectiveWeights, ScoreBreakdown, Solution
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import ArbiterTier, TeamGenome

FRONTIER = "gemini-3.1-pro"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def synthetic_breakdown(instance_id: str, fitness: float) -> ScoreBreakdown:
    feasible = fitness >= 0.0
    normalized = max(0.0, min(1.0, fitness)) if feasible else 0.0
    from darwin.problem.scorer import SCORER_VERSION as _ver

    return ScoreBreakdown(
        solution_id="mock", instance_id=instance_id, feasible=feasible, violations=[],
        raw_cost=0.0, raw_lead_time=0.0, raw_risk=0.0, weighted_objective=0.0,
        normalized_score=normalized, total_penalty=0.0 if feasible else abs(fitness),
        final_fitness=fitness, objective_weights=ObjectiveWeights.cost_only(),
        scorer_version=_ver, computed_at=_now(), diagnostics={"mock": True},
    )


def synthetic_evaluation(genome: TeamGenome, instance_id: str, fitness: float) -> GenomeEvaluation:
    sb = synthetic_breakdown(instance_id, fitness)
    return GenomeEvaluation(
        genome_id=genome.genome_id, version=genome.version, instance_id=instance_id,
        completed=sb.feasible, final_solution=Solution(solution_id="mock", instance_id=instance_id, flows=[]),
        score_breakdown=sb, fitness=sb.final_fitness, normalized_score=sb.normalized_score,
        cleared_threshold=(sb.feasible and sb.normalized_score >= 0.90),
        arbiter_tier_used=ArbiterTier.PRIMARY,
    )


class MockRunner:
    """Returns a GenomeEvaluation whose fitness is a pure function of the genome."""

    def __init__(self, fitness_fn: Callable[[TeamGenome], float]) -> None:
        self.fitness_fn = fitness_fn
        self.calls: List[tuple] = []  # (genome, persist_outcome)

    async def evaluate(self, genome, instance, weights=None, *, persist_outcome: bool = True):
        self.calls.append((genome, persist_outcome))
        return synthetic_evaluation(genome, getattr(instance, "instance_id", "i"), self.fitness_fn(genome))


# ---------------------------------------------------------------------------
# Fitness functions
# ---------------------------------------------------------------------------
def climbing_fitness(genome: TeamGenome) -> float:
    """Climbs as more agents move to the frontier model (a multi-step climb)."""
    pro = sum(1 for a in genome.agents if a.spec.model_id == FRONTIER)
    return round(0.40 + 0.10 * pro, 6)


def regressive_fitness(baseline_signature: str) -> Callable[[TeamGenome], float]:
    """Baseline is best; ANY rearrangement scores lower (tests elitism)."""
    from darwin.rearrange.operators import signature

    def fn(genome: TeamGenome) -> float:
        return 0.70 if signature(genome) == baseline_signature else 0.30

    return fn


def ceiling_fitness(genome: TeamGenome) -> float:
    """Reaches the ceiling once any agent is on the frontier model."""
    return 0.995 if any(a.spec.model_id == FRONTIER for a in genome.agents) else 0.50
