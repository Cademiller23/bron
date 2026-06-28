"""The RearrangementLoop — the always-on inner loop (rearrange, never regress).

After the initial team, the system ALWAYS attempts to improve the arrangement —
even a strong team gets at least one pass. Each round generates K rearranged
candidates (programmatically), evaluates them concurrently under B3's shared
inference gate, and adopts the best **only if it strictly beats the current best**
(elitism). So the team can never get worse and the fitness curve is monotonically
non-decreasing — the live "it's getting better" story.
"""

import asyncio
import inspect
import logging
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from darwin.constants import (
    REARRANGE_CEILING,
    REARRANGE_EPSILON,
    REARRANGE_K,
    REARRANGE_MAX_ITERS,
    REARRANGE_PATIENCE,
    SCORE_THRESHOLD,
)
from darwin.problem.schemas import ObjectiveWeights, ProblemInstance
from darwin.rearrange.generator import generate_candidates
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import TeamGenome

logger = logging.getLogger("darwin.rearrange")


def _norm(evaluation: GenomeEvaluation) -> float:
    """normalized_score for the trace, clamped to 0 when infeasible — so the
    normalized_trace can't dip across the feasibility boundary (an infeasible
    solution can score normalized 1.0 in B1, which would otherwise look like a
    regression when a feasible candidate at 0.8 is adopted)."""
    return evaluation.normalized_score if evaluation.score_breakdown.feasible else 0.0


class RearrangementResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    best_genome: TeamGenome
    best_evaluation: GenomeEvaluation
    fitness_trace: List[float] = Field(default_factory=list)  # the climbing curve
    normalized_trace: List[float] = Field(default_factory=list)
    adopted_count: int = 0
    iterations: int = 0
    cleared_threshold: bool = False
    total_cost_usd: float = 0.0  # cumulative cost of EVERY evaluation (baseline + all candidates)


class RearrangementLoop:
    def __init__(
        self,
        runner: Any,
        store: Any = None,
        registry: Any = None,
        *,
        k: int = REARRANGE_K,
        patience: int = REARRANGE_PATIENCE,
        max_iters: int = REARRANGE_MAX_ITERS,
        ceiling: float = REARRANGE_CEILING,
        epsilon: float = REARRANGE_EPSILON,
        threshold_stop: bool = False,
        reorganizer: Optional[Callable] = None,
        event_sink: Optional[Callable] = None,
        rng: Any = None,
    ) -> None:
        self.runner = runner
        self.store = store
        self.registry = registry
        self._k = k
        self._patience = patience
        self._max_iters = max_iters
        self._ceiling = ceiling
        self._epsilon = epsilon
        self._threshold_stop = threshold_stop
        self.reorganizer = reorganizer
        self.event_sink = event_sink
        import random as _random

        self._rng = rng or _random.Random()

    async def run(
        self, genome: TeamGenome, instance: ProblemInstance, weights: Optional[ObjectiveWeights] = None
    ) -> RearrangementResult:
        weights = weights or ObjectiveWeights.cost_only()

        # Baseline (transient evaluation — not persisted).
        best_eval = await self.runner.evaluate(genome, instance, weights, persist_outcome=False)
        best_genome = genome
        fitness_trace = [best_eval.fitness]
        normalized_trace = [_norm(best_eval)]
        adopted = 0
        no_improve = 0
        iteration = 0
        total_cost = best_eval.total_cost_usd  # accrue EVERY evaluation's spend (not just the winner)

        while True:
            hints = await self._hints(best_eval.score_breakdown)
            candidates = generate_candidates(
                best_genome, self._k, rng=self._rng, hints=hints, registry=self.registry
            )
            if not candidates:
                break  # nothing valid to try (degenerate genome)

            # Evaluate all K candidates concurrently under the shared gate; each
            # never raises (a bad candidate just scores at the floor).
            evals = await asyncio.gather(
                *[self.runner.evaluate(c.genome, instance, weights, persist_outcome=False) for c in candidates]
            )
            total_cost += sum(e.total_cost_usd for e in evals)
            best_idx = max(range(len(candidates)), key=lambda i: evals[i].fitness)
            candidate, candidate_eval = candidates[best_idx], evals[best_idx]
            iteration += 1

            if candidate_eval.fitness > best_eval.fitness + self._epsilon:  # ELITISM: strict improvement only
                # record the true pre-adoption fitness as fitness_before
                best_genome = await self._commit(best_genome, candidate, candidate_eval, best_eval.fitness)
                best_eval = candidate_eval
                adopted += 1
                no_improve = 0
                fitness_trace.append(best_eval.fitness)
                normalized_trace.append(_norm(best_eval))
                self._emit(iteration, best_eval, True, candidate.description, best_genome.version)
            else:
                no_improve += 1
                fitness_trace.append(best_eval.fitness)  # plateau: the curve never dips
                normalized_trace.append(_norm(best_eval))
                self._emit(iteration, best_eval, False, candidate.description, best_genome.version)

            if self._should_stop(no_improve, iteration, best_eval):
                break

        return RearrangementResult(
            best_genome=best_genome,
            best_evaluation=best_eval,
            fitness_trace=fitness_trace,
            normalized_trace=normalized_trace,
            adopted_count=adopted,
            iterations=iteration,
            cleared_threshold=best_eval.cleared_threshold,
            total_cost_usd=total_cost,
        )

    # -- stop conditions ----------------------------------------------------
    def _should_stop(self, no_improve: int, iteration: int, best_eval: GenomeEvaluation) -> bool:
        if no_improve >= self._patience:
            return True
        if iteration >= self._max_iters:
            return True
        if best_eval.score_breakdown.feasible and best_eval.normalized_score >= self._ceiling:
            return True
        if self._threshold_stop and best_eval.cleared_threshold and best_eval.normalized_score >= SCORE_THRESHOLD:
            return True
        return False

    # -- commit (optimistic-locked, conflict-safe) --------------------------
    async def _commit(
        self, best_genome: TeamGenome, candidate, candidate_eval: GenomeEvaluation, fitness_before: float
    ) -> TeamGenome:
        if self.store is None:
            return candidate.genome  # in-memory adoption (no persistence)

        from darwin.team.genome import GenomeStatus

        status = GenomeStatus.CLEARED_THRESHOLD if candidate_eval.cleared_threshold else GenomeStatus.EVALUATED

        def derive(current: TeamGenome):
            # apply the structural rearrangement AND refresh the score fields so the
            # persisted genome's current_fitness/status aren't left stale; inject
            # the true pre-adoption fitness_before + the new fitness_after onto the
            # lineage record (the operator's derive can't know either).
            set_ops, record = candidate.derive_fn(current)
            set_ops = {
                **set_ops,
                "current_fitness": candidate_eval.fitness,
                "current_normalized_score": candidate_eval.normalized_score,
                "status": status.value,
            }
            record = record.model_copy(
                update={"fitness_before": fitness_before, "fitness_after": candidate_eval.fitness}
            )
            return set_ops, record

        try:
            return await self.store.retry_mutate(best_genome.genome_id, derive)
        except Exception as exc:  # noqa: BLE001 - commit is best-effort; keep the in-memory winner
            logger.warning("rearrangement commit failed for %s (keeping in-memory): %s", best_genome.genome_id, exc)
            return candidate.genome

    async def _hints(self, score_breakdown):
        if self.reorganizer is None:
            return None
        try:
            result = self.reorganizer(score_breakdown)
            if inspect.isawaitable(result):
                result = await result
            return result
        except Exception:  # noqa: BLE001 - steering is optional; never break the loop
            return None

    def _emit(self, iteration, ev, adopted, description, version):
        if self.event_sink is None:
            return
        self.event_sink(
            {
                "iteration": iteration,
                "best_fitness": ev.fitness,
                "normalized_score": ev.normalized_score,
                "adopted": adopted,
                "mutation_description": description if adopted else None,
                "genome_version": version,
            }
        )
