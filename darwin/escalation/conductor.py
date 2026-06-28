"""The top-level conductor — the whole brain in one call.

``solve(instance, weights, budget) -> SolveResult`` realizes the whiteboard
exactly: B4 design -> B5 always rearrange -> threshold check -> (below) B6
escalate (corpus, then curate) -> B5 again -> repeat until cleared or the budget
is exhausted. Team-growth elitism keeps an added agent only if it improved the
score (else it's rolled back); useful curated agents are promoted to the corpus.

The entire body runs inside a solve boundary: any unexpected error returns the
best-so-far. The brain always produces a scored answer — degraded at worst,
never dead.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from darwin.constants import ESCALATION_EPSILON, GENOME_FLOOR_FITNESS, SCORE_THRESHOLD
from darwin.escalation.schemas import (
    EscalationMethod,
    SolveBudget,
    SolveResult,
    SolveStatus,
)
from darwin.problem.schemas import ObjectiveWeights, ProblemInstance, ScoreBreakdown, Solution
from darwin.problem.scorer import SCORER_VERSION
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import ArbiterTier, MutationActor, MutationRecord, MutationType, TeamGenome

logger = logging.getLogger("darwin.escalation.conductor")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Conductor:
    def __init__(
        self, architect: Any, rearrangement_loop: Any, escalator: Any, corpus: Any,
        store: Any = None, *, event_sink: Optional[Callable] = None,
        threshold: float = SCORE_THRESHOLD, epsilon: float = ESCALATION_EPSILON,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.architect = architect
        self.rearrangement_loop = rearrangement_loop
        self.escalator = escalator
        self.corpus = corpus
        self.store = store
        self.event_sink = event_sink
        self._threshold = threshold
        self._epsilon = epsilon
        self._clock = clock

    async def solve(
        self, instance: ProblemInstance, weights: Optional[ObjectiveWeights] = None,
        budget: Optional[SolveBudget] = None,
    ) -> SolveResult:
        weights = weights or ObjectiveWeights.cost_only()
        budget = budget or SolveBudget()
        try:
            return await self._solve(instance, weights, budget)
        except Exception as exc:  # noqa: BLE001 - the brain always returns a result
            logger.exception("solve boundary caught an error; returning a floor result")
            return self._floor_result(instance, weights, error=f"{type(exc).__name__}: {exc}")

    # =======================================================================
    async def _solve(self, instance, weights, budget) -> SolveResult:
        start = self._clock()

        # B4 — design the initial team (never raises).
        genome = await self.architect.design_initial_team(instance, weights)

        # B5 — always rearrange.
        result = await self._rearrange(genome, instance, weights)
        best_genome = result.best_genome
        best_eval = result.best_evaluation

        full_trace: List[float] = list(result.fitness_trace)
        trace_markers: List[Dict[str, Any]] = []
        agents_added: List[Dict[str, Any]] = []
        per_round: List[Dict[str, Any]] = []
        corpus_hits = 0
        corpus_promotions = 0
        rounds = 0
        total_cost = result.total_cost_usd  # true spend across the whole rearrange, not just the winner

        self._emit_threshold(best_eval)

        # outer escalation loop (conditional — grow only when reshaping can't clear 90%)
        while (
            not best_eval.cleared_threshold
            and rounds < budget.max_escalations
            and len(best_genome.agents) < budget.max_team_size
            and (self._clock() - start) < budget.max_wall_clock_seconds
            and total_cost < budget.max_total_cost_usd
        ):
            snapshot = best_genome
            pre_fitness = best_eval.fitness

            # B6 — escalate (corpus first, then curate). Never raises.
            try:
                esc = await self.escalator.escalate(best_genome, instance, weights, best_eval)
            except Exception as exc:  # noqa: BLE001 - one bad escalation must not crash the solve
                logger.warning("escalation round errored (stopping growth): %s", exc)
                break
            if esc.method == EscalationMethod.NONE_AVAILABLE or esc.genome is None:
                break

            self._emit_escalation(esc)
            trace_markers.append({
                "index": len(full_trace), "label": f"agent added: {esc.added_spec.role_name}",
                "method": esc.method.value, "corpus_hit": esc.method == EscalationMethod.CORPUS,
            })
            if esc.method == EscalationMethod.CORPUS:
                corpus_hits += 1

            # B5 — rearrange the larger team.
            result = await self._rearrange(esc.genome, instance, weights)
            full_trace.extend(result.fitness_trace)
            round_eval = result.best_evaluation
            total_cost += result.total_cost_usd
            rounds += 1
            delta = round_eval.fitness - pre_fitness

            if delta > self._epsilon:  # TEAM-GROWTH ELITISM: keep only if it helped
                best_genome = result.best_genome
                best_eval = round_eval
                agents_added.append({"agent_id": esc.added_agent_id, "role": esc.added_spec.role_name,
                                     "method": esc.method.value})
                if esc.method == EscalationMethod.CURATED:
                    if await self.corpus.promote(esc.added_spec, delta, instance.problem_class.value, instance.instance_id):
                        corpus_promotions += 1
                elif esc.method == EscalationMethod.CORPUS and esc.corpus_entry_id:
                    await self.corpus.update_stats(esc.corpus_entry_id, delta, True)
                kept = True
            else:  # roll back — restore the pre-escalation team so it never regresses
                await self._restore(snapshot)
                if esc.method == EscalationMethod.CORPUS and esc.corpus_entry_id:
                    await self.corpus.update_stats(esc.corpus_entry_id, delta, False)
                kept = False

            per_round.append({
                "round": rounds, "method": esc.method.value, "role": esc.added_spec.role_name,
                "delta": delta, "kept": kept, "fitness_after": best_eval.fitness,
                "normalized_after": best_eval.normalized_score, "gap": esc.gap.capability_needed,
            })
            self._emit_threshold(best_eval)

        status = SolveStatus.SEALED if best_eval.cleared_threshold else SolveStatus.EXHAUSTED
        return SolveResult(
            instance_id=instance.instance_id, final_genome=best_genome, final_evaluation=best_eval,
            cleared_threshold=best_eval.cleared_threshold, status=status, escalation_rounds=rounds,
            agents_added=agents_added, corpus_hits=corpus_hits, corpus_promotions=corpus_promotions,
            full_fitness_trace=full_trace, trace_markers=trace_markers,
            total_latency_ms=(self._clock() - start) * 1000.0, total_cost_usd=total_cost,
            per_round_summary=per_round,
        )

    # =======================================================================
    async def _rearrange(self, genome, instance, weights):
        from darwin.rearrange.loop import RearrangementResult

        try:
            return await self.rearrangement_loop.run(genome, instance, weights)
        except Exception as exc:  # noqa: BLE001 - B5 should never raise, but defend
            logger.warning("rearrangement errored; flooring this genome: %s", exc)
            ev = self._floor_evaluation(genome, instance.instance_id, weights)
            return RearrangementResult(best_genome=genome, best_evaluation=ev, fitness_trace=[ev.fitness],
                                       normalized_trace=[0.0], adopted_count=0, iterations=0,
                                       cleared_threshold=False, total_cost_usd=ev.total_cost_usd)

    async def _restore(self, snapshot: TeamGenome) -> None:
        if self.store is None:
            return
        snap = snapshot.model_dump(mode="json")

        def derive(current: TeamGenome):
            set_ops = {
                "agents": snap["agents"], "edges": snap["edges"], "arbiter_id": snap["arbiter_id"],
                "current_fitness": snapshot.current_fitness,
                "current_normalized_score": snapshot.current_normalized_score,
                "status": snapshot.status.value,
            }
            record = MutationRecord(
                mutation_type=MutationType.REMOVE_AGENT, actor=MutationActor.ESCALATION,
                description="rolled back unhelpful agent (no score improvement)",
                from_version=current.version, to_version=current.version + 1, fitness_before=current.current_fitness,
            )
            return set_ops, record

        try:
            await self.store.retry_mutate(snapshot.genome_id, derive)
        except Exception as exc:  # noqa: BLE001 - rollback is best-effort
            logger.warning("rollback failed for %s (continuing): %s", snapshot.genome_id, exc)

    # -- events -------------------------------------------------------------
    def _emit_escalation(self, esc) -> None:
        if self.event_sink is None:
            return
        self.event_sink({
            "event_type": "ESCALATION", "method": esc.method.value, "gap": esc.gap.capability_needed,
            "added_role": esc.added_spec.role_name if esc.added_spec else None,
            "genome_version": esc.genome.version if esc.genome else None,
            "corpus_hit": esc.method == EscalationMethod.CORPUS,
        })

    def _emit_threshold(self, evaluation) -> None:
        if self.event_sink is None:
            return
        self.event_sink({
            "event_type": "THRESHOLD_CHECK", "normalized_score": evaluation.normalized_score,
            "cleared": evaluation.cleared_threshold,
        })

    # -- floor results (the solve boundary's last resort) -------------------
    def _floor_evaluation(self, genome, instance_id, weights) -> GenomeEvaluation:
        sb = ScoreBreakdown(
            solution_id="floor", instance_id=instance_id, feasible=False, violations=[], raw_cost=0.0,
            raw_lead_time=0.0, raw_risk=0.0, weighted_objective=0.0, normalized_score=0.0,
            total_penalty=abs(GENOME_FLOOR_FITNESS), final_fitness=GENOME_FLOOR_FITNESS,
            objective_weights=weights, scorer_version=SCORER_VERSION, computed_at=_now_iso(), diagnostics={"floor": True},
        )
        return GenomeEvaluation(
            genome_id=genome.genome_id, version=genome.version, instance_id=instance_id, completed=False,
            final_solution=Solution(solution_id="floor", instance_id=instance_id, flows=[]),
            score_breakdown=sb, fitness=GENOME_FLOOR_FITNESS, normalized_score=0.0, cleared_threshold=False,
            arbiter_tier_used=ArbiterTier.INFEASIBLE_SENTINEL, error="floored",
        )

    def _floor_result(self, instance, weights, error) -> SolveResult:
        # never read instance.instance_id raw here: an empty/absent id would make
        # Solution(min_length=1) raise INSIDE the solve boundary and re-propagate.
        iid = getattr(instance, "instance_id", None) or "unknown"
        genome = self._safe_floor_genome(instance)
        ev = self._floor_evaluation(genome, iid, weights)
        return SolveResult(
            instance_id=iid, final_genome=genome, final_evaluation=ev,
            cleared_threshold=False, status=SolveStatus.EXHAUSTED, full_fitness_trace=[ev.fitness], error=error,
        )

    def _safe_floor_genome(self, instance) -> TeamGenome:
        """A valid placeholder team for the degraded result — must never raise.

        The architect's safe-default can itself fail (e.g. a misconfigured/empty
        registry — the very fault the floor exists to survive), so fall back to a
        minimal one-agent team built against the always-seeded process-wide
        default registry, and ultimately to an unvalidated placeholder.
        """
        instance_id = getattr(instance, "instance_id", "unknown")
        try:
            return self.architect._safe_default(instance)
        except Exception:  # noqa: BLE001
            logger.warning("architect safe-default failed in the floor path; building a minimal team")
        try:
            from darwin.agent.spec import AgentSpec, InputKind, OutputKind
            from darwin.team.genome import AgentNode

            spec = AgentSpec(
                agent_id="floor_agent", role_name="floor_agent",
                role_description="Floor placeholder agent (degraded result).",
                input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION,
            )  # no registry context -> validates against the seeded default registry
            node = AgentNode(agent_id="floor_agent", spec=spec)
            return TeamGenome.create(instance_id=instance_id, agents=[node], edges=[], arbiter_id="floor_agent")
        except Exception:  # noqa: BLE001 - ultimate fallback: bypass validation entirely
            logger.warning("minimal floor team build failed; constructing an unvalidated placeholder")
            from darwin.agent.spec import AgentSpec, InputKind, OutputKind
            from darwin.team.genome import AgentNode, GenomeStatus

            spec = AgentSpec.model_construct(
                agent_id="floor_agent", role_name="floor_agent",
                role_description="Floor placeholder agent (degraded result).",
                input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION,
                model_id="floor",
            )
            node = AgentNode.model_construct(agent_id="floor_agent", spec=spec)
            return TeamGenome.model_construct(
                instance_id=instance_id, agents=[node], edges=[], arbiter_id="floor_agent",
                version=1, status=GenomeStatus.DRAFT, history=[],
            )
