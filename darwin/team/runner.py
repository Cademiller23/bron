"""The TeamRunner — the kitchen. Genome in, valid scored answer out. Always a
number, never an exception.

The entire body of :meth:`TeamRunner.evaluate` runs inside a catch-all boundary:
any exception from any layer is mapped to a floor-scored ``GenomeEvaluation`` and
logged to the genome's history as an ``EVAL_ERROR``. A genome that errors is just
a genome that scored at the floor — indistinguishable to B5/B6 from any other bad
team, so the loop is total and uninterruptible.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from darwin.agent.client import Usage
from darwin.agent.worker import AgentInput, AgentResult, WorkerAgent
from darwin.constants import (
    AGENT_MAX_RETRIES,
    ARBITER_MAX_RETRIES,
    GENOME_FLOOR_FITNESS,
    SCORE_THRESHOLD,
)
from darwin.problem.schemas import ObjectiveWeights, ProblemInstance, ScoreBreakdown, Solution
from darwin.problem.scorer import SCORER_VERSION, score as default_score
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import (
    AgentNode,
    ArbiterTier,
    GenomeStatus,
    MutationActor,
    MutationRecord,
    MutationType,
    TeamGenome,
)
from darwin.team.inference_gate import InferenceGate
from darwin.team.validation import validate

logger = logging.getLogger("darwin.team.runner")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_solution(result: Optional[AgentResult]) -> Optional[Solution]:
    """Pull the B1 Solution out of an agent's output, if it carries one."""
    if result is None or not result.success or result.output is None:
        return None
    return getattr(result.output, "solution", None)


class TeamRunner:
    def __init__(
        self,
        scorer: Optional[Callable] = None,
        model_client: Any = None,
        telemetry: Any = None,
        inference_gate: Optional[InferenceGate] = None,
        store: Any = None,
        *,
        worker_factory: Optional[Callable] = None,
        agent_max_retries: int = AGENT_MAX_RETRIES,
        arbiter_max_retries: int = ARBITER_MAX_RETRIES,
    ) -> None:
        self.scorer = scorer or default_score
        self.model_client = model_client
        self.telemetry = telemetry
        self.gate = inference_gate or InferenceGate()
        self.store = store
        self._worker_factory = worker_factory or (lambda spec, client, tel: WorkerAgent(spec, client, tel))
        self._agent_max_retries = agent_max_retries
        self._arbiter_max_retries = arbiter_max_retries

    # =======================================================================
    # Public entry point — the genome-evaluation boundary (never raises)
    # =======================================================================
    async def evaluate(
        self,
        genome: TeamGenome,
        instance: ProblemInstance,
        weights: Optional[ObjectiveWeights] = None,
        *,
        persist_outcome: bool = True,
    ) -> GenomeEvaluation:
        """Evaluate ``genome``. With ``persist_outcome=False`` the outcome is NOT
        written back to the store — used by B5 to score transient candidate
        genomes without polluting the persistent lineage."""
        weights = weights or ObjectiveWeights.cost_only()
        try:
            return await self._evaluate(genome, instance, weights, persist_outcome=persist_outcome)
        except asyncio.CancelledError:
            # Per spec §10, an asyncio cancellation is *floored* (the loop is
            # total). Use the no-I/O backstop so we don't re-await (and re-cancel)
            # inside an already-cancelling task.
            logger.warning("evaluation cancelled; flooring genome %s", genome.genome_id)
            return self._bare_floor_evaluation(genome, instance, weights, error="evaluation cancelled")
        except Exception as exc:  # noqa: BLE001 - evaluation NEVER raises
            logger.exception("evaluation boundary caught an error; flooring genome %s", genome.genome_id)
            try:
                return await self._floor_evaluation(
                    genome, instance, weights, error=f"{type(exc).__name__}: {exc}", agent_results=[]
                )
            except BaseException:  # pragma: no cover - ultimate backstop (no I/O)
                return self._bare_floor_evaluation(genome, instance, weights, error=f"{type(exc).__name__}: {exc}")

    # =======================================================================
    # The pipeline
    # =======================================================================
    async def _evaluate(
        self, genome: TeamGenome, instance: ProblemInstance, weights: ObjectiveWeights,
        *, persist_outcome: bool = True,
    ) -> GenomeEvaluation:
        # Step 1 — validate (defence in depth; the Architect validates first).
        registry = getattr(self.model_client, "registry", None)
        result = validate(genome, registry)
        if not result.valid:
            return await self._floor_evaluation(
                genome, instance, weights, error="invalid genome: " + "; ".join(result.errors), agent_results=[]
            )

        # Step 2 — topological levels.
        levels = self._topo_levels(genome)

        # Steps 3-4 — run every non-arbiter agent, level by level, concurrently
        # under the shared inference gate.
        results: Dict[str, AgentResult] = {}
        for level in levels:
            workers = [n for n in level if n.agent_id != genome.arbiter_id]
            if not workers:
                continue
            level_results = await asyncio.gather(
                *[self._run_agent_guarded(genome, node, instance, results) for node in workers]
            )
            for node, res in zip(workers, level_results):
                results[node.agent_id] = res

        # Steps 5-6 — the arbiter through the three-tier fallback.
        final_solution, tier, arbiter_result, fallback_error = await self._run_arbiter(
            genome, instance, weights, results
        )
        if arbiter_result is not None:
            results[genome.arbiter_id] = arbiter_result

        # Step 7 — score the final solution (deterministic, never raises).
        breakdown = self.scorer(instance, final_solution, weights)

        # Step 8 — assemble the evaluation (records the version that was evaluated).
        agent_results = list(results.values())
        # The 90% gate requires a FEASIBLE answer: an infeasible solution can score
        # normalized_score==1.0 (B1 gives raw_cost 0 a perfect ratio), but a team
        # that produced an infeasible answer has NOT cleared and must escalate.
        cleared = breakdown.feasible and breakdown.normalized_score >= SCORE_THRESHOLD
        completed = tier in (ArbiterTier.PRIMARY, ArbiterTier.RETRY)
        evaluation = GenomeEvaluation(
            genome_id=genome.genome_id,
            version=genome.version,
            instance_id=instance.instance_id,
            completed=completed,
            final_solution=final_solution,
            score_breakdown=breakdown,
            fitness=breakdown.final_fitness,
            normalized_score=breakdown.normalized_score,
            cleared_threshold=cleared,
            arbiter_tier_used=tier,
            agent_results=agent_results,
            total_latency_ms=sum(r.latency_ms for r in agent_results),
            total_cost_usd=sum(r.est_cost for r in agent_results),
            error=None,
        )

        # Step 9 — persist the outcome (best-effort; a persist failure must never
        # corrupt the real fitness signal that B5 reads). Skipped for transient
        # B5 candidate genomes (persist_outcome=False).
        if persist_outcome:
            await self._persist_outcome(genome, evaluation, tier, fallback_error)
        return evaluation

    # =======================================================================
    # Topological layering
    # =======================================================================
    def _topo_levels(self, genome: TeamGenome) -> List[List[AgentNode]]:
        ids = [a.agent_id for a in genome.agents]
        indeg = {i: 0 for i in ids}
        adj: Dict[str, List[str]] = {i: [] for i in ids}
        for e in genome.edges:
            if e.from_agent_id == e.to_agent_id:
                continue
            adj[e.from_agent_id].append(e.to_agent_id)
            indeg[e.to_agent_id] += 1
        node_index = genome.node_index
        levels: List[List[AgentNode]] = []
        current = sorted(i for i in ids if indeg[i] == 0)
        while current:
            levels.append([node_index[i] for i in current])
            nxt: List[str] = []
            for u in current:
                for v in adj[u]:
                    indeg[v] -= 1
                    if indeg[v] == 0:
                        nxt.append(v)
            current = sorted(set(nxt))
        return levels

    # =======================================================================
    # Agent execution: retry once, then route around
    # =======================================================================
    async def _run_agent_guarded(
        self, genome: TeamGenome, node: AgentNode, instance: ProblemInstance, results: Dict[str, AgentResult]
    ) -> AgentResult:
        worker = self._worker_factory(node.spec, self.model_client, self.telemetry)
        agent_input = self._assemble_input(genome, node, instance, results)
        last: Optional[AgentResult] = None
        for _attempt in range(self._agent_max_retries + 1):
            async with self.gate.acquire():  # every model call respects the global ceiling
                try:
                    last = await worker.run(agent_input)
                except Exception as exc:  # noqa: BLE001 - a contract-violating worker
                    # must not floor the whole genome; route around it. (CancelledError
                    # is a BaseException and still propagates to the boundary.)
                    last = self._failed_result(node, f"worker raised {type(exc).__name__}: {exc}")
            if last.success:
                return last
        # route around: return the last (failed) result; downstream ignores it
        return last  # type: ignore[return-value]

    @staticmethod
    def _failed_result(node: AgentNode, error: str) -> AgentResult:
        return AgentResult(
            agent_id=node.agent_id,
            role_name=node.spec.role_name,
            model_id=node.spec.model_id,
            success=False,
            output=None,
            raw_text="",
            num_repairs=0,
            latency_ms=0.0,
            usage=Usage(),
            est_cost=0.0,
            error=error,
            produced_at=_now_iso(),
        )

    def _assemble_input(
        self, genome: TeamGenome, node: AgentNode, instance: ProblemInstance, results: Dict[str, AgentResult]
    ) -> AgentInput:
        from darwin.agent.spec import InputKind

        sibling_outputs: List[Dict[str, Any]] = []
        draft: Optional[Solution] = None
        seen_upstream: set = set()
        for edge in genome.upstream_edges(node.agent_id):
            upstream = results.get(edge.from_agent_id)
            if upstream is None or not upstream.success or upstream.output is None:
                continue  # the DAG tolerates missing contributions by design
            if edge.from_agent_id in seen_upstream:
                continue  # don't feed the same upstream agent's output twice
            seen_upstream.add(edge.from_agent_id)
            sibling_outputs.append(
                {
                    "from_agent_id": edge.from_agent_id,
                    "edge_type": edge.edge_type.value,
                    "output": upstream.output.model_dump(mode="json"),
                }
            )
            if (
                node.spec.input_contract == InputKind.PROBLEM_PLUS_DRAFT
                and draft is None
                and edge.edge_type.value == "PASSES_PROPOSAL"
            ):
                sol = _extract_solution(upstream)
                if sol is not None:
                    draft = sol
        return AgentInput(
            instance=instance,
            sibling_outputs=sibling_outputs,
            draft=draft,
            team_genome_id=genome.genome_id,
        )

    # =======================================================================
    # The three-tier arbiter fallback
    # =======================================================================
    async def _run_arbiter(
        self,
        genome: TeamGenome,
        instance: ProblemInstance,
        weights: ObjectiveWeights,
        results: Dict[str, AgentResult],
    ) -> Tuple[Solution, ArbiterTier, Optional[AgentResult], Optional[str]]:
        arbiter_node = genome.node_index[genome.arbiter_id]
        arbiter_input = self._assemble_input(genome, arbiter_node, instance, results)

        # Tier 1 — retry (n attempts = 1 + arbiter_max_retries).
        last_result: Optional[AgentResult] = None
        last_error = "arbiter did not produce a usable solution"
        for attempt in range(self._arbiter_max_retries + 1):
            worker = self._worker_factory(arbiter_node.spec, self.model_client, self.telemetry)
            async with self.gate.acquire():
                try:
                    last_result = await worker.run(arbiter_input)
                except Exception as exc:  # noqa: BLE001 - fall through to the next tier
                    last_result = self._failed_result(arbiter_node, f"arbiter raised {type(exc).__name__}: {exc}")
            sol = _extract_solution(last_result)
            if last_result.success and sol is not None:
                tier = ArbiterTier.PRIMARY if attempt == 0 else ArbiterTier.RETRY
                return sol, tier, last_result, None
            if last_result.error:
                last_error = last_result.error

        # Tier 2 — best available scored feasible proposal.
        best_solution: Optional[Solution] = None
        best_score = float("-inf")
        for pid in genome.proposer_ids():
            candidate = _extract_solution(results.get(pid))
            if candidate is None:
                continue
            breakdown = self.scorer(instance, candidate, weights)
            if breakdown.feasible and breakdown.normalized_score > best_score:
                best_score = breakdown.normalized_score
                best_solution = candidate
        if best_solution is not None:
            return (
                best_solution,
                ArbiterTier.BEST_PROPOSAL_FALLBACK,
                last_result,
                f"arbiter failed ({last_error}); used best feasible proposal",
            )

        # Tier 3 — infeasible-but-scored sentinel (scores at the floor).
        sentinel = Solution(
            solution_id=f"sentinel-{genome.genome_id}",
            instance_id=instance.instance_id,
            flows=[],
            produced_by="runner:sentinel",
        )
        return (
            sentinel,
            ArbiterTier.INFEASIBLE_SENTINEL,
            last_result,
            f"arbiter failed ({last_error}); no usable proposal — infeasible sentinel",
        )

    # =======================================================================
    # Persistence (best-effort, optimistic-locked)
    # =======================================================================
    async def _persist_outcome(
        self,
        genome: TeamGenome,
        evaluation: GenomeEvaluation,
        tier: ArbiterTier,
        fallback_error: Optional[str],
    ) -> None:
        if self.store is None:
            return
        status = GenomeStatus.CLEARED_THRESHOLD if evaluation.cleared_threshold else GenomeStatus.EVALUATED

        def derive(current: TeamGenome):
            set_ops = {
                "current_fitness": evaluation.fitness,
                "current_normalized_score": evaluation.normalized_score,
                "status": status.value,
            }
            # A clean evaluation only updates the score fields (no history entry —
            # the lineage records structural mutations + fallback/error events, so
            # repeated evaluations of one genome don't bloat history). A fallback
            # IS a lineage-worthy event and is recorded.
            record = None
            if tier in (ArbiterTier.BEST_PROPOSAL_FALLBACK, ArbiterTier.INFEASIBLE_SENTINEL):
                record = MutationRecord(
                    mutation_type=MutationType.ARBITER_FALLBACK_USED,
                    actor=MutationActor.RUNNER,
                    description=f"arbiter degraded to {tier.value}",
                    from_version=current.version,
                    to_version=current.version + 1,
                    fitness_before=current.current_fitness,
                    fitness_after=evaluation.fitness,
                    error=fallback_error,
                )
            return set_ops, record

        try:
            await self.store.retry_mutate(genome.genome_id, derive)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.warning("outcome persistence failed for %s (continuing): %s", genome.genome_id, exc)

    # =======================================================================
    # Floor evaluations
    # =======================================================================
    async def _floor_evaluation(
        self,
        genome: TeamGenome,
        instance: ProblemInstance,
        weights: ObjectiveWeights,
        error: str,
        agent_results: List[AgentResult],
    ) -> GenomeEvaluation:
        evaluation = self._bare_floor_evaluation(genome, instance, weights, error, agent_results)
        # log the error to the genome's history (best-effort)
        if self.store is not None:
            def derive(current: TeamGenome):
                record = MutationRecord(
                    mutation_type=MutationType.EVAL_ERROR,
                    actor=MutationActor.RUNNER,
                    description="evaluation error -> floor score",
                    from_version=current.version,
                    to_version=current.version + 1,
                    fitness_before=current.current_fitness,
                    fitness_after=evaluation.fitness,
                    error=error,
                )
                return {"current_fitness": evaluation.fitness, "current_normalized_score": 0.0,
                        "status": GenomeStatus.EVALUATED.value}, record

            try:
                await self.store.retry_mutate(genome.genome_id, derive)
            except Exception as exc:  # noqa: BLE001
                logger.warning("EVAL_ERROR persistence failed for %s: %s", genome.genome_id, exc)
        return evaluation

    def _bare_floor_evaluation(
        self,
        genome: TeamGenome,
        instance: ProblemInstance,
        weights: ObjectiveWeights,
        error: str,
        agent_results: Optional[List[AgentResult]] = None,
    ) -> GenomeEvaluation:
        """Construct a floor evaluation with NO I/O — the ultimate backstop."""
        sentinel = Solution(
            solution_id=f"floor-{genome.genome_id}",
            instance_id=instance.instance_id,
            flows=[],
            produced_by="runner:floor",
        )
        breakdown = self._synthetic_floor_breakdown(instance.instance_id, sentinel.solution_id, weights)
        return GenomeEvaluation(
            genome_id=genome.genome_id,
            version=genome.version,
            instance_id=instance.instance_id,
            completed=False,
            final_solution=sentinel,
            score_breakdown=breakdown,
            fitness=breakdown.final_fitness,
            normalized_score=0.0,
            cleared_threshold=False,
            arbiter_tier_used=ArbiterTier.INFEASIBLE_SENTINEL,
            agent_results=agent_results or [],
            total_latency_ms=sum(r.latency_ms for r in (agent_results or [])),
            total_cost_usd=sum(r.est_cost for r in (agent_results or [])),
            error=error,
        )

    @staticmethod
    def _synthetic_floor_breakdown(
        instance_id: str, solution_id: str, weights: ObjectiveWeights
    ) -> ScoreBreakdown:
        return ScoreBreakdown(
            solution_id=solution_id,
            instance_id=instance_id,
            feasible=False,
            violations=[],
            raw_cost=0.0,
            raw_lead_time=0.0,
            raw_risk=0.0,
            weighted_objective=0.0,
            normalized_score=0.0,
            total_penalty=abs(GENOME_FLOOR_FITNESS),
            final_fitness=GENOME_FLOOR_FITNESS,
            objective_weights=weights,
            scorer_version=SCORER_VERSION,
            computed_at=_now_iso(),
            diagnostics={"floor": True},
        )
