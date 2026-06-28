"""The Architect — the meta-agent that designs teams (never solves the problem).

It reads a problem, decides dynamically how to decompose it, authors a fresh
``AgentSpec`` for each part (coining the role, writing the job, assigning the best
model), and assembles them into an initial ``TeamGenome`` that B3 runs. On
escalation it authors a single targeted agent for a scorer-diagnosed gap.

Enterprise guarantee: **degraded, never dead.** A hopeless design falls back to a
minimal safe team; the Architect never crashes the system.
"""

import logging
from typing import Any, List, Optional, Tuple

from pydantic import ValidationError

from darwin.agent.parsing import extract_json, try_parse_json
from darwin.agent.registry import default_registry
from darwin.agent.spec import AgentSpec, InputKind, OutputKind, ThinkingLevel
from darwin.architect.assembly import _slugify, assemble
from darwin.architect.prompts import (
    SYSTEM_PROMPT,
    build_analysis_prompt,
    build_design_prompt,
    build_gap_prompt,
    repair_prompt,
)
from darwin.architect.schemas import (
    AgentSpecDraft,
    ArchitectTeamDesign,
    CuratedAgentDraft,
    EdgeDraft,
    ProblemAnalysis,
)
from darwin.constants import ARCHITECT_MODEL_ID, DEFAULT_TIMEOUT_S, FAST_MODEL_ID, MAX_DESIGN_REPAIRS
from darwin.problem.schemas import (
    ObjectiveWeights,
    ProblemInstance,
    ViolationType,
)
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import AgentNode, Edge, EdgeType, MutationActor, TeamGenome
from darwin.team.validation import validate

logger = logging.getLogger("darwin.architect")

_DESIGN_MAX_TOKENS = 4096
_RISK_WEAK_THRESHOLD = 0.45  # raw_risk above this => resilience is a weak dimension


class Architect:
    def __init__(
        self,
        client: Any,
        store: Any = None,
        registry: Any = None,
        *,
        architect_model_id: str = ARCHITECT_MODEL_ID,
        fast_model_id: str = FAST_MODEL_ID,
        max_repairs: int = MAX_DESIGN_REPAIRS,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        self.client = client
        self.store = store
        self.registry = registry or getattr(client, "registry", None) or default_registry()
        self.architect_model_id = architect_model_id
        self.fast_model_id = fast_model_id
        self._max_repairs = max_repairs
        self._timeout = timeout

    # =======================================================================
    # design_initial_team — always returns a valid, persisted genome, never raises
    # =======================================================================
    async def design_initial_team(self, instance: ProblemInstance, weights: ObjectiveWeights) -> TeamGenome:
        try:
            genome = await self._design(instance, weights)
        except Exception:  # noqa: BLE001 - the Architect never crashes the system
            logger.exception("architect design errored; falling back to the safe-default team")
            genome = self._safe_default(instance)
        await self._persist(genome)
        return genome

    async def _design(self, instance: ProblemInstance, weights: ObjectiveWeights) -> TeamGenome:
        analysis = await self._analyze(instance, weights)
        user = build_design_prompt(analysis, self.registry)
        last_error: Optional[str] = None

        for _attempt in range(self._max_repairs + 1):
            design, raw, perr = await self._call_structured(user, ArchitectTeamDesign)
            if design is None:
                last_error = perr or "no valid design JSON"
                user = repair_prompt(user, last_error)
                continue
            try:
                genome = assemble(design, instance.instance_id, self.registry)
            except Exception as exc:  # assembly / spec-validation (e.g. bad model_id)
                last_error = f"assembly error: {exc}"
                user = repair_prompt(user, last_error)
                continue
            result = validate(genome, self.registry)
            if result.valid:
                return genome
            last_error = "; ".join(result.errors)
            user = repair_prompt(user, last_error)

        logger.warning("architect exhausted design repairs (%s); using safe default", last_error)
        return self._safe_default(instance)

    async def _analyze(self, instance: ProblemInstance, weights: ObjectiveWeights) -> ProblemAnalysis:
        try:
            analysis, _raw, _err = await self._call_structured(
                build_analysis_prompt(instance, weights), ProblemAnalysis
            )
            if analysis is not None:
                return analysis
        except Exception:  # noqa: BLE001 - analysis is best-effort
            logger.warning("architect analysis call failed; using a heuristic analysis", exc_info=True)
        return self._heuristic_analysis(instance, weights)

    # =======================================================================
    # curate_agent_for_gap — the escalation entry point (scorer-driven)
    # =======================================================================
    async def curate_agent_for_gap(
        self, genome: TeamGenome, instance: ProblemInstance, evaluation: GenomeEvaluation
    ) -> Tuple[AgentSpec, List[Edge]]:
        diagnosis = self.diagnose(evaluation)
        user = build_gap_prompt(genome, evaluation, diagnosis, self.registry)
        last_error: Optional[str] = None

        for _attempt in range(self._max_repairs + 1):
            curated, _raw, perr = await self._call_structured(user, CuratedAgentDraft)
            if curated is None:
                last_error = perr or "no valid curated-agent JSON"
                user = repair_prompt(user, last_error)
                continue
            try:
                spec, edges = self._build_curated(curated.agent, curated.edges, genome)
                candidate = self._with_added_agent(genome, spec, edges)
            except Exception as exc:  # noqa: BLE001
                last_error = f"build error: {exc}"
                user = repair_prompt(user, last_error)
                continue
            result = validate(candidate, self.registry)
            if result.valid:
                return spec, edges
            last_error = "; ".join(result.errors)
            user = repair_prompt(user, last_error)

        logger.warning("architect exhausted gap-curation repairs (%s); using a heuristic agent", last_error)
        return self._heuristic_curated_agent(genome, diagnosis)

    # -- scorer-driven diagnosis -------------------------------------------
    @staticmethod
    def diagnose(evaluation: GenomeEvaluation) -> str:
        sb = evaluation.score_breakdown
        vtypes = {v.violation_type for v in sb.violations}
        if ViolationType.DEMAND_UNMET in vtypes:
            return "persistent DEMAND_UNMET — the team needs a demand-coverage specialist"
        if vtypes & {ViolationType.OVER_ARC_CAPACITY, ViolationType.OVER_NODE_CAPACITY}:
            return "capacity violations — the team needs a capacity-rebalancing agent"
        if sb.raw_risk > _RISK_WEAK_THRESHOLD:
            return "high disruption risk (weak resilience) — the team needs a disruption-risk modeler"
        if sb.feasible and evaluation.normalized_score < 0.90:
            return "cost far from the known optimum — the team needs a cost-reduction specialist"
        return "below threshold — the team needs a stronger specialist or arbitrator"

    # =======================================================================
    # Structured model call (parse + validate, no repair here — callers repair)
    # =======================================================================
    async def _call_structured(self, user: str, schema_cls):
        schema = schema_cls.model_json_schema()
        response = await self.client.complete(
            self.architect_model_id, SYSTEM_PROMPT, user, schema, ThinkingLevel.HIGH.value,
            _DESIGN_MAX_TOKENS, timeout=self._timeout,
        )
        if response.error is not None:
            return None, response.raw_text, response.error
        data = response.parsed if response.parsed is not None else self._extract(response.raw_text)
        if data is None:
            return None, response.raw_text, "no parseable JSON in model output"
        try:
            return schema_cls.model_validate(data), response.raw_text, None
        except ValidationError as exc:
            return None, response.raw_text, f"schema validation: {str(exc)[:300]}"

    @staticmethod
    def _extract(raw_text: str):
        span = extract_json(raw_text)
        if span is None:
            return None
        value = try_parse_json(span)
        return value if isinstance(value, (dict, list)) else None

    # =======================================================================
    # Curated-agent assembly + validation
    # =======================================================================
    def _unique_agent_id(self, role_name: str, genome: TeamGenome) -> str:
        # dedup against existing agent_ids AND role_names — the curated agent's
        # role_name is set to this id, so it must not collide with either.
        taken = {a.agent_id for a in genome.agents} | {a.spec.role_name for a in genome.agents}
        base = _slugify(role_name)
        agent_id = base
        i = 2
        while agent_id in taken:
            agent_id = f"{base}-{i}"
            i += 1
        return agent_id

    def _build_curated(
        self, draft: AgentSpecDraft, edge_drafts: List[EdgeDraft], genome: TeamGenome
    ) -> Tuple[AgentSpec, List[Edge]]:
        # the new agent's id is unique; use it as role_name too so it never
        # duplicates an existing role (contract: curate does not duplicate a role).
        agent_id = self._unique_agent_id(draft.role_name, genome)
        spec = AgentSpec.model_validate(
            {
                "agent_id": agent_id, "role_name": agent_id,
                "role_description": draft.role_description, "input_contract": draft.input_contract.value,
                "output_contract": draft.output_contract.value, "model_id": draft.model_id,
                "thinking_level": draft.thinking_level.value, "created_by": "architect", "spec_version": "1.0.0",
            },
            context={"registry": self.registry},
        )
        # existing agents win the role->id binding (setdefault) so a colliding new
        # role can never silently re-point an edge meant for an existing agent.
        role_to_id = {n.spec.role_name: n.agent_id for n in genome.agents}
        role_to_id.setdefault(draft.role_name, agent_id)
        role_to_id.setdefault(spec.role_name, agent_id)

        edges: List[Edge] = []
        for ed in edge_drafts:
            src = role_to_id.get(ed.from_role_name)
            dst = role_to_id.get(ed.to_role_name)
            if src is None or dst is None:
                raise ValueError(f"curated edge references unknown role: {ed.from_role_name!r} -> {ed.to_role_name!r}")
            edges.append(Edge(from_agent_id=src, to_agent_id=dst, edge_type=ed.edge_type))
        # guarantee the new agent is actually wired in (its edges may have resolved
        # entirely to existing agents on a role collision).
        if not any(agent_id in (e.from_agent_id, e.to_agent_id) for e in edges):
            edges.append(Edge(from_agent_id=agent_id, to_agent_id=genome.arbiter_id, edge_type=EdgeType.FEEDS_ARBITER))
        return spec, edges

    def _with_added_agent(self, genome: TeamGenome, spec: AgentSpec, edges: List[Edge]) -> TeamGenome:
        node = AgentNode(agent_id=spec.agent_id, spec=spec)
        data = genome.model_dump()
        data["agents"] = data["agents"] + [node.model_dump()]
        data["edges"] = data["edges"] + [e.model_dump() for e in edges]
        return TeamGenome.model_validate(data, context={"registry": self.registry})

    def _heuristic_curated_agent(self, genome: TeamGenome, diagnosis: str) -> Tuple[AgentSpec, List[Edge]]:
        role = "demand_coverage_specialist"
        if "capacity" in diagnosis:
            role = "capacity_rebalancer"
        elif "resilience" in diagnosis or "risk" in diagnosis:
            role = "disruption_risk_modeler"
        elif "cost" in diagnosis:
            role = "cost_reduction_specialist"
        agent_id = self._unique_agent_id(role, genome)
        spec = AgentSpec.model_validate(
            {
                "agent_id": agent_id, "role_name": agent_id,
                "role_description": (
                    f"Reason DIRECTLY about the problem to address: {diagnosis}. Produce a complete, "
                    "feasible Solution; never write or call a solver."
                ),
                "input_contract": InputKind.FULL_PROBLEM.value, "output_contract": OutputKind.FULL_SOLUTION.value,
                "model_id": self._safe_model_id(), "thinking_level": ThinkingLevel.MEDIUM.value,
                "created_by": "architect", "spec_version": "1.0.0",
            },
            context={"registry": self.registry},
        )
        edges = [Edge(from_agent_id=agent_id, to_agent_id=genome.arbiter_id, edge_type=EdgeType.FEEDS_ARBITER)]
        return spec, edges

    def _safe_model_id(self) -> str:
        """A guaranteed registry-legal model id so the last-line-of-defense
        fallbacks can never themselves raise (degraded, never dead)."""
        from darwin.constants import DEFAULT_MODEL_ID

        for candidate in (self.fast_model_id, DEFAULT_MODEL_ID):
            if self.registry.contains(candidate):
                return candidate
        ids = self.registry.all_ids()
        if not ids:  # truly empty registry — nothing is runnable
            raise RuntimeError("registry is empty; cannot build a fallback team")
        return ids[0]

    # =======================================================================
    # Heuristics & safe default
    # =======================================================================
    def _heuristic_analysis(self, instance: ProblemInstance, weights: ObjectiveWeights) -> ProblemAnalysis:
        objs = sorted(
            [("cost", weights.cost_weight), ("lead_time", weights.lead_time_weight), ("risk", weights.risk_weight)],
            key=lambda kv: -kv[1],
        )
        dominant = [name for name, w in objs if w >= objs[0][1] - 1e-9 and w > 0]
        parts = min(4, max(2, 1 + len(dominant)))
        return ProblemAnalysis(
            problem_class=instance.problem_class.value,
            dominant_objectives=dominant or ["cost"],
            binding_constraints=([c.constraint_type.value for c in instance.additional_constraints] or ["capacity", "demand_satisfaction"]),
            difficulty_estimate=instance.metadata.difficulty.value,
            suggested_part_count=parts,
            rationale="heuristic analysis (model analysis unavailable)",
        )

    def _safe_default(self, instance: ProblemInstance) -> TeamGenome:
        proposer = AgentSpec.model_validate(
            {
                "agent_id": "cost_minimizer", "role_name": "cost_minimizer",
                "role_description": (
                    "Reason DIRECTLY about the problem and produce a complete, feasible, low-cost Solution "
                    "(flows / open_facilities / routes). Never write or call a solver."
                ),
                "input_contract": InputKind.FULL_PROBLEM.value, "output_contract": OutputKind.FULL_SOLUTION.value,
                "model_id": self._safe_model_id(), "thinking_level": ThinkingLevel.MEDIUM.value,
                "created_by": "architect", "spec_version": "1.0.0",
            },
            context={"registry": self.registry},
        )
        arbiter = AgentSpec.model_validate(
            {
                "agent_id": "arbitrator", "role_name": "arbitrator",
                "role_description": (
                    "Synthesize the final Solution from the sibling proposals (pass the best proposal through). "
                    "Reason directly; never write or call a solver."
                ),
                "input_contract": InputKind.SIBLING_OUTPUTS.value, "output_contract": OutputKind.ARBITRATION.value,
                "model_id": self._safe_model_id(), "thinking_level": ThinkingLevel.LOW.value,
                "created_by": "architect", "spec_version": "1.0.0",
            },
            context={"registry": self.registry},
        )
        nodes = [AgentNode(agent_id="cost_minimizer", spec=proposer), AgentNode(agent_id="arbitrator", spec=arbiter)]
        edges = [Edge(from_agent_id="cost_minimizer", to_agent_id="arbitrator", edge_type=EdgeType.FEEDS_ARBITER)]
        return TeamGenome.create(
            instance_id=instance.instance_id, agents=nodes, edges=edges, arbiter_id="arbitrator",
            actor=MutationActor.ARCHITECT, description="safe-default minimal team (proposer -> arbitrator)",
        )

    async def _persist(self, genome: TeamGenome) -> None:
        if self.store is None:
            return
        try:
            await self.store.save_new(genome)
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            logger.warning("architect could not persist genome %s: %s", genome.genome_id, exc)
