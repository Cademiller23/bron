"""The WorkerAgent — the single reusable unit of intelligence (the atom).

One generic agent: it takes a structured problem (or a piece of one) plus a role
description (the ``AgentSpec``), calls a model through the model-agnostic client,
and returns strict structured output — never free text. Every later phase
composes these atoms.

Non-negotiable: the worker produces *candidate* solutions; it **never grades
them**. It imports no scorer and computes no fitness — quality judgment is B1's
deterministic scorer's exclusive job. The worker's only contract is "return
valid, schema-conforming structured output, or fail gracefully."
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger("darwin.agent.worker")

from darwin.agent.client import ModelClient, Usage
from darwin.agent.outputs import OutputUnion, output_model_for
from darwin.agent.parsing import extract_json, try_parse_json
from darwin.agent.spec import AgentSpec, OutputKind
from darwin.agent.telemetry import TelemetrySink
from darwin.constants import (
    DEFAULT_TIMEOUT_S,
    MAX_REPAIRS,
    TELEMETRY_RAW_OUTPUT_MAX_CHARS,
)
from darwin.problem.schemas import ProblemInstance, Solution

# Constant across every agent — the role is what varies, the guardrail does not.
GUARDRAIL_PREAMBLE = (
    "You are one agent in a team solving a supply-chain optimization problem. "
    "Output ONLY JSON conforming to the provided schema. Never include prose, "
    "markdown, or explanation outside the JSON. Your output will be parsed by a machine."
)

_DEFAULT_INSTRUCTION = {
    OutputKind.FULL_SOLUTION: "Produce a complete solution (flows / open_facilities / routes) for the problem.",
    OutputKind.PARTIAL_SOLUTION: "Produce a partial solution for your assigned sub-problem.",
    OutputKind.CRITIQUE: "Critique the provided sibling output(s); list concrete, located issues.",
    OutputKind.CONSTRAINT_REPORT: "Report which constraints you suspect the candidate violates (a hint, not a verdict).",
    OutputKind.ARBITRATION: "Synthesize one final solution from the provided sibling outputs.",
    OutputKind.DECOMPOSITION: "Propose a decomposition of the problem into named sub-problems.",
}


class AgentInput(BaseModel):
    """What a worker is given, per its spec's ``input_contract``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: ProblemInstance  # the relevant B1 problem (or sub-problem)
    sub_problem_id: Optional[str] = None
    sibling_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    draft: Optional[Solution] = None  # a current candidate to improve
    instruction: str = ""  # optional override of the per-output-kind default task line
    team_genome_id: Optional[str] = None  # stamped by B3 for telemetry provenance


class AgentResult(BaseModel):
    """The frozen result of one ``run()`` — never raised, always returned.

    Note: there is deliberately **no fitness field** — grading is B1's job.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str
    role_name: str
    model_id: str
    success: bool
    output: Optional[OutputUnion] = None
    raw_text: str = ""
    num_repairs: int = 0
    latency_ms: float = 0.0
    usage: Usage = Field(default_factory=Usage)
    est_cost: float = 0.0
    error: Optional[str] = None
    produced_at: str = ""


class WorkerAgent:
    def __init__(
        self,
        spec: AgentSpec,
        client: ModelClient,
        telemetry: TelemetrySink,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_repairs: int = MAX_REPAIRS,
    ) -> None:
        self.spec = spec
        self.client = client
        self.telemetry = telemetry
        self._timeout = timeout
        self._max_repairs = max_repairs

    async def run(self, agent_input: AgentInput) -> AgentResult:
        """The invocation pipeline (§6). **Never raises into the caller.**

        A thin backstop wraps the whole pipeline so that *any* unexpected error
        (e.g. prompt assembly choking on a cyclic ``sibling_outputs``) still
        returns a graceful ``AgentResult`` and writes one telemetry doc — the
        never-raises / telemetry-on-every-path invariants hold unconditionally.
        """
        try:
            return await self._run_pipeline(agent_input)
        except Exception as exc:  # noqa: BLE001 - the whole point is to never escape
            logger.exception("worker pipeline raised; returning graceful failure")
            result = AgentResult(
                agent_id=self.spec.agent_id,
                role_name=self.spec.role_name,
                model_id=self.spec.model_id,
                success=False,
                output=None,
                raw_text="",
                num_repairs=0,
                latency_ms=0.0,
                usage=Usage(),
                est_cost=0.0,
                error=f"worker pipeline error: {type(exc).__name__}: {exc}",
                produced_at=datetime.now(timezone.utc).isoformat(),
            )
            try:
                await self._log(result, agent_input)
            except Exception:  # pragma: no cover - _log is itself guarded
                logger.warning("telemetry also failed inside the backstop")
            return result

    async def _run_pipeline(self, agent_input: AgentInput) -> AgentResult:
        output_model = output_model_for(self.spec.output_contract)
        schema = output_model.model_json_schema()
        system = self._build_system_prompt()
        base_user = self._build_user_message(agent_input)

        tokens_in = 0
        tokens_out = 0
        latency_ms = 0.0
        num_repairs = 0
        last_error: Optional[str] = None
        last_raw = ""
        output: Optional[BaseModel] = None
        user = base_user

        while True:
            response = await self.client.complete(
                self.spec.model_id,
                system,
                user,
                schema,
                self.spec.thinking_level.value,
                self.spec.max_output_tokens,
                timeout=self._timeout,
            )
            latency_ms += response.latency_ms
            tokens_in += response.usage.tokens_in
            tokens_out += response.usage.tokens_out
            last_raw = response.raw_text

            if response.error is not None:
                # Transport/API failure — already retried inside the client.
                # Re-calling won't fix it, so fail gracefully (no repair).
                last_error = response.error
                break

            # Obtain JSON. Pathological output (e.g. deeply-nested brackets that
            # make json.loads recurse) must degrade to a repair/failure, never
            # crash run() — so the extractor and validator are both guarded.
            iter_error: Optional[str] = None
            data = None
            if response.parsed is not None:
                data = response.parsed
            else:
                try:
                    data = self._extract(response.raw_text)
                except Exception as exc:  # noqa: BLE001 - run() must never raise
                    iter_error = f"output extraction error: {type(exc).__name__}"

            if data is not None:
                try:
                    output = output_model.model_validate(data)
                    break  # success
                except Exception as exc:  # ValidationError or any unexpected error
                    iter_error = self._short_error(exc)
            elif iter_error is None:
                iter_error = "no parseable JSON found in model output"

            last_error = iter_error
            if num_repairs >= self._max_repairs:
                break  # repairs exhausted -> graceful failure
            num_repairs += 1
            user = self._build_repair_message(base_user, last_error)

        usage = Usage(tokens_in=tokens_in, tokens_out=tokens_out)
        success = output is not None
        try:
            est_cost = self.client.estimate_cost(self.spec.model_id, usage)
        except Exception:  # cost estimation must never abort the run / lose telemetry
            est_cost = 0.0
        result = AgentResult(
            agent_id=self.spec.agent_id,
            role_name=self.spec.role_name,
            model_id=self.spec.model_id,
            success=success,
            output=output,
            raw_text=last_raw,
            num_repairs=num_repairs,
            latency_ms=latency_ms,
            usage=usage,
            est_cost=est_cost,
            error=None if success else last_error,
            produced_at=datetime.now(timezone.utc).isoformat(),
        )
        await self._log(result, agent_input)
        return result

    # -- prompt assembly --------------------------------------------------------
    def _build_system_prompt(self) -> str:
        return f"{GUARDRAIL_PREAMBLE}\n\nYour role:\n{self.spec.role_description}"

    def _build_user_message(self, agent_input: AgentInput) -> str:
        parts: List[str] = ["PROBLEM (canonical JSON):", agent_input.instance.model_dump_json()]
        if agent_input.sub_problem_id:
            parts.append(f"SUB_PROBLEM_ID: {agent_input.sub_problem_id}")
        if agent_input.draft is not None:
            parts.append("CURRENT DRAFT (candidate to improve):")
            parts.append(agent_input.draft.model_dump_json())
        if agent_input.sibling_outputs:
            parts.append("SIBLING OUTPUTS:")
            parts.append(json.dumps(agent_input.sibling_outputs, default=str))
        instruction = agent_input.instruction or _DEFAULT_INSTRUCTION[self.spec.output_contract]
        parts.append(f"TASK: {instruction}")
        return "\n".join(parts)

    def _build_repair_message(self, base_user: str, error: str) -> str:
        return (
            f"{base_user}\n\nYour previous output failed validation: {error}\n"
            "Return corrected JSON only, conforming to the schema. No prose, no markdown."
        )

    @staticmethod
    def _extract(raw_text: str):
        span = extract_json(raw_text)
        if span is None:
            return None
        value = try_parse_json(span)
        return value if isinstance(value, (dict, list)) else None

    @staticmethod
    def _short_error(exc: ValidationError) -> str:
        text = str(exc).replace("\n", " ")
        return text[:300]

    # -- telemetry (§9) ---------------------------------------------------------
    async def _log(self, result: AgentResult, agent_input: AgentInput) -> None:
        record = {
            "invocation_id": uuid.uuid4().hex,
            "agent_id": result.agent_id,
            "role_name": result.role_name,
            "model_id": result.model_id,
            "thinking_level": self.spec.thinking_level.value,
            "instance_id": agent_input.instance.instance_id,
            "team_genome_id": agent_input.team_genome_id,
            "input_kind": self.spec.input_contract.value,
            "output_kind": self.spec.output_contract.value,
            "success": result.success,
            "num_repairs": result.num_repairs,
            "latency_ms": result.latency_ms,
            "tokens_in": result.usage.tokens_in,
            "tokens_out": result.usage.tokens_out,
            "est_cost": result.est_cost,
            "raw_output": (result.raw_text or "")[:TELEMETRY_RAW_OUTPUT_MAX_CHARS],
            "validated": result.success,
            "error": result.error,
            # Filled later when the team's assembled solution is scored by B1:
            "scorer_fitness": None,
            "scorer_version": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # Defense in depth: the worker owns the never-raises guarantee, so even a
        # non-conforming sink that raises degrades to a local log rather than
        # breaking run(). (``Exception`` is caught — asyncio.CancelledError, a
        # BaseException, still propagates for cooperative cancellation.)
        try:
            await self.telemetry.log_invocation(record)
        except Exception as exc:  # noqa: BLE001
            logger.warning("telemetry sink raised (degraded to local log): %s", exc)
            logger.info("agent_invocation(local): %s", record)
