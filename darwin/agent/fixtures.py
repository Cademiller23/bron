"""Canned model responses and test doubles for deterministic, API-free tests.

Nothing here touches the network. The ``ScriptedProvider`` plugs into a real
``ModelClient`` so worker tests exercise the full client → adapter → response
path while feeding pre-canned outcomes (valid JSON, prose-wrapped JSON,
malformed JSON, schema violations, transport errors).
"""

import json
from typing import Any, Dict, List, Optional, Union

from darwin.agent.client import ModelClient, ModelProvider, ModelResponse, ProviderError, Usage
from darwin.agent.outputs import FullSolutionOutput
from darwin.agent.registry import Provider, default_registry, reset_default_registry
from darwin.agent.spec import AgentSpec, InputKind, OutputKind
from darwin.agent.telemetry import InMemoryTelemetrySink
from darwin.agent.worker import AgentInput, WorkerAgent
from darwin.constants import DEFAULT_MODEL_ID
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import FlowAssignment, ProblemInstance, Solution

# ---------------------------------------------------------------------------
# Canned valid output (a real, scorable FullSolutionOutput for golden transport)
# ---------------------------------------------------------------------------
def valid_full_solution_json(instance_id: str = "golden-transportation") -> str:
    out = FullSolutionOutput(
        solution=Solution(
            solution_id="cand-1",
            instance_id=instance_id,
            flows=[FlowAssignment(arc_id="S1-D1", quantity=8.0), FlowAssignment(arc_id="S2-D2", quantity=7.0)],
            produced_by="worker-test",
        ),
        rationale="route each sink from its cheapest source",
    )
    return out.model_dump_json()


def response(
    raw_text: str = "",
    parsed: Optional[Dict[str, Any]] = None,
    tokens_in: int = 40,
    tokens_out: int = 25,
    latency_ms: float = 12.0,
    finish_reason: str = "stop",
    error: Optional[str] = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> ModelResponse:
    return ModelResponse(
        raw_text=raw_text,
        parsed=parsed,
        usage=Usage(tokens_in=tokens_in, tokens_out=tokens_out),
        latency_ms=latency_ms,
        model_id=model_id,
        finish_reason=finish_reason,
        error=error,
    )


def native_schema_response(instance_id: str = "golden-transportation") -> ModelResponse:
    """A native-schema-mode success: both raw_text and parsed are populated."""
    text = valid_full_solution_json(instance_id)
    return response(raw_text=text, parsed=json.loads(text))


def prose_wrapped_response(instance_id: str = "golden-transportation") -> ModelResponse:
    """Valid JSON buried in prose/markdown — the extractor must recover it."""
    text = valid_full_solution_json(instance_id)
    wrapped = f"Sure! Here is the solution:\n```json\n{text}\n```\nHope this helps."
    return response(raw_text=wrapped, parsed=None)


def malformed_json_response() -> ModelResponse:
    return response(raw_text="{ this is not valid json ", parsed=None)


def schema_violation_response() -> ModelResponse:
    """Valid JSON, wrong shape for FullSolutionOutput (extra/unknown structure)."""
    bad = {"totally": "wrong", "shape": [1, 2, 3]}
    return response(raw_text=json.dumps(bad), parsed=bad)


def extra_field_response(instance_id: str = "golden-transportation") -> ModelResponse:
    """A *complete, valid* FullSolutionOutput PLUS one hallucinated extra key.

    Isolates the ``extra="forbid"`` rung: the payload is otherwise perfect, so
    only the hallucinated field should cause a validation failure."""
    payload = json.loads(valid_full_solution_json(instance_id))
    payload["hallucinated_field"] = {"confidence": 0.99}
    return response(raw_text=json.dumps(payload), parsed=payload)


# ---------------------------------------------------------------------------
# Scripted provider + client
# ---------------------------------------------------------------------------
class ScriptedProvider(ModelProvider):
    """Returns / raises pre-canned outcomes in order; records every call."""

    def __init__(self, script: List[Union[ModelResponse, BaseException]]) -> None:
        self._script = list(script)
        self._i = 0
        self.calls: List[Dict[str, Any]] = []

    async def raw_complete(self, entry, system, user, response_schema, thinking_level, max_output_tokens):
        self.calls.append(
            {
                "model_id": entry.model_id,
                "system": system,
                "user": user,
                "response_schema": response_schema,
                "thinking_level": thinking_level,
                "max_output_tokens": max_output_tokens,
            }
        )
        item = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        if isinstance(item, BaseException):
            raise item
        return item


async def _noop_sleep(_seconds: float) -> None:
    return None


def scripted_client(
    script: List[Union[ModelResponse, BaseException]],
    *,
    registry=None,
    provider: Provider = Provider.GEMINI,
    **client_kwargs,
) -> ModelClient:
    """A real ModelClient whose dispatched provider is scripted (no network)."""
    prov = ScriptedProvider(script)
    return ModelClient(
        registry=registry or default_registry(),
        adapters={provider: prov},
        sleep=_noop_sleep,
        retry_base_delay=0.0,
        **client_kwargs,
    )


# ---------------------------------------------------------------------------
# Spec / input / worker builders
# ---------------------------------------------------------------------------
def make_spec(
    *,
    agent_id: str = "agent-1",
    role_name: str = "cost_minimizer",
    role_description: str = "Minimize total transportation cost while meeting all demand.",
    input_contract: InputKind = InputKind.FULL_PROBLEM,
    output_contract: OutputKind = OutputKind.FULL_SOLUTION,
    **overrides,
) -> AgentSpec:
    reset_default_registry()
    return AgentSpec(
        agent_id=agent_id,
        role_name=role_name,
        role_description=role_description,
        input_contract=input_contract,
        output_contract=output_contract,
        **overrides,
    )


def make_input(instance: Optional[ProblemInstance] = None, **overrides) -> AgentInput:
    return AgentInput(instance=instance or golden_transportation(), **overrides)


def make_worker(script: List[Union[ModelResponse, BaseException]], spec: Optional[AgentSpec] = None, **client_kwargs):
    """Build (worker, telemetry, client) wired with a scripted provider."""
    spec = spec or make_spec()
    telemetry = InMemoryTelemetrySink()
    client = scripted_client(script, **client_kwargs)
    worker = WorkerAgent(spec, client, telemetry)
    return worker, telemetry, client


# Need BaseException name in annotations above; re-export for convenience.
__all__ = [
    "valid_full_solution_json",
    "response",
    "native_schema_response",
    "prose_wrapped_response",
    "malformed_json_response",
    "schema_violation_response",
    "extra_field_response",
    "ScriptedProvider",
    "scripted_client",
    "make_spec",
    "make_input",
    "make_worker",
]
