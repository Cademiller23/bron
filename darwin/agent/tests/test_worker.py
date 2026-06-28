"""§12.5 WorkerAgent pipeline tests — mocked client, the largest suite."""

import pytest

from darwin.agent import fixtures as F
from darwin.agent.client import ErrorCategory, ModelClient, ProviderError
from darwin.agent.outputs import FullSolutionOutput
from darwin.agent.registry import (
    CapabilityTier,
    ModelEntry,
    Provider,
    default_registry,
    reset_default_registry,
)
from darwin.agent.spec import OutputKind
from darwin.agent.telemetry import InMemoryTelemetrySink
from darwin.agent.worker import AgentResult, WorkerAgent


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


# ---------------------------------------------------------------------------
# Core pipeline paths
# ---------------------------------------------------------------------------
async def test_happy_path_native_schema():
    worker, telemetry, _ = F.make_worker([F.native_schema_response()])
    result = await worker.run(F.make_input())
    assert result.success is True
    assert isinstance(result.output, FullSolutionOutput)
    assert result.num_repairs == 0
    assert result.error is None
    assert len(telemetry.invocations) == 1


async def test_free_text_then_json_is_recovered():
    worker, _, _ = F.make_worker([F.prose_wrapped_response()])
    result = await worker.run(F.make_input())
    assert result.success is True
    assert result.num_repairs == 0  # the extractor recovered it without a repair


async def test_malformed_then_valid_repairs_once():
    worker, _, client = F.make_worker([F.malformed_json_response(), F.native_schema_response()])
    result = await worker.run(F.make_input())
    assert result.success is True
    assert result.num_repairs == 1
    assert len(client._adapters[Provider.GEMINI].calls) == 2


async def test_schema_violation_then_valid_repairs():
    worker, _, _ = F.make_worker([F.schema_violation_response(), F.native_schema_response()])
    result = await worker.run(F.make_input())
    assert result.success is True
    assert result.num_repairs == 1


async def test_hallucinated_extra_field_is_rejected_then_repaired():
    # The extra="forbid" rung: a complete, valid payload + one hallucinated key
    # must fail validation and trigger a repair.
    worker, _, _ = F.make_worker([F.extra_field_response(), F.native_schema_response()])
    result = await worker.run(F.make_input())
    assert result.success is True
    assert result.num_repairs == 1


async def test_hallucinated_extra_field_alone_fails_gracefully():
    worker, telemetry, _ = F.make_worker([F.extra_field_response()])
    result = await worker.run(F.make_input())
    assert result.success is False
    assert result.error is not None
    assert len(telemetry.invocations) == 1


async def test_pathological_nested_output_never_crashes_run():
    # Deeply-nested brackets make json.loads recurse past the limit; run() must
    # degrade gracefully (never raise) and still write exactly one telemetry doc.
    for raw in ("[" * 1100, "[" * 1100 + "]" * 1100):
        worker, telemetry, _ = F.make_worker([F.response(raw_text=raw, parsed=None)])
        result = await worker.run(F.make_input())  # must not raise
        assert result.success is False
        assert len(telemetry.invocations) == 1


async def test_cyclic_sibling_outputs_does_not_crash_run():
    # Prompt assembly (json.dumps of sibling_outputs) runs before the loop; a
    # cyclic value must hit the run() backstop, not escape.
    cyclic = {}
    cyclic["self"] = cyclic
    worker, telemetry, _ = F.make_worker([F.native_schema_response()])
    result = await worker.run(F.make_input(sibling_outputs=[cyclic]))  # must not raise
    assert result.success is False
    assert "pipeline error" in (result.error or "")
    assert len(telemetry.invocations) == 1


async def test_raising_telemetry_sink_does_not_break_run(caplog):
    import logging

    class _BoomSink:
        async def log_invocation(self, record):
            raise RuntimeError("sink exploded")

        async def save_corpus_spec(self, record):
            return None

    spec = F.make_spec()
    client = F.scripted_client([F.native_schema_response()])
    worker = WorkerAgent(spec, client, _BoomSink())
    with caplog.at_level(logging.WARNING, logger="darwin.agent.worker"):
        result = await worker.run(F.make_input())  # must not raise
    assert result.success is True
    assert any("degraded to local log" in r.message for r in caplog.records)


async def test_repairs_exhausted_fails_gracefully():
    worker, telemetry, client = F.make_worker([F.malformed_json_response()])  # repeats forever
    result = await worker.run(F.make_input())  # must not raise
    assert result.success is False
    assert result.output is None
    assert result.error is not None
    assert result.num_repairs == 2  # MAX_REPAIRS
    assert len(client._adapters[Provider.GEMINI].calls) == 3  # 1 + 2 repairs
    assert len(telemetry.invocations) == 1


async def test_transport_error_fails_gracefully_without_repair():
    worker, _, client = F.make_worker([F.response(error="[TIMEOUT] model call exceeded 30s")])
    result = await worker.run(F.make_input())
    assert result.success is False
    assert "TIMEOUT" in result.error
    assert result.num_repairs == 0  # transport errors are not repaired
    assert len(client._adapters[Provider.GEMINI].calls) == 1


async def test_rate_limit_then_success_via_client_retry():
    worker, _, _ = F.make_worker([ProviderError(ErrorCategory.RATE_LIMIT, "429"), F.native_schema_response()])
    result = await worker.run(F.make_input())
    assert result.success is True  # the client retried under the hood
    assert result.num_repairs == 0


async def test_auth_error_fails_fast():
    worker, _, client = F.make_worker([ProviderError(ErrorCategory.AUTH, "401 missing key"), F.native_schema_response()])
    result = await worker.run(F.make_input())
    assert result.success is False
    assert "AUTH" in result.error
    assert len(client._adapters[Provider.GEMINI].calls) == 1  # no pointless retries


# ---------------------------------------------------------------------------
# Provenance, prompt assembly, cost
# ---------------------------------------------------------------------------
async def test_result_output_union_survives_round_trip():
    # B3 may serialize/reload results; the output member type must not be coerced.
    from darwin.agent.outputs import FullSolutionOutput

    worker, _, _ = F.make_worker([F.native_schema_response()])
    result = await worker.run(F.make_input())
    assert isinstance(result.output, FullSolutionOutput)
    reloaded = AgentResult.model_validate(result.model_dump())
    assert isinstance(reloaded.output, FullSolutionOutput)
    assert reloaded == result


async def test_result_carries_full_provenance():
    spec = F.make_spec(agent_id="cap-aud-7")
    worker, _, _ = F.make_worker([F.native_schema_response()], spec=spec)
    result = await worker.run(F.make_input())
    assert result.agent_id == "cap-aud-7"
    assert result.role_name == spec.role_name
    assert result.model_id == spec.model_id
    assert result.latency_ms > 0.0
    assert result.est_cost > 0.0
    assert result.usage.tokens_in > 0
    assert result.produced_at


async def test_system_prompt_has_guardrail_and_role():
    spec = F.make_spec(role_description="Audit node capacities and flag overflows.")
    worker, _, _ = F.make_worker([F.native_schema_response()], spec=spec)
    system = worker._build_system_prompt()
    assert "Output ONLY JSON" in system  # guardrail preamble
    assert "Audit node capacities" in system  # the role


async def test_user_message_includes_problem_and_task():
    worker, _, _ = F.make_worker([F.native_schema_response()])
    user = worker._build_user_message(F.make_input())
    assert "PROBLEM (canonical JSON)" in user
    assert "golden-transportation" in user
    assert "TASK:" in user


# ---------------------------------------------------------------------------
# Model-agnosticism (§14): switching model_id changes provider, not worker.py
# ---------------------------------------------------------------------------
async def test_worker_is_model_agnostic_across_providers():
    from darwin.agent.spec import AgentSpec, InputKind

    # Register a fleet model behind the OpenAI-compatible provider...
    default_registry().register(
        ModelEntry(
            model_id="fleet-max", provider=Provider.OPENAI_COMPAT, endpoint="https://do/v1",
            capability_tier=CapabilityTier.FRONTIER, est_cost_per_1k_in=0.001, est_cost_per_1k_out=0.002,
        )
    )
    spec = AgentSpec(
        agent_id="a", role_name="cost_minimizer", role_description="Minimize cost.",
        input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION, model_id="fleet-max",
    )
    # ...and the *same* WorkerAgent code drives it, only the provider changes.
    client = F.scripted_client([F.native_schema_response()], provider=Provider.OPENAI_COMPAT)
    worker = WorkerAgent(spec, client, InMemoryTelemetrySink())
    result = await worker.run(F.make_input())
    assert result.success is True
    assert result.model_id == "fleet-max"


# ---------------------------------------------------------------------------
# Telemetry on every path + no-grading invariant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "script",
    [
        [F.native_schema_response()],  # success
        [F.malformed_json_response(), F.native_schema_response()],  # repaired success
        [F.malformed_json_response()],  # failure
        [F.response(error="[TIMEOUT] x")],  # transport failure
    ],
)
async def test_telemetry_written_exactly_once_on_every_path(script):
    worker, telemetry, _ = F.make_worker(script)
    await worker.run(F.make_input())
    assert len(telemetry.invocations) == 1
    doc = telemetry.invocations[0]
    for key in ("invocation_id", "agent_id", "model_id", "success", "num_repairs", "est_cost", "instance_id", "created_at"):
        assert key in doc


async def test_telemetry_records_team_genome_id():
    worker, telemetry, _ = F.make_worker([F.native_schema_response()])
    await worker.run(F.make_input(team_genome_id="genome-42"))
    assert telemetry.invocations[0]["team_genome_id"] == "genome-42"


async def test_worker_never_grades(monkeypatch):
    # If the worker ever called the scorer, this patched scorer would explode.
    import darwin.problem.scorer as scorer_mod

    def _boom(*a, **k):
        raise AssertionError("the worker must never call the scorer")

    monkeypatch.setattr(scorer_mod, "score", _boom)
    worker, _, _ = F.make_worker([F.native_schema_response()])
    result = await worker.run(F.make_input())
    assert result.success is True

    # structural: the result has no fitness field, and worker.py imports no scorer
    assert "fitness" not in AgentResult.model_fields
    assert not any("fitness" in f for f in AgentResult.model_fields)
    import darwin.agent.worker as worker_mod

    assert "score" not in vars(worker_mod)
    assert "scorer" not in vars(worker_mod)
