"""Tests for the managed-agent provider (Interactions API / Antigravity).

Fully offline: a fake client captures the ``interactions.create`` kwargs and
returns a canned Interaction, so the request shape, response parsing, usage
mapping, status handling, environment reuse, and error categorization are all
asserted without a key or a network. Mirrors §12.4's provider-adapter style.
"""

from types import SimpleNamespace

import pytest

from darwin.agent.client import ErrorCategory, ModelClient, ProviderError, Usage
from darwin.agent.fixtures import valid_full_solution_json
from darwin.agent.outputs import FullSolutionOutput
from darwin.agent.providers.gemini_agent import GeminiAgentProvider
from darwin.agent.registry import (
    CapabilityTier,
    ModelEntry,
    ModelRegistry,
    Provider,
)

SCHEMA = FullSolutionOutput.model_json_schema()
AGENT_ID = "antigravity-preview-05-2026"


def _entry(model_id: str = AGENT_ID) -> ModelEntry:
    return ModelEntry(
        model_id=model_id,
        provider=Provider.GEMINI_AGENT,
        capability_tier=CapabilityTier.FRONTIER,
        supports_native_schema=True,
    )


class _FakeInteractions:
    def __init__(self, interaction):
        self.interaction = interaction
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.interaction


def _interaction(text, *, status="completed", env_id="env_abc123", tin=11, tout=7):
    return SimpleNamespace(
        id="int_1",
        status=status,
        output_text=text,
        environment_id=env_id,
        usage=SimpleNamespace(total_input_tokens=tin, total_output_tokens=tout),
        steps=[],
    )


def _client_with(interaction):
    fake = _FakeInteractions(interaction)
    # Async surface only (client.aio.interactions), mirroring the real SDK.
    client = SimpleNamespace(aio=SimpleNamespace(interactions=fake))
    return client, fake


async def test_request_shape_and_parse():
    text = valid_full_solution_json()
    client, fake = _client_with(_interaction(text))
    provider = GeminiAgentProvider(genai_client=client)

    resp = await provider.raw_complete(_entry(), "sys", "usr", SCHEMA, "high", 512)

    sent = fake.calls[0]
    assert sent["agent"] == AGENT_ID  # the gene's model_id is the agent id
    assert sent["input"] == "usr"
    assert sent["system_instruction"] == "sys"
    assert sent["response_format"]["json_schema"]["schema"] == SCHEMA  # schema carried
    assert sent["generation_config"]["max_output_tokens"] == 512
    assert sent["environment"] == "remote"  # fresh sandbox by default
    assert resp.parsed is not None
    assert resp.parsed["solution"]["instance_id"] == "golden-transportation"
    assert resp.usage == Usage(tokens_in=11, tokens_out=7)
    assert resp.finish_reason == "completed"


async def test_environment_not_reused_by_default():
    client, fake = _client_with(_interaction(valid_full_solution_json()))
    provider = GeminiAgentProvider(genai_client=client)  # reuse_environment=False
    await provider.raw_complete(_entry(), "s", "u", SCHEMA, "high", 64)
    await provider.raw_complete(_entry(), "s", "u", SCHEMA, "high", 64)
    assert [c["environment"] for c in fake.calls] == ["remote", "remote"]


async def test_environment_reused_when_enabled():
    client, fake = _client_with(_interaction(valid_full_solution_json(), env_id="env_xyz"))
    provider = GeminiAgentProvider(genai_client=client, reuse_environment=True)
    await provider.raw_complete(_entry(), "s", "u", SCHEMA, "high", 64)
    await provider.raw_complete(_entry(), "s", "u", SCHEMA, "high", 64)
    # First call provisions a fresh remote sandbox; the second resumes it by id.
    assert [c["environment"] for c in fake.calls] == ["remote", "env_xyz"]


async def test_sync_only_client_runs_off_the_event_loop():
    # A client exposing only the sync surface must still work (via to_thread).
    captured = {}

    class _SyncInteractions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _interaction(valid_full_solution_json())

    client = SimpleNamespace(interactions=_SyncInteractions())
    resp = await GeminiAgentProvider(genai_client=client).raw_complete(
        _entry(), "s", "u", SCHEMA, "high", 64
    )
    assert captured["agent"] == AGENT_ID
    assert resp.parsed is not None


async def test_failed_status_raises():
    client, _ = _client_with(_interaction("", status="failed"))
    with pytest.raises(ProviderError) as exc:
        await GeminiAgentProvider(genai_client=client).raw_complete(
            _entry(), "s", "u", SCHEMA, "high", 64
        )
    assert exc.value.category == ErrorCategory.OTHER


async def test_budget_exceeded_status_raises_bad_request():
    client, _ = _client_with(_interaction("", status="budget_exceeded"))
    with pytest.raises(ProviderError) as exc:
        await GeminiAgentProvider(genai_client=client).raw_complete(
            _entry(), "s", "u", SCHEMA, "high", 64
        )
    assert exc.value.category == ErrorCategory.BAD_REQUEST


@pytest.mark.parametrize(
    "code,category",
    [
        (429, ErrorCategory.RATE_LIMIT),
        (401, ErrorCategory.AUTH),
        (503, ErrorCategory.SERVER),
        (404, ErrorCategory.BAD_REQUEST),
    ],
)
async def test_maps_api_error_to_category(code, category):
    class _Boom:
        async def create(self, **kwargs):
            err = RuntimeError("api error")
            err.code = code
            raise err

    client = SimpleNamespace(aio=SimpleNamespace(interactions=_Boom()))
    with pytest.raises(ProviderError) as exc:
        await GeminiAgentProvider(genai_client=client).raw_complete(
            _entry(), "s", "u", SCHEMA, "high", 64
        )
    assert exc.value.category == category


async def test_client_dispatches_gemini_agent_provider():
    # The ModelClient resolves the GEMINI_AGENT provider and routes the call,
    # so a managed agent is dispatchable as a model gene end-to-end.
    registry = ModelRegistry({AGENT_ID: _entry()})
    client_obj, fake = _client_with(_interaction(valid_full_solution_json()))
    mc = ModelClient(
        registry=registry,
        adapters={Provider.GEMINI_AGENT: GeminiAgentProvider(genai_client=client_obj)},
    )
    resp = await mc.complete(AGENT_ID, "s", "u", SCHEMA, "high", 128)
    assert resp.error is None
    assert resp.parsed is not None
    assert fake.calls[0]["agent"] == AGENT_ID


def test_managed_agent_is_an_optin_fleet_model():
    from darwin.routing.fleet import managed_profile, profile

    m = managed_profile(AGENT_ID)
    assert m.provider == Provider.GEMINI_AGENT
    assert m.tier == CapabilityTier.FRONTIER
    assert m.endpoint == ""  # Google-hosted; no endpoint required
    # Opt-in by design: it is NOT in the deterministic curated fleet.
    with pytest.raises(KeyError):
        profile(AGENT_ID)


def test_install_managed_agents_makes_it_dispatchable():
    from darwin.routing.fleet import install_managed_agents

    reg = ModelRegistry()
    install_managed_agents(reg)
    assert reg.contains(AGENT_ID)
    assert reg.get(AGENT_ID).provider == Provider.GEMINI_AGENT
