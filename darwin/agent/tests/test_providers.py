"""§12.4 Provider-adapter tests — mocked transport, no network."""

import json
from types import SimpleNamespace

import pytest

from darwin.agent.client import ErrorCategory, ModelClient, ModelProvider, ModelResponse, ProviderError, Usage
from darwin.agent.fixtures import valid_full_solution_json
from darwin.agent.outputs import FullSolutionOutput
from darwin.agent.providers.gemini import GeminiProvider
from darwin.agent.providers.openai_compat import OpenAICompatProvider
from darwin.agent.registry import (
    CapabilityTier,
    ModelEntry,
    ModelRegistry,
    Provider,
    default_registry,
    reset_default_registry,
)
from darwin.constants import DEFAULT_MODEL_ID

SCHEMA = FullSolutionOutput.model_json_schema()


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


# ---------------------------------------------------------------------------
# Gemini adapter — installed-SDK (generate_content) path
# ---------------------------------------------------------------------------
class _FakeGenerateContent:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __call__(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        return self.response


def _fake_gemini_client(text):
    # Faithful to the real SDK: finish_reason is a FinishReason enum, not a str.
    from google.genai import types

    resp = SimpleNamespace(
        text=text,
        usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=7),
        candidates=[SimpleNamespace(finish_reason=types.FinishReason.STOP)],
    )
    gc = _FakeGenerateContent(resp)
    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=gc)))
    return client, gc


async def test_gemini_generate_content_request_shape_and_parse():
    text = valid_full_solution_json()
    client, gc = _fake_gemini_client(text)
    provider = GeminiProvider(genai_client=client)
    entry = default_registry().get(DEFAULT_MODEL_ID)

    resp = await provider.raw_complete(entry, "sys", "usr", SCHEMA, "low", 512)

    config = gc.calls[0]["config"]
    assert config.response_json_schema == SCHEMA  # schema carried
    assert config.response_mime_type == "application/json"
    assert config.thinking_config.thinking_budget == 1024  # "low" mapped
    assert config.temperature is None  # NEVER set low (Gemini 3.x looping guard)
    assert resp.parsed is not None and resp.parsed["solution"]["instance_id"] == "golden-transportation"
    assert resp.usage == Usage(tokens_in=11, tokens_out=7)
    assert resp.finish_reason == "STOP"  # the FinishReason enum's clean .name


async def test_gemini_thinking_budget_mapping():
    for level, budget in [("minimal", 0), ("low", 1024), ("medium", -1), ("high", 24576)]:
        client, gc = _fake_gemini_client(valid_full_solution_json())
        await GeminiProvider(genai_client=client).raw_complete(
            default_registry().get(DEFAULT_MODEL_ID), "s", "u", SCHEMA, level, 256
        )
        assert gc.calls[0]["config"].thinking_config.thinking_budget == budget


@pytest.mark.parametrize(
    "code,category",
    [(429, ErrorCategory.RATE_LIMIT), (401, ErrorCategory.AUTH), (503, ErrorCategory.SERVER), (404, ErrorCategory.BAD_REQUEST)],
)
async def test_gemini_maps_api_error_to_category(code, category):
    async def _boom(*, model, contents, config):
        err = RuntimeError("api error")
        err.code = code
        raise err

    client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content=_boom)))
    provider = GeminiProvider(genai_client=client)
    with pytest.raises(ProviderError) as exc:
        await provider.raw_complete(default_registry().get(DEFAULT_MODEL_ID), "s", "u", SCHEMA, "low", 64)
    assert exc.value.category == category


# ---------------------------------------------------------------------------
# Gemini adapter — forward-compatible Interactions API path
# ---------------------------------------------------------------------------
import pytest as _pytest

@_pytest.mark.skip(reason="interactions.create signature is incompatible in google-genai 2.x; "
                          "the adapter deliberately forces the generate_content path (see gemini.py).")
async def test_gemini_interactions_path_when_available():
    captured = {}

    class _FakeInteractions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text=valid_full_solution_json(),
                usage=SimpleNamespace(tokens_in=3, tokens_out=2),
                finish_reason="stop",
            )

    client = SimpleNamespace(interactions=_FakeInteractions())
    resp = await GeminiProvider(genai_client=client).raw_complete(
        default_registry().get(DEFAULT_MODEL_ID), "sys", "usr", SCHEMA, "high", 512
    )
    assert captured["response_format"]["schema"] == SCHEMA
    assert captured["thinking_level"] == "high"
    assert resp.parsed is not None
    assert resp.usage == Usage(tokens_in=3, tokens_out=2)


# ---------------------------------------------------------------------------
# OpenAI-compatible adapter
# ---------------------------------------------------------------------------
def _openai_response(text):
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4},
    }


async def test_openai_compat_request_shape_and_parse():
    captured = {}

    async def fake_post(url, headers, body):
        captured.update(url=url, headers=headers, body=body)
        return 200, _openai_response(valid_full_solution_json())

    entry = ModelEntry(
        model_id="fleet-max", provider=Provider.OPENAI_COMPAT, endpoint="https://do.example/v1",
        capability_tier=CapabilityTier.FRONTIER,
    )
    provider = OpenAICompatProvider(post_json=fake_post, api_key="sk-test")
    resp = await provider.raw_complete(entry, "sys", "usr", SCHEMA, "high", 777)

    assert captured["url"] == "https://do.example/v1/chat/completions"  # base_url from registry
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["body"]["response_format"]["json_schema"]["schema"] == SCHEMA  # json-schema mode
    # strict=False: our schemas have optional fields, which real OpenAI strict mode rejects (400)
    assert captured["body"]["response_format"]["json_schema"]["strict"] is False
    assert captured["body"]["reasoning_effort"] == "high"  # thinking_level -> reasoning_effort
    assert captured["body"]["max_tokens"] == 777
    assert "temperature" not in captured["body"]  # never set low
    assert resp.parsed is not None
    assert resp.usage == Usage(tokens_in=9, tokens_out=4)


@pytest.mark.parametrize(
    "status,category",
    [(429, ErrorCategory.RATE_LIMIT), (401, ErrorCategory.AUTH), (503, ErrorCategory.SERVER), (400, ErrorCategory.BAD_REQUEST)],
)
async def test_openai_compat_maps_status_to_category(status, category):
    async def fake_post(url, headers, body):
        return status, {"error": {"message": "x"}}

    entry = ModelEntry(model_id="m", provider=Provider.OPENAI_COMPAT, endpoint="https://x/v1")
    with pytest.raises(ProviderError) as exc:
        await OpenAICompatProvider(post_json=fake_post, api_key="k").raw_complete(entry, "s", "u", SCHEMA, "low", 64)
    assert exc.value.category == category


async def test_openai_compat_reasoning_effort_mapping():
    seen = {}

    async def fake_post(url, headers, body):
        seen[body["model"]] = body["reasoning_effort"]
        return 200, _openai_response(valid_full_solution_json())

    entry = ModelEntry(model_id="m", provider=Provider.OPENAI_COMPAT, endpoint="https://x/v1")
    for level, effort in [("minimal", "minimal"), ("low", "low"), ("medium", "medium"), ("high", "high")]:
        await OpenAICompatProvider(post_json=fake_post, api_key="k").raw_complete(entry, "s", "u", SCHEMA, level, 64)
    assert seen["m"] == "high"  # last call


# ---------------------------------------------------------------------------
# Dispatch — ModelClient routes purely on the registry's provider
# ---------------------------------------------------------------------------
class _Marker(ModelProvider):
    def __init__(self, tag):
        self.tag = tag

    async def raw_complete(self, entry, system, user, response_schema, thinking_level, max_output_tokens):
        return ModelResponse(raw_text=self.tag, model_id=entry.model_id, finish_reason="stop")


async def test_client_dispatches_by_provider():
    reg = ModelRegistry(
        {
            "g": ModelEntry(model_id="g", provider=Provider.GEMINI),
            "o": ModelEntry(model_id="o", provider=Provider.OPENAI_COMPAT, endpoint="https://x/v1"),
        }
    )
    client = ModelClient(
        registry=reg,
        adapters={Provider.GEMINI: _Marker("GEM"), Provider.OPENAI_COMPAT: _Marker("OAI")},
    )
    g = await client.complete("g", "s", "u", SCHEMA, "low", 64)
    o = await client.complete("o", "s", "u", SCHEMA, "low", 64)
    assert g.raw_text == "GEM"
    assert o.raw_text == "OAI"
