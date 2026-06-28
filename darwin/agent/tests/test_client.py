"""ModelClient resilience (§10): timeout, retry/backoff, fail-fast, circuit breaker."""

import asyncio

import pytest

from darwin.agent.client import ErrorCategory, ModelClient, ModelProvider, ModelResponse, ProviderError, Usage
from darwin.agent.fixtures import ScriptedProvider, response
from darwin.agent.outputs import FullSolutionOutput
from darwin.agent.registry import ModelEntry, ModelRegistry, Provider, default_registry, reset_default_registry
from darwin.constants import DEFAULT_MODEL_ID

SCHEMA = FullSolutionOutput.model_json_schema()


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


async def _noop_sleep(_):
    return None


def _client(script, **kw):
    return ModelClient(
        registry=default_registry(),
        adapters={Provider.GEMINI: ScriptedProvider(script)},
        sleep=_noop_sleep,
        retry_base_delay=0.0,
        **kw,
    )


def _provider(client):
    return client._adapters[Provider.GEMINI]


async def test_timeout_returns_graceful_error():
    class _Hang(ModelProvider):
        async def raw_complete(self, *a, **k):
            await asyncio.sleep(10)

    client = ModelClient(registry=default_registry(), adapters={Provider.GEMINI: _Hang()}, sleep=_noop_sleep)
    resp = await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64, timeout=0.01)
    assert resp.error is not None and "TIMEOUT" in resp.error


async def test_rate_limit_then_success_retries():
    client = _client([ProviderError(ErrorCategory.RATE_LIMIT, "429"), response(raw_text="ok")])
    resp = await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64)
    assert resp.error is None and resp.raw_text == "ok"
    assert len(_provider(client).calls) == 2  # retried once


async def test_auth_error_fails_fast_without_retry():
    client = _client([ProviderError(ErrorCategory.AUTH, "401"), response(raw_text="ok")])
    resp = await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64)
    assert resp.error is not None and "AUTH" in resp.error
    assert len(_provider(client).calls) == 1  # no pointless retries


async def test_server_error_retries_then_succeeds():
    client = _client(
        [ProviderError(ErrorCategory.SERVER, "503"), ProviderError(ErrorCategory.SERVER, "503"), response(raw_text="ok")]
    )
    resp = await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64)
    assert resp.error is None and resp.raw_text == "ok"
    assert len(_provider(client).calls) == 3


async def test_retries_exhausted_returns_error():
    client = _client([ProviderError(ErrorCategory.SERVER, "503")] * 10, max_retries=2)
    resp = await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64)
    assert resp.error is not None and "SERVER" in resp.error
    assert len(_provider(client).calls) == 3  # 1 initial + 2 retries


async def test_circuit_breaker_marks_model_degraded_then_recovers():
    reg = default_registry()
    client = ModelClient(
        registry=reg,
        adapters={Provider.GEMINI: ScriptedProvider([ProviderError(ErrorCategory.SERVER, "x"), ProviderError(ErrorCategory.SERVER, "x"), response(raw_text="ok")])},
        sleep=_noop_sleep,
        retry_base_delay=0.0,
        max_retries=0,  # one attempt per complete() so each call is one "failure"
        circuit_threshold=2,
    )
    assert reg.is_degraded(DEFAULT_MODEL_ID) is False
    await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64)  # fail 1
    assert reg.is_degraded(DEFAULT_MODEL_ID) is False
    await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64)  # fail 2 -> degraded
    assert reg.is_degraded(DEFAULT_MODEL_ID) is True
    ok = await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64)  # success -> recover
    assert ok.error is None
    assert reg.is_degraded(DEFAULT_MODEL_ID) is False


async def test_unknown_model_id_returns_error_not_raises():
    client = ModelClient(registry=default_registry())
    resp = await client.complete("ghost-model", "s", "u", SCHEMA, "low", 64)  # must not raise
    assert resp.error is not None and "ghost-model" in resp.error


async def test_transient_transport_error_is_retried():
    client = _client([ProviderError(ErrorCategory.TRANSIENT, "connection reset"), response(raw_text="ok")])
    resp = await client.complete(DEFAULT_MODEL_ID, "s", "u", SCHEMA, "low", 64)
    assert resp.error is None and resp.raw_text == "ok"
    assert len(_provider(client).calls) == 2  # transient faults are retriable


async def test_estimate_cost_uses_registry_rates():
    client = ModelClient(registry=default_registry())
    entry = default_registry().get(DEFAULT_MODEL_ID)
    expected = (1000 / 1000) * entry.est_cost_per_1k_in + (500 / 1000) * entry.est_cost_per_1k_out
    assert client.estimate_cost(DEFAULT_MODEL_ID, Usage(tokens_in=1000, tokens_out=500)) == expected
