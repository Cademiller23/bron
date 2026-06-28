"""The OpenAI-compatible provider adapter — for the rest of the fleet.

Drives any ``/v1/chat/completions`` endpoint (MAX, MiniMax, DigitalOcean GenAI,
and even Gemini via its compatibility layer). Written in B2 so the contract is
honest and the hosted fleet is a *config change, not a code change* in B7; it is
exercised mainly there.

* ``base_url`` / ``api_key`` come from the registry entry / environment.
* Structured output via ``response_format`` JSON-schema mode.
* ``thinking_level`` is mapped to OpenAI's ``reasoning_effort`` (which Gemini's
  compat layer, in turn, maps back to ``thinking_level``).
"""

import os
from time import perf_counter
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from darwin.agent.client import (
    ErrorCategory,
    ModelProvider,
    ModelResponse,
    ProviderError,
    Usage,
    is_transient_transport_error,
)
from darwin.agent.parsing import try_parse_json
from darwin.agent.registry import ModelEntry

# Canonical thinking_level -> OpenAI reasoning_effort.
_THINKING_TO_EFFORT: Dict[str, str] = {
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
}

# An injectable async transport: (url, headers, body) -> (status_code, json_dict).
PostJson = Callable[[str, Dict[str, str], Dict[str, Any]], Awaitable[Tuple[int, Dict[str, Any]]]]


class OpenAICompatProvider(ModelProvider):
    def __init__(
        self,
        post_json: Optional[PostJson] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self._post_json = post_json  # inject in tests to avoid the network
        self._api_key = api_key
        self._api_key_env = api_key_env

    async def raw_complete(
        self,
        entry: ModelEntry,
        system: str,
        user: str,
        response_schema: Dict[str, Any],
        thinking_level: str,
        max_output_tokens: int,
    ) -> ModelResponse:
        base = (entry.endpoint or "").rstrip("/")
        url = f"{base}/chat/completions"
        # Key precedence: an explicitly injected key (tests) > the model's own
        # api_key_env from the registry entry (so MAX and MiniMax authenticate with
        # different keys behind one adapter) > the adapter's default env var.
        if self._api_key is not None:
            api_key = self._api_key
        else:
            env_var = getattr(entry, "api_key_env", "") or self._api_key_env
            api_key = os.environ.get(env_var, "")
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        body: Dict[str, Any] = {
            "model": entry.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_output_tokens,
            "reasoning_effort": _THINKING_TO_EFFORT.get(thinking_level, "medium"),
            # strict=False: our schemas legitimately have optional fields (with
            # defaults), which real OpenAI "strict" mode rejects with a 400. The
            # schema is still a strong steer, and parse+validate enforces the rest.
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "agent_output", "schema": response_schema, "strict": False},
            },
            # NOTE: temperature intentionally omitted (provider default).
        }

        start = perf_counter()
        status, data = await self._do_post(url, headers, body)
        latency_ms = (perf_counter() - start) * 1000.0

        if status != 200:
            raise self._map_status(status, data)

        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
            finish = choice.get("finish_reason", "stop")
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(ErrorCategory.OTHER, f"unexpected response shape: {exc}")

        usage_obj = data.get("usage") or {}
        usage = Usage(
            tokens_in=int(usage_obj.get("prompt_tokens", 0) or 0),
            tokens_out=int(usage_obj.get("completion_tokens", 0) or 0),
        )
        parsed = try_parse_json(text)
        return ModelResponse(
            raw_text=text,
            parsed=parsed if isinstance(parsed, dict) else None,
            usage=usage,
            latency_ms=latency_ms,
            model_id=entry.model_id,
            finish_reason=str(finish),
        )

    async def _do_post(
        self, url: str, headers: Dict[str, str], body: Dict[str, Any]
    ) -> Tuple[int, Dict[str, Any]]:
        if self._post_json is not None:
            return await self._post_json(url, headers, body)
        import httpx  # pragma: no cover - real network path

        async with httpx.AsyncClient(timeout=None) as http:  # the client owns the timeout
            try:
                resp = await http.post(url, headers=headers, json=body)
            except httpx.TransportError as exc:
                # connection reset / DNS / pool failure -> retriable, not a crash
                raise ProviderError(ErrorCategory.TRANSIENT, f"{type(exc).__name__}: {exc}") from exc
            try:
                payload = resp.json()
            except Exception:
                payload = {"error": {"message": resp.text}}
            return resp.status_code, payload

    @staticmethod
    def _map_status(status: int, data: Dict[str, Any]) -> ProviderError:
        message = ""
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                message = err.get("message", "")
            elif isinstance(err, str):
                message = err
        message = message or f"HTTP {status}"
        if status == 429:
            return ProviderError(ErrorCategory.RATE_LIMIT, message, status)
        if status in (401, 403):
            return ProviderError(ErrorCategory.AUTH, message, status)
        if 500 <= status < 600:
            return ProviderError(ErrorCategory.SERVER, message, status)
        if 400 <= status < 500:
            return ProviderError(ErrorCategory.BAD_REQUEST, message, status)
        return ProviderError(ErrorCategory.OTHER, message, status)
