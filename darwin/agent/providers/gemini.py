"""The native Gemini provider adapter (the first concrete provider).

Spec target: the new Interactions API + ``thinking_level`` string dial. The
installed ``google-genai`` (1.47) still uses ``models.generate_content`` +
``ThinkingConfig(thinking_budget=...)``, so this adapter **prefers** the
Interactions API when a future SDK exposes ``client.interactions`` and otherwise
falls back to the (currently "Legacy"-labelled) ``generate_content`` surface,
translating the canonical ``thinking_level`` to that surface's knob.

Key disciplines (§2):
* Structured output via the JSON Schema (``response_json_schema`` /
  ``response_format`` schema) — the strongest rung of the strictness ladder.
* ``temperature`` is **never set low** (left at the SDK default) — on Gemini 3.x
  a low temperature causes looping/degradation; reproducibility is the scorer's
  job, not the sampler's.
"""

import inspect
from time import perf_counter
from typing import Any, Dict, Tuple

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

# Canonical thinking_level -> Gemini legacy thinking_budget (tokens).
# -1 == dynamic ("let the model decide"), the natural meaning of the default.
_THINKING_BUDGET: Dict[str, int] = {
    "minimal": 0,
    "low": 1024,
    "medium": -1,
    "high": 24576,
}


def _parsed_dict(text: str):
    value = try_parse_json(text)
    return value if isinstance(value, dict) else None


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class GeminiProvider(ModelProvider):
    def __init__(self, genai_client: Any = None, api_key_env: str = "GEMINI_API_KEY") -> None:
        # An injected client makes the adapter fully unit-testable without a key.
        self._client = genai_client
        self._api_key_env = api_key_env

    def _get_client(self):
        if self._client is None:  # pragma: no cover - requires a real key/SDK
            from google import genai

            self._client = genai.Client()  # reads GEMINI_API_KEY from the env
        return self._client

    async def raw_complete(
        self,
        entry: ModelEntry,
        system: str,
        user: str,
        response_schema: Dict[str, Any],
        thinking_level: str,
        max_output_tokens: int,
    ) -> ModelResponse:
        client = self._get_client()
        start = perf_counter()
        try:
            if hasattr(client, "interactions"):
                text, usage, finish = await self._via_interactions(
                    client, entry, system, user, response_schema, thinking_level, max_output_tokens
                )
            else:
                text, usage, finish = await self._via_generate_content(
                    client, entry, system, user, response_schema, thinking_level, max_output_tokens
                )
        except ProviderError:
            raise
        except Exception as exc:  # normalize SDK/transport errors to a category
            raise self._map_error(exc)

        latency_ms = (perf_counter() - start) * 1000.0
        return ModelResponse(
            raw_text=text or "",
            parsed=_parsed_dict(text),
            usage=usage,
            latency_ms=latency_ms,
            model_id=entry.model_id,
            finish_reason=finish or "stop",
        )

    # -- the installed-SDK path -------------------------------------------------
    async def _via_generate_content(
        self, client, entry, system, user, response_schema, thinking_level, max_output_tokens
    ) -> Tuple[str, Usage, str]:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=response_schema,
            max_output_tokens=max_output_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=_THINKING_BUDGET.get(thinking_level, -1)
            ),
            # NOTE: temperature intentionally left unset (Gemini 3.x default).
        )
        response = await _maybe_await(
            client.aio.models.generate_content(model=entry.model_id, contents=user, config=config)
        )
        text = getattr(response, "text", None) or ""
        usage = self._extract_usage(getattr(response, "usage_metadata", None))
        finish = ""
        candidates = getattr(response, "candidates", None)
        if candidates:
            reason = getattr(candidates[0], "finish_reason", "")
            # real SDK returns a FinishReason enum; prefer its clean .name
            finish = getattr(reason, "name", None) or (str(reason) if reason else "")
        return text, usage, finish

    # -- the forward-compatible Interactions API path ---------------------------
    async def _via_interactions(
        self, client, entry, system, user, response_schema, thinking_level, max_output_tokens
    ) -> Tuple[str, Usage, str]:
        interaction = await _maybe_await(
            client.interactions.create(
                model=entry.model_id,
                input=user,
                system=system,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": response_schema,
                },
                thinking_level=thinking_level,
                max_output_tokens=max_output_tokens,
            )
        )
        text = getattr(interaction, "output_text", None) or ""
        usage = self._extract_usage(getattr(interaction, "usage", None))
        finish = str(getattr(interaction, "finish_reason", "") or "")
        return text, usage, finish

    @staticmethod
    def _extract_usage(meta) -> Usage:
        if meta is None:
            return Usage()
        tokens_in = (
            getattr(meta, "prompt_token_count", None)
            or getattr(meta, "tokens_in", None)
            or getattr(meta, "input_tokens", None)
            or 0
        )
        tokens_out = (
            getattr(meta, "candidates_token_count", None)
            or getattr(meta, "tokens_out", None)
            or getattr(meta, "output_tokens", None)
            or 0
        )
        return Usage(tokens_in=int(tokens_in or 0), tokens_out=int(tokens_out or 0))

    @staticmethod
    def _map_error(exc: Exception) -> ProviderError:
        code = getattr(exc, "code", None)
        if code is None:
            code = getattr(exc, "status_code", None)
        if isinstance(code, str) and code.isdigit():
            code = int(code)
        message = str(exc)
        if code == 429:
            return ProviderError(ErrorCategory.RATE_LIMIT, message, 429)
        if code in (401, 403):
            return ProviderError(ErrorCategory.AUTH, message, code)
        if isinstance(code, int) and 500 <= code < 600:
            return ProviderError(ErrorCategory.SERVER, message, code)
        if isinstance(code, int) and 400 <= code < 500:
            return ProviderError(ErrorCategory.BAD_REQUEST, message, code)
        if is_transient_transport_error(exc):
            return ProviderError(ErrorCategory.TRANSIENT, f"{type(exc).__name__}: {message}")
        return ProviderError(ErrorCategory.OTHER, f"{type(exc).__name__}: {message}")
