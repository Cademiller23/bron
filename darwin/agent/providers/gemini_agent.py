"""The Gemini *managed-agent* provider — the Interactions API / Antigravity.

Where :class:`~darwin.agent.providers.gemini.GeminiProvider` calls a model
(``generate_content`` → one completion), this adapter calls a Google-**hosted
agent** (``client.interactions.create`` → an autonomous run inside an ephemeral,
Google-managed Linux sandbox that can reason, browse the web, and execute code).
The agent id (e.g. ``antigravity-preview-05-2026``) lives in ``entry.model_id``,
so to the rest of Darwin a managed agent is just another **model gene**: the
worker, the runner, and the evolution loop are untouched — only the dispatched
provider differs.

Verified against the live API (``google-genai >= 2.3``):
``client.interactions.create(agent=…, input=…, system_instruction=…,
response_format={"type":"json_schema","json_schema":{…}}, generation_config=…,
environment=…)`` → an Interaction with ``.status``, ``.output_text`` (SDK-added),
``.environment_id``, ``.steps`` and ``.usage`` (``total_input_tokens`` /
``total_output_tokens``).

Disciplines kept identical to the model adapter:
* Structured output via JSON Schema (``response_format`` json_schema) — the
  agent is still asked to emit a scorable ``Solution``; ``output_text`` is parsed
  with the same robust extractor, never trusted to be clean JSON.
* ``temperature`` is never set low.
* Every transport/API error is normalized to a categorized :class:`ProviderError`
  so :class:`ModelClient`'s resilience policy (timeout/retry/circuit-breaker)
  applies unchanged.
"""

import asyncio
from time import perf_counter
from typing import Any, Dict, Optional, Tuple

from darwin.agent.client import (
    ErrorCategory,
    ModelProvider,
    ModelResponse,
    ProviderError,
    Usage,
    is_transient_transport_error,
)
from darwin.agent.parsing import try_parse_json

# The default general-purpose managed agent (current pinned version). Used only
# when an entry's model_id is empty; normally the gene supplies the agent id.
DEFAULT_AGENT_ID = "antigravity-preview-05-2026"

# Interaction.status values that mean "no usable answer" — raised, not parsed.
_TERMINAL_FAILURE = {"failed", "cancelled"}


def _parsed_dict(text: str):
    value = try_parse_json(text)
    return value if isinstance(value, dict) else None


class GeminiAgentProvider(ModelProvider):
    """Adapter for a Google-hosted managed agent via the Interactions API.

    ``reuse_environment`` is the stateful-memory hook: when True the adapter
    threads the sandbox's ``environment_id`` into the next call so files, code,
    and terminal state survive across invocations. It defaults to **False** —
    each genome evaluation is independent, so a fresh ``"remote"`` sandbox per
    call is the correct (and safely concurrent) default; the evolution loop never
    wants one agent's leftover state leaking into another's run.
    """

    def __init__(
        self,
        genai_client: Any = None,
        api_key_env: str = "GEMINI_API_KEY",
        *,
        reuse_environment: bool = False,
    ) -> None:
        # An injected client makes the adapter fully unit-testable without a key.
        self._client = genai_client
        self._api_key_env = api_key_env
        self._reuse_environment = reuse_environment
        self._environment_id: Optional[str] = None

    def _get_client(self):
        if self._client is None:  # pragma: no cover - requires a real key/SDK
            from google import genai

            self._client = genai.Client()  # reads GEMINI_API_KEY from the env
        return self._client

    async def raw_complete(
        self,
        entry,
        system: str,
        user: str,
        response_schema: Dict[str, Any],
        thinking_level: str,  # unused: a managed agent governs its own reasoning
        max_output_tokens: int,
    ) -> ModelResponse:
        client = self._get_client()
        agent_id = entry.model_id or DEFAULT_AGENT_ID
        start = perf_counter()
        try:
            interaction = await self._create_interaction(
                client, agent_id, system, user, response_schema, max_output_tokens
            )
            text, usage, finish, env_id = self._read_interaction(interaction)
        except ProviderError:
            raise
        except Exception as exc:  # normalize SDK/transport errors to a category
            raise self._map_error(exc)

        if self._reuse_environment and env_id:
            self._environment_id = env_id

        latency_ms = (perf_counter() - start) * 1000.0
        return ModelResponse(
            raw_text=text or "",
            parsed=_parsed_dict(text),
            usage=usage,
            latency_ms=latency_ms,
            model_id=entry.model_id,
            finish_reason=finish or "stop",
        )

    # -- the Interactions API call ----------------------------------------------
    async def _create_interaction(
        self, client, agent_id, system, user, response_schema, max_output_tokens
    ):
        # A persisted environment id resumes the sandbox; otherwise provision a
        # fresh remote one. (Independent runs => fresh by default; see __init__.)
        environment = self._environment_id or "remote"
        kwargs = dict(
            agent=agent_id,
            input=user,
            system_instruction=system,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "darwin_solution", "schema": response_schema},
            },
            generation_config={"max_output_tokens": max_output_tokens},
            environment=environment,
        )
        # Prefer the async surface (client.aio.interactions); if only the sync
        # surface exists, run it off the event loop so a real (blocking) sandbox
        # call never stalls concurrent genome evaluations.
        aio = getattr(client, "aio", None)
        aio_interactions = getattr(aio, "interactions", None) if aio is not None else None
        if aio_interactions is not None:
            return await aio_interactions.create(**kwargs)
        return await asyncio.to_thread(lambda: client.interactions.create(**kwargs))

    # -- response extraction ----------------------------------------------------
    def _read_interaction(self, interaction) -> Tuple[str, Usage, str, Optional[str]]:
        status = str(getattr(interaction, "status", "") or "").lower()
        text = getattr(interaction, "output_text", None) or ""
        if status in _TERMINAL_FAILURE:
            # A failed/cancelled run with no answer: surface it so the client can
            # route around the gene (non-retriable — the agent itself gave up).
            raise ProviderError(
                ErrorCategory.OTHER, f"managed agent interaction status={status!r}"
            )
        if status == "budget_exceeded":
            raise ProviderError(
                ErrorCategory.BAD_REQUEST, "managed agent exceeded its budget"
            )
        usage = self._extract_usage(getattr(interaction, "usage", None))
        env_id = getattr(interaction, "environment_id", None)
        return text, usage, status or "completed", env_id

    @staticmethod
    def _extract_usage(meta) -> Usage:
        if meta is None:
            return Usage()
        tokens_in = (
            getattr(meta, "total_input_tokens", None)
            or getattr(meta, "input_tokens", None)
            or getattr(meta, "prompt_token_count", None)
            or 0
        )
        tokens_out = (
            getattr(meta, "total_output_tokens", None)
            or getattr(meta, "output_tokens", None)
            or getattr(meta, "candidates_token_count", None)
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
