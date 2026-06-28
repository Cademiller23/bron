"""The model-agnostic client — the abstraction the worker calls.

The worker calls ``ModelClient.complete(...)`` and never knows which provider
ran. The client looks the ``model_id`` up in the registry, dispatches to the
matching provider adapter, applies the §10 resilience policy (per-call timeout,
bounded retry/backoff for 429/5xx, fail-fast on auth, a consecutive-failure
circuit breaker), and normalizes everything into a uniform :class:`ModelResponse`.

Adapters return a :class:`ModelResponse` on a successful transport call and
**raise** a categorized :class:`ProviderError` on an API/transport failure;
``complete`` converts a final failure into a ``ModelResponse`` with ``error``
set, so the worker always gets a response and never has to catch exceptions.
"""

import asyncio
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from darwin.agent.registry import ModelEntry, ModelRegistry, Provider, default_registry
from darwin.constants import (
    CIRCUIT_BREAKER_THRESHOLD,
    DEFAULT_TIMEOUT_S,
    MAX_API_RETRIES,
    RETRY_BASE_DELAY_S,
)


class ErrorCategory(str, Enum):
    TIMEOUT = "TIMEOUT"
    RATE_LIMIT = "RATE_LIMIT"  # 429 -> backoff + retry
    AUTH = "AUTH"  # 401/403 -> fail fast
    SERVER = "SERVER"  # 5xx -> bounded retry
    TRANSIENT = "TRANSIENT"  # connection reset / read timeout / transport fault -> bounded retry
    BAD_REQUEST = "BAD_REQUEST"  # 4xx (non-auth/non-429) -> no retry
    OTHER = "OTHER"


# Categories worth a bounded retry with backoff.
RETRIABLE = (ErrorCategory.RATE_LIMIT, ErrorCategory.SERVER, ErrorCategory.TRANSIENT)


def is_transient_transport_error(exc: BaseException) -> bool:
    """A connection reset / read timeout / DNS failure with no HTTP status —
    worth a bounded retry rather than a terminal failure. Shared by adapters."""
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    try:
        import httpx

        if isinstance(exc, (httpx.TransportError, httpx.TimeoutException)):
            return True
    except Exception:  # pragma: no cover - httpx is installed in this project
        pass
    name = type(exc).__name__.lower()
    return any(token in name for token in ("connect", "timeout", "transport", "unavailable"))


class ProviderError(Exception):
    """A normalized provider/transport error every adapter raises."""

    def __init__(self, category: ErrorCategory, message: str, status_code: Optional[int] = None):
        super().__init__(f"[{category.value}] {message}")
        self.category = category
        self.message = message
        self.status_code = status_code


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)


class ModelResponse(BaseModel):
    """The uniform shape every provider returns — the worker never branches on
    provider type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_text: str = ""
    parsed: Optional[Dict[str, Any]] = None  # filled when the provider returns parseable JSON
    usage: Usage = Field(default_factory=Usage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    model_id: str = ""
    finish_reason: str = ""
    error: Optional[str] = None  # set only on a normalized transport failure


class ModelProvider:
    """Base class for provider adapters. One raw attempt; raise on API failure."""

    async def raw_complete(
        self,
        entry: ModelEntry,
        system: str,
        user: str,
        response_schema: Dict[str, Any],
        thinking_level: str,
        max_output_tokens: int,
    ) -> ModelResponse:  # pragma: no cover - abstract
        raise NotImplementedError


class ModelClient:
    """Registry-backed, resilient, model-agnostic dispatcher."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        adapters: Optional[Dict[Provider, ModelProvider]] = None,
        *,
        max_retries: int = MAX_API_RETRIES,
        retry_base_delay: float = RETRY_BASE_DELAY_S,
        circuit_threshold: int = CIRCUIT_BREAKER_THRESHOLD,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> None:
        self.registry = registry or default_registry()
        self._adapters: Dict[Provider, ModelProvider] = dict(adapters or {})
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._circuit_threshold = circuit_threshold
        self._sleep = sleep or asyncio.sleep
        self._consecutive_failures: Dict[str, int] = {}

    # -- adapter resolution (lazy defaults so the heavy SDKs import on demand) --
    def _adapter_for(self, provider: Provider) -> ModelProvider:
        if provider not in self._adapters:
            if provider == Provider.GEMINI:
                from darwin.agent.providers.gemini import GeminiProvider

                self._adapters[provider] = GeminiProvider()
            elif provider == Provider.OPENAI_COMPAT:
                from darwin.agent.providers.openai_compat import OpenAICompatProvider

                self._adapters[provider] = OpenAICompatProvider()
            else:  # pragma: no cover - exhaustive enum
                raise ValueError(f"no adapter for provider {provider!r}")
        return self._adapters[provider]

    def estimate_cost(self, model_id: str, usage: Usage) -> float:
        return self.registry.estimate_cost(model_id, usage.tokens_in, usage.tokens_out)

    async def complete(
        self,
        model_id: str,
        system: str,
        user: str,
        response_schema: Dict[str, Any],
        thinking_level: str,
        max_output_tokens: int,
        timeout: float = DEFAULT_TIMEOUT_S,
    ) -> ModelResponse:
        """Dispatch one logical completion, applying the resilience policy.

        Always returns a ``ModelResponse`` — a final failure is reported via the
        ``error`` field, never raised.
        """
        # Resolve registry entry + adapter defensively: an unknown model_id
        # (KeyError) or a missing provider SDK (ImportError) must come back as a
        # ModelResponse error, never raise into the caller.
        try:
            entry = self.registry.get(model_id)
            adapter = self._adapter_for(entry.provider)
        except Exception as exc:  # noqa: BLE001
            return ModelResponse(
                model_id=model_id, finish_reason="error",
                error=f"[{ErrorCategory.BAD_REQUEST.value}] resolve {model_id!r}: {type(exc).__name__}: {exc}",
            )

        attempt = 0
        last_error: ProviderError
        while True:
            try:
                response = await asyncio.wait_for(
                    adapter.raw_complete(
                        entry, system, user, response_schema, thinking_level, max_output_tokens
                    ),
                    timeout=timeout,
                )
                self._on_success(model_id)
                return response
            except asyncio.TimeoutError:
                last_error = ProviderError(ErrorCategory.TIMEOUT, f"model call exceeded {timeout}s")
                break  # a hung call is not retried — return a graceful timeout
            except ProviderError as exc:
                last_error = exc
                retriable = exc.category in RETRIABLE
                if exc.category == ErrorCategory.AUTH:
                    break  # fail fast — almost always a missing/expired key
                if retriable and attempt < self._max_retries:
                    await self._sleep(self._backoff(attempt))
                    attempt += 1
                    continue
                break
            except Exception as exc:  # normalize any unexpected adapter error
                last_error = ProviderError(ErrorCategory.OTHER, f"{type(exc).__name__}: {exc}")
                break

        self._on_failure(model_id)
        return ModelResponse(
            raw_text="",
            parsed=None,
            usage=Usage(),
            latency_ms=0.0,
            model_id=model_id,
            finish_reason="error",
            error=f"[{last_error.category.value}] {last_error.message}",
        )

    def _backoff(self, attempt: int) -> float:
        return self._retry_base_delay * (2 ** attempt)

    # -- circuit breaker -----------------------------------------------------
    def _on_success(self, model_id: str) -> None:
        self._consecutive_failures[model_id] = 0
        if self.registry.is_degraded(model_id):
            self.registry.set_degraded(model_id, False)  # recovered

    def _on_failure(self, model_id: str) -> None:
        count = self._consecutive_failures.get(model_id, 0) + 1
        self._consecutive_failures[model_id] = count
        if count >= self._circuit_threshold and not self.registry.is_degraded(model_id):
            self.registry.set_degraded(model_id, True)
