"""MongoDB telemetry & the corpus seed.

Every worker invocation is written to ``agent_invocations`` — the substrate the
live screen animates from, B6 escalates over, and B8 analyses. The
``agent_corpus`` collection is *seeded* here (plumbing only) and populated by B6.

Two non-negotiables (§9):
* **Fire-and-forget-safe.** A logging failure must NEVER break ``run()`` — a
  Mongo hiccup degrades to a local log line, not a crashed agent.
* **Log on every path** — success, repaired-success, and failure alike.
"""

import logging
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger("darwin.agent.telemetry")


class TelemetrySink(Protocol):
    """The interface the worker logs through. Implementations must be
    failure-safe: ``log_invocation`` must never raise."""

    async def log_invocation(self, record: Dict[str, Any]) -> None: ...

    async def save_corpus_spec(self, record: Dict[str, Any]) -> None: ...


class NullTelemetrySink:
    """Discards everything (used when telemetry is disabled)."""

    async def log_invocation(self, record: Dict[str, Any]) -> None:  # noqa: D401
        return None

    async def save_corpus_spec(self, record: Dict[str, Any]) -> None:
        return None


class InMemoryTelemetrySink:
    """Captures records in memory — for tests and offline runs."""

    def __init__(self) -> None:
        self.invocations: List[Dict[str, Any]] = []
        self.corpus: List[Dict[str, Any]] = []

    async def log_invocation(self, record: Dict[str, Any]) -> None:
        self.invocations.append(record)

    async def save_corpus_spec(self, record: Dict[str, Any]) -> None:
        self.corpus.append(record)


class MongoTelemetrySink:
    """Motor-backed sink writing to ``agent_invocations`` / ``agent_corpus``.

    Accepts injected collections (motor ``AsyncIOMotorCollection`` or any object
    exposing an awaitable ``insert_one``) so it is fully unit-testable, or build
    one lazily from a connection URI via :meth:`from_uri`.
    """

    INVOCATIONS = "agent_invocations"
    CORPUS = "agent_corpus"

    def __init__(self, invocations: Any, corpus: Any) -> None:
        self._invocations = invocations
        self._corpus = corpus

    @classmethod
    def from_uri(
        cls, uri: str, db_name: str = "darwin"
    ) -> "MongoTelemetrySink":  # pragma: no cover - requires a real/embedded Mongo
        from motor.motor_asyncio import AsyncIOMotorClient

        db = AsyncIOMotorClient(uri)[db_name]
        return cls(db[cls.INVOCATIONS], db[cls.CORPUS])

    async def log_invocation(self, record: Dict[str, Any]) -> None:
        try:
            await self._invocations.insert_one(dict(record))
        except Exception as exc:  # fire-and-forget-safe: degrade, never crash run()
            logger.warning("telemetry log_invocation failed (degraded to local log): %s", exc)
            logger.info("agent_invocation(local): %s", record)

    async def save_corpus_spec(self, record: Dict[str, Any]) -> None:
        try:
            await self._corpus.insert_one(dict(record))
        except Exception as exc:
            logger.warning("telemetry save_corpus_spec failed (degraded to local log): %s", exc)


def make_telemetry(uri: Optional[str] = None) -> TelemetrySink:
    """Build a telemetry sink: Mongo when a URI is given, else in-memory."""
    if uri:  # pragma: no cover - requires a real/embedded Mongo
        return MongoTelemetrySink.from_uri(uri)
    return InMemoryTelemetrySink()
