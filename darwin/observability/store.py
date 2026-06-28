"""Durable persistence for the event stream and the scorer-version history.

``run_events`` is the ordered narrative (index ``(run_id, sequence_number)``);
``scorer_versions`` is the self-improving scorer's weight + correlation history
(index ``(scorer_version, timestamp)``). The domain collections (genomes,
agent_invocations, agent_corpus) stay authoritative; this is the narrative layer.

Every write is failure-safe (a Mongo hiccup degrades to a warning, never crashes
the solve — the live bus subscribers still flow and full history is best-effort).
"""

import logging
from typing import Any, Dict, List, Optional

from darwin.observability.events import RunEvent

logger = logging.getLogger("darwin.observability.store")


class EventStore:
    RUN_EVENTS = "run_events"
    SCORER_VERSIONS = "scorer_versions"

    def __init__(self, run_events: Any, scorer_versions: Any = None) -> None:
        self._events = run_events
        self._scorer_versions = scorer_versions

    @classmethod
    def from_uri(cls, uri: str, db_name: str = "darwin") -> "EventStore":  # pragma: no cover - needs a server
        from motor.motor_asyncio import AsyncIOMotorClient

        db = AsyncIOMotorClient(uri)[db_name]
        return cls(db[cls.RUN_EVENTS], db[cls.SCORER_VERSIONS])

    async def ensure_indexes(self) -> None:  # pragma: no cover - needs a server
        await self._events.create_index([("run_id", 1), ("sequence_number", 1)])
        if self._scorer_versions is not None:
            await self._scorer_versions.create_index([("scorer_version", 1), ("timestamp", 1)])

    # -- run_events ----------------------------------------------------------
    async def append(self, event: RunEvent) -> bool:
        try:
            await self._events.insert_one(event.to_doc())
            return True
        except Exception as exc:  # noqa: BLE001 - durable append is best-effort
            logger.warning("run_events append failed (degraded): %s", exc)
            return False

    async def load_run(self, run_id: str) -> List[RunEvent]:
        try:
            cursor = self._events.find({"run_id": run_id}).sort("sequence_number", 1)
            docs = await cursor.to_list(length=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_events load_run failed (degraded to empty): %s", exc)
            return []
        out: List[RunEvent] = []
        for doc in docs:
            try:
                out.append(RunEvent.from_doc(doc))
            except Exception:  # a corrupt row must not poison the whole replay
                continue
        out.sort(key=lambda e: e.sequence_number)  # re-sort on the VALIDATED int key
        return out

    async def load_since(self, run_id: str, after_sequence: int) -> List[RunEvent]:
        """Events with ``sequence_number > after_sequence`` in order — the
        catch-up a reconnecting WebSocket client resumes from.

        Defined as ``load_run`` filtered on the VALIDATED int sequence, so the
        resume catch-up is byte-for-byte a *subset* of the full replay — they can
        never disagree (a raw ``$gt`` on the unvalidated document would diverge
        from ``load_run`` on a coercible-but-non-int sequence, leaving a gap the
        full replay doesn't have). Correctness over a marginal index optimization.
        """
        events = await self.load_run(run_id)  # validated, corrupt-skipped, sorted
        return [e for e in events if e.sequence_number > after_sequence]

    async def max_sequence(self, run_id: str) -> int:
        """The highest sequence_number for a run, or -1 if none."""
        events = await self.load_run(run_id)
        return events[-1].sequence_number if events else -1

    # -- scorer_versions -----------------------------------------------------
    async def save_scorer_version(self, record: Dict[str, Any]) -> bool:
        if self._scorer_versions is None:
            return False
        try:
            await self._scorer_versions.insert_one(dict(record))
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("scorer_versions save failed (degraded): %s", exc)
            return False

    async def load_scorer_versions(self) -> List[Dict[str, Any]]:
        if self._scorer_versions is None:
            return []
        try:
            cursor = self._scorer_versions.find({}).sort("timestamp", 1)
            return await cursor.to_list(length=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("scorer_versions load failed (degraded to empty): %s", exc)
            return []
