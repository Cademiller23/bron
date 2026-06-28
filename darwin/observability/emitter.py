"""The EventEmitter — the one call every phase makes at its key moments.

``emit(event_type, payload)`` does two things: durably append the event to
``run_events`` (failure-safe) and publish it to the in-process bus (non-blocking).
It is **fire-and-forget-safe**: like B2's telemetry, a Mongo or bus failure
degrades to a warning and NEVER blocks or crashes the solve. ``sequence_number``
is assigned monotonically per run with no ``await`` between read and increment, so
even concurrently-firing phases get a strictly increasing, gap-free order.

One emitter per run (it owns the run's sequence counter). The Conductor (B6) is
the natural owner; it threads the emitter through the narrative.
"""

import logging
from typing import Any, Dict, Optional

from darwin.observability.bus import EventBus
from darwin.observability.events import RunEvent, RunEventType
from darwin.observability.store import EventStore

logger = logging.getLogger("darwin.observability.emitter")


class EventEmitter:
    def __init__(self, run_id: str, store: Optional[EventStore] = None,
                 bus: Optional[EventBus] = None, *, start_sequence: int = 0) -> None:
        self.run_id = run_id
        self._store = store
        self._bus = bus
        self._seq = start_sequence

    @property
    def next_sequence(self) -> int:
        return self._seq

    async def emit(
        self, event_type: Any, payload: Optional[Dict[str, Any]] = None, *, description: str = ""
    ) -> Optional[RunEvent]:
        """Append + publish one event. ``event_type`` may be a ``RunEventType`` or
        its string value (so callers like the Conductor need no B8 import). Returns
        the event, or None if it could not be constructed (a malformed payload or an
        unknown event type must never break the caller)."""
        try:
            et = event_type if isinstance(event_type, RunEventType) else RunEventType(str(event_type))
            # NOTE: read-and-increment of self._seq happens with NO await between
            # them, so the sequence is atomic under asyncio (monotonic, gap-free).
            event = RunEvent(
                run_id=self.run_id, sequence_number=self._seq, event_type=et,
                description=description, payload=_jsonable(payload or {}),
            )
            self._seq += 1
        except Exception as exc:  # noqa: BLE001 - a bad event must not crash the solve
            logger.warning("event construction failed (skipped): %s", exc)
            return None

        if self._bus is not None:
            try:
                self._bus.publish(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("event publish failed (degraded): %s", exc)
        if self._store is not None:
            try:
                await self._store.append(event)
            except Exception as exc:  # noqa: BLE001 - store is already failure-safe; belt + suspenders
                logger.warning("event append failed (degraded): %s", exc)
        return event


def _jsonable(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort coercion so a payload always survives the frozen RunEvent (and
    later JSON serialization). Non-serializable values fall back to ``str``."""
    out: Dict[str, Any] = {}
    for k, v in payload.items():
        out[str(k)] = _coerce(v)
    return out


def _coerce(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _coerce(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_coerce(x) for x in v]
    try:
        return str(v)
    except Exception:  # noqa: BLE001
        return None
