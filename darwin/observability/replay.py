"""Replay — re-emit a past run from ``run_events`` in exact order.

Because every moment is durably logged with references to the authoritative
domain state, any past run can be reconstructed on the screen. ``replay_run``
reads ``run_events`` for a ``run_id`` in ``sequence_number`` order and re-publishes
them to the bus — at original pace (from the recorded timestamps) or accelerated.
Invaluable as a pre-recorded demo backup (if live inference wobbles on stage,
replay a real prior run) and for debugging.

Order is guaranteed: ``store.load_run`` returns events sorted by the validated
integer ``sequence_number``, so replay preserves the original sequence regardless
of pace.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable, Optional

from darwin.observability.events import RunEvent

logger = logging.getLogger("darwin.observability.replay")

_MAX_DELAY_S = 5.0  # cap any single inter-event wait so a long real gap doesn't stall a replay


def _delta_seconds(a: str, b: str) -> float:
    try:
        return max(0.0, (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds())
    except Exception:  # noqa: BLE001 - an unparseable timestamp just means no pacing
        return 0.0


async def replay_run(
    store: Any,
    run_id: str,
    *,
    bus: Any = None,
    emit: Optional[Callable[[RunEvent], Any]] = None,
    speed: Optional[float] = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> int:
    """Re-emit run ``run_id`` in sequence order. Returns the number of events replayed.

    * ``bus`` — re-publish each event to this :class:`EventBus` (the live screen).
    * ``emit`` — alternatively, call this coroutine/callable per event.
    * ``speed`` — ``None`` (default) replays as fast as possible (order preserved);
      ``> 0`` paces by the recorded timestamps divided by ``speed`` (2.0 = 2× faster).
    """
    events = await store.load_run(run_id)  # already sorted by sequence_number
    prev_ts: Optional[str] = None
    for ev in events:
        if speed is not None and speed > 0 and prev_ts is not None:
            delay = _delta_seconds(prev_ts, ev.timestamp) / speed
            if delay > 0:
                await sleep(min(delay, _MAX_DELAY_S))
        prev_ts = ev.timestamp
        if bus is not None:
            try:
                bus.publish(ev)
            except Exception as exc:  # noqa: BLE001 - a bad subscriber must not stop the replay
                logger.warning("replay publish failed (continuing): %s", exc)
        if emit is not None:
            try:
                res = emit(ev)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as exc:  # noqa: BLE001
                logger.warning("replay emit callback failed (continuing): %s", exc)
    return len(events)
