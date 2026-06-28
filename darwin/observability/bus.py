"""An in-process async pub/sub bus — the live fan-out to the screen.

The emitter publishes each ``RunEvent`` here; the WebSocket server (and replay)
subscribe and forward to clients. Publishing is **synchronous and non-blocking**
(``put_nowait``), so a slow or broken subscriber can NEVER block the solve or the
other subscribers — a full queue drops the oldest event for that subscriber only
(the live screen prefers recency; full history lives durably in ``run_events``).
"""

import asyncio
import logging
from typing import List, Optional

from darwin.observability.events import RunEvent

logger = logging.getLogger("darwin.observability.bus")

_CLOSE = object()  # sentinel that ends a subscription's async iteration


class Subscription:
    """One subscriber's bounded queue + async iterator."""

    def __init__(self, bus: "EventBus", maxsize: int) -> None:
        self._bus = bus
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.dropped = 0
        self._closed = False

    def _put(self, event: RunEvent) -> None:
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            # drop the OLDEST to make room for the newest (keep the stream moving)
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except Exception:  # noqa: BLE001 - never let a backpressure race break publish
                pass
            self.dropped += 1

    async def get(self, timeout: Optional[float] = None) -> RunEvent:
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout)

    async def __aiter__(self):
        while True:
            item = await self._queue.get()
            if item is _CLOSE:
                return
            yield item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(_CLOSE)
        except asyncio.QueueFull:
            try:  # force room for the sentinel so the iterator can finish
                self._queue.get_nowait()
                self._queue.put_nowait(_CLOSE)
            except Exception:  # noqa: BLE001
                pass
        self._bus._remove(self)


class EventBus:
    """Fan-out of ``RunEvent``s to all live subscribers."""

    def __init__(self, default_maxsize: int = 1000) -> None:
        self._subs: List[Subscription] = []
        self._default_maxsize = default_maxsize
        self._closed = False

    def subscribe(self, maxsize: Optional[int] = None) -> Subscription:
        sub = Subscription(self, maxsize or self._default_maxsize)
        if self._closed:
            # subscribing after shutdown returns an already-closed subscription
            # (sentinel pre-queued) so its async-for terminates at once, never hangs
            sub.close()
            return sub
        self._subs.append(sub)
        return sub

    def publish(self, event: RunEvent) -> None:
        """Fan ``event`` out to every subscriber. Never raises — one bad
        subscriber must not break the publish or the others."""
        for sub in list(self._subs):  # snapshot: a subscriber may close mid-publish
            try:
                sub._put(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("bus publish to a subscriber failed (skipped): %s", exc)

    def _remove(self, sub: Subscription) -> None:
        try:
            self._subs.remove(sub)
        except ValueError:
            pass

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)

    def close(self) -> None:
        self._closed = True
        for sub in list(self._subs):
            sub.close()
