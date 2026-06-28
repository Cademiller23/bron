"""Bounded global inference concurrency — the shared semaphore.

A single ``InferenceGate`` is created once at startup and passed to every
``TeamRunner`` (and thus shared across every concurrent genome evaluation in B5).
Every model call acquires the gate, so total in-flight calls across the ENTIRE
swarm can never exceed the ceiling — a wide swarm degrades gracefully (excess
calls queue, latency rises) instead of saturating the endpoint and producing
timeouts that masquerade as bad genomes (which would corrupt B5's evolutionary
signal).
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from darwin.constants import MAX_CONCURRENT_INFERENCE


class InferenceGate:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT_INFERENCE) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self.max_concurrent = max_concurrent
        self._sem: Optional[asyncio.Semaphore] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None  # the loop the sem is bound to
        self._in_flight = 0
        self._peak = 0

    def _semaphore(self) -> asyncio.Semaphore:
        # On Python 3.9 an asyncio.Semaphore binds to the loop it was created
        # under; reusing one gate across asyncio.run() boundaries (different loops)
        # would otherwise raise "got Future attached to a different loop" under
        # contention and spuriously floor a whole generation. So (re)create the
        # semaphore whenever the running loop changes.
        loop = asyncio.get_running_loop()
        if self._sem is None or self._loop is not loop:
            self._sem = asyncio.Semaphore(self.max_concurrent)
            self._loop = loop
            self._in_flight = 0
        return self._sem

    @asynccontextmanager
    async def acquire(self):
        sem = self._semaphore()
        await sem.acquire()
        self._in_flight += 1
        if self._in_flight > self._peak:
            self._peak = self._in_flight
        try:
            yield
        finally:
            self._in_flight -= 1
            sem.release()

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def peak_concurrency(self) -> int:
        return self._peak

    def reset_peak(self) -> None:
        self._peak = self._in_flight
