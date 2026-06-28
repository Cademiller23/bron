"""The WebSocket bridge — the bus to the TypeScript face.

A connected client subscribes to the live bus and receives each ``RunEvent`` as
JSON; the face renders the org chart (genome snapshots), the climbing curve
(fitness), the model panel ("MAX served X%"), and the LiveKit voice (each event's
``description``). A reconnecting client resumes from its last ``sequence_number``
(a ``run_events`` catch-up), then rejoins the live stream — no gap, no duplicate.

The transport is injected (an async ``send(dict)``) so the bridge is fully
unit-testable without a real socket; ``WebSocketServer.serve`` wires the real
``websockets`` library around the same :class:`ClientSession`.
"""

import logging
from typing import Any, Awaitable, Callable, Optional

from darwin.observability.events import RunEvent

logger = logging.getLogger("darwin.observability.websocket")

Send = Callable[[dict], Awaitable[None]]


class ClientSession:
    """One connected client: optional resume catch-up, then the live stream.

    A send failure (client gone) stops THIS session cleanly and never affects the
    bus, the solve, or the other clients."""

    def __init__(self, send: Send, bus: Any, store: Any = None, run_id: Optional[str] = None) -> None:
        self._send = send
        self._bus = bus
        self._store = store
        self._run_id = run_id

    async def _safe_send(self, event: RunEvent) -> bool:
        try:
            await self._send(event.model_dump(mode="json"))
            return True
        except Exception as exc:  # noqa: BLE001 - the client disconnected; stop this session only
            logger.info("client send failed (closing session): %s", exc)
            return False

    async def run(self, last_sequence: Optional[int] = None) -> None:
        # Subscribe FIRST so no event published during the catch-up read can slip
        # through the gap between the store read and the live subscription. Then
        # replay the missed range from the store, then forward live events while
        # de-duplicating anything already delivered in the catch-up (an event can be
        # in BOTH the store and the buffered live stream). Result: no gap, no dup.
        sub = self._bus.subscribe()
        try:
            sent_through = last_sequence if last_sequence is not None else -1
            # 1. resume: replay everything the client missed, in order
            if self._store is not None and self._run_id and last_sequence is not None:
                for ev in await self._store.load_since(self._run_id, last_sequence):
                    if not await self._safe_send(ev):
                        return
                    sent_through = max(sent_through, ev.sequence_number)
            # 2. live: forward, skipping any sequence already sent in the catch-up
            async for ev in sub:
                if ev.sequence_number <= sent_through:
                    continue  # already delivered -> no duplicate at the boundary
                if not await self._safe_send(ev):
                    break
                sent_through = max(sent_through, ev.sequence_number)
        finally:
            sub.close()


class WebSocketServer:
    """Wires the live bus to connected clients. ``serve`` uses the real
    ``websockets`` library; the streaming core is :class:`ClientSession`."""

    def __init__(self, bus: Any, store: Any = None) -> None:
        self._bus = bus
        self._store = store

    async def handle(self, send: Send, *, run_id: Optional[str] = None,
                     last_sequence: Optional[int] = None) -> None:
        """Drive one client connection (testable: pass any async ``send``)."""
        await ClientSession(send, self._bus, self._store, run_id).run(last_sequence)

    async def serve(self, host: str = "0.0.0.0", port: int = 8765) -> None:  # pragma: no cover - real socket
        import json

        import websockets

        async def _handler(websocket):
            run_id, last_sequence = None, None
            try:  # an optional first message: {"run_id": "...", "last_sequence": N}
                first = await websocket.recv()
                hello = json.loads(first)
                run_id, last_sequence = hello.get("run_id"), hello.get("last_sequence")
            except Exception:  # noqa: BLE001 - no/!json hello -> just go live
                pass

            async def _send(doc: dict) -> None:
                await websocket.send(json.dumps(doc))

            await self.handle(_send, run_id=run_id, last_sequence=last_sequence)

        async with websockets.serve(_handler, host, port):
            import asyncio

            await asyncio.Future()  # serve forever
