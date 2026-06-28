"""Serve the live evolution to the TypeScript face over a WebSocket.

Runs ``Conductor.solve`` with an event emitter on an in-process bus, translates
each darwin event into the face's ``RunEvent`` shape (see ``frontend_bridge.py``),
and streams the result to every connected client. Point the frontend at it:

    NEXT_PUBLIC_AGENT_RUN_WS=ws://localhost:8765

Run it:

    # deterministic, no API key (mock stack) — the safe demo:
    DARWIN_DEMO=1 venv/bin/python -m darwin.observability.serve_frontend

    # live models (needs GEMINI_API_KEY; uses the real Architect + swarm):
    venv/bin/python -m darwin.observability.serve_frontend

Each browser connection starts its OWN fresh run, so reloading the page replays
the whole story from the top. Events are paced (``PACE_MS``, default 350 ms) so the
org chart and the climbing curve are legible in ~90 seconds — the B10 legibility
rule. Config via env: ``HOST``, ``PORT``, ``PACE_MS``, ``DARWIN_DEMO``, ``PROBLEM``
(``transportation`` | ``f1``).
"""

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Optional, Tuple
from uuid import uuid4

from darwin.observability.bus import EventBus
from darwin.observability.emitter import EventEmitter
from darwin.observability.frontend_bridge import EventTranslator

logger = logging.getLogger("darwin.observability.serve_frontend")

Send = Callable[[dict], Awaitable[None]]


def _demo_enabled() -> bool:
    if os.environ.get("DARWIN_DEMO", "").strip() in ("1", "true", "yes"):
        return True
    return not os.environ.get("GEMINI_API_KEY")  # no key -> safe deterministic stack


# ---------------------------------------------------------------------------
# Stack builders — each returns (conductor, instance, weights). Imports are lazy
# so importing this module stays cheap and offline-safe.
# ---------------------------------------------------------------------------
def build_demo_stack(emitter: EventEmitter) -> Tuple[Any, Any, Any]:
    """A deterministic, no-network run: base team scores 0.7, escalation grows one
    agent, the larger team clears 0.90 — the full rearrange→grow→break arc."""
    from darwin.escalation.conductor import Conductor
    from darwin.escalation.corpus import AgentCorpus
    from darwin.escalation.embedding import KeywordEmbedder
    from darwin.escalation.escalator import Escalator
    from darwin.escalation.fixtures import (
        CorpusFakeCollection,
        MockConductorArchitect,
        MockRearrangementLoop,
        base_genome,
    )

    class _Instance:
        def __init__(self, pc: str, iid: str) -> None:
            self.instance_id = iid
            self.problem_class = type("PC", (), {"value": pc})()

    corpus = AgentCorpus(CorpusFakeCollection(), KeywordEmbedder())
    architect = MockConductorArchitect(base_genome())
    escalator = Escalator(corpus, architect, store=None)
    loop = MockRearrangementLoop(lambda g: 0.7 if len(g.agents) == 4 else 0.95)
    conductor = Conductor(architect, loop, escalator, corpus, emitter=emitter)
    return conductor, _Instance("supply_chain", "demo-supply-chain"), None


def build_real_stack(emitter: EventEmitter) -> Tuple[Any, Any, Any]:
    """The real brain: live Architect + swarm over the model registry."""
    from darwin.agent.client import ModelClient
    from darwin.agent.telemetry import InMemoryTelemetrySink
    from darwin.architect.architect import Architect
    from darwin.escalation.conductor import Conductor
    from darwin.escalation.corpus import AgentCorpus
    from darwin.escalation.embedding import KeywordEmbedder
    from darwin.escalation.escalator import Escalator
    from darwin.escalation.fixtures import CorpusFakeCollection
    from darwin.problem.schemas import ObjectiveWeights
    from darwin.rearrange.loop import RearrangementLoop
    from darwin.team import fixtures as TF
    from darwin.team.inference_gate import InferenceGate
    from darwin.team.runner import TeamRunner

    problem = os.environ.get("PROBLEM", "transportation").strip().lower()
    if problem == "f1":
        # F1 rides inside the engine's Solution shape: the architect is F1-aware via
        # problem_class=F1_CALENDAR, and the injected scorer decodes the calendar
        # back out of the Solution and scores it (carbon/revenue/feasibility).
        from darwin.problem.f1_problem import build_f1_instance
        from darwin.problem.f1_scorer import score_f1_solution

        instance, scorer = build_f1_instance(), score_f1_solution
    else:
        from darwin.problem.fixtures import golden_transportation

        instance, scorer = golden_transportation(), None  # None -> B1's default scorer

    store = TF.new_store()
    client = ModelClient()
    architect = Architect(client, store=store)
    runner = TeamRunner(model_client=client, telemetry=InMemoryTelemetrySink(),
                        inference_gate=InferenceGate(4), store=store, scorer=scorer)
    loop = RearrangementLoop(runner, store=store, registry=None, k=4)
    corpus = AgentCorpus(CorpusFakeCollection(), KeywordEmbedder())
    escalator = Escalator(corpus, architect, store=store)
    conductor = Conductor(architect, loop, escalator, corpus, store=store, emitter=emitter)
    return conductor, instance, ObjectiveWeights.balanced()


# ---------------------------------------------------------------------------
# One client connection = one run, streamed and paced.
# ---------------------------------------------------------------------------
async def run_one_session(send: Send, *, pace_s: float = 0.35, record_path: Optional[str] = None) -> None:
    import json

    bus = EventBus()
    emitter = EventEmitter(run_id=uuid4().hex, store=None, bus=bus)
    translator = EventTranslator()
    builder = build_demo_stack if _demo_enabled() else build_real_stack
    conductor, instance, weights = builder(emitter)

    # RECORD: capture exactly what the face consumes, so a real run can be replayed
    # later instantly (no keys, no waiting) — the on-stage safety net.
    recorder = open(record_path, "w") if record_path else None  # noqa: SIM115
    sub = bus.subscribe()
    solve = asyncio.create_task(conductor.solve(instance, weights))
    try:
        done = False
        while not done:
            try:
                ev = await sub.get(timeout=0.25)
            except asyncio.TimeoutError:
                if solve.done():
                    break  # solve finished and the queue is drained
                continue
            for fe in translator.translate(ev.event_type, ev.payload, ev.description):
                await send(fe)
                if recorder:
                    recorder.write(json.dumps(fe) + "\n")
                if fe.get("type") == "run_complete":
                    done = True
                if pace_s:
                    await asyncio.sleep(pace_s)
    finally:
        if recorder:
            recorder.close()
        sub.close()
        if not solve.done():
            solve.cancel()
        try:
            await solve
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 - solve is best-effort here
            pass


async def replay_session(send: Send, path: str, *, pace_s: float = 0.35) -> int:
    """Stream a previously RECORDed run (JSONL of frontend events) to a client.

    No conductor, no models, no Mongo — instant and byte-identical every time. The
    on-stage default when a live run is too slow or risky. Returns events sent."""
    import json

    with open(path) as f:
        events = [json.loads(line) for line in f if line.strip()]
    for fe in events:
        await send(fe)
        if pace_s:
            await asyncio.sleep(pace_s)
    return len(events)


async def serve(host: str = "0.0.0.0", port: int = 8765, pace_s: float = 0.35) -> None:  # pragma: no cover - real socket
    import json

    import websockets

    replay_path = os.environ.get("REPLAY", "").strip() or None
    record_path = os.environ.get("RECORD", "").strip() or None

    async def _handler(websocket):
        async def _send(doc: dict) -> None:
            await websocket.send(json.dumps(doc))

        try:
            if replay_path:
                logger.info("client connected — replaying %s", replay_path)
                await replay_session(_send, replay_path, pace_s=pace_s)
            else:
                logger.info("client connected — starting a run")
                await run_one_session(_send, pace_s=pace_s, record_path=record_path)
        except websockets.ConnectionClosed:
            logger.info("client disconnected mid-run")

    mode = f"replay {replay_path}" if replay_path else ("demo" if _demo_enabled() else "live")
    logger.info("serving darwin -> frontend on ws://%s:%d (mode=%s)", host, port, mode)
    async with websockets.serve(_handler, host, port):
        await asyncio.Future()  # serve forever


def main() -> None:  # pragma: no cover - entrypoint
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8765"))
    pace_s = float(os.environ.get("PACE_MS", "350")) / 1000.0
    try:
        asyncio.run(serve(host, port, pace_s))
    except KeyboardInterrupt:
        logger.info("shutting down")


if __name__ == "__main__":  # pragma: no cover
    main()
