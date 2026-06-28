"""Darwin Phase B8 — Persistence, Telemetry & the Self-Improving Scorer.

The capstone that makes the whole brain observable and replayable. Two things:

1. **Persistence & telemetry** — every meaningful moment of a solve is a
   structured ``RunEvent`` durably appended (in order) to ``run_events`` and
   published to an in-process bus. That single ordered stream drives the live
   TypeScript screen (over the WebSocket bridge) and lets ``replay`` reconstruct
   exactly what the system tried and why. Domain state (genomes, invocations,
   corpus) stays authoritative; this is the narrative layer. Emission is
   async, non-blocking, and failure-safe — it never blocks or crashes the solve.

2. **The self-improving scorer** (optional second-order loop) — tunes B1's
   ``ObjectiveWeights`` when they stop predicting true optimality, anchored to the
   oracle (never an LLM). The primary scorer stays deterministic math; only the
   weights move, toward agreement with verifiable ground truth.

Frozen handoff: ``EventEmitter.emit``, ``RunEvent``/``RunEventType``, ``EventBus``,
``EventStore``, ``replay_run``, ``WebSocketServer``/``ClientSession``, and
``SelfImprovingScorer``. ``Conductor.solve`` (B6) accepts an optional ``emitter``
and narrates the full run through it.
"""

from darwin.observability.bus import EventBus, Subscription
from darwin.observability.emitter import EventEmitter
from darwin.observability.events import RunEvent, RunEventType
from darwin.observability.replay import replay_run
from darwin.observability.self_improving_scorer import (
    CalibrationSample,
    RetuneResult,
    SelfImprovingScorer,
    bump_version,
    correlation,
    predicted_goodness,
    retune,
    spearman,
)
from darwin.observability.store import EventStore
from darwin.observability.websocket_server import ClientSession, WebSocketServer

__all__ = [
    # events + bus
    "RunEvent", "RunEventType", "EventBus", "Subscription",
    # emitter + store
    "EventEmitter", "EventStore",
    # websocket + replay
    "WebSocketServer", "ClientSession", "replay_run",
    # self-improving scorer
    "SelfImprovingScorer", "CalibrationSample", "RetuneResult", "correlation", "spearman",
    "retune", "predicted_goodness", "bump_version",
]
