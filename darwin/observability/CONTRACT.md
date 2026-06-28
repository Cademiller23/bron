# Darwin B8 — Frozen Contract (Persistence, Telemetry & the Self-Improving Scorer)

> The capstone that makes the whole brain observable and replayable. B8 changes
> **no phase's logic** — it threads an event emitter through the Conductor and
> emits at the key moments, and adds the optional self-improving scorer. With B8
> done, **B1–B8 are the complete working brain**: a self-organizing, model-aware
> swarm, now fully observable and replayable, entered through `Conductor.solve`.

## Two layers, cleanly separated

- **Domain state** (what IS — authoritative): `genomes` (B3), `agent_invocations`
  (B2/B7), `agent_corpus` (B6). Unchanged.
- **Event log** (what HAPPENED, in order — the narrative): `run_events` (new).
  Events reference domain objects by **id + version**; the domain collections stay
  authoritative. One ordered stream is the single source of narrative truth for
  the screen and for replay — no second guess.

## The event stream (`events.py`)

`RunEvent{event_id, run_id, sequence_number, timestamp, event_type, description,
payload}` — frozen, `extra="forbid"`. `sequence_number` is **monotonic per run**
(the replay order key). `RunEventType` is the narrative: `RUN_STARTED`,
`GROUNDING_DONE` (B9), `TEAM_DESIGNED` (B4), `GENOME_EVALUATED` (B3),
`REARRANGE_CANDIDATE_SCORED`/`REARRANGE_ADOPTED` (B5), `THRESHOLD_CHECK` (B6),
`ESCALATION_CORPUS_HIT`/`ESCALATION_CURATED`/`AGENT_ROLLED_BACK` (B6),
`MODEL_PANEL_UPDATE` (B7), `SCORER_RETUNED` (B8), `RUN_SEALED`/`RUN_EXHAUSTED`.

## The emitter + bus + store

- **`EventEmitter.emit(event_type, payload)`** — assigns the next sequence,
  publishes to the bus (non-blocking), and durably appends to `run_events`. It is
  **fire-and-forget-safe**: a Mongo/bus/construction failure degrades to a warning
  and NEVER blocks or crashes the solve. The read-and-increment of the sequence has
  **no `await` between** read and increment, so under `asyncio` it is atomic →
  monotonic and gap-free even under concurrent emits. `event_type` may be a
  `RunEventType` or its string value (callers need no B8 import). Payloads are
  coerced to JSON-able (non-serializable values fall back to `str`).
- **`EventBus`** — in-process pub/sub. `publish` is synchronous and non-blocking
  (`put_nowait`); a slow/broken subscriber can never block the solve or the others
  — a full queue **drops the oldest** event for that subscriber only (the live
  screen prefers recency; full history is durable in `run_events`).
- **`EventStore`** — Motor-backed `run_events` (index `(run_id, sequence_number)`)
  + `scorer_versions` (index `(scorer_version, timestamp)`); fake-collection
  testable. `load_run`/`load_since` re-sort on the **validated** integer sequence
  (a corrupt row is skipped, never poisons the replay). Every method is failure-safe.

## The WebSocket bridge + replay

- **`WebSocketServer`/`ClientSession`** — bridges the bus to the TypeScript face
  (org chart, climbing curve, model panel, voice). A reconnecting client resumes
  from its last `sequence_number` (a `run_events` catch-up) then rejoins live —
  **no gap, no duplicate** at the boundary (subscribe-then-catch-up-then-dedup).
  The transport is injected, so the bridge is fully unit-testable; a dead client
  ends its own session cleanly and never affects the bus or other clients.
- **`replay_run(store, run_id, bus|emit, speed)`** — re-emits a stored run in
  exact `sequence_number` order, at original pace (from recorded timestamps) or
  accelerated (`speed`>0). The pre-recorded demo backup: if live inference wobbles
  on stage, replay a real prior run.

## The self-improving scorer (`self_improving_scorer.py`) — the second-order loop

The airtight answer to "how is this recursive, not just a search loop?"

1. **Calibrate** — Spearman rank correlation between the scorer's goodness (a
   function of the weights) and each solution's TRUE goodness (distance-to-optimum
   from B1's OR-Tools **oracle**). High = well-calibrated; degraded = mis-weighted.
2. **Re-tune** — when degraded, gradient-free search over the `ObjectiveWeights`
   simplex for the weights that best correlate with the truth; adopt, bump
   `scorer_version` (B1 stamps every score with it), persist to `scorer_versions`,
   emit `SCORER_RETUNED`. **Bounded** (only when degraded, under `max_retunes`,
   and only if materially better).
3. **The firewall (non-negotiable):** anchored to the **oracle, never an LLM**.
   The primary scorer (B1 `score()`) stays deterministic math; only the *weights*
   move. This is what keeps the recursion legitimate — not "GPT grading GPT." The
   `scorer_versions` history is the second money-shot curve: predictive validity
   *rising* — the system getting better at knowing what "better" means.

## Threading (additive, opt-in — no logic change)

`Conductor(..., emitter=None)`: `None` is byte-identical to pre-B8. With an
`EventEmitter` injected, the solve narrates the full stream (RUN_STARTED →
TEAM_DESIGNED → GENOME_EVALUATED/MODEL_PANEL_UPDATE → THRESHOLD_CHECK →
ESCALATION_*/AGENT_ROLLED_BACK → REARRANGE_ADOPTED → RUN_SEALED/EXHAUSTED). Every
emit is awaited but failure-safe — narration can never affect the solve's result.

## Handoff

**Frozen:** `RunEvent`/`RunEventType`, `EventEmitter.emit`, `EventBus`,
`EventStore`, `WebSocketServer`/`ClientSession`, `replay_run`,
`SelfImprovingScorer`, and `Conductor`'s optional `emitter`. The enhancement layer
(B9–B13) emits through this same stream, so the screen and replay stay the single
source of truth.
