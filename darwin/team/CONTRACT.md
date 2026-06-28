# Darwin B3 — Frozen Contract (the kitchen)

> B4 (the Architect, which authors genomes) and B5 (rearrangement, which mutates
> them) build against **four frozen surfaces**. As long as these hold, the
> Architect can invent any team and the runner runs it with **zero code changes**.

## The one guarantee

`TeamRunner.evaluate(genome, instance, weights) -> GenomeEvaluation`:
**genome in, valid scored answer out — always a number, never an exception.**
Every failure path returns a `GenomeEvaluation` with a real (floored) `fitness`;
the error is logged to the genome's `history` as an `EVAL_ERROR`. A genome that
errors is just a genome that scored at the floor — indistinguishable to B5/B6.

## Surface 1 — `TeamGenome` (the recipe card)

A small DAG: `agents` (each an `AgentNode` carrying an embedded, self-contained
B2 `AgentSpec`), `edges` (`PASSES_PROPOSAL` / `CHECKS` / `FEEDS_ARBITER` /
`SENDS_FEEDBACK`), and one `arbiter_id`. Frozen in memory, `extra="forbid"`.
Carries `version` (the optimistic-lock token), `current_fitness` /
`current_normalized_score`, `status`, and an embedded `history` of
`MutationRecord`s — so the entire evolutionary lineage is **one query**.
`validate(genome)` rejects cycles, non-terminal/missing arbiters, orphans,
unregistered models, and contract mismatches *before* a run.

## Surface 2 — `GenomeStore` (atomic, conflict-safe persistence)

Genomes are **mutated in place** with optimistic locking:

- `mutate(genome_id, expected_version, set_ops, record)` — one atomic
  `find_one_and_update({_id, version}, {$set, $inc:{version:1}, $push:{history}})`.
  Returns the after-document, or `None` if a concurrent writer already advanced
  the version (reload and retry).
- `retry_mutate(genome_id, derive_fn, max_attempts)` — load → derive → mutate,
  reload-and-retry on conflict. **No lost updates, no torn writes.**

Typed edits live in `mutations.py` (`add_edge` / `rearrange_edge` / `swap_model`
/ `retarget_arbiter` / `add_agent` / `remove_agent`), each a `derive_fn` that
rebuilds a *validated* candidate so an illegal edit is rejected before it writes.

## Surface 3 — `GenomeEvaluation` (the runner's output)

Always carries a real `fitness` + a full B1 `ScoreBreakdown`,
`normalized_score`, `cleared_threshold` (**feasible AND `normalized_score >=
0.90`** — an infeasible empty solution scores `normalized 1.0` but does *not*
clear), `arbiter_tier_used`, per-agent `agent_results`, latency, cost, and
`version` (exactly what was evaluated). `completed` means the arbiter answered
(Tier 1), not "evaluation didn't crash" (it never crashes).

## Surface 4 — `InferenceGate` (shared global concurrency bound)

One semaphore, created at startup, passed to **every** `TeamRunner`. Every model
call acquires it, so total in-flight calls across the whole swarm can never
exceed `MAX_CONCURRENT_INFERENCE` — a wide swarm degrades gracefully instead of
saturating the endpoint and producing timeouts that masquerade as bad genomes.

## Resilience guarantees (the three pre-flight tests gate every run)

- **Optimistic-lock contention** — concurrent writers can't clobber each other.
- **Three-tier arbiter fallback** — retry (×2) → best feasible proposal →
  infeasible sentinel; always a scored answer; fallbacks logged.
- **Saturation at swarm width** — peak in-flight ≤ the ceiling; no spurious
  floor-scores from infrastructure.

A single bad non-arbiter agent is survived: **retry once, then route around**;
the DAG tolerates missing upstream contributions by design.

## Handoff to B4 / B5

B4 authors `AgentSpec`s (roles, descriptions, per-job `model_id`) and emits the
initial genome B3 validates, stores, and runs. B5 proposes mutated candidates via
the typed mutations, evaluates them concurrently through the runner under the
shared gate, keeps the best, and — if the best `normalized_score < 0.90` — hands
to B6 (escalation) to add corpus/curated agents before rearranging again.

**These four surfaces are frozen. Changing them is a breaking change for B4–B6.**
