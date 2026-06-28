# Darwin B5 — Frozen Contract (the Rearrangement Loop)

> One frozen surface: `RearrangementLoop.run(genome, instance, weights) ->
> RearrangementResult`. The always-on inner loop that climbs the score by
> reshaping the team — **always runs, never regresses, never crashes.**

## Two defining properties

- **Always runs.** After the initial team, the system *always* attempts at least
  one rearrangement pass — even a team that already clears 0.90 (it might reach
  higher).
- **Never regresses (elitism).** A candidate is adopted **only if it strictly
  beats the current best** (`fitness > best + EPSILON`). A bad rearrangement is
  simply not adopted, so the team can never get worse and the `fitness_trace` is
  monotonically **non-decreasing** — the live "it's getting better" curve.

## The loop (programmatic, not LLM-per-candidate)

Each round: generate `K` rearranged candidates (fast mutation operators over the
genome graph) → evaluate them **concurrently** under B3's shared inference gate
(`persist_outcome=False`, unpersisted) → adopt the best if it strictly improves →
commit it as an atomic optimistic-locked B3 mutation (`REARRANGER` record with
`fitness_before`/`after`) → repeat. Stops on plateau (`PATIENCE`), ceiling,
threshold (per policy), or `MAX_ITERS`.

The LLM's creativity is spent in B4 (design) and B6 (curation); B5's inner loop
is **programmatic** so it runs fast and renders live. An optional lightweight
`reorganizer` (one cheap call per *round*, not per candidate) may bias which
operators to try; the generator works fully without it.

## The operators — reshape only, never add/remove an agent

The **agent set is invariant** under B5 (growth is strictly B6's job):
`reassign_model`, `redirect_edge`, `reorder_pipeline`, `swap_arbiter`. Every
candidate is validated (still a DAG, still a single reachable arbiter); invalid
ones are discarded and resampled. Commit derives are **relative** (recompute on
the freshly-loaded genome), so adoption is conflict-safe under the optimistic
lock.

## Guarantees (why it's safe on stage)

- **Never crashes**: B3's runner never raises (a failing candidate floor-scores
  and is never adopted); the loop tolerates floor scores and commit failures.
- **Bounded concurrency**: `K` candidates × `M` agents stay under the shared
  semaphore, so a wide round degrades gracefully (no timeouts masquerading as bad
  genomes — exactly B3's saturation-test scenario).
- **Clean lineage**: transient candidates are never persisted; only adopted
  winners land in the genome's `history`.
- **Live story**: every round emits `{iteration, best_fitness, normalized_score,
  adopted, mutation_description, genome_version}` for the org-chart animation, the
  climbing curve, and the voice narration.

## Handoff (the whiteboard loop)

B4 designs → **B5 always rearranges** → threshold check (≥0.90 done; else B6
escalates by adding a corpus/curated agent) → B5 again on the larger team →
repeat. **`RearrangementLoop.run` and `RearrangementResult` are frozen.**
