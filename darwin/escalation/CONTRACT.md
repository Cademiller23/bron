# Darwin B6 — Frozen Contract (Threshold Gate, Escalation & the Conductor)

> One frozen surface: `Conductor.solve(instance, weights, budget) -> SolveResult`.
> The whole brain in one call — **always returns a real score, never regresses,
> never crashes, and gets better across problems.**

## The whiteboard loop (exactly)

```
B4 design  →  B5 ALWAYS rearrange  →  threshold gate (≥0.90?)
                                          │  yes → SEALED, return
                                          │  no  → B6 escalate (GROW the team)
                                          │           1. corpus reuse  (cheap)
                                          │           2. else curate    (B4)
                                          │        → B5 rearrange the larger team
                                          │        → keep iff it improved, else ROLL BACK
                                          └──────────── repeat until cleared or budget out
```

B5 reshapes a **fixed** agent set (the inner loop). B6 is the only thing that
**grows** the team (the heavier outer loop). The two loops alternate.

## Three defining properties

- **Always returns a real score.** Every boundary degrades rather than raises.
  `solve` returns a `SolveResult` whose `final_evaluation.fitness` is a real
  number — `SEALED` if it cleared 0.90, `EXHAUSTED` (best-so-far) if the budget
  ran out, or a floored result if an unexpected error hit the solve boundary.
- **Never regresses (team-growth elitism).** After escalating + rearranging, the
  added agent is kept **only if** `round_fitness > pre_escalation_fitness + ε`.
  Otherwise the team is **rolled back** to the pre-escalation snapshot (an
  optimistic-locked `REMOVE_AGENT` mutation). Growing the team can never make the
  answer worse.
- **Gets better across problems (the corpus).** A curated agent that proves
  useful is **promoted** to the MongoDB corpus with a performance record + a
  semantic embedding. On later problems the escalator searches the corpus
  **first** and reuses a proven agent instead of inventing one. Cold → always
  curate; warm → increasingly reuse. That compounding is the genuine
  "self-improvement across problems."

## Escalation — two ordered steps (`Escalator.escalate`)

1. **Corpus reuse.** `diagnose_gap` (deterministic, from the `ScoreBreakdown`)
   produces one `GapDescription` that is BOTH the corpus vector-search query and
   the curation seed. The corpus is searched (`$vectorSearch` → brute-force cosine
   fallback); the best candidate above the similarity threshold whose role isn't
   already on the team is added as an `ADD_AGENT_FROM_CORPUS` mutation, validly
   wired to the arbiter. Returns `method=CORPUS`.
2. **Curate.** Only if the corpus has nothing usable, B4 authors a new targeted
   agent, added as `ADD_CURATED_AGENT`. Returns `method=CURATED`.
3. Else `method=NONE_AVAILABLE` (the conductor stops growing).

Every addition is pre-validated (still a DAG, single reachable arbiter, legal
models/contracts) **before** the atomic optimistic-locked commit. Any failure
(bad wiring, commit conflict, curation error) degrades — that candidate is
skipped or the step returns `NONE_AVAILABLE`; the solve never crashes.

## The gap diagnosis (deterministic — the scorer diagnoses, not a model)

`diagnose_gap(evaluation, problem_class)` is a **pure function** of the
`ScoreBreakdown` (no model, no randomness, no wall-clock). It ranks weak
dimensions and routes:

| Condition                              | Weak dimension | Role kind |
|----------------------------------------|----------------|-----------|
| has violations + `DEMAND_UNMET`        | FEASIBILITY    | proposer  |
| has violations + capacity overflow     | FEASIBILITY    | checker   |
| `raw_risk > 0.45`                      | RESILIENCE     | specialist|
| feasible & `normalized < 0.90`         | COST           | proposer  |
| `norm_lead > 0.50`                     | LEAD_TIME      | specialist|

`severity = 1.0` if infeasible, else `max(0, 0.90 − normalized)`.

## The corpus (`AgentCorpus`) — the MongoDB story

- **search** embeds `gap.capability_needed`, fetches via Atlas `$vectorSearch`
  (falls back to a brute-force cosine scan when no index/server), filters by
  `similarity ≥ threshold`, ranks by
  `combined = sim · (1 + max(0, avg_fitness)) · log(times_reused + 2)`, with a
  `×1.25` boost when the problem class matches. Any failure → `[]`.
- **promote** upserts by `role_name` (idempotent — one row per role), maintaining
  a correct **running average** of the fitness contribution and incrementing
  `times_reused` / `success_count`, `$addToSet` the problem class. Any failure →
  `False` (the escalator simply curates instead).
- **update_stats** records a reuse outcome (success or failure) and updates the
  running average; an unknown id → `False`. A corrupt row is skipped, never fatal.
- Pluggable `Embedder`: `KeywordEmbedder` (deterministic, offline, real cosine)
  for tests/demo; `VoyageEmbedder` for production semantic quality. The SAME
  embedder embeds both the stored description and the query.

## Budget (`SolveBudget`) — bounded growth

`max_escalations`, `max_team_size`, `max_wall_clock_seconds`, `max_total_cost_usd`.
The outer loop halts as soon as any bound is reached; the partial best is returned
as `EXHAUSTED`. Defaults live in `darwin/constants.py`.

## `SolveResult` (frozen) — the whole-brain output + live story

`instance_id`, `final_genome`, `final_evaluation`, `cleared_threshold`, `status`,
`escalation_rounds`, `agents_added[]`, `corpus_hits`, `corpus_promotions`,
`full_fitness_trace` (the continuous climbing curve across B5 *and* B6),
`trace_markers[]` (where agents were added), `total_latency_ms`, `total_cost_usd`,
`per_round_summary[]`, `error`.

## Guarantees (why it's safe on stage)

- **Never crashes**: B4 (never raises), B5 (never raises), the escalator (degrades
  to `NONE_AVAILABLE`), the corpus (degrades to empty/no-op), and the rollback
  (best-effort) are each wrapped; the `solve` boundary floors any residual error.
- **Never regresses**: elitism + snapshot rollback guarantee team-level
  non-regression even though B5 may have reshaped the larger team.
- **Clean lineage**: every kept addition and every rollback is an atomic
  optimistic-locked B3 mutation with a labeled `MutationRecord`
  (`ESCALATION` actor).
- **Honest compounding**: the corpus starts empty; reuse only happens after a
  genuine promotion. `corpus_hits` counts corpus-supplied escalations;
  `corpus_promotions` counts useful curated agents written back.

## Handoff

**`Conductor.solve`, `SolveResult`, `SolveBudget`, `Escalator.escalate`,
`EscalationResult`, `AgentCorpus`, `diagnose_gap`, and `GapDescription` are
frozen.** This is the top of the stack — the entry point the demo/API calls.
