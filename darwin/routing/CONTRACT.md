# Darwin B7 — Frozen Contract (Multi-Model Routing & the Model Registry)

> The "model-aware" layer woven through B2–B6. B7 does **not** rebuild B2's
> model-agnostic client or B3's model gene — it **configures** the fleet,
> **formalizes** `model_id` as an evolvable gene, and **adds a cost/latency
> penalty** to the *selection* fitness so the gene matters. That penalty is the
> crux: without it the search would put the frontier model everywhere (more
> capability is never penalized); with it the swarm *discovers* — under a real
> budget — that mechanical roles belong on the cheap-fast MAX-served model and
> only the rare hard decisions (the arbiter, the Architect) need a frontier model.

## The one rule that makes B7 real: two fitnesses, two decisions

- **TASK fitness** = B1's `normalized_score` (Q ∈ [0,1]). This, and *only* this,
  drives B6's **0.90 gate**. A team clears by solving the problem, never by being
  cheap to run. (Distinguish *solution cost* — dollars to ship goods, inside Q —
  from *inference cost* — dollars/latency to run the agents, the penalty below.)
- **SELECTION fitness** = `efficiency_adjusted_fitness` = `fitness − λ_cost·C_norm
  − λ_latency·L_norm` (cost/latency from B3's `total_cost_usd`/`total_latency_ms`,
  normalized per round to [0,1]). The base is the raw `fitness` — which **equals
  Q for feasible teams** (where the cost trim matters) and is a large negative
  penalty for infeasible ones, so the bounded penalty (≤ 0.1) only ever tie-breaks
  among teams of equal raw fitness and can never promote a more-violated infeasible
  team. This drives argmax choices — B5's "keep the best candidate" and B6's
  team-growth elitism — so among teams of *similar task quality*, the cheaper one
  is preferred, while B5's non-decreasing fitness curve is preserved.

**The threshold guard (lexicographic, provable).** `compare`/`best_index` rank by
`(clears_the_gate?, efficiency_adjusted_fitness)`. A team that clears 0.90 always
outranks one that doesn't, *regardless* of efficiency — so efficiency can never
sacrifice clearing the gate. Below threshold Q dominates (λ small); once teams
clear, the efficiency term trims expensive models that aren't needed.

**Adoption = hold quality, cut cost.** `improves(candidate, incumbent)` adopts a
grown/reshaped team only if it strictly improves the efficiency-adjusted fitness
**and never trades raw quality Q down** (it may cross the gate, never drop below
it, never reduce Q among same-status teams). So B5's non-decreasing curve and the
gate-on-raw-Q rule both hold even while the model search makes the team cheaper.

## The curated fleet (`fleet.py`) — ~5 models, one interface

`FleetModel` carries the routing/deploy metadata B2's frozen `ModelEntry` doesn't
(api_key_env, default_thinking_level, hf id) and projects to a `ModelEntry` via
`to_registry_entry()`. Tiers reuse B2's `CapabilityTier` (**`CHEAP` is the spec's
"FAST" tier**):

| Tier | Models | Role |
|------|--------|------|
| FAST (CHEAP) | `max-llama-3.1-8b` (Modular MAX), `minimax-m2` | the workhorses — bulk mechanical/proposer calls |
| MID | `gemini-3.5-flash`, `max-llama-3.3-70b` (MAX) | proposers / objective specialists |
| FRONTIER | `gemini-3.1-pro` | the arbiter + the Architect only |

`get_fleet()`, `profile(model_id)`, `by_tier(tier)`, `install_fleet(registry=None)`
(registers into the registry — idempotent), `fleet_registry()` (a fresh,
self-contained registry). The fleet is validated at import (prices ≥ 0, OPENAI
endpoints well-formed http(s), all three tiers present, no duplicate ids). Every
model is reached through B2's one `ModelClient.complete(model_id, …)` — Gemini via
the native backend, everything else (MAX/MiniMax/DigitalOcean) via the
OpenAI-compatible backend. **Adding a model is a registry entry, not code.**

## The model gene (`gene.py`)

`model_id` on each `AgentNode.spec` — no schema change. `genotype(genome)` /
`model_of(genome, agent_id)` read it. Model-aware operators share B5's
`op(genome, rng, registry) -> Optional[CandidateRearrangement]` signature, change
**only** model_ids (agent set + wiring invariant, validity preserved), skip
degraded models, and emit `SWAP_MODEL` via B3's optimistic-locked `store` with a
**relative** derive (recomputed on the freshly-loaded genome → conflict-safe):

- `downgrade_mechanical` — a checker → a cheaper FAST model (drives work onto MAX).
- `upgrade_critical` — the arbiter → a FRONTIER model.
- `swap_to_tier(genome, agent_id, tier, registry)` — building block (cheapest in tier).
- `retier_to_policy` — a random agent → the cheapest model in its policy tier.
- `rebalance_genotype` — compound: cheap periphery + strong center in one move.

`MODEL_AWARE_OPERATORS` is the list B5's generator samples (via `extra_operators`).

## The warm-start policy (`policy.py`)

`classify_role(spec, is_arbiter)` derives the role kind from `output_contract` +
arbiter position (`AgentSpec` has no explicit role_kind). `suggest_tier`:
checker→FAST, proposer/specialist/decomposer→MID, arbiter/architect→FRONTIER.
`suggest_model` returns the cheapest model in the preferred tier (walking a
fallback chain, restrictable to `allowed_ids` so it never suggests a model the
caller's registry lacks). This is the *reason* step; the penalized search is the
*verify* step.

## Observability (`observability.py`)

`aggregate(invocations, registry, fleet) -> ModelEconomics`: calls/cost/latency
per model, backend, and tier; `max_served_share` (MAX = OPENAI_COMPAT models the
fleet serves from a HF id — distinguished from MiniMax); the headline *"Modular
MAX served X% of all calls."* Pure and failure-safe — non-dict/malformed/non-finite
records are skipped, an empty stream degrades to zeros. `aggregate_sink` filters a
telemetry sink by instance/genome.

## The wiring (additive, opt-in — no contract break)

- `RearrangementLoop(..., selector=None, extra_operators=None)`: `selector=None`
  is byte-identical to pre-B7 (max-argmax-fitness + strict-ε adoption). Inject
  `EfficiencyStrategy()` + `MODEL_AWARE_OPERATORS` for B7 behavior.
- `generate_candidates(..., extra_operators=None)`: samples the model-aware
  operators alongside the base set.
- `Conductor(..., comparator=None)`: `None` keeps the raw-delta elitism; inject
  `EfficiencyStrategy().improves` for the guarded comparator. **The 0.90 gate stays
  on raw Q regardless.**

## Honesty constraint

Curate **~5** models and demo on that tractable tier; the genotype space is
`(#models)^(#agents)` — five keeps it in the low thousands, which the penalty-guided
search converges on. The method *scales* to a catalog of thousands (a single
registry entry per model, all behind two interfaces) — **never claim to search
thousands live.**

## Handoff

**Frozen:** `get_fleet`/`install_fleet`/`profile`/`by_tier`, `suggest_model`/
`suggest_tier`/`classify_role`, `efficiency_adjusted_fitness`/`compare`/`improves`
+ `EfficiencyStrategy`/`RawFitnessStrategy`, `MODEL_AWARE_OPERATORS`, and
`aggregate`/`ModelEconomics`. B8 streams the per-model economics to MongoDB and
renders the model panel.
