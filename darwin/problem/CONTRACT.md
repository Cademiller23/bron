# Darwin B1 — Frozen Contract (v1.0.0)

> This document defines the **locked** data contract that B2–B8 build against.
> The scorer carries `scorer_version = "1.0.0"`. Changing any field semantics
> below is a breaking change and must bump the version.

The whole point of B1: turn any supply-chain optimization problem into **one
canonical object** every agent reads identically, and given any proposed
solution return **one number in under a millisecond** where `1.0` (100%) means
"you matched the known optimum" — using **pure deterministic arithmetic, never
an LLM**.

## The three nouns

| Type | Role |
|------|------|
| `ProblemInstance` | the canonical, validated problem (nodes + arcs + constraints + known optimum) |
| `Solution` | a proposed answer (`flows` for network-flow classes, `routes` for VRP) |
| `ScoreBreakdown` | what `score()` returns — feasibility, raw objectives, `final_fitness` |

All models are **frozen** (immutable) and **reject unknown fields**
(`extra="forbid"`). Construction validates: referential integrity, unique ids,
non-negative costs/capacities/demands, `risk_score ∈ [0,1]`, no NaN/inf, and
`total_supply ≥ total_demand` (else the instance is rejected as structurally
infeasible).

## The one number: `final_fitness`

`score(instance, solution, weights) -> ScoreBreakdown` runs these steps in this
order, deterministically:

1. **Structural validation** — malformed solution ⇒ scores dead-last
   (`final_fitness = -1e15`), never crashes.
2. **Feasibility** — one `Violation` per breach (arc/node capacity, supply,
   demand, conservation, closed-facility, lead-time, declared constraints).
3. **Raw objectives** — `raw_cost`, `raw_lead_time` (max over used arcs),
   `raw_risk` (deterministic resilience).
4. **Blend** — each objective scale-normalized, then weighted by
   `ObjectiveWeights` (the dial B8 turns).
5. **Normalize** — `normalized_score = min(1.0, optimum / achieved_cost)`.
6. **Penalize** — feasible ⇒ `final_fitness = normalized_score ∈ [0,1]`;
   infeasible ⇒ `final_fitness = -(penalty) < 0`.
7. **Stamp** — `scorer_version`, `computed_at` (metadata only — never in the number).

### Guaranteed invariants (unit-tested)

- **Determinism**: identical inputs ⇒ byte-identical `final_fitness`;
  order-independent (shuffling `flows` cannot change the score).
- **No model / no randomness / no wall-clock** ever enters the number.
- **Feasible beats infeasible**: every feasible solution ranks above every
  infeasible one; more violations ⇒ strictly lower; larger magnitude ⇒ never higher.
- **Optimum ⇒ 1.0**, twice-optimal cost ⇒ 0.5.

## Resilience (`raw_risk ∈ [0,1]`, lower is better)

`raw_risk = 0.3·C + 0.3·E + 0.4·W` where:
- **C** = demand-weighted single-largest-supplier share (concentration),
- **E** = `Σ(flow·risk)/Σflow` over used arcs (exposure),
- **W** = worst-case single-failure unmet demand (remove the biggest source,
  recompute deliverable demand via exact max-flow).

## Enumerations (locked)

`NodeType{SOURCE,TRANSSHIPMENT,SINK}` ·
`ProblemClass{TRANSPORTATION,TRANSSHIPMENT,FACILITY_LOCATION,VEHICLE_ROUTING}` ·
`Difficulty{EASY,MEDIUM,HARD}` ·
`ConstraintType{CAPACITY,DEMAND_SATISFACTION,FLOW_CONSERVATION,LEAD_TIME_LIMIT,SINGLE_SOURCE,MUTUAL_EXCLUSION,CUSTOM}` ·
`OptimumSource{BENCHMARK_LABEL,SOLVER_VERIFIED,UNKNOWN}` ·
`ViolationType{OVER_ARC_CAPACITY,OVER_NODE_CAPACITY,SUPPLY_EXCEEDED,DEMAND_UNMET,CONSERVATION_BROKEN,CLOSED_FACILITY_USED,LEAD_TIME_EXCEEDED,MALFORMED_SOLUTION,CUSTOM_CONSTRAINT}`

## Extension points (declared, not built this weekend)

- Multi-period / inventory dynamics / stochastic demand — out of scope.
- VEHICLE_ROUTING is **supported** as a parallel scorer branch producing the
  same `ScoreBreakdown` (continuous-Euclidean metric).

**The contract above is frozen. B2 (the worker agent) builds strictly against it.**
