# Darwin B4 — Frozen Contract (the Architect / curator)

> The Architect is the **source of all invention**. B5 (rearrangement) and B6
> (escalation) build against two frozen capabilities. The Architect **never
> solves the problem** — it designs the team that will.

## The meta-agent principle

The Architect is an agent whose entire job is **designing other agents**. It
reads a problem, decides dynamically how to decompose it (the "3 parts" / "6
parts" — never a fixed menu), authors a fresh `AgentSpec` for each part (coining
the role, writing the job, assigning the best model), and assembles them into an
initial `TeamGenome`. "Curate a new agent" literally means *the Architect emits a
new `AgentSpec` and the runner spins it up* — no human in the loop.

## Capability 1 — `design_initial_team(instance, weights) -> TeamGenome`

Pipeline: analyze → author (`ArchitectTeamDesign`) → deterministically assemble
(`assembly.py`) → B3 `validate` → bounded repair (`MAX_DESIGN_REPAIRS`) →
persist. **Always returns a valid, persisted, version-1 `TeamGenome` and never
raises.** On repair exhaustion or any error it falls back to a minimal **safe
default** (a `cost_minimizer` proposer → a trivial `arbitrator`) that is always
valid and runnable. *Degraded, never dead.*

## Capability 2 — `curate_agent_for_gap(genome, instance, evaluation) -> (AgentSpec, [Edge])`

The escalation entry point B6 calls. **The deterministic scorer diagnoses; the
frontier Architect prescribes**: B1's `ScoreBreakdown` says exactly what's missing
(`DEMAND_UNMET` → demand-coverage specialist, high `raw_risk` → disruption-risk
modeler, capacity violations → capacity rebalancer, cost far from optimum → cost
specialist). Returns **exactly one** new agent that does not duplicate an
existing role and whose suggested wiring keeps the genome valid. Never crashes
(heuristic fallback).

## Guardrails (what keeps invention production-shaped)

- **The schema**: every authored agent must carry a `role`, `input_contract`,
  `output_contract`, and a **registry-legal** `model_id`.
- **B3 `validate`**: malformed teams (cycles, missing arbiter, orphans, bad
  models) are repaired or rejected at the boundary — they never reach the runner.
- **The direct-decision reminder** (mandatory in the system prompt): every
  authored agent reasons directly; it never writes or calls a solver.

## Tiering (the B7 seed)

The Architect itself runs on the **frontier** reasoner (`gemini-3.1-pro`) — rare,
hard design calls. The agents it authors run on **fast** `gemini-3.5-flash`;
deep-reasoning roles (the arbitrator) may get the frontier model. Each
assignment is justified in `why_this_model` — model-aware curation, and the seed
of B7's cost/latency-penalized model search.

## Handoff

B4 hands B5 a valid, persisted version-1 genome. B5 always rearranges it; B6 calls
`curate_agent_for_gap` when escalation needs a new agent. **These two
capabilities and the `ArchitectTeamDesign` schema are frozen.**
