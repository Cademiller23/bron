# Darwin B2 — Frozen Contract (the atom)

> B3 (the team runner) and B4 (the Architect) build against **four frozen
> surfaces and nothing else**. As long as these hold, B3 can wire any topology
> of these atoms and B4 can author the specs that fill them, without ever
> reaching inside the worker.

## The non-negotiable rule

The worker produces **candidate** solutions; it **never grades them**. It
imports no scorer and computes no fitness — all quality judgment is B1's
deterministic scorer's job. The worker's only contract is: *return valid,
schema-conforming structured output, or fail gracefully.*

## Surface 1 — `AgentSpec` (the role the Architect fills, B4)

Frozen, `extra="forbid"`. Fields: `agent_id`, `role_name` (slug),
`role_description`, `input_contract` (`InputKind`), `output_contract`
(`OutputKind`), `model_id` (must be in the registry — **fail-fast at
construction**), `thinking_level` (`minimal|low|medium|high`),
`max_output_tokens` (>0), `tool_names`, `created_by`, `spec_version`.

## Surface 2 — `AgentResult` (what B3 reads)

Frozen. `{agent_id, role_name, model_id, success, output, raw_text, num_repairs,
latency_ms, usage{tokens_in,tokens_out}, est_cost, error, produced_at}`.
**No fitness field — by design.** `run()` returns this on *every* path and
**never raises** into the caller. B3 relies on `success`, `output`, and the
cost/latency fields.

## Surface 3 — `ModelClient` (one shared, registry-backed instance)

`await client.complete(model_id, system, user, response_schema, thinking_level,
max_output_tokens, timeout) -> ModelResponse`. Dispatches purely on the
registry's `provider`, applies timeout / bounded-retry (429, 5xx) / fail-fast
(401/403) / circuit-breaker, and always returns a uniform `ModelResponse`
(never raises). B3 passes **one** client to every worker so model assignment is
centralized; **switching `model_id` changes the provider with no change to
`worker.py`** (Gemini is just the first registry entry).

## Surface 4 — the six output schemas

`FullSolutionOutput`, `PartialSolutionOutput`, `CritiqueOutput`,
`ConstraintReportOutput`, `ArbitrationOutput`, `DecompositionOutput`. Each is
frozen, `extra="forbid"`, shallow + enum-rich, and is both the response JSON
Schema handed to the model and the validator the return is checked against.
`FullSolutionOutput`/`ArbitrationOutput` wrap a B1 `Solution` the scorer grades.

## Invariants (unit-tested, §12)

- **Never raises**: success, repaired-success, parse-fail, schema-violation,
  transport-error, timeout, auth → all return an `AgentResult`.
- **Structured-output ladder**: native JSON-schema → parse+validate
  (`extra="forbid"`) → bounded repair (`MAX_REPAIRS=2`) → graceful degradation.
- **No low temperature** on Gemini 3.x (looping guard); reproducibility comes
  from the scorer, not the sampler.
- **Telemetry on every path**, failure-safe (a Mongo hiccup degrades to a local
  log). Every invocation captures usage + `est_cost` for B7's router.
- **Model-agnostic**: `worker.py` never names a provider.

## Handoff to B3

B3 instantiates `WorkerAgent(spec, client, telemetry)` per `AgentSpec` in a
genome, calls `await worker.run(agent_input)`, passes one shared `ModelClient`,
and stamps `agent_input.team_genome_id` for telemetry provenance.

**These four surfaces are frozen. `scorer_version`-style stability applies:
changing them is a breaking change for B3–B8.**
