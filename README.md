# Darwin — Supply-Chain Backend (Phases B1–B8)

Eight phases are built and frozen — **B1–B8 are the complete working brain**:
- **B1 — the Problem Loader & Scorer** (`darwin/problem/`): the world + the judge.
- **B2 — the Worker Agent** (`darwin/agent/`): the model-agnostic atom every team is built from. See [`darwin/agent/CONTRACT.md`](darwin/agent/CONTRACT.md).
- **B3 — the Team Genome & Team Runner** (`darwin/team/`): the kitchen — a DAG of B2 atoms, mutated in place with optimistic locking, executed into one scored answer that's always a number and never an exception. See [`darwin/team/CONTRACT.md`](darwin/team/CONTRACT.md).
- **B4 — the Architect** (`darwin/architect/`): the meta-agent that *designs* teams (it never solves the problem). See [`darwin/architect/CONTRACT.md`](darwin/architect/CONTRACT.md).
- **B5 — the Rearrangement Loop** (`darwin/rearrange/`): the always-on inner loop that climbs the score by reshaping the team — always runs, never regresses. See [`darwin/rearrange/CONTRACT.md`](darwin/rearrange/CONTRACT.md).
- **B6 — the Threshold Gate, Escalation & Conductor** (`darwin/escalation/`): the heavier outer loop and the top-level `Conductor.solve` entry point — grows the team (corpus reuse, then curation) only when rearrangement can't clear 0.90, keeps an added agent only if it helped, and gets better across problems via the agent corpus. See [`darwin/escalation/CONTRACT.md`](darwin/escalation/CONTRACT.md).
- **B7 — Multi-Model Routing & the Model Registry** (`darwin/routing/`): the "model-aware" layer — a curated ~5-model fleet (FAST/MID/FRONTIER) behind one interface, `model_id` as an evolvable gene, and a cost/latency penalty on the *selection* fitness so the swarm discovers "cheap-fast for mechanical roles, frontier for the arbiter." See [`darwin/routing/CONTRACT.md`](darwin/routing/CONTRACT.md).
- **B8 — Persistence, Telemetry & the Self-Improving Scorer** (`darwin/observability/`): the observability capstone — every moment of a solve is a durable, ordered `RunEvent` (event sourcing) that drives the live screen over a WebSocket bus and replays exactly; plus the optional second-order loop that tunes B1's objective weights toward the oracle (never an LLM). See [`darwin/observability/CONTRACT.md`](darwin/observability/CONTRACT.md).

---

## Phase B1: The Problem Loader & Scorer

B1 is the foundation every other Darwin phase stands on. It has exactly two jobs:

1. **Load** any supply-chain optimization problem (IndustryOR, Mamo, CVRPLIB, or
   generated live) into **one canonical, validated, in-memory `ProblemInstance`**
   that every agent reads identically.
2. **Score** any proposed `Solution` and return **one number in under a
   millisecond** — `final_fitness`, where `1.0` means "you matched the known
   optimum" — using **pure deterministic math, never an LLM.**

Everything downstream (the rearrangement loop B5, the threshold gate B6, the
evolutionary selection, the live fitness curve) consumes that one number. If B1
is wrong, every curve is a lie; if B1 is slow, the live demo stalls. So B1 is
built first, tested hardest, and **frozen as a contract** before B2 begins — see
[`darwin/problem/CONTRACT.md`](darwin/problem/CONTRACT.md).

> The single most important rule: **the scorer is arithmetic, not judgment.** No
> model call ever touches the fitness number — that is what makes the recursion
> story airtight when a judge asks "how is this not GPT grading GPT?"

## Layout

```
darwin/problem/
  schemas.py        # the canonical data model — the frozen contract
  resilience.py     # deterministic resilience/risk metric (the differentiator)
  scorer.py         # the deterministic fitness function (score())
  flow.py           # exact pure-Python max-flow / min-cost-flow primitives
  oracle.py         # OR-Tools (+ pure-Python fallback) ground-truth solver
  loader.py         # top-level loader + caching
  adapters/         # industryor.py, mamo.py, cvrplib.py, generated.py
  generator.py      # live fresh-instance generator (seeded, optimum-attached)
  fixtures.py       # hand-built golden instances (optima computed by hand)
  demo_instances.py # curated, oracle-verified demo suite (spans difficulty)
  data/             # sample raw files, one per source format
  tests/            # the full unit-test suite (mirrors the modules)
```

## Install & test

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest          # 565 tests (B1–B8), ~4s
```

`ortools` (B1 oracle), `google-genai` / `motor` / `httpx` (B2) are all **lazy,
optional imports**: the oracle falls back to exact pure-Python, telemetry falls
back to in-memory, and the Gemini provider is only touched by the gated
integration test — so the package never hard-depends on a heavy binary and the
whole unit suite runs offline.

## Quick start

```python
from darwin.problem import score, ObjectiveWeights
from darwin.problem.loader import load_instance
from darwin.problem import oracle

inst = load_instance("industryor", "darwin/problem/data/industryor_sample.json")

# Never trust a benchmark label blindly — verify it with the oracle first.
agrees, labeled, solver_value, _ = oracle.verify_label(inst)

# Score any solution; the oracle's optimal solution scores exactly 1.0.
breakdown = score(inst, oracle.solve_optimum(inst).solution)
print(breakdown.feasible, breakdown.normalized_score, breakdown.final_fitness)
```

Generate a fresh, never-seen instance on stage (with a verified optimum attached
the instant it is created):

```python
from darwin.problem.generator import generate_instance
from darwin.problem.schemas import ProblemClass

inst = generate_instance(seed=20260627, problem_class=ProblemClass.TRANSPORTATION)
assert inst.known_optimum.verified            # oracle already confirmed it
```

## Scope decision (§12)

The spine is the **network-flow family** (transportation / transshipment /
facility location) — clean known optima, maximally legible, and where org
structure genuinely changes the answer. **Vehicle routing (CVRP) is included** as
a parallel scorer branch producing the same `ScoreBreakdown` (continuous
Euclidean metric, exact brute-force oracle), so the routes-on-a-map visual is
available without compromising the bulletproof network-flow core.

## Acceptance criteria status

- ✅ Every §8 test passes, including the Hypothesis property invariants.
- ✅ 8 demo instances loaded and oracle-label-verified, spanning difficulty;
  some clear the threshold by rearrangement (EASY) and some force escalation
  (HARD) — see `demo_instances.curated_demo_instances()`.
- ✅ The live generator produces a fresh, feasible, optimum-attached instance on
  a new seed in well under a second.
- ✅ Scoring returns in < 1 ms (~36 µs on the golden instance) and is
  byte-for-byte deterministic and order-independent.
- ✅ No scorer code path calls a model, uses randomness, or puts wall-clock time
  in the number.
- ✅ The schema is frozen and documented as the contract for B2–B8.

---

## Phase B2: The Worker Agent (the atom)

B2 is the single reusable unit of intelligence every team is built from: one
generic, **model-agnostic** agent that takes an `AgentSpec` (the role B4's
Architect authors) + an `AgentInput`, calls a model through the registry-backed
`ModelClient`, and returns **strict structured output** as an `AgentResult` —
never free text, never a fitness number.

> The worker produces *candidates*; it **never grades them.** It imports no
> scorer and computes no fitness — quality judgment stays B1's alone. That is
> what keeps the recursion story airtight.

```
darwin/agent/
  spec.py              # AgentSpec — the contract the Architect fills (frozen)
  outputs.py           # the six structured output schemas + OutputUnion
  registry.py          # ModelRegistry/ModelEntry — maps model_id -> provider (B7 extends)
  client.py            # ModelClient (timeout/retry/circuit-breaker) + ModelResponse
  providers/gemini.py        # native Gemini adapter (schema mode, thinking dial, no low temp)
  providers/openai_compat.py # OpenAI /v1/chat/completions adapter (the rest of the fleet)
  parsing.py           # robust JSON extractor (fences + balanced-bracket scan)
  telemetry.py         # MongoDB sink (failure-safe) + corpus seed
  worker.py            # WorkerAgent.run — the invocation pipeline + AgentInput/AgentResult
  fixtures.py          # canned responses + ScriptedProvider for API-free tests
  tests/               # the full §12 suite (112 tests, all offline)
```

```python
from darwin.agent import WorkerAgent, AgentInput, ModelClient, AgentSpec, InputKind, OutputKind
from darwin.agent.telemetry import InMemoryTelemetrySink
from darwin.problem.fixtures import golden_transportation

spec = AgentSpec(agent_id="a1", role_name="cost_minimizer",
                 role_description="Minimize total transportation cost.",
                 input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION)
worker = WorkerAgent(spec, ModelClient(), InMemoryTelemetrySink())
result = await worker.run(AgentInput(instance=golden_transportation()))  # never raises
# result.success / result.output (a FullSolutionOutput) / result.est_cost / result.usage
```

**Model-agnostic by construction:** Gemini is one registry entry; switching
`model_id` to a hosted-fleet model (B7) changes the provider with **no change to
`worker.py`**. Set `GEMINI_API_KEY` to run the one gated integration test that
makes a real call and feeds the result into the B1 scorer. The four frozen
handoff surfaces for B3 are documented in
[`darwin/agent/CONTRACT.md`](darwin/agent/CONTRACT.md).

---

## Phase B3: The Team Genome & Team Runner (the kitchen)

B3 defines the **genome** (a DAG of B2 atoms wired by edges, with one arbiter) and
the **team runner** that executes it into one scored answer. Its contract is
exact: **genome in, valid scored answer out — always a number, never an
exception.**

```
darwin/team/
  genome.py            # TeamGenome / AgentNode / Edge / MutationRecord + enums (frozen)
  evaluation.py        # GenomeEvaluation — always a real fitness + a ScoreBreakdown
  validation.py        # DAG / single-arbiter / reachability / model / contract checks
  store.py             # GenomeStore — atomic optimistic-locked mutate / retry_mutate
  mutations.py         # typed edits (add/swap/retarget/remove) B5/B6 call
  inference_gate.py    # shared semaphore bounding global inference concurrency
  runner.py            # TeamRunner.evaluate — topo exec, 3-tier arbiter fallback, never-raises boundary
  fixtures.py          # FakeMongoCollection (atomic) + fixture genomes + scripted workers
  tests/               # incl. the THREE MANDATORY pre-flight tests (61 tests, offline)
```

Three resilience properties gate every run (the mandatory pre-flight tests):
**optimistic-lock contention** (concurrent writers can't clobber each other),
**three-tier arbiter fallback** (retry → best feasible proposal → infeasible
sentinel; always a scored answer), and **saturation at swarm width** (a shared
semaphore bounds peak in-flight calls so the fitness signal reflects topology,
not infrastructure). Genomes are **mutated in place** in MongoDB via atomic
`find_one_and_update({_id, version})`, with the lineage embedded in `history` so
the evolutionary tree is one query. The four frozen handoff surfaces for B4/B5
are documented in [`darwin/team/CONTRACT.md`](darwin/team/CONTRACT.md).

---

## Phase B4: The Architect (the curator / meta-agent)

B4 is the meta-agent whose entire job is **designing other agents** — it never
solves the problem itself. It reads a problem, decides dynamically how to
decompose it (no fixed menu of roles), authors a fresh `AgentSpec` for each part,
and assembles them into an initial `TeamGenome` B3 runs.

```
darwin/architect/
  schemas.py     # ProblemAnalysis / AgentSpecDraft / EdgeDraft / ArchitectTeamDesign / CuratedAgentDraft
  assembly.py    # deterministic design -> TeamGenome (slugify, role->id, layout, validate)
  prompts.py     # the meta-role system prompt + model catalog + direct-decision reminder
  architect.py   # design_initial_team / curate_agent_for_gap / repair loop / safe-default fallback
  tests/         # happy / repair / safe-default / model-legality / direct-decision / curate (30 tests)
```

`design_initial_team` **always** returns a valid, persisted genome and never
raises — on repair exhaustion or any error it falls back to a minimal **safe
default** (*degraded, never dead*). `curate_agent_for_gap` diagnoses the missing
capability straight from B1's deterministic `ScoreBreakdown` and authors **one**
targeted agent. The Architect runs on the **frontier** `gemini-3.1-pro`; the
agents it authors run on **fast** `gemini-3.5-flash`. See
[`darwin/architect/CONTRACT.md`](darwin/architect/CONTRACT.md).

## Phase B5: The Rearrangement Loop (the always-on inner loop)

B5 climbs the score by **reshaping** the team (reorder the pipeline, redirect an
edge, swap the arbiter, reassign a model) — never adding agents (that's B6). It
**always runs** at least one pass and **never regresses** (elitism: adopt only
strict improvements), so the fitness curve is monotonically non-decreasing — the
demo money-shot (38→50→62→78).

```
darwin/rearrange/
  operators.py    # 4 agent-set-invariant ops (reassign_model/redirect_edge/reorder_pipeline/swap_arbiter)
  generator.py    # K distinct, valid candidates (de-duped, agent-set invariant)
  reorganizer.py  # optional LLM steering (one call per round, not per candidate) + heuristic
  loop.py         # RearrangementLoop.run — generate -> evaluate-concurrently -> elitist adopt/commit
  tests/          # climbs / never-regresses / plateau / ceiling / bounded / commits / unpersisted (32 tests)
```

The inner loop is **programmatic** (fast, live), candidates are evaluated
**unpersisted** under B3's shared gate, and only adopted winners are committed as
atomic optimistic-locked mutations with full `REARRANGER` lineage. See
[`darwin/rearrange/CONTRACT.md`](darwin/rearrange/CONTRACT.md).

Together B4→B5 are the whiteboard loop: the Architect designs, B5 always
rearranges, and (in B6) escalation grows the team only when rearrangement alone
can't clear 90%.

---

## Phase B6: The Threshold Gate, Escalation & the Conductor (the whole brain)

B6 is the top of the stack: one call, `Conductor.solve(instance, weights, budget)
-> SolveResult`, runs the entire whiteboard loop. B4 designs → B5 **always**
rearranges → the 0.90 gate is checked → and only when reshaping can't clear it,
B6 **grows** the team. Growth is two ordered steps: reuse a proven agent from the
**corpus** (cheap), else **curate** a new one (B4). The larger team is rearranged
again, and the added agent is **kept only if it strictly improved the score** —
otherwise the team is **rolled back** (team-growth elitism, never-regress). Useful
curated agents are **promoted** to the corpus, so the system gets better across
problems: cold it always curates; warm it increasingly reuses.

```
darwin/escalation/
  schemas.py     # GapDescription, CorpusEntry, EscalationResult, SolveBudget, SolveResult (frozen)
  diagnosis.py   # diagnose_gap — deterministic gap from the ScoreBreakdown (the scorer diagnoses)
  embedding.py   # KeywordEmbedder (offline, real cosine) + VoyageEmbedder (prod) + cosine_similarity
  corpus.py      # AgentCorpus — $vectorSearch→brute-force search, promote/update_stats running averages
  escalator.py   # Escalator.escalate — corpus-first then curate, valid wiring, optimistic-locked add
  conductor.py   # Conductor.solve — the outer loop, gate, elitism + rollback, budgets, solve boundary
  tests/         # schemas/diagnosis/corpus/escalator/conductor + an offline compounding integration (49 tests)
```

The **agent corpus** is the genuine "gets better across problems" mechanism and
the strongest MongoDB story: each useful agent is stored with a performance
record and a semantic embedding (Atlas Vector Search, with a brute-force cosine
fallback), and later problems search it first. Every boundary **degrades rather
than crashes** — `solve` always returns a `SolveResult` with a real fitness:
`SEALED` if it cleared 0.90, `EXHAUSTED` (best-so-far) if a budget ran out. See
[`darwin/escalation/CONTRACT.md`](darwin/escalation/CONTRACT.md).

---

## Phase B7: Multi-Model Routing & the Model Registry (the model-aware layer)

B7 is a **thin layer woven through B2–B6**, not a new subsystem. B2 already gave
the model-agnostic client and B3 already made `model_id` a gene; B7 configures the
curated fleet, formalizes the gene, and adds a **cost/latency penalty to the
*selection* fitness**. That penalty is the crux: without it the search would put
the frontier model on every agent (more capability is never penalized), so there
would be nothing to discover. With it, the swarm discovers — under a real budget —
that mechanical roles belong on the cheap-fast **Modular MAX**-served model and
only the rare hard decisions (the arbiter, the Architect) need a frontier model.

```
darwin/routing/
  fleet.py          # the curated ~5-model fleet (FAST/MID/FRONTIER) -> ModelEntry; one interface
  policy.py         # role-kind -> tier -> model warm-start (the principled first pass)
  efficiency.py     # the penalized SELECTION fitness + the guarded lexicographic comparator
  gene.py           # model_id as an evolvable gene + the model-aware operators (SWAP_MODEL)
  observability.py  # per-model/backend/tier economics -> "Modular MAX served X% of all calls"
  tests/            # fleet/policy/efficiency/gene/observability/wiring/provider-keys + gated integration (78 tests)
```

**Two fitnesses, two decisions** (the central rule): the **0.90 gate** is judged
on raw task `normalized_score`; **selection** uses `efficiency_adjusted_fitness =
Q − λ·(cost,latency)` with a **lexicographic threshold guard** that *provably*
never lets efficiency sacrifice clearing the gate (a clearing team always outranks
a non-clearing one). Below threshold Q dominates and the search climbs; once teams
clear, the penalty trims expensive models — **quality held, cost cut**.

The wiring is **additive and opt-in**: `RearrangementLoop(selector=…,
extra_operators=…)` and `Conductor(comparator=…)` default to the exact pre-B7
behavior, so the whole prior suite is unchanged. We **curate five models and demo
on that tractable tier**, and frame the catalog of thousands as the vision —
adding any model is a single registry entry, since every backend already speaks
one of two interfaces. See [`darwin/routing/CONTRACT.md`](darwin/routing/CONTRACT.md).

---

## Phase B8: Persistence, Telemetry & the Self-Improving Scorer (the capstone)

B8 makes the whole brain **observable and replayable**, and adds the optional
second-order loop. It changes **no phase's logic** — it threads an event emitter
through the Conductor, which emits at each key moment. With B8 done, **B1–B8 are
the complete working brain**, entered through `Conductor.solve`.

```
darwin/observability/
  events.py                 # RunEvent + the event-type narrative (event sourcing)
  bus.py                    # in-process async pub/sub (slow-subscriber-safe)
  emitter.py                # EventEmitter.emit — durable append + publish, failure-safe, monotonic
  store.py                  # Motor run_events / scorer_versions (fake-collection testable)
  websocket_server.py       # the bus -> TypeScript face bridge (resume-from-sequence, no gap/dup)
  replay.py                 # re-emit a past run in exact order (the pre-recorded demo backup)
  self_improving_scorer.py  # the second-order loop: calibrate vs the oracle, re-tune the weights
  tests/                    # events/bus/store/emitter/conductor/websocket/replay/scorer (57 tests)
```

**Event sourcing.** Domain state (genomes, invocations, corpus) stays
authoritative; a single ordered stream of `RunEvent`s (`run_events`) is the
narrative the screen animates and replay reads — events reference domain objects
by id + version. Emission is async, **non-blocking, and failure-safe** (a Mongo or
bus hiccup degrades, never blocks or crashes the solve), and `sequence_number` is
monotonic and gap-free even under concurrent emits. A reconnecting client resumes
from its last `sequence_number` with no gap and no duplicate.

**The self-improving scorer** is the airtight answer to "how is this recursive,
not just a search loop?" It calibrates the scorer's ranking against the OR-Tools
**oracle** (Spearman correlation) and, when the weights stop predicting true
optimality, re-tunes B1's `ObjectiveWeights` toward the oracle — **anchored to
ground truth, never an LLM**, with the primary scorer staying deterministic math.
The `scorer_versions` history is the second money-shot curve: the system getting
better at knowing what "better" means. See
[`darwin/observability/CONTRACT.md`](darwin/observability/CONTRACT.md).
# bron
