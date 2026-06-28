// Agent-run protocol + reducer + drivers.
//
// The agents page renders entirely from `RunState`, and the only way to mutate
// `RunState` is by feeding `RunEvent`s through `applyRunEvent`. Two drivers
// produce those events:
//
//   - `createSimulatedRun()` — a scripted timeline (no backend needed). Used
//     for the demo so the whole evolution story plays automatically.
//   - `createSocketRun(url)` — connects to the Python brain's WebSocket and
//     forwards each JSON message as a `RunEvent`.
//
// Because both drivers emit the *same* `RunEvent` shape, swapping the
// simulation for the real backend is a one-line change in the page (pick the
// driver based on `NEXT_PUBLIC_AGENT_RUN_WS`). Keep the backend's WebSocket
// messages matching the `RunEvent` union below and nothing else has to change.

export type GenomeNodeKind = "agent" | "optional" | "output" | "input" | "userData";

export type GenomeNode = {
  id: string;
  model: string;
  description: string;
  kind?: GenomeNodeKind;
  terminal?: boolean;
};

export type GenomeEdge = {
  from: string;
  to: string;
  primary?: boolean;
};

export type RunAgent = {
  id: string;
  role: string;
  model: string;
  origin: "seed" | "grown";
};

export type Iteration = {
  id: number;
  name: string;
  description: string;
  fitness: number;
  nodes: GenomeNode[];
  edges: GenomeEdge[];
};

export type LogKind = "info" | "score" | "corpus" | "success";

export type LogEntry = {
  id: number;
  text: string;
  kind: LogKind;
};

export type Candidate = {
  label: string;
  nodes: GenomeNode[];
  edges: GenomeEdge[];
  fitness: number | null;
};

export type RunStatus = "idle" | "running" | "complete";

export type RunState = {
  status: RunStatus;
  phase: string;
  problemLabel: string;
  threshold: number;
  bestFitness: number;
  generations: Iteration[];
  activeGenerationId: number | null;
  candidate: Candidate | null;
  corpus: RunAgent[];
  log: LogEntry[];
  nextLogId: number;
};

export type RunEvent =
  | { type: "run_started"; problemLabel: string; threshold: number; corpus: RunAgent[] }
  | { type: "phase"; text: string }
  | { type: "log"; text: string; kind?: LogKind }
  | { type: "candidate"; label: string; nodes: GenomeNode[]; edges: GenomeEdge[] }
  | { type: "candidate_scored"; fitness: number }
  | { type: "generation_committed"; iteration: Iteration }
  | { type: "agent_created"; agent: RunAgent }
  | { type: "run_complete"; bestFitness: number };

const maxLogEntries = 9;

export const initialRunState: RunState = {
  status: "idle",
  phase: "Waiting for problem",
  problemLabel: "",
  threshold: 0.75,
  bestFitness: 0,
  generations: [],
  activeGenerationId: null,
  candidate: null,
  corpus: [],
  log: [],
  nextLogId: 1,
};

function pushLog(state: RunState, text: string, kind: LogKind): RunState {
  const entry: LogEntry = { id: state.nextLogId, text, kind };
  const log = [...state.log, entry].slice(-maxLogEntries);
  return { ...state, log, nextLogId: state.nextLogId + 1 };
}

export function applyRunEvent(state: RunState, event: RunEvent): RunState {
  switch (event.type) {
    case "run_started":
      return {
        ...initialRunState,
        status: "running",
        phase: "Decomposing problem",
        problemLabel: event.problemLabel,
        threshold: event.threshold,
        corpus: event.corpus,
        nextLogId: 1,
      };
    case "phase":
      return { ...state, phase: event.text };
    case "log":
      return pushLog(state, event.text, event.kind ?? "info");
    case "candidate":
      return {
        ...state,
        candidate: {
          label: event.label,
          nodes: event.nodes,
          edges: event.edges,
          fitness: null,
        },
      };
    case "candidate_scored":
      return state.candidate
        ? { ...state, candidate: { ...state.candidate, fitness: event.fitness } }
        : state;
    case "generation_committed":
      return {
        ...state,
        generations: [...state.generations, event.iteration],
        activeGenerationId: event.iteration.id,
        candidate: null,
        bestFitness: Math.max(state.bestFitness, event.iteration.fitness),
      };
    case "agent_created":
      return { ...state, corpus: [...state.corpus, event.agent] };
    case "run_complete":
      return {
        ...state,
        status: "complete",
        bestFitness: Math.max(state.bestFitness, event.bestFitness),
        candidate: null,
      };
    default:
      return state;
  }
}

// Convenience selector: what the graph should actually draw right now.
export function displayedGenome(state: RunState): {
  nodes: GenomeNode[];
  edges: GenomeEdge[];
} | null {
  if (state.candidate) {
    return { nodes: state.candidate.nodes, edges: state.candidate.edges };
  }
  const active = state.generations.find((gen) => gen.id === state.activeGenerationId);
  return active ? { nodes: active.nodes, edges: active.edges } : null;
}

export type RunDriver = {
  start: (emit: (event: RunEvent) => void) => void;
  stop: () => void;
};

// ---------------------------------------------------------------------------
// Team definitions used by the simulated timeline.
// ---------------------------------------------------------------------------

const userDataNode: GenomeNode = {
  id: "user-data",
  model: "User Data",
  description: "uploaded dataset",
  kind: "optional",
};

const webSearchNode: GenomeNode = {
  id: "web-search",
  model: "Web Search",
  description: "live source lookup",
  kind: "optional",
};

const seedNodes: GenomeNode[] = [
  { id: "input", model: "Input", description: "user request", kind: "optional" },
  userDataNode,
  webSearchNode,
  { id: "ingest", model: "Ingest Agent", description: "captures events" },
  { id: "router", model: "Router Agent", description: "assigns tasks" },
  { id: "planner", model: "Planner Model", description: "builds sequence" },
  { id: "reasoner", model: "Reasoner Model", description: "solves state" },
  { id: "tools", model: "Tool Agent", description: "executes calls" },
  { id: "review", model: "Review Agent", description: "validates output" },
  { id: "output", model: "Output", description: "answer + actions", kind: "output", terminal: true },
];

const linearEdges: GenomeEdge[] = [
  { from: "input", to: "ingest" },
  { from: "user-data", to: "ingest", primary: false },
  { from: "ingest", to: "router" },
  { from: "router", to: "planner" },
  { from: "planner", to: "reasoner" },
  { from: "reasoner", to: "tools" },
  { from: "tools", to: "review" },
  { from: "review", to: "output" },
];

const parallelEdges: GenomeEdge[] = [
  { from: "input", to: "ingest" },
  { from: "user-data", to: "ingest", primary: false },
  { from: "ingest", to: "router" },
  { from: "router", to: "planner" },
  { from: "router", to: "reasoner" },
  { from: "router", to: "tools" },
  { from: "planner", to: "review" },
  { from: "reasoner", to: "review" },
  { from: "tools", to: "review" },
  { from: "review", to: "output" },
];

const rearrangedNodes: GenomeNode[] = [
  { id: "input", model: "Input", description: "user request", kind: "optional" },
  userDataNode,
  webSearchNode,
  { id: "router", model: "Router Agent", description: "assigns tasks" },
  { id: "planner", model: "Planner Model", description: "builds sequence" },
  { id: "reasoner", model: "Reasoner Model", description: "solves state" },
  { id: "tools", model: "Tool Agent", description: "executes calls" },
  { id: "review", model: "Review Agent", description: "validates output" },
  { id: "output", model: "Output", description: "answer + actions", kind: "output", terminal: true },
];

const rearrangedEdges: GenomeEdge[] = [
  { from: "input", to: "router" },
  { from: "user-data", to: "router", primary: false },
  { from: "router", to: "planner" },
  { from: "router", to: "reasoner" },
  { from: "planner", to: "tools" },
  { from: "reasoner", to: "tools" },
  { from: "tools", to: "review" },
  { from: "review", to: "output" },
];

const grownNodes: GenomeNode[] = [
  { id: "input", model: "Input", description: "user request", kind: "optional" },
  userDataNode,
  webSearchNode,
  { id: "router", model: "Router Agent", description: "assigns tasks" },
  { id: "planner", model: "Planner Model", description: "builds sequence" },
  { id: "reasoner", model: "Reasoner Model", description: "solves state" },
  { id: "tools", model: "Tool Agent", description: "executes calls" },
  { id: "risk", model: "Risk Balancer", description: "stress-tests plan" },
  { id: "review", model: "Review Agent", description: "validates output" },
  { id: "output", model: "Output", description: "answer + actions", kind: "output", terminal: true },
];

const grownEdges: GenomeEdge[] = [
  { from: "input", to: "router" },
  { from: "user-data", to: "router", primary: false },
  { from: "router", to: "planner" },
  { from: "router", to: "reasoner" },
  { from: "router", to: "tools" },
  { from: "planner", to: "risk" },
  { from: "reasoner", to: "risk" },
  { from: "tools", to: "risk" },
  { from: "risk", to: "review" },
  { from: "review", to: "output" },
];

const seedCorpus: RunAgent[] = [
  { id: "ingest", role: "captures events", model: "max-small", origin: "seed" },
  { id: "router", role: "assigns tasks", model: "minimax", origin: "seed" },
  { id: "planner", role: "builds sequence", model: "gemini-pro", origin: "seed" },
  { id: "reasoner", role: "solves state", model: "gemini-pro", origin: "seed" },
  { id: "tools", role: "executes calls", model: "max-small", origin: "seed" },
  { id: "review", role: "validates output", model: "minimax", origin: "seed" },
];

type TimelineStep = { delay: number; event: RunEvent };

function buildTimeline(problemLabel: string): TimelineStep[] {
  return [
    { delay: 200, event: { type: "run_started", problemLabel, threshold: 0.75, corpus: seedCorpus } },
    { delay: 500, event: { type: "log", text: `Problem received — ${problemLabel}`, kind: "info" } },
    { delay: 900, event: { type: "phase", text: "Decomposing problem into 6 subproblems" } },
    { delay: 1100, event: { type: "log", text: "Decomposed into 6 subproblems", kind: "info" } },

    // Generation 1 — spawn + score two candidates, pick the best.
    { delay: 1000, event: { type: "phase", text: "Generation 1 — spawning agent swarm" } },
    { delay: 700, event: { type: "candidate", label: "Gen 1 · candidate A", nodes: seedNodes, edges: linearEdges } },
    { delay: 1500, event: { type: "phase", text: "Scoring candidate A against constraints" } },
    { delay: 900, event: { type: "candidate_scored", fitness: 0.42 } },
    { delay: 700, event: { type: "log", text: "Candidate A (sequential) scored 0.42", kind: "score" } },
    { delay: 900, event: { type: "candidate", label: "Gen 1 · candidate B", nodes: seedNodes, edges: parallelEdges } },
    { delay: 1500, event: { type: "phase", text: "Scoring candidate B against constraints" } },
    { delay: 900, event: { type: "candidate_scored", fitness: 0.5 } },
    { delay: 700, event: { type: "log", text: "Candidate B (parallel) scored 0.50 — best", kind: "score" } },
    {
      delay: 900,
      event: {
        type: "generation_committed",
        iteration: {
          id: 1,
          name: "Generation 1",
          description: "Parallel procedural pipeline",
          fitness: 0.5,
          nodes: seedNodes,
          edges: parallelEdges,
        },
      },
    },
    { delay: 600, event: { type: "log", text: "Generation 1 best fitness 0.50", kind: "info" } },

    // Generation 2 — rearrange the team it already has.
    { delay: 1200, event: { type: "phase", text: "Generation 2 — rearranging topology" } },
    { delay: 700, event: { type: "candidate", label: "Gen 2 · rearranged", nodes: rearrangedNodes, edges: rearrangedEdges } },
    { delay: 1600, event: { type: "candidate_scored", fitness: 0.62 } },
    {
      delay: 800,
      event: {
        type: "generation_committed",
        iteration: {
          id: 2,
          name: "Generation 2",
          description: "Rearranged topology",
          fitness: 0.62,
          nodes: rearrangedNodes,
          edges: rearrangedEdges,
        },
      },
    },
    { delay: 600, event: { type: "log", text: "Generation 2 best fitness 0.62 (rearranged)", kind: "info" } },

    // Stuck-detection → tier-two capability growth.
    { delay: 1200, event: { type: "phase", text: "0.62 below threshold 0.75 — checking agent corpus" } },
    { delay: 1000, event: { type: "log", text: "Rearranging insufficient — checking corpus", kind: "corpus" } },
    { delay: 1200, event: { type: "log", text: "Missing capability: multi-supplier risk balancing", kind: "corpus" } },
    {
      delay: 1200,
      event: {
        type: "agent_created",
        agent: { id: "risk", role: "stress-tests plan", model: "gemini-pro", origin: "grown" },
      },
    },
    { delay: 200, event: { type: "log", text: "Created agent: Risk Balancer → gemini-pro", kind: "corpus" } },

    // Generation 3 — re-evolve with the enriched corpus, break the ceiling.
    { delay: 1300, event: { type: "phase", text: "Generation 3 — re-evolving with new agent" } },
    { delay: 700, event: { type: "candidate", label: "Gen 3 · grown", nodes: grownNodes, edges: grownEdges } },
    { delay: 1700, event: { type: "candidate_scored", fitness: 0.78 } },
    {
      delay: 800,
      event: {
        type: "generation_committed",
        iteration: {
          id: 3,
          name: "Generation 3",
          description: "Grew Risk Balancer agent",
          fitness: 0.78,
          nodes: grownNodes,
          edges: grownEdges,
        },
      },
    },
    { delay: 600, event: { type: "log", text: "Generation 3 best fitness 0.78 ✓ above threshold", kind: "success" } },
    { delay: 1100, event: { type: "phase", text: "Converged — selected best team" } },
    { delay: 600, event: { type: "run_complete", bestFitness: 0.78 } },
  ];
}

export function createSimulatedRun(problemLabel = "Type 2 supply-chain instance (6 parts)"): RunDriver {
  const timers: ReturnType<typeof setTimeout>[] = [];

  return {
    start(emit) {
      const steps = buildTimeline(problemLabel);
      let elapsed = 0;
      for (const step of steps) {
        elapsed += step.delay;
        timers.push(setTimeout(() => emit(step.event), elapsed));
      }
    },
    stop() {
      while (timers.length > 0) {
        const timer = timers.pop();
        if (timer) {
          clearTimeout(timer);
        }
      }
    },
  };
}

// Backend driver: forwards the Python brain's WebSocket messages as RunEvents.
// The brain just needs to send JSON matching the RunEvent union above.
export function createSocketRun(url: string): RunDriver {
  let socket: WebSocket | null = null;

  return {
    start(emit) {
      socket = new WebSocket(url);
      socket.addEventListener("message", (message) => {
        try {
          const event = JSON.parse(message.data as string) as RunEvent;
          emit(event);
        } catch {
          // Ignore malformed frames rather than breaking the run.
        }
      });
    },
    stop() {
      socket?.close();
      socket = null;
    },
  };
}

// ---------------------------------------------------------------------------
// JSON driver: replays a recorded F1 run from /public/data (no backend needed).
// Falls back to the simulated run if the file can't be loaded.
// ---------------------------------------------------------------------------
export function createJsonRun(
  url = "/data/f1_run.json",
  fallback: () => RunDriver = () => createSimulatedRun(),
): RunDriver {
  const timers: ReturnType<typeof setTimeout>[] = [];
  let stopped = false;

  function delayFor(event: RunEvent): number {
    switch (event.type) {
      case "generation_committed":
        return 1400;
      case "agent_created":
        return 1100;
      case "candidate_scored":
        return 900;
      case "candidate":
        return 800;
      case "phase":
        return 700;
      case "run_complete":
        return 600;
      default:
        return 450;
    }
  }

  return {
    start(emit) {
      void (async () => {
        let events: RunEvent[];
        try {
          const res = await fetch(url, { cache: "no-store" });
          if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
          events = (await res.json()) as RunEvent[];
          if (!Array.isArray(events) || events.length === 0) {
            throw new Error("empty or malformed run file");
          }
        } catch (error) {
          console.warn("createJsonRun: falling back to simulated run:", error);
          const driver = fallback();
          driver.start(emit);
          this.stop = driver.stop;
          return;
        }

        if (stopped) return;
        let elapsed = 0;
        for (const event of events) {
          elapsed += delayFor(event);
          timers.push(setTimeout(() => emit(event), elapsed));
        }
      })();
    },
    stop() {
      stopped = true;
      while (timers.length > 0) {
        const timer = timers.pop();
        if (timer) clearTimeout(timer);
      }
    },
  };
}
