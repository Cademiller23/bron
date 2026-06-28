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
// Orchestrator (Gemini 3.5 Flash) branches outward; subagents are Mutable Operators.
// ---------------------------------------------------------------------------

const ORCHESTRATOR_MODEL = "Gemini 3.5 Flash";

const orchestratorNode: GenomeNode = {
  id: "orchestrator",
  model: ORCHESTRATOR_MODEL,
  description: "Orchestrator Agent",
  kind: "output",
  terminal: true,
};

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

function mutableOperator(id: string, specialization: string): GenomeNode {
  return {
    id,
    model: "Mutable Operator",
    description: specialization,
    kind: "agent",
  };
}

// Generation 1 — two levels: orchestrator → three specialist operators.
const gen1Nodes: GenomeNode[] = [
  orchestratorNode,
  mutableOperator("routing-op", "routing constraints"),
  mutableOperator("scheduling-op", "week placement"),
  mutableOperator("clustering-op", "region clustering"),
  userDataNode,
];

const gen1FlatEdges: GenomeEdge[] = [
  { from: "orchestrator", to: "routing-op" },
  { from: "orchestrator", to: "scheduling-op" },
  { from: "orchestrator", to: "clustering-op" },
  { from: "user-data", to: "routing-op", primary: false },
];

// Generation 1 candidate B — three levels under the orchestrator.
const gen1TieredNodes: GenomeNode[] = [
  orchestratorNode,
  mutableOperator("routing-op", "routing constraints"),
  mutableOperator("scheduling-op", "week placement"),
  mutableOperator("leg-op", "leg distance cap"),
  mutableOperator("carbon-op", "carbon proxy"),
  mutableOperator("window-op", "weather windows"),
  userDataNode,
];

const gen1TieredEdges: GenomeEdge[] = [
  { from: "orchestrator", to: "routing-op" },
  { from: "orchestrator", to: "scheduling-op" },
  { from: "routing-op", to: "leg-op" },
  { from: "routing-op", to: "carbon-op" },
  { from: "scheduling-op", to: "window-op" },
  { from: "user-data", to: "scheduling-op", primary: false },
];

// Generation 2 — three levels with a wider middle tier.
const gen2Nodes: GenomeNode[] = [
  orchestratorNode,
  mutableOperator("routing-op", "routing constraints"),
  mutableOperator("scheduling-op", "week placement"),
  mutableOperator("clustering-op", "region clustering"),
  mutableOperator("leg-op", "leg distance cap"),
  mutableOperator("carbon-op", "carbon proxy"),
  mutableOperator("window-op", "weather windows"),
  mutableOperator("revenue-op", "revenue peaks"),
  userDataNode,
  webSearchNode,
];

const gen2Edges: GenomeEdge[] = [
  { from: "orchestrator", to: "routing-op" },
  { from: "orchestrator", to: "scheduling-op" },
  { from: "orchestrator", to: "clustering-op" },
  { from: "routing-op", to: "leg-op" },
  { from: "routing-op", to: "carbon-op" },
  { from: "scheduling-op", to: "window-op" },
  { from: "scheduling-op", to: "revenue-op" },
  { from: "user-data", to: "routing-op", primary: false },
  { from: "web-search", to: "clustering-op", primary: false },
];

// Generation 3 — four levels; grown operator deepens the clustering branch.
const gen3Nodes: GenomeNode[] = [
  orchestratorNode,
  mutableOperator("routing-op", "routing constraints"),
  mutableOperator("scheduling-op", "week placement"),
  mutableOperator("clustering-op", "region clustering"),
  mutableOperator("leg-op", "leg distance cap"),
  mutableOperator("carbon-op", "carbon proxy"),
  mutableOperator("window-op", "weather windows"),
  mutableOperator("revenue-op", "revenue peaks"),
  mutableOperator("region-op", "swing sequencing"),
  mutableOperator("swing-op", "intra-region order"),
  userDataNode,
  webSearchNode,
];

const gen3Edges: GenomeEdge[] = [
  { from: "orchestrator", to: "routing-op" },
  { from: "orchestrator", to: "scheduling-op" },
  { from: "orchestrator", to: "clustering-op" },
  { from: "routing-op", to: "leg-op" },
  { from: "routing-op", to: "carbon-op" },
  { from: "scheduling-op", to: "window-op" },
  { from: "scheduling-op", to: "revenue-op" },
  { from: "clustering-op", to: "region-op" },
  { from: "region-op", to: "swing-op" },
  { from: "user-data", to: "routing-op", primary: false },
  { from: "web-search", to: "clustering-op", primary: false },
];

const seedCorpus: RunAgent[] = [
  { id: "routing-op", role: "routing constraints", model: "max-small", origin: "seed" },
  { id: "scheduling-op", role: "week placement", model: "minimax", origin: "seed" },
  { id: "clustering-op", role: "region clustering", model: "gemini-pro", origin: "seed" },
  { id: "leg-op", role: "leg distance cap", model: "max-small", origin: "seed" },
  { id: "window-op", role: "weather windows", model: "minimax", origin: "seed" },
];

type TimelineStep = { delay: number; event: RunEvent };

function buildTimeline(problemLabel: string): TimelineStep[] {
  return [
    { delay: 200, event: { type: "run_started", problemLabel, threshold: 0.75, corpus: seedCorpus } },
    { delay: 500, event: { type: "log", text: `Problem received — ${problemLabel}`, kind: "info" } },
    { delay: 900, event: { type: "phase", text: "Decomposing problem into 6 subproblems" } },
    { delay: 1100, event: { type: "log", text: "Decomposed into 6 subproblems", kind: "info" } },

    // Generation 1 — spawn + score two candidates, pick the deeper topology.
    { delay: 1000, event: { type: "phase", text: "Generation 1 — spawning mutable operators" } },
    { delay: 700, event: { type: "candidate", label: "Gen 1 · candidate A", nodes: gen1Nodes, edges: gen1FlatEdges } },
    { delay: 1500, event: { type: "phase", text: "Scoring candidate A against constraints" } },
    { delay: 900, event: { type: "candidate_scored", fitness: 0.42 } },
    { delay: 700, event: { type: "log", text: "Candidate A (2-level star) scored 0.42", kind: "score" } },
    { delay: 900, event: { type: "candidate", label: "Gen 1 · candidate B", nodes: gen1TieredNodes, edges: gen1TieredEdges } },
    { delay: 1500, event: { type: "phase", text: "Scoring candidate B against constraints" } },
    { delay: 900, event: { type: "candidate_scored", fitness: 0.5 } },
    { delay: 700, event: { type: "log", text: "Candidate B (3-level tree) scored 0.50 — best", kind: "score" } },
    {
      delay: 900,
      event: {
        type: "generation_committed",
        iteration: {
          id: 1,
          name: "Generation 1",
          description: "3-level operator tree",
          fitness: 0.5,
          nodes: gen1TieredNodes,
          edges: gen1TieredEdges,
        },
      },
    },
    { delay: 600, event: { type: "log", text: "Generation 1 best fitness 0.50", kind: "info" } },

    // Generation 2 — rearrange into a wider 3-level topology.
    { delay: 1200, event: { type: "phase", text: "Generation 2 — rearranging operator topology" } },
    { delay: 700, event: { type: "candidate", label: "Gen 2 · rearranged", nodes: gen2Nodes, edges: gen2Edges } },
    { delay: 1600, event: { type: "candidate_scored", fitness: 0.62 } },
    {
      delay: 800,
      event: {
        type: "generation_committed",
        iteration: {
          id: 2,
          name: "Generation 2",
          description: "Wider 3-level operator mesh",
          fitness: 0.62,
          nodes: gen2Nodes,
          edges: gen2Edges,
        },
      },
    },
    { delay: 600, event: { type: "log", text: "Generation 2 best fitness 0.62 (rearranged)", kind: "info" } },

    // Stuck-detection → tier-two capability growth.
    { delay: 1200, event: { type: "phase", text: "0.62 below threshold 0.75 — checking operator corpus" } },
    { delay: 1000, event: { type: "log", text: "Rearranging insufficient — checking corpus", kind: "corpus" } },
    { delay: 1200, event: { type: "log", text: "Missing capability: intra-region swing sequencing", kind: "corpus" } },
    {
      delay: 1200,
      event: {
        type: "agent_created",
        agent: { id: "swing-op", role: "intra-region order", model: "gemini-pro", origin: "grown" },
      },
    },
    { delay: 200, event: { type: "log", text: "Created Mutable Operator: swing-op → gemini-pro", kind: "corpus" } },

    // Generation 3 — re-evolve with the enriched corpus, break the ceiling.
    { delay: 1300, event: { type: "phase", text: "Generation 3 — re-evolving with new operator" } },
    { delay: 700, event: { type: "candidate", label: "Gen 3 · grown", nodes: gen3Nodes, edges: gen3Edges } },
    { delay: 1700, event: { type: "candidate_scored", fitness: 0.78 } },
    {
      delay: 800,
      event: {
        type: "generation_committed",
        iteration: {
          id: 3,
          name: "Generation 3",
          description: "4-level tree with swing operator",
          fitness: 0.78,
          nodes: gen3Nodes,
          edges: gen3Edges,
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
