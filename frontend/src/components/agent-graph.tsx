"use client";

import Link from "next/link";
import {
  Activity,
  ArrowRight,
  Boxes,
  Database,
  FileText,
  GitBranch,
  PanelLeft,
  PanelRight,
  RotateCcw,
  X,
} from "lucide-react";
import {
  type FocusEvent,
  type PointerEvent,
  type WheelEvent,
  useCallback,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  useState,
} from "react";

import {
  type GenomeEdge,
  type GenomeNode,
  type GenomeNodeKind,
  type RunDriver,
  type RunEvent,
  applyRunEvent,
  createSimulatedRun,
  createSocketRun,
  displayedGenome,
  initialRunState,
} from "@/lib/agent-run";
import {
  formatFileSize,
  getAttachedDataset,
  type AttachedDatasetMetadata,
} from "@/lib/attached-dataset";
import {
  CONVERSATION_SUMMARY_UPDATED_EVENT,
  readConversationSummary,
} from "@/lib/conversation-summary";

type LaidNode = GenomeNode & {
  x: number;
  y: number;
  width: number;
  height: number;
  entry: boolean;
  visualKind: GenomeNodeKind;
};

type GraphViewBox = {
  height: number;
  width: number;
  x: number;
  y: number;
};

const centerX = 600;
const topY = 96;
const rowGap = 132;
const colGap = 300;
const terminalWidth = 280;
const optionalNodeWidth = 260;
const nodeWidth = 244;
const terminalHeight = 96;
const optionalNodeHeight = 92;
const nodeHeight = 78;
const sidecarGap = 52;

function isPrimaryEdge(edge: GenomeEdge) {
  return edge.primary !== false;
}

function normalizeNodeText(value: string) {
  return value.toLowerCase().replace(/[\s_-]+/g, "");
}

function findOrchestrator(nodes: GenomeNode[], edges: GenomeEdge[]) {
  const geminiOrchestrator = nodes.find((node) => {
    const text = normalizeNodeText(`${node.id} ${node.model} ${node.description}`);
    return text.includes("gemini35flash") || text.includes("orchestrator");
  });
  if (geminiOrchestrator) {
    return geminiOrchestrator;
  }

  const explicit = nodes.find((node) => {
    const text = normalizeNodeText(`${node.id} ${node.model} ${node.description}`);
    return text.includes("arbiter");
  });
  if (explicit) {
    return explicit;
  }

  const terminalOutput = nodes.find((node) => node.terminal && node.kind === "output");
  if (terminalOutput) {
    return terminalOutput;
  }

  const source = nodes.find((node) => !edges.some((edge) => edge.to === node.id));
  return source ?? null;
}

/** Flip inbound arbiter edges so the orchestrator always branches outward. */
function normalizeEdgesFromOrchestrator(edges: GenomeEdge[], orchestratorId?: string) {
  if (!orchestratorId) {
    return edges;
  }

  return edges.map((edge) => {
    if (edge.to === orchestratorId) {
      return { ...edge, from: orchestratorId, to: edge.from };
    }
    return edge;
  });
}

function isOrchestratorNode(node: GenomeNode, orchestratorId?: string) {
  if (orchestratorId && node.id === orchestratorId) {
    return true;
  }

  const text = normalizeNodeText(`${node.id} ${node.model} ${node.description}`);
  return (
    text.includes("gemini35flash") ||
    text.includes("orchestrator") ||
    text.includes("arbiter")
  );
}

/** Relabel backend genomes for display: one orchestrator, mutable operators below. */
function normalizeGenomeForDisplay(genome: { nodes: GenomeNode[]; edges: GenomeEdge[] } | null) {
  if (!genome) {
    return genome;
  }

  const orchestrator = findOrchestrator(genome.nodes, genome.edges);
  const orchestratorId = orchestrator?.id;

  return {
    nodes: genome.nodes.map((node) => {
      const kind = inferNodeKind(node, 0, orchestratorId);
      if (kind === "optional" || kind === "input" || kind === "userData") {
        return node;
      }

      if (isOrchestratorNode(node, orchestratorId) || (node.terminal && kind === "output")) {
        return {
          ...node,
          model: "Gemini 3.5 Flash",
          description: "Orchestrator Agent",
          kind: "output" as const,
          terminal: true,
        };
      }

      return {
        ...node,
        model: "Mutable Operator",
        description: node.description || node.model,
        kind: "agent" as const,
        terminal: false,
      };
    }),
    edges: genome.edges,
  };
}

function inferNodeKind(
  node: GenomeNode,
  rankValue: number,
  orchestratorId?: string,
): GenomeNodeKind {
  if (node.kind === "optional" || node.kind === "input" || node.kind === "userData") {
    return "optional";
  }
  if (node.kind === "output") {
    return "output";
  }
  if (node.kind === "agent") {
    return "agent";
  }

  const normalizedText = normalizeNodeText(`${node.id} ${node.model}`);
  if (
    normalizedText.includes("userdata") ||
    normalizedText.includes("dataset") ||
    normalizedText.includes("input") ||
    normalizedText.includes("websearch") ||
    normalizedText.includes("search") ||
    normalizedText.includes("template")
  ) {
    return "optional";
  }
  if (normalizedText.includes("output")) {
    return "output";
  }
  if (node.id === orchestratorId) {
    return "output";
  }
  if (node.terminal) {
    return rankValue === 0 ? "optional" : "output";
  }

  return "agent";
}

function nodeSizeFor(kind: GenomeNodeKind) {
  if (kind === "output") {
    return { width: terminalWidth, height: terminalHeight };
  }
  if (kind === "optional" || kind === "input" || kind === "userData") {
    return { width: optionalNodeWidth, height: optionalNodeHeight };
  }

  return { width: nodeWidth, height: nodeHeight };
}

function nodeRadiusFor(kind: GenomeNodeKind) {
  if (kind === "output") {
    return 28;
  }
  if (kind === "optional" || kind === "input" || kind === "userData") {
    return 14;
  }

  return 6;
}

function nodeFrameFor(kind: GenomeNodeKind) {
  switch (kind) {
    case "output":
      return {
        fill: "#171006",
        stroke: "#fbbf24",
        strokeOpacity: ".95",
        strokeWidth: "1.7",
        dividerOpacity: ".34",
        titleFill: "#fff7d6",
        titleClassName: "font-sans text-[20px] font-semibold",
        descriptionFill: "rgba(254,240,138,0.68)",
      };
    case "optional":
    case "input":
    case "userData":
      return {
        fill: "#061414",
        stroke: "#5eead4",
        strokeOpacity: ".88",
        strokeWidth: "1.35",
        dividerOpacity: ".28",
        titleFill: "#d7fff8",
        titleClassName: "font-sans text-[18px] font-semibold",
        descriptionFill: "rgba(153,246,228,0.64)",
      };
    default:
      return {
        fill: "#050505",
        stroke: "white",
        strokeOpacity: ".5",
        strokeWidth: "1",
        dividerOpacity: ".16",
        titleFill: "white",
        titleClassName: "font-sans text-[17px] font-medium",
        descriptionFill: "rgba(255,255,255,0.48)",
      };
  }
}

// Orchestrator (Gemini 3.5 Flash / arbiter) stays pinned at the top; mutable
// operators stack below by depth along outward-branching edges.
function layoutGenome(nodes: GenomeNode[], edges: GenomeEdge[]) {
  const orchestrator = findOrchestrator(nodes, edges);
  const orchestratorId = orchestrator?.id;
  const displayEdges = normalizeEdgesFromOrchestrator(edges, orchestratorId);

  const linkedNodeIds = new Set<string>();
  displayEdges.forEach((edge) => {
    linkedNodeIds.add(edge.from);
    linkedNodeIds.add(edge.to);
  });

  const sidecarNodes = nodes.filter((node) => {
    const visualKind = inferNodeKind(node, 0, orchestratorId);
    return visualKind === "optional" && !linkedNodeIds.has(node.id);
  });
  const rankedNodes = nodes.filter((node) => !sidecarNodes.includes(node));

  const depth = new Map<string, number>();
  if (orchestratorId) {
    depth.set(orchestratorId, 0);
    let changed = true;
    while (changed) {
      changed = false;
      for (const edge of displayEdges) {
        if (depth.has(edge.from) && !depth.has(edge.to)) {
          depth.set(edge.to, (depth.get(edge.from) ?? 0) + 1);
          changed = true;
        }
        if (depth.has(edge.to) && !depth.has(edge.from)) {
          depth.set(edge.from, (depth.get(edge.to) ?? 0) + 1);
          changed = true;
        }
      }
    }
  }

  rankedNodes.forEach((node) => {
    if (!depth.has(node.id)) {
      depth.set(node.id, 1);
    }
  });

  const rank = depth;
  const byDepth = new Map<number, GenomeNode[]>();
  rankedNodes.forEach((node) => {
    const value = depth.get(node.id) ?? 1;
    const group = byDepth.get(value);
    if (group) {
      group.push(node);
    } else {
      byDepth.set(value, [node]);
    }
  });

  const laidNodes: LaidNode[] = [];

  byDepth.forEach((group, value) => {
    group.forEach((node, index) => {
      const offset = index - (group.length - 1) / 2;
      const visualKind = inferNodeKind(node, value, orchestratorId);
      const { width, height } = nodeSizeFor(visualKind);
      laidNodes.push({
        ...node,
        x: centerX + offset * colGap,
        y: topY + value * rowGap,
        width,
        height,
        entry: visualKind === "optional" && normalizeNodeText(node.model).includes("input"),
        visualKind,
      });
    });
  });

  const rankedRightEdge =
    laidNodes.length > 0
      ? Math.max(...laidNodes.map((laidNode) => laidNode.x + laidNode.width / 2))
      : centerX + terminalWidth / 2;

  sidecarNodes.forEach((node, index) => {
    const visualKind = inferNodeKind(node, 0, orchestratorId);
    const { width, height } = nodeSizeFor(visualKind);
    laidNodes.push({
      ...node,
      x: rankedRightEdge + sidecarGap + width / 2,
      y: topY + index * (height + 22),
      width,
      height,
      entry: false,
      visualKind,
    });
  });

  const maxRank = Math.max(0, ...laidNodes.map((node) => rank.get(node.id) ?? 0));

  return { laidNodes, rank, maxRank, orchestratorId, displayEdges };
}

function edgePoint(from: LaidNode, to: LaidNode) {
  const halfWidth = from.width / 2;
  const halfHeight = from.height / 2;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const scale = Math.min(
    Math.abs(dx) > 0 ? halfWidth / Math.abs(dx) : Number.POSITIVE_INFINITY,
    Math.abs(dy) > 0 ? halfHeight / Math.abs(dy) : Number.POSITIVE_INFINITY,
  );

  return {
    x: from.x + dx * scale,
    y: from.y + dy * scale,
  };
}

function connectorPath(start: { x: number; y: number }, end: { x: number; y: number }) {
  const dx = Math.abs(end.x - start.x);
  const dy = Math.abs(end.y - start.y);

  if (dy > dx * 1.5) {
    return `M ${start.x} ${start.y} C ${start.x} ${start.y + dy * 0.42}, ${end.x} ${
      end.y - dy * 0.42
    }, ${end.x} ${end.y}`;
  }

  return `M ${start.x} ${start.y} L ${end.x} ${end.y}`;
}

const rankStepSeconds = 1.4;
const edgeTravelSeconds = 0.85;
const nodeProcessingSeconds = 0.55;
const defaultViewBoxHeight = 900;

const initialViewBox: GraphViewBox = {
  x: 160,
  y: -24,
  width: 880,
  height: defaultViewBoxHeight,
};

const minZoomWidth = 520;
const maxZoomWidth = 1400;

function formatFitness(fitness: number) {
  return `${Math.round(fitness * 100)}%`;
}

function isDefaultViewBox(viewBox: GraphViewBox) {
  return (
    Math.abs(viewBox.x - initialViewBox.x) < 0.5 &&
    Math.abs(viewBox.y - initialViewBox.y) < 0.5 &&
    Math.abs(viewBox.width - initialViewBox.width) < 0.5 &&
    Math.abs(viewBox.height - initialViewBox.height) < 0.5
  );
}

function readAttachedDatasetMetadata() {
  if (typeof window === "undefined") {
    return null;
  }

  return getAttachedDataset()?.metadata ?? null;
}

function hasUploadedUserDataNode(nodes: GenomeNode[]) {
  return nodes.some((node) => {
    const normalizedText = normalizeNodeText(`${node.id} ${node.model}`);
    return normalizedText.includes("userdata") || normalizedText.includes("uploadeddataset");
  });
}

function createUserDataNode(metadata: AttachedDatasetMetadata): GenomeNode {
  return {
    id: "user-data",
    model: "User Data",
    description: `${formatFileSize(metadata.size)} uploaded`,
    kind: "optional",
  };
}

function withUploadedUserData(
  genome: { nodes: GenomeNode[]; edges: GenomeEdge[] } | null,
  metadata: AttachedDatasetMetadata | null,
) {
  if (!metadata) {
    return genome;
  }

  const baseGenome = genome ?? { nodes: [], edges: [] };
  if (hasUploadedUserDataNode(baseGenome.nodes)) {
    return baseGenome;
  }

  return {
    nodes: [...baseGenome.nodes, createUserDataNode(metadata)],
    edges: baseGenome.edges,
  };
}

function createDriver(): RunDriver {
  const socketUrl = process.env.NEXT_PUBLIC_AGENT_RUN_WS;
  // Swap point: set NEXT_PUBLIC_AGENT_RUN_WS to the Python brain's WebSocket to
  // drive the page from the real backend instead of the scripted simulation.
  return socketUrl ? createSocketRun(socketUrl) : createSimulatedRun();
}

export function AgentGraph() {
  const [runState, dispatch] = useReducer(applyRunEvent, initialRunState);
  const emit = useCallback((event: RunEvent) => dispatch(event), []);
  const driverRef = useRef<RunDriver | null>(null);

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isContextSidebarOpen, setIsContextSidebarOpen] = useState(false);
  const [isTopMenuExpanded, setIsTopMenuExpanded] = useState(false);
  const [isRunProgressMounted, setIsRunProgressMounted] = useState(false);
  const [isRunProgressVisible, setIsRunProgressVisible] = useState(false);
  const [manualSelection, setManualSelection] = useState<number | null>(null);
  const [conversationSummary, setConversationSummary] = useState(
    readConversationSummary,
  );
  const [attachedDatasetMetadata, setAttachedDatasetMetadata] = useState(
    readAttachedDatasetMetadata,
  );
  const [viewBox, setViewBox] = useState<GraphViewBox>(initialViewBox);
  const [isPanning, setIsPanning] = useState(false);
  const lastPointerPosition = useRef<{ x: number; y: number } | null>(null);
  const viewBoxRef = useRef<GraphViewBox>(initialViewBox);
  const zoomAnimationFrame = useRef<number | null>(null);
  const runProgressAnimationFrame = useRef<number | null>(null);
  const runProgressCloseTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const thresholdFocusTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastThresholdFlashGenerationId = useRef<number | null>(null);

  // Start the run (simulation, or the real backend if the env URL is set), and
  // allow manual/backend injection through a window hook + custom event.
  useEffect(() => {
    const driver = createDriver();
    driverRef.current = driver;
    driver.start(emit);

    function handleInjectedEvent(event: Event) {
      const customEvent = event as CustomEvent<RunEvent>;
      if (customEvent.detail) {
        emit(customEvent.detail);
      }
    }

    const runWindow = window as Window & { f1AgentRun?: { emit: (event: RunEvent) => void } };
    runWindow.f1AgentRun = { emit };
    window.addEventListener("agent-run:event", handleInjectedEvent);

    return () => {
      driver.stop();
      delete runWindow.f1AgentRun;
      window.removeEventListener("agent-run:event", handleInjectedEvent);

      if (zoomAnimationFrame.current) {
        cancelAnimationFrame(zoomAnimationFrame.current);
      }
      if (runProgressAnimationFrame.current) {
        cancelAnimationFrame(runProgressAnimationFrame.current);
      }
      if (runProgressCloseTimeout.current) {
        clearTimeout(runProgressCloseTimeout.current);
      }
      if (thresholdFocusTimeout.current) {
        clearTimeout(thresholdFocusTimeout.current);
      }
    };
  }, [emit]);

  useEffect(() => {
    const metadataRefreshTimeout = setTimeout(() => {
      setAttachedDatasetMetadata(readAttachedDatasetMetadata());
    }, 0);

    function handleConversationSummaryUpdate(event: Event) {
      const summary = (event as CustomEvent<string>).detail;
      setConversationSummary(
        typeof summary === "string" ? summary : readConversationSummary(),
      );
    }

    window.addEventListener(
      CONVERSATION_SUMMARY_UPDATED_EVENT,
      handleConversationSummaryUpdate,
    );

    return () => {
      clearTimeout(metadataRefreshTimeout);
      window.removeEventListener(
        CONVERSATION_SUMMARY_UPDATED_EVENT,
        handleConversationSummaryUpdate,
      );
    };
  }, []);

  useEffect(() => {
    viewBoxRef.current = viewBox;
  }, [viewBox]);

  // The generation in view follows the run, unless the user manually picked one.
  const viewGenerationId = manualSelection ?? runState.activeGenerationId;
  const viewedGeneration =
    runState.generations.find((generation) => generation.id === viewGenerationId) ?? null;

  // What's actually on the graph right now: a live candidate if we're scoring
  // one, otherwise the generation the user is viewing.
  const { drawGenome, drawKey } = useMemo(() => {
    if (runState.candidate) {
      return {
        drawGenome: normalizeGenomeForDisplay(
          withUploadedUserData(
            { nodes: runState.candidate.nodes, edges: runState.candidate.edges },
            attachedDatasetMetadata,
          ),
        ),
        drawKey: `candidate-${runState.candidate.label}`,
      };
    }
    if (viewedGeneration) {
      return {
        drawGenome: normalizeGenomeForDisplay(
          withUploadedUserData(
            { nodes: viewedGeneration.nodes, edges: viewedGeneration.edges },
            attachedDatasetMetadata,
          ),
        ),
        drawKey: `gen-${viewedGeneration.id}`,
      };
    }
    const fallback = displayedGenome(runState);
    return {
      drawGenome: normalizeGenomeForDisplay(
        withUploadedUserData(fallback, attachedDatasetMetadata),
      ),
      drawKey: "empty",
    };
  }, [attachedDatasetMetadata, runState, viewedGeneration]);

  const { laidNodes, rank, maxRank, displayEdges } = useMemo(
    () => layoutGenome(drawGenome?.nodes ?? [], drawGenome?.edges ?? []),
    [drawGenome],
  );

  const nodeMap = useMemo(() => {
    const map = new Map<string, LaidNode>();
    laidNodes.forEach((node) => map.set(node.id, node));
    return map;
  }, [laidNodes]);

  const drawEdges = displayEdges;

  const flowCycleSeconds =
    maxRank * rankStepSeconds + edgeTravelSeconds + nodeProcessingSeconds + 1.2;

  const edgeBeginFor = (edge: GenomeEdge) => (rank.get(edge.from) ?? 0) * rankStepSeconds;

  const fitnessBarPercent = Math.round((runState.bestFitness || 0) * 100);
  const thresholdPercent = Math.round(runState.threshold * 100);

  function animateViewBox(targetViewBox: GraphViewBox) {
    if (zoomAnimationFrame.current) {
      cancelAnimationFrame(zoomAnimationFrame.current);
    }

    const startViewBox = viewBoxRef.current;
    const startedAt = performance.now();
    const durationMs = 140;

    function tick(now: number) {
      const progress = Math.min(1, (now - startedAt) / durationMs);
      const easedProgress = 1 - (1 - progress) ** 3;
      const nextViewBox = {
        x: startViewBox.x + (targetViewBox.x - startViewBox.x) * easedProgress,
        y: startViewBox.y + (targetViewBox.y - startViewBox.y) * easedProgress,
        width: startViewBox.width + (targetViewBox.width - startViewBox.width) * easedProgress,
        height:
          startViewBox.height + (targetViewBox.height - startViewBox.height) * easedProgress,
      };

      viewBoxRef.current = nextViewBox;
      setViewBox(nextViewBox);

      if (progress < 1) {
        zoomAnimationFrame.current = requestAnimationFrame(tick);
      }
    }

    zoomAnimationFrame.current = requestAnimationFrame(tick);
  }

  function handlePointerDown(event: PointerEvent<SVGSVGElement>) {
    if (event.button !== 0) {
      return;
    }

    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    if (zoomAnimationFrame.current) {
      cancelAnimationFrame(zoomAnimationFrame.current);
    }
    lastPointerPosition.current = { x: event.clientX, y: event.clientY };
    setIsPanning(true);
  }

  function handlePointerMove(event: PointerEvent<SVGSVGElement>) {
    if (!isPanning || !lastPointerPosition.current) {
      return;
    }

    event.preventDefault();
    const lastPosition = lastPointerPosition.current;
    const deltaX = event.clientX - lastPosition.x;
    const deltaY = event.clientY - lastPosition.y;
    const bounds = event.currentTarget.getBoundingClientRect();

    lastPointerPosition.current = { x: event.clientX, y: event.clientY };

    setViewBox((currentViewBox) => ({
      ...currentViewBox,
      x: currentViewBox.x - (deltaX * currentViewBox.width) / bounds.width,
      y: currentViewBox.y - (deltaY * currentViewBox.height) / bounds.height,
    }));
  }

  function handlePointerUp(event: PointerEvent<SVGSVGElement>) {
    event.currentTarget.releasePointerCapture(event.pointerId);
    lastPointerPosition.current = null;
    setIsPanning(false);
  }

  function handleWheel(event: WheelEvent<SVGSVGElement>) {
    event.preventDefault();

    const bounds = event.currentTarget.getBoundingClientRect();
    const pointerX = (event.clientX - bounds.left) / bounds.width;
    const pointerY = (event.clientY - bounds.top) / bounds.height;
    const zoomFactor = event.deltaY > 0 ? 1.12 : 0.88;

    const currentViewBox = viewBoxRef.current;
    const nextWidth = Math.min(maxZoomWidth, Math.max(minZoomWidth, currentViewBox.width * zoomFactor));
    const scale = nextWidth / currentViewBox.width;
    const nextHeight = currentViewBox.height * scale;
    const focalX = currentViewBox.x + pointerX * currentViewBox.width;
    const focalY = currentViewBox.y + pointerY * currentViewBox.height;

    animateViewBox({
      x: focalX - pointerX * nextWidth,
      y: focalY - pointerY * nextHeight,
      width: nextWidth,
      height: nextHeight,
    });
  }

  function resetView() {
    animateViewBox(initialViewBox);
  }

  function handleTopMenuBlur(event: FocusEvent<HTMLDivElement>) {
    const nextFocusedElement = event.relatedTarget;
    if (
      !(nextFocusedElement instanceof Node) ||
      !event.currentTarget.contains(nextFocusedElement)
    ) {
      setIsTopMenuExpanded(false);
    }
  }

  function openRunProgress() {
    setIsContextSidebarOpen(false);
    setIsTopMenuExpanded(false);

    if (runProgressCloseTimeout.current) {
      clearTimeout(runProgressCloseTimeout.current);
    }
    if (runProgressAnimationFrame.current) {
      cancelAnimationFrame(runProgressAnimationFrame.current);
    }

    setIsRunProgressMounted(true);
    runProgressAnimationFrame.current = requestAnimationFrame(() => {
      setIsRunProgressVisible(true);
    });
  }

  function closeRunProgress() {
    if (runProgressCloseTimeout.current) {
      clearTimeout(runProgressCloseTimeout.current);
    }
    if (runProgressAnimationFrame.current) {
      cancelAnimationFrame(runProgressAnimationFrame.current);
    }

    setIsRunProgressVisible(false);
    runProgressCloseTimeout.current = setTimeout(() => {
      setIsRunProgressMounted(false);
    }, 90);
  }

  const hasMovedView = !isDefaultViewBox(viewBox);
  const isRunProgressActive = isRunProgressVisible;
  const hasReachedThreshold = runState.bestFitness >= runState.threshold;
  const finalGeneration =
    runState.generations[runState.generations.length - 1] ?? null;
  const hasContextPanelContent = Boolean(conversationSummary || attachedDatasetMetadata);
  const isFinalGenerationDisplayed = Boolean(
    hasReachedThreshold &&
      !runState.candidate &&
      finalGeneration &&
      viewedGeneration?.id === finalGeneration.id,
  );

  useEffect(() => {
    if (!hasReachedThreshold) {
      lastThresholdFlashGenerationId.current = null;
      if (thresholdFocusTimeout.current) {
        clearTimeout(thresholdFocusTimeout.current);
        thresholdFocusTimeout.current = null;
      }
      return;
    }

    if (
      !finalGeneration ||
      runState.candidate ||
      lastThresholdFlashGenerationId.current === finalGeneration.id
    ) {
      return;
    }

    lastThresholdFlashGenerationId.current = finalGeneration.id;
    if (thresholdFocusTimeout.current) {
      clearTimeout(thresholdFocusTimeout.current);
    }

    thresholdFocusTimeout.current = setTimeout(() => {
      setManualSelection(finalGeneration.id);
      setIsSidebarOpen(true);
      thresholdFocusTimeout.current = null;
    }, 0);
  }, [finalGeneration, hasReachedThreshold, runState.candidate]);

  return (
    <main
      className={`relative min-h-screen overflow-hidden bg-black text-white ${
        isPanning ? "select-none" : ""
      }`}
    >
      <h1 className="sr-only">Interconnected agent node graph</h1>

      {/* Run HUD pinned to the bottom so the graph stays unobstructed. */}
      <section className="pointer-events-none absolute bottom-6 left-1/2 z-20 w-[min(560px,calc(100vw-7rem))] -translate-x-1/2 rounded-md border border-white/14 bg-black/82 px-4 py-3 shadow-[0_18px_50px_rgba(0,0,0,0.38)] backdrop-blur-xl">
        <div className="flex items-center justify-between gap-3">
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/38">
            {runState.status === "complete" ? "Converged" : "Evolving swarm"}
          </div>
          <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/38">
            best {formatFitness(runState.bestFitness)} · target {thresholdPercent}%
          </div>
        </div>
        <p className="mt-2 truncate text-sm leading-6 text-white/82">{runState.phase}</p>
        <div className="relative mt-2 h-1.5 w-full overflow-hidden rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-white/80 transition-[width] duration-500 ease-out"
            style={{ width: `${fitnessBarPercent}%` }}
          />
          <div
            className="absolute top-0 h-full w-px bg-white/60"
            style={{ left: `${thresholdPercent}%` }}
          />
        </div>
        {runState.candidate ? (
          <div className="mt-2 font-mono text-[10px] text-white/50">
            evaluating {runState.candidate.label}
            {runState.candidate.fitness !== null
              ? ` — scored ${formatFitness(runState.candidate.fitness)}`
              : " — scoring…"}
          </div>
        ) : null}
        {isFinalGenerationDisplayed && finalGeneration ? (
          <div
            aria-live="polite"
            className="mt-2 flex items-center justify-between gap-3 border-t border-white/10 pt-2"
          >
            <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-white/78">
              Final swarm displayed on graph
            </span>
            <span className="shrink-0 font-mono text-[10px] text-white/46">
              {finalGeneration.name} · {formatFitness(finalGeneration.fitness)}
            </span>
          </div>
        ) : null}
      </section>

      {hasReachedThreshold ? (
        <Link
          href="/flight"
          aria-label="Open result visualization"
          transitionTypes={["nav-forward"]}
          className="absolute bottom-5 right-5 z-30 inline-flex h-10 items-center gap-2 rounded-md border border-white/20 bg-white px-4 font-mono text-[11px] font-semibold uppercase tracking-[0.14em] text-black shadow-[0_18px_50px_rgba(0,0,0,0.42)] outline-none transition hover:bg-white/88 focus-visible:ring-2 focus-visible:ring-white/40"
        >
          Next
          <ArrowRight className="size-4" />
        </Link>
      ) : null}

      <button
        type="button"
        aria-label="Open iterations"
        onClick={() => setIsSidebarOpen(true)}
        className={`absolute left-5 top-5 z-30 inline-flex h-9 items-center gap-2 rounded-md border border-white/14 bg-black/80 px-3 font-mono text-[11px] uppercase tracking-[0.14em] text-white/72 outline-none transition hover:border-white/28 hover:text-white focus-visible:ring-2 focus-visible:ring-white/35 ${
          isSidebarOpen ? "pointer-events-none opacity-0" : "opacity-100"
        }`}
      >
        <PanelLeft className="size-4" />
        Iterations
      </button>

      <div
        onBlur={handleTopMenuBlur}
        onFocus={() => setIsTopMenuExpanded(true)}
        onMouseEnter={() => setIsTopMenuExpanded(true)}
        onMouseLeave={() => setIsTopMenuExpanded(false)}
        className={`absolute right-5 top-5 z-30 h-9 transition-[width,opacity] duration-200 ease-out ${
          isTopMenuExpanded ? "w-[min(32rem,calc(100vw-2.5rem))]" : "w-9"
        } ${
          isRunProgressActive ? "pointer-events-none opacity-0" : "opacity-100"
        }`}
      >
        <button
          type="button"
          aria-label="Open top controls"
          aria-expanded={isTopMenuExpanded}
          onClick={() => setIsTopMenuExpanded(true)}
          className={`absolute right-0 top-0 inline-flex size-9 items-center justify-center rounded-md border border-white/14 bg-black/80 text-white/72 outline-none transition hover:border-white/28 hover:text-white focus-visible:ring-2 focus-visible:ring-white/35 ${
            isTopMenuExpanded
              ? "pointer-events-none scale-95 opacity-0"
              : "scale-100 opacity-100"
          }`}
        >
          <Activity className="size-4" />
        </button>

        <div
          aria-label="Top controls"
          aria-hidden={!isTopMenuExpanded}
          className={`absolute right-0 top-0 flex items-center justify-end gap-2 transition-[transform,opacity] duration-200 ease-out ${
            isTopMenuExpanded
              ? "translate-x-0 scale-100 opacity-100"
              : "pointer-events-none translate-x-2 scale-[0.98] opacity-0"
          }`}
        >
          {hasContextPanelContent ? (
            <button
              type="button"
              aria-label="Open context"
              tabIndex={isTopMenuExpanded && !isContextSidebarOpen ? 0 : -1}
              onClick={() => {
                closeRunProgress();
                setIsContextSidebarOpen(true);
              }}
              className={`inline-flex h-9 items-center gap-2 rounded-md border border-white/14 bg-black/80 px-3 font-mono text-[11px] uppercase tracking-[0.14em] text-white/72 outline-none transition hover:border-white/28 hover:text-white focus-visible:ring-2 focus-visible:ring-white/35 ${
                isContextSidebarOpen ? "pointer-events-none opacity-0" : "opacity-100"
              }`}
            >
              <PanelRight className="size-4" />
              Context
            </button>
          ) : null}

          {hasMovedView ? (
            <button
              type="button"
              tabIndex={isTopMenuExpanded ? 0 : -1}
              onClick={resetView}
              className="inline-flex h-9 items-center gap-2 rounded-md border border-white/14 bg-black/80 px-3 font-mono text-[11px] uppercase tracking-[0.14em] text-white/72 outline-none transition hover:border-white/28 hover:text-white focus-visible:ring-2 focus-visible:ring-white/35"
            >
              <RotateCcw className="size-4" />
              Reset view
            </button>
          ) : null}

          <button
            type="button"
            aria-label="Open run progress"
            tabIndex={isTopMenuExpanded ? 0 : -1}
            onClick={openRunProgress}
            className="inline-flex h-9 items-center gap-2 rounded-md border border-white/14 bg-black/80 px-3 font-mono text-[11px] uppercase tracking-[0.14em] text-white/72 outline-none transition hover:border-white/28 hover:text-white focus-visible:ring-2 focus-visible:ring-white/35"
          >
            <Activity className="size-4" />
            Run Progress
          </button>
        </div>
      </div>

      <aside
        aria-label="Iterations"
        aria-hidden={!isSidebarOpen}
        className={`absolute inset-y-0 left-0 z-40 flex w-[300px] flex-col border-r border-white/12 bg-black/92 transform-gpu will-change-transform transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${
          isSidebarOpen
            ? "translate-x-0"
            : "pointer-events-none -translate-x-[calc(100%+1px)]"
        }`}
      >
          <div className="flex h-14 items-center justify-between border-b border-white/10 px-4">
            <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.16em] text-white/70">
              <GitBranch className="size-4" />
              Generations
            </div>
            <button
              type="button"
              onClick={() => setIsSidebarOpen(false)}
              aria-label="Close iterations"
              className="inline-flex size-8 items-center justify-center rounded-md text-white/60 outline-none transition hover:bg-white/8 hover:text-white focus-visible:ring-2 focus-visible:ring-white/35"
            >
              <X className="size-4" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-3">
            <div className="space-y-2">
              {runState.generations.length === 0 ? (
                <div className="rounded-md border border-dashed border-white/12 px-3 py-4 text-center font-mono text-[10px] uppercase tracking-[0.14em] text-white/30">
                  awaiting first generation
                </div>
              ) : null}
              {runState.generations.map((generation) => {
                const isViewedGeneration = generation.id === viewGenerationId;
                const isFinalGeneration =
                  hasReachedThreshold && generation.id === finalGeneration?.id;

                return (
                  <button
                    key={generation.id}
                    type="button"
                    onClick={() => setManualSelection(generation.id)}
                    className={`group relative w-full overflow-hidden rounded-md border px-3 py-3 text-left outline-none transition focus-visible:ring-2 focus-visible:ring-white/35 ${
                      isViewedGeneration
                        ? "border-white/44 bg-white/[0.075] text-white"
                        : "border-white/12 bg-black text-white/54 hover:border-white/28 hover:bg-white/[0.035] hover:text-white/80"
                    }`}
                  >
                    {isFinalGeneration ? (
                      <div className="result-plotted-flash pointer-events-none absolute left-4 right-4 top-3 z-10 rounded-sm border border-emerald-200/34 bg-emerald-950/90 px-3 py-2 text-center shadow-[0_14px_28px_rgba(0,0,0,0.34)] backdrop-blur-md">
                        <div className="font-mono text-[9px] uppercase tracking-[0.16em] text-emerald-100/86">
                          Result plotted
                        </div>
                        <div className="mt-0.5 truncate text-[11px] text-emerald-50/78">
                          Final swarm is on the graph
                        </div>
                      </div>
                    ) : null}
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="flex min-w-0 items-center gap-2">
                          <div className="truncate text-sm font-medium">{generation.name}</div>
                          {isFinalGeneration ? (
                            <span className="shrink-0 rounded-sm border border-emerald-300/24 bg-emerald-300/10 px-1.5 py-0.5 font-mono text-[8px] uppercase tracking-[0.12em] text-emerald-100/74">
                              final
                            </span>
                          ) : null}
                        </div>
                        <div className="mt-1 truncate text-xs text-white/38">
                          {generation.description}
                        </div>
                      </div>
                      <div className="shrink-0 rounded-sm border border-white/12 px-1.5 py-0.5 font-mono text-[10px] text-white/36">
                        {formatFitness(generation.fitness)}
                      </div>
                    </div>
                    <div className="mt-3 flex items-center gap-2">
                      <span
                        className={`h-px flex-1 ${
                          isViewedGeneration ? "bg-white/22" : "bg-white/10"
                        }`}
                      />
                      <span
                        className={`font-mono text-[9px] uppercase tracking-[0.14em] ${
                          isFinalGeneration && isViewedGeneration
                            ? "text-emerald-100/60"
                            : "text-white/30"
                        }`}
                      >
                        {isFinalGeneration && isViewedGeneration
                          ? "final swarm shown"
                          : isViewedGeneration
                            ? "active"
                            : "select"}
                      </span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex max-h-[50vh] shrink-0 flex-col border-t border-white/10 px-4 py-3">
            <div className="mb-2 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-white/50">
              <Boxes className="size-3.5" />
              Corpus of agents · {runState.corpus.length}
            </div>
            <div className="min-h-0 space-y-1 overflow-y-auto">
              {runState.corpus.map((agent) => (
                <div
                  key={agent.id}
                  className="flex items-center justify-between gap-2 rounded-sm border border-white/10 px-2 py-1.5"
                >
                  <div className="min-w-0">
                    <div className="truncate text-xs text-white/72">{agent.id}</div>
                    <div className="truncate font-mono text-[9px] text-white/34">{agent.model}</div>
                  </div>
                  <span
                    className={`shrink-0 rounded-sm px-1 py-0.5 font-mono text-[8px] uppercase tracking-[0.12em] ${
                      agent.origin === "grown"
                        ? "border border-white/40 text-white/80"
                        : "border border-white/12 text-white/36"
                    }`}
                  >
                    {agent.origin}
                  </span>
                </div>
              ))}
            </div>
          </div>
      </aside>

      {hasContextPanelContent ? (
        <aside
          aria-label="Context"
          aria-hidden={!isContextSidebarOpen}
          className={`absolute inset-y-0 right-0 z-40 flex w-[360px] max-w-[calc(100vw-1.5rem)] flex-col border-l border-white/12 bg-black/94 shadow-[-24px_0_70px_rgba(0,0,0,0.42)] transform-gpu will-change-transform transition-transform duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] ${
            isContextSidebarOpen
              ? "translate-x-0"
              : "pointer-events-none translate-x-[calc(100%+1px)]"
          }`}
        >
          <div className="flex h-14 items-center justify-between border-b border-white/10 px-4">
            <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.16em] text-white/70">
              <PanelRight className="size-4" />
              Context
            </div>
            <button
              type="button"
              onClick={() => setIsContextSidebarOpen(false)}
              aria-label="Close context"
              className="inline-flex size-8 items-center justify-center rounded-md text-white/60 outline-none transition hover:bg-white/8 hover:text-white focus-visible:ring-2 focus-visible:ring-white/35"
            >
              <X className="size-4" />
            </button>
          </div>

          <div className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
            {attachedDatasetMetadata ? (
              <section className="rounded-md border border-cyan-200/18 bg-cyan-300/[0.045] px-3 py-3">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-cyan-100/72">
                  <Database className="size-3.5" />
                  User data
                </div>
                <div className="mt-3 min-w-0">
                  <div className="truncate text-sm text-cyan-50/86">
                    {attachedDatasetMetadata.name}
                  </div>
                  <div className="mt-1 font-mono text-[10px] text-cyan-100/44">
                    {formatFileSize(attachedDatasetMetadata.size)} · {attachedDatasetMetadata.type}
                  </div>
                </div>
              </section>
            ) : null}

            {conversationSummary ? (
              <section className="rounded-md border border-white/12 bg-white/[0.035] px-3 py-3">
                <div className="flex items-center gap-2 font-mono text-[10px] uppercase tracking-[0.16em] text-white/46">
                  <FileText className="size-3.5" />
                  Conversation summary
                </div>
                <p className="mt-3 text-sm leading-6 text-white/68">{conversationSummary}</p>
              </section>
            ) : null}
          </div>
        </aside>
      ) : null}

      {isRunProgressMounted ? (
        <div
          aria-label="Run Progress"
          className={`absolute right-5 top-5 z-40 w-[340px] origin-top-right transform-gpu rounded-lg border border-white/14 bg-black/95 shadow-2xl shadow-black/40 will-change-transform transition-[transform,opacity] duration-100 ease-out ${
            isRunProgressVisible
              ? "translate-y-0 scale-100 opacity-100"
              : "-translate-y-1 scale-[0.99] opacity-0"
          }`}
        >
          <div className="flex h-12 items-center justify-between border-b border-white/10 px-4">
            <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-[0.16em] text-white/70">
              <Activity className="size-4" />
              Run Progress
            </div>
            <button
              type="button"
              onClick={closeRunProgress}
              aria-label="Close run progress"
              className="inline-flex size-8 items-center justify-center rounded-md text-white/60 outline-none transition hover:bg-white/8 hover:text-white focus-visible:ring-2 focus-visible:ring-white/35"
            >
              <X className="size-4" />
            </button>
          </div>

          <div className="border-b border-white/10 px-4 py-3">
            <div className="truncate text-sm text-white/72">{runState.phase}</div>
            <div className="mt-1 font-mono text-[11px] text-white/46">
              {runState.problemLabel || "—"}
            </div>
          </div>

          <div className="max-h-[280px] space-y-2.5 overflow-y-auto px-4 py-4">
            {runState.log.length === 0 ? (
              <div className="font-mono text-[11px] text-white/36">no activity yet</div>
            ) : null}
            {runState.log.map((entry) => (
              <div key={entry.id} className="flex items-start gap-3">
                <div
                  className={`mt-1.5 size-2 shrink-0 rounded-full ${
                    entry.kind === "success"
                      ? "bg-white"
                      : entry.kind === "corpus"
                        ? "bg-white/70"
                        : entry.kind === "score"
                          ? "bg-white/55"
                          : "bg-white/30"
                  }`}
                />
                <div className="text-sm leading-5 text-white/74">{entry.text}</div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <svg
        viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.width} ${viewBox.height}`}
        role="img"
        aria-label="Monochrome top-down procedural pipeline graph of interconnected agent nodes"
        className={`absolute inset-0 h-full w-full touch-none px-3 py-4 ${
          isPanning ? "cursor-grabbing" : "cursor-grab"
        }`}
        preserveAspectRatio="xMidYMid meet"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        onWheel={handleWheel}
      >
        <defs>
          <pattern id="major-grid" width="48" height="48" patternUnits="userSpaceOnUse">
            <path d="M 48 0 H 0 V 48" fill="none" stroke="white" strokeOpacity=".055" />
          </pattern>
          <pattern id="minor-grid" width="12" height="12" patternUnits="userSpaceOnUse">
            <path d="M 12 0 H 0 V 12" fill="none" stroke="white" strokeOpacity=".025" />
          </pattern>
          <radialGradient id="grid-mask" cx="50%" cy="50%" r="62%">
            <stop offset="0%" stopColor="white" stopOpacity="1" />
            <stop offset="72%" stopColor="white" stopOpacity=".8" />
            <stop offset="100%" stopColor="white" stopOpacity="0" />
          </radialGradient>
          <mask id="grid-fade">
            <rect
              x={viewBox.x - viewBox.width}
              y={viewBox.y - viewBox.height}
              width={viewBox.width * 3}
              height={viewBox.height * 3}
              fill="url(#grid-mask)"
            />
          </mask>
          <pattern
            id="terminal-texture"
            width="12"
            height="12"
            patternUnits="userSpaceOnUse"
          >
            <path d="M 12 0 L 0 12" stroke="#fbbf24" strokeOpacity=".12" strokeWidth=".8" />
            <circle cx="3" cy="3" r=".9" fill="#fef08a" fillOpacity=".2" />
          </pattern>
          <pattern
            id="optional-node-texture"
            width="16"
            height="16"
            patternUnits="userSpaceOnUse"
          >
            <circle cx="3" cy="3" r="1.2" fill="#5eead4" fillOpacity=".22" />
            <path d="M 0 16 L 16 0" stroke="#5eead4" strokeOpacity=".08" strokeWidth=".8" />
          </pattern>
          <marker
            id="arrow"
            markerHeight="9"
            markerWidth="9"
            orient="auto"
            refX="8"
            refY="4.5"
            viewBox="0 0 9 9"
          >
            <path d="M 0 0 L 9 4.5 L 0 9 z" fill="white" fillOpacity=".72" />
          </marker>
          <marker
            id="arrow-muted"
            markerHeight="8"
            markerWidth="8"
            orient="auto"
            refX="7"
            refY="4"
            viewBox="0 0 8 8"
          >
            <path d="M 0 0 L 8 4 L 0 8 z" fill="white" fillOpacity=".3" />
          </marker>
        </defs>
        <g mask="url(#grid-fade)">
          <rect
            x={viewBox.x - viewBox.width}
            y={viewBox.y - viewBox.height}
            width={viewBox.width * 3}
            height={viewBox.height * 3}
            fill="url(#major-grid)"
          />
          <rect
            x={viewBox.x - viewBox.width}
            y={viewBox.y - viewBox.height}
            width={viewBox.width * 3}
            height={viewBox.height * 3}
            fill="url(#minor-grid)"
            opacity=".6"
          />
        </g>
        <g>
          {drawEdges.map((edge) => {
            const from = nodeMap.get(edge.from);
            const to = nodeMap.get(edge.to);

            if (!from || !to) {
              return null;
            }

            const start = edgePoint(from, to);
            const end = edgePoint(to, from);
            const primary = isPrimaryEdge(edge);

            return (
              <path
                key={`${drawKey}-${edge.from}-${edge.to}`}
                data-edge-from={edge.from}
                data-edge-to={edge.to}
                d={connectorPath(start, end)}
                fill="none"
                stroke="white"
                strokeOpacity={primary ? ".48" : ".22"}
                strokeWidth={primary ? "1.35" : "1"}
                strokeLinecap="round"
                markerEnd={primary ? "url(#arrow)" : "url(#arrow-muted)"}
              />
            );
          })}
        </g>

        <g>
          {laidNodes.map((node, index) => {
            const frame = nodeFrameFor(node.visualKind);
            const radius = nodeRadiusFor(node.visualKind);
            const isOptional = node.visualKind === "optional";
            const isTerminal = node.visualKind === "output";

            return (
              <g
                key={`${drawKey}-${node.id}`}
                data-node-id={node.id}
                data-node-kind={node.visualKind}
                className="agent-node-enter"
                style={{ animationDelay: `${index * 70}ms` }}
              >
                {isTerminal ? (
                  <rect
                    x={node.x - node.width / 2 - 3}
                    y={node.y - node.height / 2 - 3}
                    width={node.width + 6}
                    height={node.height + 6}
                    rx={radius + 3}
                    fill="none"
                    stroke="#fbbf24"
                    strokeOpacity=".2"
                    strokeWidth="6"
                  />
                ) : null}
                <rect
                  data-node-frame={node.id}
                  x={node.x - node.width / 2}
                  y={node.y - node.height / 2}
                  width={node.width}
                  height={node.height}
                  rx={radius}
                  fill={frame.fill}
                  stroke={frame.stroke}
                  strokeOpacity={frame.strokeOpacity}
                  strokeWidth={frame.strokeWidth}
                />
                {isTerminal ? (
                  <>
                    <rect
                      x={node.x - node.width / 2 + 10}
                      y={node.y - node.height / 2 + 10}
                      width={node.width - 20}
                      height={node.height - 20}
                      rx="22"
                      fill="url(#terminal-texture)"
                    />
                    <rect
                      x={node.x - node.width / 2 + 14}
                      y={node.y - node.height / 2 + 14}
                      width={node.width - 28}
                      height={node.height - 28}
                      rx="19"
                      fill="none"
                      stroke="#fbbf24"
                      strokeOpacity=".28"
                      strokeWidth=".9"
                    />
                  </>
                ) : null}
                {isOptional ? (
                  <>
                    <rect
                      x={node.x - node.width / 2 + 9}
                      y={node.y - node.height / 2 + 9}
                      width={node.width - 18}
                      height={node.height - 18}
                      rx="10"
                      fill="url(#optional-node-texture)"
                    />
                    <path
                      d={`M ${node.x - node.width / 2 + 28} ${node.y - 22} H ${
                        node.x + node.width / 2 - 28
                      } M ${node.x - node.width / 2 + 28} ${node.y + 22} H ${
                        node.x + node.width / 2 - 28
                      }`}
                      fill="none"
                      stroke="#5eead4"
                      strokeOpacity=".18"
                      strokeWidth=".9"
                      strokeDasharray="4 5"
                    />
                  </>
                ) : null}
                <line
                  x1={node.x - node.width / 2 + 20}
                  y1={node.y + 5}
                  x2={node.x + node.width / 2 - 20}
                  y2={node.y + 5}
                  stroke={isTerminal ? "#fbbf24" : isOptional ? "#5eead4" : "white"}
                  strokeOpacity={frame.dividerOpacity}
                />
                <text
                  x={node.x}
                  y={node.y - 8}
                  textAnchor="middle"
                  fill={frame.titleFill}
                  className={frame.titleClassName}
                >
                  {node.model}
                </text>
                <text
                  x={node.x}
                  y={node.y + 26}
                  textAnchor="middle"
                  fill={frame.descriptionFill}
                  className="font-mono text-[10px]"
                >
                  {node.description}
                </text>
              </g>
            );
          })}
        </g>

        <g>
          {drawEdges
            .filter((edge) => isPrimaryEdge(edge))
            .map((edge) => {
              const node = nodeMap.get(edge.to);

              if (!node) {
                return null;
              }

              const beginTime = edgeBeginFor(edge);
              const holdStart = (beginTime + edgeTravelSeconds) / flowCycleSeconds;
              const holdVisible = (beginTime + edgeTravelSeconds + 0.12) / flowCycleSeconds;
              const holdEnd =
                (beginTime + edgeTravelSeconds + nodeProcessingSeconds) / flowCycleSeconds;

              return (
                <rect
                  key={`processing-${drawKey}-${edge.from}-${edge.to}`}
                  x={node.x - node.width / 2 - 4}
                  y={node.y - node.height / 2 - 4}
                  width={node.width + 8}
                  height={node.height + 8}
                  rx={nodeRadiusFor(node.visualKind) + 3}
                  fill="none"
                  stroke="white"
                  strokeOpacity="0"
                  strokeWidth="1"
                >
                  <animate
                    attributeName="stroke-opacity"
                    values="0;0;0.92;0.92;0"
                    keyTimes={`0;${holdStart};${holdVisible};${holdEnd};1`}
                    dur={`${flowCycleSeconds}s`}
                    begin="0s"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="stroke-width"
                    values="1;1;3;3;1"
                    keyTimes={`0;${holdStart};${holdVisible};${holdEnd};1`}
                    dur={`${flowCycleSeconds}s`}
                    begin="0s"
                    repeatCount="indefinite"
                  />
                </rect>
              );
            })}
        </g>

        <g>
          {drawEdges
            .filter((edge) => isPrimaryEdge(edge))
            .map((edge) => {
              const from = nodeMap.get(edge.from);
              const to = nodeMap.get(edge.to);

              if (!from || !to) {
                return null;
              }

              const start = edgePoint(from, to);
              const end = edgePoint(to, from);
              const beginTime = edgeBeginFor(edge);
              const moveStart = beginTime / flowCycleSeconds;
              const moveEnd = (beginTime + edgeTravelSeconds) / flowCycleSeconds;
              const holdEnd =
                (beginTime + edgeTravelSeconds + nodeProcessingSeconds) / flowCycleSeconds;

              return (
                <circle key={`flow-${drawKey}-${edge.from}-${edge.to}`} r="3.4" fill="white" opacity=".82">
                  <animateMotion
                    dur={`${flowCycleSeconds}s`}
                    begin="0s"
                    path={connectorPath(start, end)}
                    keyPoints={`0;0;1;1;1`}
                    keyTimes={`0;${moveStart};${moveEnd};${holdEnd};1`}
                    calcMode="linear"
                    repeatCount="indefinite"
                  />
                  <animate
                    attributeName="opacity"
                    values="0;0;.9;.9;0;0"
                    keyTimes={`0;${moveStart};${moveStart + 0.015};${holdEnd};${Math.min(
                      holdEnd + 0.04,
                      0.98,
                    )};1`}
                    dur={`${flowCycleSeconds}s`}
                    begin="0s"
                    repeatCount="indefinite"
                  />
                </circle>
              );
            })}
        </g>
      </svg>
    </main>
  );
}
