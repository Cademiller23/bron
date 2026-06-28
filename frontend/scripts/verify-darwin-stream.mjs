// Verify the Darwin brain's live WebSocket stream against the frontend's OWN
// reducer (src/lib/agent-run.ts). Transpiles the reducer with the local
// TypeScript, connects to NEXT_PUBLIC_AGENT_RUN_WS (or ws://localhost:8765),
// feeds every received frame through applyRunEvent, and asserts the run reaches
// a valid completed RunState. No browser, no Next server needed.
//
//   node scripts/verify-darwin-stream.mjs

import { readFileSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import ts from "typescript";

const url = process.env.NEXT_PUBLIC_AGENT_RUN_WS ?? "ws://localhost:8765";

const src = readFileSync(new URL("../src/lib/agent-run.ts", import.meta.url), "utf8");
const js = ts.transpileModule(src, {
  compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
}).outputText;
const dir = mkdtempSync(join(tmpdir(), "darwin-verify-"));
const file = join(dir, "agent-run.mjs");
writeFileSync(file, js);
const { initialRunState, applyRunEvent, displayedGenome } = await import(pathToFileURL(file).href);

const received = [];
let state = initialRunState;

await new Promise((resolve, reject) => {
  const ws = new WebSocket(url);
  const timer = setTimeout(() => { ws.close(); reject(new Error("timed out waiting for run_complete")); }, 30000);
  ws.addEventListener("message", (m) => {
    const event = JSON.parse(m.data);
    received.push(event.type);
    state = applyRunEvent(state, event);
    if (event.type === "run_complete") { clearTimeout(timer); ws.close(); resolve(); }
  });
  ws.addEventListener("error", (e) => { clearTimeout(timer); reject(new Error("ws error: " + (e.message ?? e))); });
});

const fail = (msg) => { console.error("FAIL:", msg); process.exit(1); };

if (state.status !== "complete") fail(`status=${state.status}`);
if (!(state.bestFitness >= 0.9)) fail(`bestFitness=${state.bestFitness}`);
if (state.generations.length < 2) fail(`generations=${state.generations.length}`);
if (state.corpus.length < 1) fail(`corpus empty`);
if (state.threshold !== 0.9) fail(`threshold=${state.threshold}`);
const drawn = displayedGenome(state);
if (!drawn || drawn.nodes.length < 1) fail("no displayable genome");
if (!drawn.nodes.some((n) => n.terminal)) fail("no terminal (arbiter) node");

console.log("OK — frontend reducer accepted the live Darwin stream");
console.log("  events:", received.length, "unique:", [...new Set(received)].join(", "));
console.log("  problemLabel:", state.problemLabel);
console.log("  generations:", state.generations.map((g) => g.fitness).join(" -> "));
console.log("  bestFitness:", state.bestFitness, "corpus size:", state.corpus.length);
console.log("  final genome nodes:", drawn.nodes.length, "edges:", drawn.edges.length);
