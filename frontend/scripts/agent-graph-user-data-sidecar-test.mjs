import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { chromium } from "playwright";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const defaultUrl = "http://localhost:3000";
const alternateUrl = "http://localhost:3001";
const testPort = 3127;
const attachedDatasetStorageKey = "f1-agent-attached-dataset";
const conversationSummaryStorageKey = "f1-agent-conversation-summary";

function agentsUrl(baseUrl) {
  return new URL("/agents", baseUrl).toString();
}

async function canUseServer(baseUrl) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 800);

  try {
    const response = await fetch(agentsUrl(baseUrl), { signal: controller.signal });
    return response.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}

async function waitForServer(baseUrl, serverProcess) {
  const startedAt = Date.now();
  const logs = [];

  serverProcess.stdout.on("data", (chunk) => logs.push(chunk.toString()));
  serverProcess.stderr.on("data", (chunk) => logs.push(chunk.toString()));

  while (Date.now() - startedAt < 25_000) {
    if (await canUseServer(baseUrl)) {
      return;
    }

    if (serverProcess.exitCode !== null) {
      throw new Error(`Next dev exited early.\n${logs.join("")}`);
    }

    await new Promise((resolve) => setTimeout(resolve, 250));
  }

  throw new Error(`Timed out waiting for ${baseUrl}.\n${logs.join("")}`);
}

async function resolveBaseUrl() {
  if (process.env.AGENT_GRAPH_TEST_URL) {
    return { baseUrl: process.env.AGENT_GRAPH_TEST_URL, serverProcess: null };
  }

  for (const baseUrl of [defaultUrl, alternateUrl]) {
    if (await canUseServer(baseUrl)) {
      return { baseUrl, serverProcess: null };
    }
  }

  const baseUrl = `http://127.0.0.1:${testPort}`;
  const serverProcess = spawn(
    process.execPath,
    ["node_modules/next/dist/bin/next", "dev", "--hostname", "127.0.0.1", "--port", String(testPort)],
    {
      cwd: projectRoot,
      env: { ...process.env, NEXT_TELEMETRY_DISABLED: "1" },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  await waitForServer(baseUrl, serverProcess);
  return { baseUrl, serverProcess };
}

const { baseUrl, serverProcess } = await resolveBaseUrl();
const browser = await chromium.launch({ headless: true });

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  await page.addInitScript(
    ({ datasetKey, summaryKey }) => {
      window.sessionStorage.setItem(
        datasetKey,
        JSON.stringify({
          lastModified: 1782622800000,
          name: "race-strategy-input.csv",
          size: 24576,
          type: "text/csv",
        }),
      );
      window.sessionStorage.setItem(
        summaryKey,
        "Optimize a race strategy with uploaded stint and tire degradation data.",
      );
    },
    {
      datasetKey: attachedDatasetStorageKey,
      summaryKey: conversationSummaryStorageKey,
    },
  );
  await page.goto(agentsUrl(baseUrl), { waitUntil: "networkidle" });
  await page.waitForFunction(() => Boolean(window.f1AgentRun));

  // Let the built-in simulation's initial run_started event fire before
  // injecting the focused test graph.
  await page.waitForTimeout(300);

  await page.evaluate(() => {
    window.f1AgentRun.emit({
      type: "run_started",
      problemLabel: "unlinked user data test",
      threshold: 0.75,
      corpus: [],
    });
    window.f1AgentRun.emit({
      type: "candidate",
      label: "unlinked user data",
      nodes: [
        { id: "input", model: "Input", description: "user request", kind: "optional" },
        {
          id: "web-search",
          model: "Web Search",
          description: "live source lookup",
          kind: "optional",
        },
        { id: "planner", model: "Planner Model", description: "builds sequence" },
        {
          id: "output",
          model: "Output",
          description: "answer + actions",
          kind: "output",
          terminal: true,
        },
      ],
      edges: [
        { from: "input", to: "planner" },
        { from: "planner", to: "output" },
      ],
    });
  });

  await page.waitForSelector('[data-node-frame="user-data"]');
  await page.waitForSelector('[data-node-frame="web-search"]');
  await page.locator('button[aria-label="Open top controls"]').hover();
  await page.waitForFunction(() => {
    const contextButton = document.querySelector('button[aria-label="Open context"]');
    return (
      contextButton instanceof HTMLButtonElement &&
      contextButton.tabIndex === 0 &&
      getComputedStyle(contextButton).pointerEvents !== "none"
    );
  });
  await page.locator('button[aria-label="Open context"]').click();
  await page.waitForSelector('aside[aria-label="Context"]');

  const result = await page.evaluate(() => {
    function readFrame(frame) {
      return {
        x: Number(frame.getAttribute("x")),
        y: Number(frame.getAttribute("y")),
        width: Number(frame.getAttribute("width")),
        height: Number(frame.getAttribute("height")),
      };
    }

    const userDataFrame = document.querySelector('[data-node-frame="user-data"]');
    const webSearchFrame = document.querySelector('[data-node-frame="web-search"]');
    const inputNode = document.querySelector('[data-node-id="input"]');
    const mainFrames = Array.from(document.querySelectorAll("[data-node-frame]")).filter(
      (frame) => frame.closest("[data-node-kind]")?.getAttribute("data-node-kind") !== "optional",
    );
    const userDataEdges = document.querySelectorAll(
      'path[data-edge-from="user-data"], path[data-edge-to="user-data"]',
    );
    const webSearchEdges = document.querySelectorAll(
      'path[data-edge-from="web-search"], path[data-edge-to="web-search"]',
    );
    const contextSidebar = document.querySelector('aside[aria-label="Context"]');
    const bottomSummary = Array.from(document.querySelectorAll("main > section")).find((section) =>
      section.textContent?.includes("Conversation summary"),
    );

    if (!userDataFrame || !webSearchFrame || mainFrames.length === 0 || !contextSidebar) {
      return null;
    }

    const userData = readFrame(userDataFrame);
    const webSearch = readFrame(webSearchFrame);
    const mainRightEdge = Math.max(...mainFrames.map((frame) => {
      const box = readFrame(frame);
      return box.x + box.width;
    }));

    return {
      userData,
      webSearch,
      mainRightEdge,
      userDataEdgeCount: userDataEdges.length,
      webSearchEdgeCount: webSearchEdges.length,
      inputKind: inputNode?.getAttribute("data-node-kind"),
      isRightSidecar: userData.x > mainRightEdge && webSearch.x > mainRightEdge,
      contextHasSummary: Boolean(
        contextSidebar.textContent?.includes("Optimize a race strategy"),
      ),
      contextHasFile: Boolean(
        contextSidebar.textContent?.includes("race-strategy-input.csv"),
      ),
      bottomSummaryExists: Boolean(bottomSummary),
    };
  });

  if (!result) {
    throw new Error("Could not read graph node frames.");
  }
  if (!result.isRightSidecar) {
    throw new Error(
      `Expected unlinked User Data to render to the right. Got x=${result.userData.x}, mainRightEdge=${result.mainRightEdge}.`,
    );
  }
  if (result.userDataEdgeCount !== 0) {
    throw new Error(`Expected no User Data edges. Found ${result.userDataEdgeCount}.`);
  }
  if (result.webSearchEdgeCount !== 0) {
    throw new Error(`Expected no Web Search edges. Found ${result.webSearchEdgeCount}.`);
  }
  if (result.inputKind !== "optional") {
    throw new Error(`Expected Input to be rendered as optional. Got ${result.inputKind}.`);
  }
  if (!result.contextHasSummary || !result.contextHasFile) {
    throw new Error("Expected context drawer to contain the summary and uploaded file metadata.");
  }
  if (result.bottomSummaryExists) {
    throw new Error("Expected conversation summary to be removed from the bottom overlay.");
  }

  console.log(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
  if (serverProcess) {
    serverProcess.kill("SIGTERM");
  }
}
