import { spawn } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const currentFile = fileURLToPath(import.meta.url);
const workspaceRoot = path.resolve(path.dirname(currentFile), "..");
const [firstArg, ...remainingArgs] = process.argv.slice(2);
const hasExplicitMode = firstArg === "dev" || firstArg === "start";
const mode = hasExplicitMode ? firstArg : "dev";
const nextArgs = hasExplicitMode ? remainingArgs : process.argv.slice(2);
const shutdownSignalExitCodes = new Map([
  ["SIGINT", 130],
  ["SIGTERM", 143],
]);

let isShuttingDown = false;
let exitCode = 0;

const childProcesses = [
  {
    name: "next",
    command: process.execPath,
    args: [path.join(workspaceRoot, "node_modules/next/dist/bin/next"), mode, ...nextArgs],
  },
  {
    name: "voice-agent",
    command: process.execPath,
    args: [path.join(workspaceRoot, "agents/voice-agent.mjs"), mode],
  },
].map(({ name, command, args }) => {
  const child = spawn(command, args, {
    cwd: workspaceRoot,
    env: process.env,
    stdio: "inherit",
  });

  child.once("exit", (code, signal) => {
    if (isShuttingDown) {
      return;
    }

    exitCode = code ?? (signal ? (shutdownSignalExitCodes.get(signal) ?? 1) : 1);
    console.error(`[dev] ${name} exited. Stopping the remaining process.`);
    shutdown(signal ?? "SIGTERM");
  });

  return child;
});

function shutdown(signal = "SIGTERM") {
  if (isShuttingDown) {
    return;
  }

  isShuttingDown = true;

  for (const child of childProcesses) {
    if (!child.killed && child.exitCode === null && child.signalCode === null) {
      child.kill(signal);
    }
  }
}

process.once("SIGINT", () => {
  exitCode = shutdownSignalExitCodes.get("SIGINT") ?? 130;
  shutdown("SIGINT");
});

process.once("SIGTERM", () => {
  exitCode = shutdownSignalExitCodes.get("SIGTERM") ?? 143;
  shutdown("SIGTERM");
});

process.on("exit", () => {
  for (const child of childProcesses) {
    if (!child.killed && child.exitCode === null && child.signalCode === null) {
      child.kill("SIGTERM");
    }
  }
});

Promise.all(
  childProcesses.map(
    (child) =>
      new Promise((resolve) => {
        child.once("close", resolve);
      }),
  ),
).then(() => {
  process.exit(exitCode);
});
