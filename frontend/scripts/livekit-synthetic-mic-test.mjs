import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const currentFile = fileURLToPath(import.meta.url);
const workspaceRoot = path.resolve(path.dirname(currentFile), "..");
const defaultOutputDir = path.join(workspaceRoot, ".codex", "media", "livekit-synthetic");

function parseArgs(argv) {
  const options = {
    url: "http://localhost:3000/speech",
    outputDir: defaultOutputDir,
    durationMs: 18000,
    screenshotEveryMs: 3000,
    headless: true,
    channel: "chrome",
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    const nextValue = () => {
      const value = argv[index + 1];

      if (!value || value.startsWith("--")) {
        throw new Error(`${arg} requires a value.`);
      }

      index += 1;
      return value;
    };

    if (arg === "--url") {
      options.url = nextValue();
    } else if (arg === "--out") {
      options.outputDir = path.resolve(workspaceRoot, nextValue());
    } else if (arg === "--duration-ms") {
      options.durationMs = Number(nextValue());
    } else if (arg === "--screenshot-every-ms") {
      options.screenshotEveryMs = Number(nextValue());
    } else if (arg === "--headed") {
      options.headless = false;
    } else if (arg === "--channel") {
      options.channel = nextValue();
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (!Number.isFinite(options.durationMs) || options.durationMs < 1000) {
    throw new Error("--duration-ms must be at least 1000.");
  }

  if (!Number.isFinite(options.screenshotEveryMs) || options.screenshotEveryMs < 250) {
    throw new Error("--screenshot-every-ms must be at least 250.");
  }

  return options;
}

function printHelp() {
  console.log(`Usage: npm run test:livekit:synthetic -- [options]

Options:
  --url <url>                    Speech page URL. Default: http://localhost:3000/speech
  --out <dir>                    Screenshot/report directory. Default: .codex/media/livekit-synthetic
  --duration-ms <ms>             Total run duration. Default: 18000
  --screenshot-every-ms <ms>     Screenshot cadence. Default: 3000
  --headed                       Run Chrome visibly instead of headless.
  --channel <name>               Playwright browser channel. Default: chrome
`);
}

function installSyntheticMicAndRtcProbe() {
  const OriginalRTCPeerConnection = window.RTCPeerConnection;
  const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;

  window.__liveKitSyntheticProbe = {
    createdStreams: 0,
    getUserMediaCalls: [],
    peerConnections: [],
    lastRms: 0,
    statsErrors: [],
  };

  class ProbedRTCPeerConnection extends OriginalRTCPeerConnection {
    constructor(...args) {
      super(...args);
      window.__liveKitSyntheticProbe.peerConnections.push(this);
    }
  }

  window.RTCPeerConnection = ProbedRTCPeerConnection;

  navigator.mediaDevices.getUserMedia = async (constraints) => {
    if (!constraints?.audio) {
      return new MediaStream();
    }

    const audioContext = new AudioContextConstructor();
    await audioContext.resume();

    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    const analyser = audioContext.createAnalyser();
    const destination = audioContext.createMediaStreamDestination();

    oscillator.type = "sine";
    oscillator.frequency.value = 440;
    gain.gain.value = 0.2;
    analyser.fftSize = 2048;

    oscillator.connect(gain);
    gain.connect(analyser);
    gain.connect(destination);
    oscillator.start();

    const samples = new Float32Array(analyser.fftSize);
    const updateRms = () => {
      analyser.getFloatTimeDomainData(samples);
      let sum = 0;

      for (const sample of samples) {
        sum += sample * sample;
      }

      window.__liveKitSyntheticProbe.lastRms = Math.sqrt(sum / samples.length);
      window.requestAnimationFrame(updateRms);
    };

    updateRms();
    window.__liveKitSyntheticProbe.createdStreams += 1;
    window.__liveKitSyntheticProbe.getUserMediaCalls.push({
      audio: Boolean(constraints.audio),
      video: Boolean(constraints.video),
      at: Date.now(),
    });
    window.__liveKitSyntheticProbe.audioContext = audioContext;
    window.__liveKitSyntheticProbe.syntheticStream = destination.stream;

    return destination.stream;
  };
}

async function collectPageProbe(page) {
  return page.evaluate(async () => {
    const probe = window.__liveKitSyntheticProbe;
    const peerConnectionStats = [];
    const senderTracks = [];

    for (const pc of probe?.peerConnections ?? []) {
      for (const sender of pc.getSenders()) {
        if (sender.track) {
          senderTracks.push({
            kind: sender.track.kind,
            enabled: sender.track.enabled,
            muted: sender.track.muted,
            readyState: sender.track.readyState,
            label: sender.track.label,
          });
        }
      }

      try {
        const stats = await pc.getStats();
        const selectedStats = [];

        stats.forEach((report) => {
          if (
            report.type === "outbound-rtp" ||
            report.type === "media-source" ||
            report.type === "track"
          ) {
            selectedStats.push({
              id: report.id,
              type: report.type,
              kind: report.kind,
              mediaType: report.mediaType,
              packetsSent: report.packetsSent,
              bytesSent: report.bytesSent,
              totalSamplesSent: report.totalSamplesSent,
              totalAudioEnergy: report.totalAudioEnergy,
              audioLevel: report.audioLevel,
            });
          }
        });
        peerConnectionStats.push(selectedStats);
      } catch (error) {
        probe.statsErrors.push(String(error));
      }
    }

    return {
      at: Date.now(),
      bodyText: document.body.innerText,
      createdStreams: probe?.createdStreams ?? 0,
      getUserMediaCalls: probe?.getUserMediaCalls ?? [],
      rms: probe?.lastRms ?? 0,
      peerConnections: probe?.peerConnections?.length ?? 0,
      senderTracks,
      peerConnectionStats,
    };
  });
}

function flattenStats(samples) {
  return samples.flatMap((sample) =>
    sample.peerConnectionStats.flatMap((pcStats) =>
      pcStats
        .filter((stat) => stat.kind === "audio" || stat.mediaType === "audio")
        .map((stat) => ({ ...stat, at: sample.at })),
    ),
  );
}

function maxMetricDelta(stats, metric) {
  const valuesById = new Map();

  for (const stat of stats) {
    const value = Number(stat[metric]);

    if (!Number.isFinite(value)) {
      continue;
    }

    const current = valuesById.get(stat.id) ?? { min: value, max: value };
    current.min = Math.min(current.min, value);
    current.max = Math.max(current.max, value);
    valuesById.set(stat.id, current);
  }

  return Math.max(0, ...Array.from(valuesById.values(), ({ min, max }) => max - min));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const runId = new Date().toISOString().replace(/[:.]/g, "-");
  const outputDir = path.join(options.outputDir, runId);
  const screenshotsDir = path.join(outputDir, "screenshots");

  await fs.mkdir(screenshotsDir, { recursive: true });

  const browser = await chromium.launch({
    headless: options.headless,
    channel: options.channel,
    args: ["--use-fake-ui-for-media-stream"],
  });

  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    permissions: ["microphone"],
  });
  await context.addInitScript(installSyntheticMicAndRtcProbe);

  const page = await context.newPage();
  const samples = [];
  const screenshots = [];

  try {
    await page.goto(options.url, { waitUntil: "networkidle", timeout: 30000 });
    await page.screenshot({ path: path.join(screenshotsDir, "00-before-click.png") });
    screenshots.push(path.join(screenshotsDir, "00-before-click.png"));
    await page.click('button[aria-label="Start voice session"]');

    const startedAt = Date.now();
    let screenshotIndex = 1;
    let nextScreenshotAt = options.screenshotEveryMs;

    while (Date.now() - startedAt < options.durationMs) {
      const remainingMs = options.durationMs - (Date.now() - startedAt);

      await page.waitForTimeout(Math.min(500, Math.max(0, remainingMs)));

      const elapsedMs = Date.now() - startedAt;
      const sample = await collectPageProbe(page);

      samples.push(sample);

      if (elapsedMs >= nextScreenshotAt || elapsedMs >= options.durationMs) {
        const screenshotPath = path.join(
          screenshotsDir,
          `${String(screenshotIndex).padStart(2, "0")}-${elapsedMs}ms.png`,
        );

        await page.screenshot({ path: screenshotPath });
        screenshots.push(screenshotPath);
        screenshotIndex += 1;
        nextScreenshotAt += options.screenshotEveryMs;
      }
    }
  } finally {
    await browser.close();
  }

  const audioStats = flattenStats(samples);
  const maxRms = Math.max(0, ...samples.map((sample) => sample.rms));
  const bytesSentDelta = maxMetricDelta(audioStats, "bytesSent");
  const packetsSentDelta = maxMetricDelta(audioStats, "packetsSent");
  const agentJoined = samples.some((sample) => /voice agent connected|Agent:/i.test(sample.bodyText));
  const silenceDetected = samples.some((sample) => /detects silence/i.test(sample.bodyText));
  const finalStatusText = samples.at(-1)?.bodyText ?? "";
  const failures = [];

  if (samples.length === 0) {
    failures.push("No probe samples were collected.");
  }

  if (!samples.some((sample) => sample.createdStreams > 0)) {
    failures.push("navigator.mediaDevices.getUserMedia was not called for audio.");
  }

  if (maxRms < 0.02) {
    failures.push(`Synthetic mic RMS stayed too low (${maxRms.toFixed(4)}).`);
  }

  if (packetsSentDelta <= 0 && bytesSentDelta <= 0) {
    failures.push("Outbound WebRTC audio stats did not increase.");
  }

  if (!agentJoined) {
    failures.push("The LiveKit agent did not visibly join before the timeout.");
  }

  if (silenceDetected) {
    failures.push("The page showed the LiveKit silence warning during the synthetic mic run.");
  }

  const report = {
    ok: failures.length === 0,
    url: options.url,
    outputDir,
    screenshots,
    summary: {
      maxRms,
      bytesSentDelta,
      packetsSentDelta,
      agentJoined,
      silenceDetected,
      finalStatusText,
    },
    failures,
    samples,
  };

  const reportPath = path.join(outputDir, "report.json");
  await fs.writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`);

  console.log(JSON.stringify({ ...report.summary, ok: report.ok, failures, reportPath }, null, 2));

  if (!report.ok) {
    process.exitCode = 1;
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
