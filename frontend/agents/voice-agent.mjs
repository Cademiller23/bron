import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import voiceAgentDefaults from "../voice-agent.defaults.json" with { type: "json" };
import {
  AutoSubscribe,
  cli,
  defineAgent,
  inference,
  llm,
  ServerOptions,
  voice,
} from "@livekit/agents";
import { LLM as OpenAILLM } from "@livekit/agents-plugin-openai";
import { z } from "zod";

const currentFile = fileURLToPath(import.meta.url);
const workspaceRoot = path.resolve(path.dirname(currentFile), "..");

function loadEnvFile(fileName) {
  const envPath = path.join(workspaceRoot, fileName);

  if (!fs.existsSync(envPath)) {
    return;
  }

  const lines = fs.readFileSync(envPath, "utf8").split(/\r?\n/);

  for (const line of lines) {
    const trimmedLine = line.trim();

    if (!trimmedLine || trimmedLine.startsWith("#")) {
      continue;
    }

    const separatorIndex = trimmedLine.indexOf("=");

    if (separatorIndex === -1) {
      continue;
    }

    const key = trimmedLine.slice(0, separatorIndex).trim();
    let value = trimmedLine.slice(separatorIndex + 1).trim();

    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }

    process.env[key] ??= value;
  }
}

loadEnvFile(".env.local");
loadEnvFile(".env");

function requiredEnv(name) {
  const value = process.env[name];

  if (!value) {
    throw new Error(`${name} is required for the LiveKit voice agent.`);
  }

  return value;
}

function optionalNumberEnv(name, fallback) {
  const value = process.env[name];
  const parsedValue = value ? Number(value) : Number.NaN;

  return Number.isFinite(parsedValue) ? parsedValue : fallback;
}

function boundedNumberEnv(name, fallback, min, max) {
  return Math.min(max, Math.max(min, optionalNumberEnv(name, fallback)));
}

const livekitUrl = requiredEnv("LIVEKIT_URL");
const livekitApiKey = requiredEnv("LIVEKIT_API_KEY");
const livekitApiSecret = requiredEnv("LIVEKIT_API_SECRET");
const digitalOceanApiKey = requiredEnv("DIGITALOCEAN_MODEL_API_KEY");
const digitalOceanBaseUrl =
  process.env.DIGITALOCEAN_MODEL_BASE_URL ?? "https://inference.do-ai.run/v1";
const chatModel = process.env.DIGITALOCEAN_CHAT_MODEL ?? "qwen3-coder-flash";
const sttModel =
  process.env.LIVEKIT_AGENT_STT_MODEL ?? "deepgram/flux-general-en";
const ttsModel = process.env.LIVEKIT_AGENT_TTS_MODEL ?? "cartesia/sonic-3.5";
const ttsVoice =
  process.env.LIVEKIT_AGENT_TTS_VOICE || "a5136bf9-224c-4d76-b823-52bd5efcffcc";
const agentName =
  process.env.LIVEKIT_AGENT_NAME ||
  process.env.LIVEKIT_AGENT_ID ||
  "f1-voice-agent";
const agentId = process.env.LIVEKIT_AGENT_ID || agentName;
const CUSTOM_DATASET_UPLOAD_TOPIC = "lebronsseiur.custom_dataset_upload";
const START_AGENT_WORK_TOPIC = "lebronsseiur.start_agent_work";
const VOICE_DEBUG_TOPIC = "lebronsseiur.voice_debug";
const START_AGENT_WORK_PHRASE =
  "Ok, redirecting you to Coach Bron. Stand by for context switch.";
const turnDetectionMode = process.env.LIVEKIT_AGENT_TURN_DETECTION || "stt";
const vadConfig = {
  minSpeechDurationMs: boundedNumberEnv(
    "LIVEKIT_AGENT_VAD_MIN_SPEECH_MS",
    voiceAgentDefaults.vad.minSpeechDurationMs,
    20,
    500,
  ),
  minSilenceDurationMs: boundedNumberEnv(
    "LIVEKIT_AGENT_VAD_MIN_SILENCE_MS",
    voiceAgentDefaults.vad.minSilenceDurationMs,
    40,
    200,
  ),
  activationThreshold: boundedNumberEnv(
    "LIVEKIT_AGENT_VAD_ACTIVATION_THRESHOLD",
    voiceAgentDefaults.vad.activationThreshold,
    0.1,
    0.8,
  ),
};
const eotTimeoutMs = boundedNumberEnv(
  "LIVEKIT_AGENT_EOT_TIMEOUT_MS",
  voiceAgentDefaults.eotTimeoutMs,
  500,
  60000,
);
const endpointConfig = {
  minDelayMs: boundedNumberEnv(
    "LIVEKIT_AGENT_ENDPOINT_MIN_DELAY_MS",
    voiceAgentDefaults.endpoint.minDelayMs,
    20,
    1000,
  ),
  maxDelayMs: boundedNumberEnv(
    "LIVEKIT_AGENT_ENDPOINT_MAX_DELAY_MS",
    voiceAgentDefaults.endpoint.maxDelayMs,
    80,
    5000,
  ),
};

function logVoiceEvent(event, payload = {}) {
  console.info(`[voice-agent] ${event}`, payload);
}

function describeError(error) {
  if (error instanceof Error) {
    const statusCode = error.statusCode ?? error.status;
    const requestId = error.requestId;
    const details = [
      error.message,
      statusCode ? `status ${statusCode}` : null,
      requestId ? `request ${requestId}` : null,
    ].filter(Boolean);

    return details.join(" | ");
  }

  if (error && typeof error === "object") {
    const fields = Object.fromEntries(
      Object.getOwnPropertyNames(error).map((key) => [key, error[key]]),
    );
    const nestedError =
      error.error && typeof error.error === "object" ? error.error : undefined;
    const nestedFields = nestedError
      ? Object.fromEntries(
          Object.getOwnPropertyNames(nestedError).map((key) => [
            key,
            nestedError[key],
          ]),
        )
      : {};
    const statusCode =
      error.statusCode ??
      error.status ??
      fields.statusCode ??
      fields.status ??
      nestedFields.statusCode ??
      nestedFields.status;
    const message =
      typeof error.message === "string"
        ? error.message
        : typeof fields.message === "string"
          ? fields.message
          : typeof nestedFields.message === "string"
            ? nestedFields.message
            : JSON.stringify({ ...fields, error: nestedFields });

    return [message, statusCode ? `status ${statusCode}` : null]
      .filter(Boolean)
      .join(" | ");
  }

  return String(error);
}

export default defineAgent({
  entry: async (ctx) => {
    function sendVoiceDebugEvent(event, payload = {}) {
      void ctx.room.localParticipant
        ?.sendText(
          JSON.stringify({
            type: "voice_debug",
            event,
            payload,
            timestamp: Date.now(),
          }),
          { topic: VOICE_DEBUG_TOPIC },
        )
        .catch(() => undefined);
    }

    const session = new voice.AgentSession({
      vad: new inference.VAD({
        model: "silero",
        minSpeechDuration: vadConfig.minSpeechDurationMs,
        minSilenceDuration: vadConfig.minSilenceDurationMs,
        activationThreshold: vadConfig.activationThreshold,
      }),
      stt: new inference.STT({
        model: sttModel,
        language: "en",
        apiKey: livekitApiKey,
        apiSecret: livekitApiSecret,
        modelOptions: {
          eager_eot_threshold: boundedNumberEnv(
            "LIVEKIT_AGENT_EAGER_EOT_THRESHOLD",
            0.3,
            0.1,
            0.9,
          ),
          eot_timeout_ms: eotTimeoutMs,
          language_hint: "en",
        },
      }),
      llm: new OpenAILLM({
        apiKey: digitalOceanApiKey,
        baseURL: digitalOceanBaseUrl,
        model: chatModel,
        temperature: 0.35,
        maxCompletionTokens: optionalNumberEnv(
          "LIVEKIT_AGENT_MAX_COMPLETION_TOKENS",
          80,
        ),
      }),
      tts: new inference.TTS({
        model: ttsModel,
        voice: ttsVoice,
        apiKey: livekitApiKey,
        apiSecret: livekitApiSecret,
        modelOptions: {
          speed: process.env.LIVEKIT_AGENT_TTS_SPEED || "normal",
          max_buffer_delay_ms: boundedNumberEnv(
            "LIVEKIT_AGENT_TTS_MAX_BUFFER_DELAY_MS",
            20,
            0,
            1000,
          ),
        },
      }),
      aecWarmupDuration: null,
      turnHandling: {
        turnDetection: turnDetectionMode,
        preemptiveGeneration: {
          enabled: true,
          preemptiveTts: true,
          maxSpeechDuration: 8000,
          maxRetries: 3,
        },
        interruption: {
          enabled: false,
        },
        endpointing: {
          minDelay: endpointConfig.minDelayMs,
          maxDelay: endpointConfig.maxDelayMs,
        },
      },
      ttsTextTransforms: ["filter_markdown", "filter_emoji"],
    });

    session.on(voice.AgentSessionEventTypes.UserInputTranscribed, (event) => {
      const payload = {
        final: event.isFinal,
        length: event.transcript.length,
        transcript: event.transcript.slice(0, 160),
      };

      logVoiceEvent("user_input_transcribed", payload);
      sendVoiceDebugEvent("user_input_transcribed", payload);
    });
    session.on(voice.AgentSessionEventTypes.UserStateChanged, (event) => {
      const payload = {
        from: event.oldState,
        to: event.newState,
      };

      logVoiceEvent("user_state_changed", payload);
      sendVoiceDebugEvent("user_state_changed", payload);
    });
    session.on(voice.AgentSessionEventTypes.AgentStateChanged, (event) => {
      const payload = {
        from: event.oldState,
        to: event.newState,
      };

      logVoiceEvent("agent_state_changed", payload);
      sendVoiceDebugEvent("agent_state_changed", payload);
    });
    session.on(voice.AgentSessionEventTypes.SpeechCreated, (event) => {
      const payload = {
        source: event.source,
        userInitiated: event.userInitiated,
      };

      logVoiceEvent("speech_created", payload);
      sendVoiceDebugEvent("speech_created", payload);
    });
    session.on(voice.AgentSessionEventTypes.FunctionToolsExecuted, (event) => {
      const payload = {
        tools: event.functionCalls.map((call) => call.name),
      };

      logVoiceEvent("function_tools_executed", payload);
      sendVoiceDebugEvent("function_tools_executed", payload);
    });
    session.on(voice.AgentSessionEventTypes.MetricsCollected, (event) => {
      const payload = {
        type: event.metrics.type,
        label: event.metrics.label,
        requestId: event.metrics.requestId,
      };

      logVoiceEvent("metrics_collected", payload);
      sendVoiceDebugEvent("metrics_collected", payload);
    });
    session.on(voice.AgentSessionEventTypes.Error, (event) => {
      const payload = {
        error: describeError(event.error),
        source: event.source?.constructor?.name,
      };

      logVoiceEvent("session_error", payload);
      sendVoiceDebugEvent("session_error", payload);
    });

    const agent = new voice.Agent({
      id: agentId,
      instructions: `You are Bron, a fast voice agent for a problem-solving app. Speak in short, confident sentences under 18 words unless asked for detail. Ask one clear follow-up when the user is vague. Do not mention implementation details unless asked.

Tool rules:
- If the user asks to upload, attach, select, choose, add, import, load, provide, or send any file, document, PDF, DOC, DOCX, spreadsheet, CSV, JSON, dataset, source data, custom dataset, their own data file, file selector, file picker, paperclip, upload button, attachment button, addition button, or additional file, call show_custom_dataset_upload before saying anything else. After the tool call, say exactly: "Oh, I'll pull up the addition button below. Just press it and add your custom dataset in any format."
- Upload/file/document requests are not handoff requests. If a turn asks for upload/file/document controls, call only show_custom_dataset_upload and do not call start_agent_work in that same turn.
- If the user clearly says they are done explaining, asks you to get to work, start working, run it, build it, solve it, proceed, go ahead, move on, quit voice, switch to Coach Bron, or go to the next step, and they are not asking to upload/select/add a file or document, call start_agent_work before saying anything else. After the tool call, say exactly: "${START_AGENT_WORK_PHRASE}"
- Do not call start_agent_work while you still need a critical clarification. If you cannot call a tool, still respond naturally and state what you are doing.`,
      tools: {
        show_custom_dataset_upload: llm.tool({
          description:
            "Show the custom dataset upload control when the user asks to upload, attach, select, choose, add, import, load, provide, or send a file, document, PDF, DOC, DOCX, custom dataset, their own data, source data, spreadsheet, CSV, JSON, Excel file, file selector, file picker, paperclip, attachment button, upload button, addition button, or additional file.",
          parameters: z.object({
            reason: z.string().optional(),
          }),
          execute: async () => {
            await ctx.room.localParticipant?.sendText(
              JSON.stringify({ type: "show_custom_dataset_upload" }),
              { topic: CUSTOM_DATASET_UPLOAD_TOPIC },
            );
            const payload = { topic: CUSTOM_DATASET_UPLOAD_TOPIC };

            logVoiceEvent("tool_sent_show_custom_dataset_upload", payload);
            sendVoiceDebugEvent("tool_sent_show_custom_dataset_upload", payload);

            return "The custom dataset upload button is visible below the voice control.";
          },
        }),
        start_agent_work: llm.tool({
          description:
            "Use when the user is done explaining and wants the app to stop the voice conversation, summarize the request, move to the agent work graph, proceed, go ahead, move on, quit voice, switch to Coach Bron, or go to the next step. Do not use for upload, file, or document selection requests.",
          parameters: z.object({
            reason: z.string().optional(),
          }),
          execute: async () => {
            await ctx.room.localParticipant?.sendText(
              JSON.stringify({ type: "start_agent_work" }),
              { topic: START_AGENT_WORK_TOPIC },
            );
            const payload = { topic: START_AGENT_WORK_TOPIC };

            logVoiceEvent("tool_sent_start_agent_work", payload);
            sendVoiceDebugEvent("tool_sent_start_agent_work", payload);

            return START_AGENT_WORK_PHRASE;
          },
        }),
      },
    });

    await ctx.connect(undefined, AutoSubscribe.AUDIO_ONLY);
    const roomConnectedPayload = {
      room: ctx.room.name,
      agentName,
      chatModel,
      sttModel,
      ttsModel,
      turnDetectionMode,
      vadConfig,
      eotTimeoutMs,
      endpointConfig,
    };

    logVoiceEvent("room_connected", roomConnectedPayload);
    sendVoiceDebugEvent("room_connected", roomConnectedPayload);

    await session.start({
      agent,
      room: ctx.room,
      outputOptions: {
        syncTranscription: false,
      },
    });

    // Do not speak on join. The first audio should be a direct response to the user.
  },
});

if (process.argv[1] === currentFile) {
  delete process.env.LIVEKIT_AGENT_NAME;

  cli.runApp(
    new ServerOptions({
      agent: currentFile,
      wsURL: livekitUrl,
      apiKey: livekitApiKey,
      apiSecret: livekitApiSecret,
      agentName,
    }),
  );
}
