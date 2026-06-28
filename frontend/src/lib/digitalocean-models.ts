import "server-only";

import { getDigitalOceanModelEnv } from "@/lib/env";

export type ConversationMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type DigitalOceanChatResponse = {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
};

type DigitalOceanModelsResponse = {
  data?: Array<{
    id?: string;
  }>;
};

type DigitalOceanSpeechResponse = {
  data?: Array<{
    b64_json?: string;
  }>;
};

const speechCache = new Map<string, string>();

const chatModelFallbacks = [
  "qwen3-coder-flash",
  "mistral-3-14B",
  "openai-gpt-5.4-nano",
  "openai-gpt-4o-mini",
  "openai-gpt-oss-20b",
];

const summaryModelFallbacks = [
  "qwen3-coder-flash",
  "mistral-3-14B",
  "alibaba-qwen3-32b",
  "llama3.3-70b-instruct",
  "openai-gpt-oss-20b",
  "llama-4-maverick",
];

const intentModelFallbacks = [
  "qwen3-coder-flash",
  "mistral-3-14B",
  "openai-gpt-5.4-nano",
  "openai-gpt-4o-mini",
  "openai-gpt-oss-20b",
];

export type ConversationIntent =
  | "none"
  | "show_dataset_upload"
  | "start_agent_work";

function normalizeBaseUrl(baseUrl: string) {
  return baseUrl.replace(/\/+$/, "");
}

async function listAvailableModels() {
  const { apiKey, baseUrl } = getDigitalOceanModelEnv();
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/models`, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
    },
  });

  if (!response.ok) {
    return [];
  }

  const data = (await response.json()) as DigitalOceanModelsResponse;

  return data.data?.map((model) => model.id).filter((id): id is string => Boolean(id)) ?? [];
}

async function resolveAvailableModel(model: string, fallbacks: string[]) {
  const availableModels = await listAvailableModels();

  if (availableModels.length === 0 || availableModels.includes(model)) {
    return model;
  }

  const fallback = fallbacks.find((candidate) => availableModels.includes(candidate));

  if (fallback) {
    return fallback;
  }

  throw new Error(
    `DigitalOcean model "${model}" is not available. Available models: ${availableModels.join(", ")}`,
  );
}

async function completeWithDigitalOcean({
  fallbacks,
  maxTokens,
  messages,
  model,
  temperature = 0.3,
}: {
  fallbacks: string[];
  maxTokens?: number;
  messages: ConversationMessage[];
  model: string;
  temperature?: number;
}) {
  const { apiKey, baseUrl } = getDigitalOceanModelEnv();
  const resolvedModel = await resolveAvailableModel(model, fallbacks);
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: resolvedModel,
      messages,
      temperature,
      ...(maxTokens ? { max_tokens: maxTokens } : {}),
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => "");
    throw new Error(
      `DigitalOcean model request failed (${response.status}): ${
        errorBody || response.statusText
      }`,
    );
  }

  const data = (await response.json()) as DigitalOceanChatResponse;
  const content = data.choices?.[0]?.message?.content?.trim();

  if (!content) {
    throw new Error("DigitalOcean model response did not include text.");
  }

  return content;
}

export async function continueConversation(messages: ConversationMessage[]) {
  const { chatModel } = getDigitalOceanModelEnv();

  return completeWithDigitalOcean({
    fallbacks: chatModelFallbacks,
    maxTokens: 36,
    model: chatModel,
    temperature: 0.45,
    messages: [
      {
        role: "system",
        content:
          "You are a fast voice agent. Reply in one short sentence, 12 words max. Ask one concise clarifying question when needed.",
      },
      ...messages,
    ],
  });
}

export async function summarizeConversation(messages: ConversationMessage[]) {
  const { summaryModel } = getDigitalOceanModelEnv();

  return completeWithDigitalOcean({
    fallbacks: summaryModelFallbacks,
    maxTokens: 220,
    model: summaryModel,
    temperature: 0.1,
    messages: [
      {
        role: "system",
        content:
          "Summarize what the user wants as one direct implementation brief string. Include concrete requirements, constraints, and data context. Do not include markdown.",
      },
      {
        role: "user",
        content: messages
          .map((message) => `${message.role.toUpperCase()}: ${message.content}`)
          .join("\n"),
      },
    ],
  });
}

function parseConversationIntent(content: string): ConversationIntent {
  const jsonMatch = content.match(/\{[\s\S]*\}/);

  if (!jsonMatch) {
    return "none";
  }

  try {
    const parsed = JSON.parse(jsonMatch[0]) as { intent?: unknown };

    if (
      parsed.intent === "show_dataset_upload" ||
      parsed.intent === "start_agent_work" ||
      parsed.intent === "none"
    ) {
      return parsed.intent;
    }
  } catch {
    return "none";
  }

  return "none";
}

export async function classifyConversationIntent(
  messages: ConversationMessage[],
) {
  const { chatModel } = getDigitalOceanModelEnv();
  const conversation = messages
    .slice(-8)
    .map((message) => `${message.role.toUpperCase()}: ${message.content}`)
    .join("\n");
  const content = await completeWithDigitalOcean({
    fallbacks: intentModelFallbacks,
    maxTokens: 40,
    model: chatModel,
    temperature: 0,
    messages: [
      {
        role: "system",
        content:
          'Classify the user intent for this voice app. Return only JSON: {"intent":"none"} or {"intent":"show_dataset_upload"} or {"intent":"start_agent_work"}. Use show_dataset_upload only when the user wants to upload/select/attach/add a file, document, PDF, spreadsheet, dataset, source data, or upload button. Use start_agent_work only when the user clearly wants to end voice and switch/handoff/proceed to Coach Bron or the work agent. If both are present, choose show_dataset_upload. Otherwise choose none.',
      },
      {
        role: "user",
        content: conversation,
      },
    ],
  });

  return parseConversationIntent(content);
}

export async function synthesizeSpeech(input: string) {
  const { apiKey, baseUrl, ttsInstructions, ttsModel, ttsVoice } = getDigitalOceanModelEnv();
  const cacheKey = `${ttsModel}:${ttsVoice}:${ttsInstructions}:${input}`;
  const cachedAudio = speechCache.get(cacheKey);

  if (cachedAudio) {
    return cachedAudio;
  }

  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/audio/speech`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: ttsModel,
      input,
      voice: ttsVoice,
      response_format: "mp3",
      instructions: ttsInstructions,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text().catch(() => "");
    throw new Error(
      `DigitalOcean TTS request failed (${response.status}): ${errorBody || response.statusText}`,
    );
  }

  const contentType = response.headers.get("content-type") ?? "";

  if (contentType.includes("audio/") || contentType.includes("application/octet-stream")) {
    const audioBuffer = await response.arrayBuffer();

    const audioBase64 = Buffer.from(audioBuffer).toString("base64");

    speechCache.set(cacheKey, audioBase64);
    return audioBase64;
  }

  const data = (await response.json()) as DigitalOceanSpeechResponse;
  const audioBase64 = data.data?.[0]?.b64_json;

  if (!audioBase64) {
    throw new Error("DigitalOcean TTS response did not include audio.");
  }

  speechCache.set(cacheKey, audioBase64);
  return audioBase64;
}
