import { z } from "zod";

const nonEmptyString = z.string().trim().min(1);

export function optionalEnv(name: string): string | undefined {
  const value = process.env[name];
  return value && value.trim().length > 0 ? value : undefined;
}

export function requireEnv(name: string): string {
  const parsed = nonEmptyString.safeParse(process.env[name]);

  if (!parsed.success) {
    throw new Error(`Missing required environment variable: ${name}`);
  }

  return parsed.data;
}

export function getLiveKitEnv() {
  return {
    url: requireEnv("LIVEKIT_URL"),
    apiKey: requireEnv("LIVEKIT_API_KEY"),
    apiSecret: requireEnv("LIVEKIT_API_SECRET"),
  };
}

export function getLiveKitAgentName() {
  return optionalEnv("LIVEKIT_AGENT_NAME") ?? optionalEnv("LIVEKIT_AGENT_ID") ?? "f1-voice-agent";
}

export function getDigitalOceanModelEnv() {
  return {
    apiKey: requireEnv("DIGITALOCEAN_MODEL_API_KEY"),
    baseUrl: optionalEnv("DIGITALOCEAN_MODEL_BASE_URL") ?? "https://inference.do-ai.run/v1",
    chatModel: requireEnv("DIGITALOCEAN_CHAT_MODEL"),
    summaryModel: requireEnv("DIGITALOCEAN_SUMMARY_MODEL"),
    ttsInstructions:
      optionalEnv("DIGITALOCEAN_TTS_INSTRUCTIONS") ??
      "Warm, natural, concise.",
    ttsModel: optionalEnv("DIGITALOCEAN_TTS_MODEL") ?? "qwen3-tts-voicedesign",
    ttsVoice: optionalEnv("DIGITALOCEAN_TTS_VOICE") ?? "alloy",
  };
}
