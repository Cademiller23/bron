"use client";

import { Loader2, Mic, MicOff } from "lucide-react";
import { Room, RoomEvent } from "livekit-client";
import { useCallback, useRef, useState } from "react";

import { cn } from "@/lib/utils";
import { api } from "@/trpc/react";

type VoiceState = "idle" | "connecting" | "listening" | "stopping" | "error";

const TRANSCRIPTION_ROOM = "f1-voice-transcription";
const TRANSCRIPTION_TOPIC = "lk.transcription";
const TRANSCRIPTION_FINAL_ATTRIBUTE = "lk.transcription_final";

type TranscriptSegment = {
  id: string;
  text: string;
  final: boolean;
  timestamp: number;
};

function createIdentity() {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return `voice-${crypto.randomUUID()}`;
  }

  return `voice-${Date.now()}`;
}

export function VoiceTranscriptionButton() {
  const [voiceState, setVoiceState] = useState<VoiceState>("idle");
  const [segments, setSegments] = useState<TranscriptSegment[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const roomRef = useRef<Room | null>(null);
  const tokenMutation = api.livekit.createToken.useMutation();

  const handleTranscript = useCallback((segment: TranscriptSegment) => {
    setSegments((currentSegments) => {
      const merged = new Map(currentSegments.map((currentSegment) => [currentSegment.id, currentSegment]));
      merged.set(segment.id, segment);

      return Array.from(merged.values())
        .sort((a, b) => a.timestamp - b.timestamp)
        .slice(-24);
    });
  }, []);

  const handleTranscriptionStream = useCallback(
    async (reader: AsyncIterable<string> & { info: { id: string; timestamp: number; attributes?: Record<string, string> } }) => {
      const isFinal = reader.info.attributes?.[TRANSCRIPTION_FINAL_ATTRIBUTE] === "true";
      let latestText = "";

      for await (const chunk of reader) {
        const text = chunk.trim();

        if (text.length === 0) {
          continue;
        }

        latestText = text;
        handleTranscript({
          id: reader.info.id,
          text: latestText,
          final: isFinal,
          timestamp: reader.info.timestamp,
        });
      }

      if (isFinal && latestText.length > 0) {
        handleTranscript({
          id: reader.info.id,
          text: latestText,
          final: true,
          timestamp: reader.info.timestamp,
        });
      }
    },
    [handleTranscript],
  );

  const disconnectRoom = useCallback(async () => {
    const room = roomRef.current;
    roomRef.current = null;

    if (!room) {
      return;
    }

    room.unregisterTextStreamHandler(TRANSCRIPTION_TOPIC);
    await room.localParticipant.setMicrophoneEnabled(false).catch(() => undefined);
    room.disconnect();
  }, []);

  const stopListening = useCallback(async () => {
    setVoiceState("stopping");
    await disconnectRoom();
    setVoiceState("idle");
  }, [disconnectRoom]);

  const startListening = useCallback(async () => {
    setVoiceState("connecting");
    setErrorMessage(null);
    setSegments([]);

    const room = new Room();
    roomRef.current = room;
    room.registerTextStreamHandler(TRANSCRIPTION_TOPIC, (reader) => {
      void handleTranscriptionStream(reader).catch(() => {
        setVoiceState("error");
        setErrorMessage("Could not read the LiveKit transcript stream.");
      });
    });
    room.on(RoomEvent.Disconnected, () => {
      room.unregisterTextStreamHandler(TRANSCRIPTION_TOPIC);
      if (roomRef.current === room) {
        roomRef.current = null;
        setVoiceState("idle");
      }
    });

    try {
      const identity = createIdentity();
      const { token, url } = await tokenMutation.mutateAsync({
        identity,
        room: TRANSCRIPTION_ROOM,
        name: "Voice transcription",
        metadata: JSON.stringify({ feature: "voice-to-text" }),
        canPublish: true,
        canSubscribe: true,
      });

      await room.connect(url, token);
      await room.localParticipant.setMicrophoneEnabled(true);
      setVoiceState("listening");
    } catch (error) {
      await disconnectRoom();
      setVoiceState("error");
      setErrorMessage(error instanceof Error ? error.message : "Could not start voice-to-text.");
    }
  }, [disconnectRoom, handleTranscriptionStream, tokenMutation]);

  const isBusy = voiceState === "connecting" || voiceState === "stopping";
  const isListening = voiceState === "listening";
  const buttonText =
    voiceState === "connecting"
      ? "Connecting..."
      : voiceState === "listening"
        ? "Speech-to-text on"
        : voiceState === "stopping"
          ? "Stopping..."
          : "Start speech-to-text";
  const titleText = errorMessage
    ? `Speech-to-text error: ${errorMessage}`
    : segments.length > 0
      ? "LiveKit speech-to-text is receiving transcription events."
      : "Start LiveKit speech-to-text.";

  return (
    <button
      type="button"
      disabled={isBusy}
      onClick={isListening ? stopListening : startListening}
      title={titleText}
      aria-label={buttonText}
      className={cn(
        "group inline-flex h-12 items-center gap-3 rounded-full border px-4 pr-5 text-sm font-semibold tracking-normal shadow-[0_16px_45px_oklch(0_0_0_/_0.4)] backdrop-blur-xl transition-all",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white/70 focus-visible:ring-offset-2 focus-visible:ring-offset-black",
        "disabled:pointer-events-none disabled:opacity-70",
        isListening
          ? "border-[oklch(0.76_0.19_150_/_0.85)] bg-[oklch(0.18_0.04_150_/_0.92)] text-white hover:bg-[oklch(0.21_0.05_150_/_0.94)]"
          : errorMessage
            ? "border-[oklch(0.72_0.19_30_/_0.8)] bg-[oklch(0.18_0.035_30_/_0.92)] text-white hover:bg-[oklch(0.21_0.045_30_/_0.94)]"
            : "border-white/20 bg-black/72 text-white hover:border-white/35 hover:bg-black/86",
      )}
    >
      <span
        className={cn(
          "grid size-7 place-items-center rounded-full border transition-colors",
          isListening
            ? "border-[oklch(0.76_0.19_150_/_0.55)] bg-[oklch(0.72_0.18_150)] text-black"
            : "border-white/15 bg-white text-black",
        )}
      >
        {isBusy ? (
          <Loader2 className="size-4 animate-spin" />
        ) : isListening ? (
          <MicOff className="size-4" />
        ) : (
          <Mic className="size-4" />
        )}
      </span>
      <span className="whitespace-nowrap">{buttonText}</span>
      <span
        aria-hidden="true"
        className={cn(
          "size-2 rounded-full transition-colors",
          isListening ? "bg-[oklch(0.78_0.2_150)] shadow-[0_0_16px_oklch(0.78_0.2_150)]" : "bg-white/35",
        )}
      />
    </button>
  );
}
