import type { Metadata } from "next";
import { ViewTransition } from "react";
import { SpeechInputWorkbench } from "@/components/speech-input-workbench";

export const metadata: Metadata = {
  title: "Speech Input | Lebronsseiur",
  description: "Upload optimization data and draft text-to-speech instructions.",
};

export default function SpeechPage() {
  return (
    <ViewTransition
      enter={{ "nav-forward": "nav-forward", default: "none" }}
      exit={{ "nav-forward": "nav-forward", default: "none" }}
      default="none"
    >
      <SpeechInputWorkbench />
    </ViewTransition>
  );
}
