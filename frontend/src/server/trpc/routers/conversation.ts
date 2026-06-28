import { z } from "zod";
import {
  classifyConversationIntent,
  continueConversation,
  summarizeConversation,
  synthesizeSpeech,
  type ConversationMessage,
} from "@/lib/digitalocean-models";
import { createTRPCRouter, publicProcedure } from "@/server/trpc/trpc";

const conversationMessageSchema = z.object({
  role: z.enum(["system", "user", "assistant"]),
  content: z.string().trim().min(1).max(12_000),
});

export const conversationRouter = createTRPCRouter({
  respond: publicProcedure
    .input(
      z.object({
        messages: z.array(conversationMessageSchema).min(1).max(40),
      }),
    )
    .mutation(({ input }) =>
      continueConversation(input.messages as ConversationMessage[]).then((response) => ({
        response,
      })),
    ),

  summarize: publicProcedure
    .input(
      z.object({
        messages: z.array(conversationMessageSchema).min(1).max(60),
      }),
    )
    .mutation(({ input }) =>
      summarizeConversation(input.messages as ConversationMessage[]).then((summary) => ({
        summary,
      })),
    ),

  classifyIntent: publicProcedure
    .input(
      z.object({
        messages: z.array(conversationMessageSchema).min(1).max(12),
      }),
    )
    .mutation(({ input }) =>
      classifyConversationIntent(input.messages as ConversationMessage[]).then(
        (intent) => ({
          intent,
        }),
      ),
    ),

  speak: publicProcedure
    .input(
      z.object({
        input: z.string().trim().min(1).max(2_000),
      }),
    )
    .mutation(({ input }) =>
      synthesizeSpeech(input.input).then((audioBase64) => ({
        audioBase64,
        mimeType: "audio/mpeg",
      })),
    ),
});
