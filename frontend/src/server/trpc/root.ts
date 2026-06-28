import { conversationRouter } from "@/server/trpc/routers/conversation";
import { livekitRouter } from "@/server/trpc/routers/livekit";
import { createTRPCRouter } from "@/server/trpc/trpc";

export const appRouter = createTRPCRouter({
  conversation: conversationRouter,
  livekit: livekitRouter,
});

export type AppRouter = typeof appRouter;
