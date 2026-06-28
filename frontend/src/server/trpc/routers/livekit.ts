import { z } from "zod";
import { createLiveKitRoomToken } from "@/lib/livekit";
import { createTRPCRouter, publicProcedure } from "@/server/trpc/trpc";

export const livekitRouter = createTRPCRouter({
  createToken: publicProcedure
    .input(
      z.object({
        identity: z.string().trim().min(1).max(128),
        room: z.string().trim().min(1).max(128),
        name: z.string().trim().min(1).max(128).optional(),
        metadata: z.string().max(4096).optional(),
        canPublish: z.boolean().optional(),
        canSubscribe: z.boolean().optional(),
        dispatchAgent: z.boolean().optional(),
      }),
    )
    .mutation(({ input }) => createLiveKitRoomToken(input)),
});
