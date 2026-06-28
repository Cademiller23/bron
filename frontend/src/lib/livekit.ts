import "server-only";

import { RoomAgentDispatch, RoomConfiguration } from "@livekit/protocol";
import { AccessToken } from "livekit-server-sdk";
import { getLiveKitAgentName, getLiveKitEnv } from "@/lib/env";

export type CreateLiveKitTokenInput = {
  identity: string;
  room: string;
  name?: string;
  metadata?: string;
  canPublish?: boolean;
  canSubscribe?: boolean;
  dispatchAgent?: boolean;
};

export async function createLiveKitRoomToken({
  identity,
  room,
  name,
  metadata,
  canPublish = true,
  canSubscribe = true,
  dispatchAgent = false,
}: CreateLiveKitTokenInput) {
  const { apiKey, apiSecret, url } = getLiveKitEnv();
  const token = new AccessToken(apiKey, apiSecret, {
    identity,
    name,
    metadata,
    ttl: "1h",
  });

  token.addGrant({
    room,
    roomJoin: true,
    canPublish,
    canSubscribe,
  });

  if (dispatchAgent) {
    token.roomConfig = new RoomConfiguration({
      agents: [
        new RoomAgentDispatch({
          agentName: getLiveKitAgentName(),
          metadata: JSON.stringify({ participantIdentity: identity }),
        }),
      ],
    });
  }

  return {
    token: await token.toJwt(),
    url,
  };
}
