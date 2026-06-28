import { type Circuit, haversineKm } from "@/lib/f1-circuits";

/**
 * Flight speed tuning
 *
 * Two ways to tweak speed:
 * 1. UI slider on the map (saved for the session; default below).
 * 2. Edit these constants — they control duration at speed = 1.0.
 *
 * Higher speed multiplier = faster flights (duration is divided by speed).
 * Lower LEG_* values = faster baseline. Higher = slower baseline.
 */
export const DEFAULT_FLIGHT_SPEED = 1;

export const FLIGHT_SPEED_MIN = 0.25;
export const FLIGHT_SPEED_MAX = 4;
export const FLIGHT_SPEED_STEP = 0.25;

/** Shortest hop duration at speed 1.0 (ms) */
export const LEG_MIN_MS = 600;

/** Longest hop duration at speed 1.0 (ms) */
export const LEG_MAX_MS = 2200;

/** Fixed time added to every leg at speed 1.0 (ms) */
export const LEG_BASE_MS = 500;

/** Extra ms per km of great-circle distance at speed 1.0 */
export const LEG_DISTANCE_FACTOR = 0.12;

export function legDurationMs(from: Circuit, to: Circuit, speed = DEFAULT_FLIGHT_SPEED) {
  const distanceKm = haversineKm(from, to);
  const baseDuration = Math.min(
    LEG_MAX_MS,
    Math.max(LEG_MIN_MS, LEG_BASE_MS + distanceKm * LEG_DISTANCE_FACTOR),
  );
  const clampedSpeed = Math.min(FLIGHT_SPEED_MAX, Math.max(FLIGHT_SPEED_MIN, speed));

  return Math.round(baseDuration / clampedSpeed);
}
