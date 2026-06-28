"use client";

import { Plane } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { Marker } from "react-map-gl/maplibre";

import {
  bearingBetween,
  pointAlongGreatCircle,
  type GeoPoint,
} from "@/lib/great-circle";

export type AirplaneFlightMapProps = {
  startTrigger: string | number | boolean;
  from: GeoPoint;
  to: GeoPoint;
  durationMs?: number;
  delayMs?: number;
  autoStart?: boolean;
  onComplete?: () => void;
};

export function AirplaneFlightMap({
  startTrigger,
  from,
  to,
  durationMs = 280,
  delayMs = 0,
  autoStart = false,
  onComplete,
}: AirplaneFlightMapProps) {
  const animationRef = useRef<number | null>(null);
  const delayRef = useRef<number | null>(null);
  const hasMountedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  const [progress, setProgress] = useState(0);
  const [isFlying, setIsFlying] = useState(false);

  const position = useMemo(() => pointAlongGreatCircle(from, to, progress), [from, progress, to]);
  const lookAhead = useMemo(
    () => pointAlongGreatCircle(from, to, Math.min(progress + 0.02, 1)),
    [from, progress, to],
  );
  const bearing = bearingBetween(
    { latitude: position.latitude, longitude: position.longitude },
    { latitude: lookAhead.latitude, longitude: lookAhead.longitude },
  );
  const endFadeProgress = progress > 0.9 ? (progress - 0.9) / 0.1 : 0;
  const opacity = 1 - endFadeProgress;
  const scale = (isFlying ? 1 : 0.92) * (1 - endFadeProgress * 0.72);

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    if (!hasMountedRef.current) {
      hasMountedRef.current = true;

      if (!autoStart) {
        return;
      }
    }

    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    if (delayRef.current) {
      window.clearTimeout(delayRef.current);
    }

    const run = () => {
      setProgress(0);
      setIsFlying(false);

      if (reduceMotion) {
        setProgress(1);
        onCompleteRef.current?.();
        return;
      }

      setIsFlying(true);
      const startedAt = performance.now();

      const tick = (time: number) => {
        const nextProgress = Math.min((time - startedAt) / durationMs, 1);
        setProgress(nextProgress);

        if (nextProgress < 1) {
          animationRef.current = requestAnimationFrame(tick);
          return;
        }

        setIsFlying(false);
        onCompleteRef.current?.();
      };

      animationRef.current = requestAnimationFrame(tick);
    };

    if (delayMs > 0) {
      delayRef.current = window.setTimeout(run, delayMs);
    } else {
      animationRef.current = requestAnimationFrame(run);
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }

      if (delayRef.current) {
        window.clearTimeout(delayRef.current);
      }
    };
  }, [autoStart, delayMs, durationMs, startTrigger]);

  if (progress <= 0) {
    return null;
  }

  return (
    <Marker longitude={position.longitude} latitude={position.latitude} anchor="center">
      <div
        className="pointer-events-none text-white"
        style={{
          opacity,
          transform: `rotate(${bearing - 45}deg) scale(${scale})`,
        }}
      >
        <Plane
          aria-hidden="true"
          className="size-6 fill-white stroke-white [filter:drop-shadow(0_1px_3px_rgb(0_0_0_/_0.85))]"
          strokeWidth={1.5}
        />
      </div>
    </Marker>
  );
}
