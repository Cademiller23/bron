"use client";

import { Plane } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { cn } from "@/lib/utils";

type FlightPoint = {
  x: number;
  y: number;
};

export type AirplaneFlightProps = {
  startTrigger: string | number | boolean;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  durationMs?: number;
  delayMs?: number;
  curveBend?: number;
  size?: number;
  showPath?: boolean;
  autoStart?: boolean;
  idleVisible?: boolean;
  pathLabel?: string;
  className?: string;
  onComplete?: () => void;
};

function getControlPoint(start: FlightPoint, end: FlightPoint, curveBend: number) {
  const midpoint = {
    x: (start.x + end.x) / 2,
    y: (start.y + end.y) / 2,
  };
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const length = Math.hypot(dx, dy) || 1;

  return {
    x: midpoint.x + (-dy / length) * curveBend,
    y: midpoint.y + (dx / length) * curveBend,
  };
}

function getQuadraticPoint(
  start: FlightPoint,
  control: FlightPoint,
  end: FlightPoint,
  progress: number,
) {
  const inverse = 1 - progress;

  return {
    x: inverse * inverse * start.x + 2 * inverse * progress * control.x + progress * progress * end.x,
    y: inverse * inverse * start.y + 2 * inverse * progress * control.y + progress * progress * end.y,
  };
}

function getQuadraticAngle(
  start: FlightPoint,
  control: FlightPoint,
  end: FlightPoint,
  progress: number,
  viewportWidth = 1,
  viewportHeight = 1,
) {
  const dx =
    (2 * (1 - progress) * (control.x - start.x) + 2 * progress * (end.x - control.x)) *
    viewportWidth;
  const dy =
    (2 * (1 - progress) * (control.y - start.y) + 2 * progress * (end.y - control.y)) *
    viewportHeight;

  return (Math.atan2(dy, dx) * 180) / Math.PI;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function getTrailPath(
  start: FlightPoint,
  control: FlightPoint,
  end: FlightPoint,
  progress: number,
  trailLength = 0.26,
) {
  if (progress <= 0 || progress >= 1) {
    return "";
  }

  const tailProgress = clamp(progress - trailLength, 0, 1);
  const segmentCount = 8;
  const points = Array.from({ length: segmentCount + 1 }, (_, index) => {
    const stepProgress = tailProgress + ((progress - tailProgress) * index) / segmentCount;

    return getQuadraticPoint(start, control, end, stepProgress);
  });

  return points
    .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
    .join(" ");
}

export function AirplaneFlight({
  startTrigger,
  startX,
  startY,
  endX,
  endY,
  durationMs = 280,
  delayMs = 0,
  curveBend = -16,
  size = 28,
  showPath = true,
  autoStart = false,
  idleVisible = true,
  pathLabel = "Flight path",
  className,
  onComplete,
}: AirplaneFlightProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const delayRef = useRef<number | null>(null);
  const hasMountedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
  const [viewport, setViewport] = useState({ width: 1, height: 1 });
  const [hasStarted, setHasStarted] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isFlying, setIsFlying] = useState(false);

  const start = useMemo(() => ({ x: startX, y: startY }), [startX, startY]);
  const end = useMemo(() => ({ x: endX, y: endY }), [endX, endY]);
  const control = useMemo(
    () => getControlPoint(start, end, curveBend),
    [curveBend, end, start],
  );
  const plane = getQuadraticPoint(start, control, end, progress);
  const angle = getQuadraticAngle(start, control, end, progress, viewport.width, viewport.height);
  const trailPath = getTrailPath(start, control, end, progress);
  const endFadeProgress = clamp((progress - 0.9) / 0.1, 0, 1);
  const planeOpacity = idleVisible || hasStarted ? 1 - endFadeProgress : 0;
  const planeScale = (isFlying ? 1 : 0.92) * (1 - endFadeProgress * 0.72);
  const trailOpacity = trailPath ? 0.82 * (1 - endFadeProgress) : 0;

  useEffect(() => {
    onCompleteRef.current = onComplete;
  }, [onComplete]);

  useEffect(() => {
    const container = containerRef.current;

    if (!container) {
      return;
    }

    const updateViewport = () => {
      const rect = container.getBoundingClientRect();

      setViewport({
        width: rect.width || 1,
        height: rect.height || 1,
      });
    };

    updateViewport();

    const resizeObserver = new ResizeObserver(updateViewport);
    resizeObserver.observe(container);

    return () => resizeObserver.disconnect();
  }, []);

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
      setHasStarted(true);
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

  return (
    <div
      ref={containerRef}
      className={cn("pointer-events-none absolute inset-0", className)}
      aria-label={pathLabel}
    >
      {showPath ? (
        <svg className="absolute inset-0 size-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <path
            d={trailPath}
            className="fill-none stroke-stone-200/70"
            strokeWidth="0.24"
            strokeLinecap="round"
            style={{ opacity: trailOpacity }}
          />
        </svg>
      ) : null}

      <div
        data-testid="airplane-flight-plane"
        className="absolute text-white"
        style={{
          left: `${plane.x}%`,
          top: `${plane.y}%`,
          width: size,
          height: size,
          opacity: planeOpacity,
          transform: `translate(-50%, -50%) rotate(${angle + 45}deg) scale(${planeScale})`,
          transition: isFlying ? "opacity 60ms ease-out" : "opacity 60ms ease-out, transform 60ms ease-out",
          willChange: "left, top, transform, opacity",
        }}
      >
        <Plane
          aria-hidden="true"
          className="size-full fill-white stroke-white [filter:drop-shadow(0_1px_3px_rgb(0_0_0_/_0.85))]"
          strokeWidth={1.5}
        />
      </div>
    </div>
  );
}
