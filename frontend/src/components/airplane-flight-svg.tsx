"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type FlightPoint = {
  x: number;
  y: number;
};

export type AirplaneFlightSvgProps = {
  startTrigger: string | number | boolean;
  startX: number;
  startY: number;
  endX: number;
  endY: number;
  durationMs?: number;
  delayMs?: number;
  curveBend?: number;
  size?: number;
  autoStart?: boolean;
  pathLabel?: string;
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

function getQuadraticAngle(start: FlightPoint, control: FlightPoint, end: FlightPoint, progress: number) {
  const dx = 2 * (1 - progress) * (control.x - start.x) + 2 * progress * (end.x - control.x);
  const dy = 2 * (1 - progress) * (control.y - start.y) + 2 * progress * (end.y - control.y);

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

export function AirplaneFlightSvg({
  startTrigger,
  startX,
  startY,
  endX,
  endY,
  durationMs = 280,
  delayMs = 0,
  curveBend = -12,
  size = 2.8,
  autoStart = false,
  pathLabel = "Flight path",
  onComplete,
}: AirplaneFlightSvgProps) {
  const animationRef = useRef<number | null>(null);
  const delayRef = useRef<number | null>(null);
  const hasMountedRef = useRef(false);
  const onCompleteRef = useRef(onComplete);
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
  const angle = getQuadraticAngle(start, control, end, progress);
  const trailPath = getTrailPath(start, control, end, progress);
  const endFadeProgress = clamp((progress - 0.9) / 0.1, 0, 1);
  const planeOpacity = hasStarted ? 1 - endFadeProgress : 0;
  const planeScale = (isFlying ? 1 : 0.92) * (1 - endFadeProgress * 0.72);
  const trailOpacity = trailPath ? 0.82 * (1 - endFadeProgress) : 0;

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
    <g aria-label={pathLabel}>
      {trailPath ? (
        <path
          d={trailPath}
          className="fill-none stroke-stone-200/70"
          strokeWidth={0.35}
          strokeLinecap="round"
          style={{ opacity: trailOpacity }}
        />
      ) : null}

      <g
        transform={`translate(${plane.x} ${plane.y}) rotate(${angle + 45}) scale(${planeScale})`}
        style={{ opacity: planeOpacity }}
      >
        <path
          d={`M ${-size} 0 L ${size * 0.15} ${-size * 0.35} L ${size * 0.55} ${-size * 0.15} L ${size} 0 L ${size * 0.55} ${size * 0.15} L ${size * 0.15} ${size * 0.35} Z`}
          className="fill-white stroke-white/80"
          strokeWidth={0.12}
        />
      </g>
    </g>
  );
}

export {
  getControlPoint,
  getQuadraticPoint,
};
