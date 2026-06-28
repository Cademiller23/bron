"use client";

import { useEffect, useRef, useState } from "react";

import {
  computeSeasonTotals,
  formatCarbonKg,
  formatCompactUsd,
  getEconomicsProgress,
  type SeasonEconomicsTotals,
} from "@/lib/calendar-economics";
import { type CalendarStop } from "@/lib/f1-circuits";
import { cn } from "@/lib/utils";

type TrailTheme = "red" | "green";

const ACCENT_STYLES: Record<
  TrailTheme,
  { border: string; label: string; value: string; bar: string }
> = {
  red: {
    border: "border-red-500/25",
    label: "text-red-200/70",
    value: "text-red-100",
    bar: "bg-red-500/80",
  },
  green: {
    border: "border-emerald-500/25",
    label: "text-emerald-200/70",
    value: "text-emerald-100",
    bar: "bg-emerald-500/80",
  },
};

function useAnimatedTotals(target: SeasonEconomicsTotals, durationMs = 700) {
  const frameRef = useRef<number | null>(null);
  const startRef = useRef(target);
  const startTimeRef = useRef(0);
  const [display, setDisplay] = useState(target);

  useEffect(() => {
    startRef.current = display;
    startTimeRef.current = performance.now();

    const tick = (now: number) => {
      const elapsed = now - startTimeRef.current;
      const progress = Math.min(elapsed / durationMs, 1);
      const eased = 1 - (1 - progress) ** 3;

      setDisplay({
        costUsd: startRef.current.costUsd + (target.costUsd - startRef.current.costUsd) * eased,
        carbonKg:
          startRef.current.carbonKg + (target.carbonKg - startRef.current.carbonKg) * eased,
        revenueUsd:
          startRef.current.revenueUsd + (target.revenueUsd - startRef.current.revenueUsd) * eased,
        visitedRaces: target.visitedRaces,
        completedLegs: target.completedLegs,
      });

      if (progress < 1) {
        frameRef.current = requestAnimationFrame(tick);
      }
    };

    frameRef.current = requestAnimationFrame(tick);

    return () => {
      if (frameRef.current !== null) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, [
    durationMs,
    target.carbonKg,
    target.completedLegs,
    target.costUsd,
    target.revenueUsd,
    target.visitedRaces,
  ]);

  return display;
}

type CalendarStatsLegendProps = {
  calendarLabel: string;
  calendarStops: CalendarStop[];
  trailTheme: TrailTheme;
  hasStarted: boolean;
  reachedStopIndex: number;
  activeLegIndex: number | null;
};

export function CalendarStatsLegend({
  calendarLabel,
  calendarStops,
  trailTheme,
  hasStarted,
  reachedStopIndex,
  activeLegIndex,
}: CalendarStatsLegendProps) {
  const accent = ACCENT_STYLES[trailTheme];
  const progress = getEconomicsProgress(hasStarted, reachedStopIndex, activeLegIndex);
  const totals = computeSeasonTotals(
    calendarStops,
    progress.visitedRaces,
    progress.completedLegs,
  );
  const animated = useAnimatedTotals(totals);
  const seasonProgress =
    calendarStops.length > 0 ? animated.visitedRaces / calendarStops.length : 0;

  return (
    <div
      className={cn(
        "absolute inset-x-3 bottom-3 z-10 rounded-xl border bg-black/75 px-4 py-3 shadow-lg shadow-black/40 backdrop-blur-sm",
        accent.border,
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-white/45">
          {calendarLabel} · season totals
        </p>
        <p className="text-[10px] tabular-nums text-white/40">
          {animated.visitedRaces}/{calendarStops.length} GPs
        </p>
      </div>

      <div className="mt-3 grid grid-cols-3 gap-3">
        <StatBlock
          label="Carbon"
          value={formatCarbonKg(animated.carbonKg)}
          hint="Freight emissions"
          accent={accent}
        />
        <StatBlock
          label="Cost"
          value={formatCompactUsd(animated.costUsd)}
          hint="Hosting + logistics"
          accent={accent}
        />
        <StatBlock
          label="Revenue"
          value={formatCompactUsd(animated.revenueUsd)}
          hint="Ticket sales"
          accent={accent}
        />
      </div>

      <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-white/10">
        <div
          className={cn("h-full rounded-full transition-[width] duration-500 ease-out", accent.bar)}
          style={{ width: `${seasonProgress * 100}%` }}
        />
      </div>

      <p className="mt-2 text-[10px] text-white/35">
        Placeholder values — will sync with backend when connected.
      </p>
    </div>
  );
}

function StatBlock({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: string;
  hint: string;
  accent: (typeof ACCENT_STYLES)[TrailTheme];
}) {
  return (
    <div className="min-w-0">
      <p className={cn("text-[10px] font-medium uppercase tracking-[0.14em]", accent.label)}>
        {label}
      </p>
      <p className={cn("mt-1 truncate text-base font-semibold tabular-nums sm:text-lg", accent.value)}>
        {value}
      </p>
      <p className="mt-0.5 truncate text-[10px] text-white/35">{hint}</p>
    </div>
  );
}
