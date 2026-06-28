"use client";

import { PlaneTakeoff, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Map, { Layer, Source } from "react-map-gl/maplibre";
import type { Feature, FeatureCollection, LineString } from "geojson";
import "maplibre-gl/dist/maplibre-gl.css";

import { AirplaneFlightMap } from "@/components/airplane-flight-map";
import { CalendarStatsLegend } from "@/components/calendar-stats-legend";
import { CircuitMarker, type CircuitMarkerState } from "@/components/circuit-marker";
import { Button } from "@/components/ui/button";
import {
  type CalendarStop,
  buildCalendarStops,
  currentCalendarKeys,
  formatRaceDate,
  proposedCalendarKeys,
  proposedCalendarWeeks,
} from "@/lib/f1-circuits";
import { greatCirclePoints } from "@/lib/great-circle";
import {
  DEFAULT_FLIGHT_SPEED,
  FLIGHT_SPEED_MAX,
  FLIGHT_SPEED_MIN,
  FLIGHT_SPEED_STEP,
  legDurationMs,
} from "@/lib/flight-speed";

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
const STOP_PAUSE_MS = 500;

type LegStatus = "preview" | "completed" | "upcoming";
type TrailTheme = "red" | "green";

const COMPLETED_TRAIL_COLORS: Record<TrailTheme, string> = {
  red: "rgba(239, 68, 68, 0.55)",
  green: "rgba(52, 211, 153, 0.55)",
};

function buildRouteGeoJson(
  legs: Array<{
    id: string;
    from: CalendarStop;
    to: CalendarStop;
    status: LegStatus;
  }>,
): FeatureCollection<LineString> {
  const features: Feature<LineString>[] = legs.map((leg) => ({
    type: "Feature",
    properties: {
      id: leg.id,
      status: leg.status,
    },
    geometry: {
      type: "LineString",
      coordinates: greatCirclePoints(
        { latitude: leg.from.latitude, longitude: leg.from.longitude },
        { latitude: leg.to.latitude, longitude: leg.to.longitude },
      ),
    },
  }));

  return {
    type: "FeatureCollection",
    features,
  };
}

type FlightMapPanelProps = {
  mapId: string;
  calendarLabel: string;
  calendarStops: CalendarStop[];
  trailTheme: TrailTheme;
  flightRun: number;
  activeLegIndex: number | null;
  activeLegDurationMs: number | null;
  reachedStopIndex: number;
  displayStopIndex: number;
  isPausedAtStop: boolean;
  drivesPlayback: boolean;
  onLegComplete: () => void;
};

function FlightMapPanel({
  mapId,
  calendarLabel,
  calendarStops,
  trailTheme,
  flightRun,
  activeLegIndex,
  activeLegDurationMs,
  reachedStopIndex,
  displayStopIndex,
  isPausedAtStop,
  drivesPlayback,
  onLegComplete,
}: FlightMapPanelProps) {
  const raceRoute = useMemo(
    () => calendarStops.map((circuit) => ({ circuit })),
    [calendarStops],
  );

  const raceLegs = useMemo(
    () =>
      raceRoute.slice(0, -1).map((stop, index) => ({
        from: stop.circuit,
        to: raceRoute[index + 1].circuit,
      })),
    [raceRoute],
  );

  const isFlying = activeLegIndex !== null;
  const hasStarted = flightRun > 0;
  const displayedCircuit = raceRoute[displayStopIndex].circuit;

  const getMarkerState = useCallback(
    (index: number): CircuitMarkerState => {
      if (!hasStarted) {
        return index === 0 ? "current" : "upcoming";
      }

      if (isFlying && activeLegIndex !== null) {
        if (index <= activeLegIndex) return "visited";
        if (index === activeLegIndex + 1) return "current";
        return "upcoming";
      }

      if (index < reachedStopIndex) return "visited";
      if (index === reachedStopIndex) return "current";
      return "upcoming";
    },
    [activeLegIndex, hasStarted, isFlying, reachedStopIndex],
  );

  const routeGeoJson = useMemo(() => {
    const legs = raceLegs.map((leg, index) => {
      let status: LegStatus = "upcoming";

      if (!hasStarted) {
        status = "preview";
      } else if (activeLegIndex !== null ? index < activeLegIndex : reachedStopIndex > index) {
        status = "completed";
      }

      return {
        id: `${leg.from.key}-${leg.to.key}`,
        from: leg.from,
        to: leg.to,
        status,
      };
    });

    return buildRouteGeoJson(legs);
  }, [activeLegIndex, hasStarted, raceLegs, reachedStopIndex]);

  const activeLeg = activeLegIndex !== null ? raceLegs[activeLegIndex] : null;

  return (
    <section className="relative min-h-0 min-w-0 flex-1 overflow-hidden border-white/10 [&:not(:first-child)]:border-l">
      <Map
        id={mapId}
        initialViewState={{
          longitude: 10,
          latitude: 22,
          zoom: 1.35,
        }}
        style={{ width: "100%", height: "100%" }}
        mapStyle={MAP_STYLE}
        attributionControl={false}
        renderWorldCopies={false}
      >
        <Source id={`${mapId}-race-routes`} type="geojson" data={routeGeoJson}>
          <Layer
            id={`${mapId}-race-routes-preview`}
            type="line"
            filter={["==", ["get", "status"], "preview"]}
            paint={{
              "line-color": "rgba(255,255,255,0.14)",
              "line-width": 1.5,
              "line-dasharray": [2, 2],
            }}
          />
          <Layer
            id={`${mapId}-race-routes-upcoming`}
            type="line"
            filter={["==", ["get", "status"], "upcoming"]}
            paint={{
              "line-color": "rgba(255,255,255,0.08)",
              "line-width": 1.2,
              "line-dasharray": [2, 2],
            }}
          />
          <Layer
            id={`${mapId}-race-routes-completed`}
            type="line"
            filter={["==", ["get", "status"], "completed"]}
            paint={{
              "line-color": COMPLETED_TRAIL_COLORS[trailTheme],
              "line-width": 2,
            }}
          />
        </Source>

        {raceRoute.map(({ circuit }, index) => (
          <CircuitMarker
            key={circuit.key}
            name={circuit.name}
            city={circuit.city}
            latitude={circuit.latitude}
            longitude={circuit.longitude}
            state={getMarkerState(index)}
          />
        ))}

        {activeLeg && activeLegDurationMs !== null ? (
          <AirplaneFlightMap
            key={`${flightRun}-${activeLegIndex}`}
            startTrigger={flightRun}
            autoStart
            from={{ latitude: activeLeg.from.latitude, longitude: activeLeg.from.longitude }}
            to={{ latitude: activeLeg.to.latitude, longitude: activeLeg.to.longitude }}
            durationMs={activeLegDurationMs}
            onComplete={drivesPlayback ? onLegComplete : undefined}
          />
        ) : null}
      </Map>

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_35%,rgba(0,0,0,0.45)_100%)]" />

      <div
        className={`absolute left-1/2 top-4 z-10 w-[min(92%,20rem)] -translate-x-1/2 rounded-xl border border-white/15 bg-black/70 px-4 py-3 text-center shadow-lg shadow-black/40 backdrop-blur-sm transition-opacity duration-200 ${isPausedAtStop ? "opacity-100" : "opacity-95"}`}
      >
        <p className="text-[10px] font-medium uppercase tracking-[0.18em] text-white/45">
          {calendarLabel}
        </p>
        <p className="mt-1 text-[11px] font-medium uppercase tracking-[0.2em] text-white/50">
          Lebronsseiur · Wk {displayedCircuit.week} · Round {displayStopIndex + 1} ·{" "}
          {displayedCircuit.city}
        </p>
        <p className="mt-1 text-sm font-semibold text-white sm:text-base">{displayedCircuit.grandPrix}</p>
        <p className="mt-0.5 text-xs text-red-300 sm:text-sm">{formatRaceDate(displayedCircuit.raceDate)}</p>
      </div>

      <CalendarStatsLegend
        calendarLabel={calendarLabel}
        calendarStops={calendarStops}
        trailTheme={trailTheme}
        hasStarted={hasStarted}
        reachedStopIndex={reachedStopIndex}
        activeLegIndex={activeLegIndex}
      />
    </section>
  );
}

export function FlightMap() {
  const pauseTimeoutRef = useRef<number | null>(null);
  const [flightRun, setFlightRun] = useState(0);
  const [activeLegIndex, setActiveLegIndex] = useState<number | null>(null);
  const [reachedStopIndex, setReachedStopIndex] = useState(0);
  const [displayStopIndex, setDisplayStopIndex] = useState(0);
  const [isPausedAtStop, setIsPausedAtStop] = useState(false);
  const [flightSpeed, setFlightSpeed] = useState(DEFAULT_FLIGHT_SPEED);

  const currentCalendar = useMemo(() => buildCalendarStops(currentCalendarKeys), []);
  const proposedCalendar = useMemo(
    () => buildCalendarStops(proposedCalendarKeys, proposedCalendarWeeks),
    [],
  );
  const legCount = currentCalendar.length - 1;

  const activeLegDurationMs = useMemo(() => {
    if (activeLegIndex === null) {
      return null;
    }

    const currentLeg = {
      from: currentCalendar[activeLegIndex],
      to: currentCalendar[activeLegIndex + 1],
    };
    const proposedLeg = {
      from: proposedCalendar[activeLegIndex],
      to: proposedCalendar[activeLegIndex + 1],
    };

    return Math.max(
      legDurationMs(currentLeg.from, currentLeg.to, flightSpeed),
      legDurationMs(proposedLeg.from, proposedLeg.to, flightSpeed),
    );
  }, [activeLegIndex, currentCalendar, flightSpeed, proposedCalendar]);

  const clearPauseTimeout = useCallback(() => {
    if (pauseTimeoutRef.current !== null) {
      window.clearTimeout(pauseTimeoutRef.current);
      pauseTimeoutRef.current = null;
    }
  }, []);

  const pauseAtStop = useCallback(
    (stopIndex: number, onResume: () => void) => {
      clearPauseTimeout();
      setDisplayStopIndex(stopIndex);
      setIsPausedAtStop(true);
      pauseTimeoutRef.current = window.setTimeout(() => {
        setIsPausedAtStop(false);
        onResume();
      }, STOP_PAUSE_MS);
    },
    [clearPauseTimeout],
  );

  useEffect(() => clearPauseTimeout, [clearPauseTimeout]);

  const handleStart = () => {
    clearPauseTimeout();
    setFlightRun((run) => run + 1);
    setReachedStopIndex(0);
    setActiveLegIndex(null);
    pauseAtStop(0, () => setActiveLegIndex(0));
  };

  const handleLegComplete = useCallback(() => {
    setActiveLegIndex((currentLeg) => {
      if (currentLeg === null) return null;

      const nextStop = currentLeg + 1;
      setReachedStopIndex(nextStop);

      pauseAtStop(nextStop, () => {
        if (nextStop < legCount) {
          setActiveLegIndex(nextStop);
        }
      });

      return null;
    });
  }, [legCount, pauseAtStop]);

  const sharedPanelProps = {
    flightRun,
    activeLegIndex,
    activeLegDurationMs,
    reachedStopIndex,
    displayStopIndex,
    isPausedAtStop,
    onLegComplete: handleLegComplete,
  };

  return (
    <main className="fixed inset-0 flex overflow-hidden bg-[#0a0f14]">
      <FlightMapPanel
        mapId="current-calendar"
        calendarLabel="Current calendar"
        calendarStops={currentCalendar}
        trailTheme="red"
        drivesPlayback
        {...sharedPanelProps}
      />

      <FlightMapPanel
        mapId="proposed-calendar"
        calendarLabel="Proposed calendar"
        calendarStops={proposedCalendar}
        trailTheme="green"
        drivesPlayback={false}
        {...sharedPanelProps}
      />

      <div className="pointer-events-none absolute left-4 top-4 z-20 flex flex-col gap-2">
        <div className="pointer-events-auto rounded-lg border border-white/15 bg-black/65 p-1.5 shadow-lg shadow-black/30 backdrop-blur-sm">
          <Button onClick={handleStart}>
            {flightRun === 0 ? (
              <PlaneTakeoff data-icon="inline-start" />
            ) : (
              <RotateCcw data-icon="inline-start" />
            )}
            {flightRun === 0 ? "Start season" : "Replay season"}
          </Button>
        </div>

        <div className="pointer-events-auto w-52 rounded-lg border border-white/15 bg-black/65 px-3 py-2.5 shadow-lg shadow-black/30 backdrop-blur-sm">
          <div className="flex items-center justify-between gap-2">
            <label htmlFor="flight-speed" className="text-xs font-medium text-white/80">
              Flight speed
            </label>
            <span className="text-xs tabular-nums text-white/55">{flightSpeed.toFixed(2)}×</span>
          </div>
          <input
            id="flight-speed"
            type="range"
            min={FLIGHT_SPEED_MIN}
            max={FLIGHT_SPEED_MAX}
            step={FLIGHT_SPEED_STEP}
            value={flightSpeed}
            onChange={(event) => setFlightSpeed(Number(event.target.value))}
            className="mt-2 w-full accent-red-500"
          />
          <p className="mt-1.5 text-[10px] leading-snug text-white/40">
            Slide left = slower, right = faster. Edit defaults in{" "}
            <code className="text-white/55">src/lib/flight-speed.ts</code>
          </p>
        </div>
      </div>
    </main>
  );
}
