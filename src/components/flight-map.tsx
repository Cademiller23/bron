"use client";

import { PlaneTakeoff, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Map, { Layer, Source } from "react-map-gl/maplibre";
import type { Feature, FeatureCollection, LineString } from "geojson";
import "maplibre-gl/dist/maplibre-gl.css";

import { AirplaneFlightMap } from "@/components/airplane-flight-map";
import { CircuitMarker, type CircuitMarkerState } from "@/components/circuit-marker";
import { Button } from "@/components/ui/button";
import { circuits, formatRaceDate, getCircuit } from "@/lib/f1-circuits";
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

type OptimizedStop = {
  key: string;
  round: number;
  region: string;
  revenue: number;
  legCarbonKg: number;
};

type OptimizedData = {
  totalCarbonKg: number;
  totalRevenue: number;
  stops: OptimizedStop[];
  comparison?: {
    vsBaseline?: {
      revenueDeltaPct: number;
      carbonDeltaPct: number;
      revenueDelta: number;
    };
  };
};

function buildRouteGeoJson(
  legs: Array<{
    id: string;
    from: (typeof circuits)[number];
    to: (typeof circuits)[number];
    status: LegStatus;
  }>,
): FeatureCollection<LineString> {
  const features: Feature<LineString>[] = legs.map((leg) => ({
    type: "Feature",
    properties: { id: leg.id, status: leg.status },
    geometry: {
      type: "LineString",
      coordinates: greatCirclePoints(
        { latitude: leg.from.latitude, longitude: leg.from.longitude },
        { latitude: leg.to.latitude, longitude: leg.to.longitude },
      ),
    },
  }));

  return { type: "FeatureCollection", features };
}

export function FlightMap() {
  const pauseTimeoutRef = useRef<number | null>(null);
  const [flightRun, setFlightRun] = useState(0);
  const [activeLegIndex, setActiveLegIndex] = useState<number | null>(null);
  const [reachedStopIndex, setReachedStopIndex] = useState(0);
  const [displayStopIndex, setDisplayStopIndex] = useState(0);
  const [isPausedAtStop, setIsPausedAtStop] = useState(false);
  const [flightSpeed, setFlightSpeed] = useState(DEFAULT_FLIGHT_SPEED);
  const [optimized, setOptimized] = useState<OptimizedData | null>(null);

  // Load the recorded optimized calendar; fall back to static circuit order.
  useEffect(() => {
    let cancelled = false;
    fetch("/data/f1_optimized.json", { cache: "no-store" })
      .then((res) => (res.ok ? res.json() : Promise.reject(res.status)))
      .then((data: OptimizedData) => {
        if (!cancelled && Array.isArray(data?.stops) && data.stops.length > 0) {
          setOptimized(data);
        }
      })
      .catch((error) => console.warn("flight-map: using static order:", error));
    return () => {
      cancelled = true;
    };
  }, []);

  // Route = optimized order if loaded, else the static calendar.
  const raceRoute = useMemo(() => {
    if (optimized) {
      return optimized.stops.map((stop) => {
        const circuit = getCircuit(stop.key);
        return { circuit, revenue: stop.revenue, legCarbonKg: stop.legCarbonKg };
      });
    }
    return circuits.map((circuit) => ({ circuit, revenue: 0, legCarbonKg: 0 }));
  }, [optimized]);

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
  const displayedStop = raceRoute[displayStopIndex];
  const displayedCircuit = displayedStop.circuit;

  // Cumulative carbon + revenue up to the displayed stop.
  const cumulative = useMemo(() => {
    let carbon = 0;
    let revenue = 0;
    for (let i = 0; i <= displayStopIndex && i < raceRoute.length; i++) {
      carbon += raceRoute[i].legCarbonKg ?? 0;
      revenue += raceRoute[i].revenue ?? 0;
    }
    return { carbon, revenue };
  }, [displayStopIndex, raceRoute]);

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
      return { id: `${leg.from.key}-${leg.to.key}`, from: leg.from, to: leg.to, status };
    });
    return buildRouteGeoJson(legs);
  }, [activeLegIndex, hasStarted, raceLegs, reachedStopIndex]);

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
        if (nextStop < raceLegs.length) {
          setActiveLegIndex(nextStop);
        }
      });
      return null;
    });
  }, [pauseAtStop, raceLegs.length]);

  const activeLeg = activeLegIndex !== null ? raceLegs[activeLegIndex] : null;
  const vs = optimized?.comparison?.vsBaseline;

  return (
    <main className="fixed inset-0 overflow-hidden bg-[#0a0f14]">
      <Map
        initialViewState={{ longitude: 10, latitude: 22, zoom: 1.35 }}
        style={{ width: "100%", height: "100%" }}
        mapStyle={MAP_STYLE}
        attributionControl={false}
        renderWorldCopies={false}
      >
        <Source id="race-routes" type="geojson" data={routeGeoJson}>
          <Layer
            id="race-routes-preview"
            type="line"
            filter={["==", ["get", "status"], "preview"]}
            paint={{ "line-color": "rgba(255,255,255,0.14)", "line-width": 1.5, "line-dasharray": [2, 2] }}
          />
          <Layer
            id="race-routes-upcoming"
            type="line"
            filter={["==", ["get", "status"], "upcoming"]}
            paint={{ "line-color": "rgba(255,255,255,0.08)", "line-width": 1.2, "line-dasharray": [2, 2] }}
          />
          <Layer
            id="race-routes-completed"
            type="line"
            filter={["==", ["get", "status"], "completed"]}
            paint={{ "line-color": "rgba(52,211,153,0.55)", "line-width": 2 }}
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

        {activeLeg ? (
          <AirplaneFlightMap
            key={`${flightRun}-${activeLegIndex}`}
            startTrigger={flightRun}
            autoStart
            from={{ latitude: activeLeg.from.latitude, longitude: activeLeg.from.longitude }}
            to={{ latitude: activeLeg.to.latitude, longitude: activeLeg.to.longitude }}
            durationMs={legDurationMs(activeLeg.from, activeLeg.to, flightSpeed)}
            onComplete={handleLegComplete}
          />
        ) : null}
      </Map>

      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_35%,rgba(0,0,0,0.45)_100%)]" />

      <div
        className={`absolute left-1/2 top-4 z-10 w-[min(92vw,32rem)] -translate-x-1/2 rounded-xl border border-white/15 bg-black/70 px-5 py-3 text-center shadow-lg shadow-black/40 backdrop-blur-sm transition-opacity duration-200 ${isPausedAtStop ? "opacity-100" : "opacity-95"}`}
      >
        <p className="text-[11px] font-medium uppercase tracking-[0.2em] text-white/50">
          Lebronsseiur · Round {displayStopIndex + 1} · {displayedCircuit.city}
        </p>
        <p className="mt-1 text-base font-semibold text-white sm:text-lg">{displayedCircuit.grandPrix}</p>
        <p className="mt-0.5 text-sm text-red-300">{formatRaceDate(displayedCircuit.raceDate)}</p>
      </div>

      {optimized ? (
        <div className="absolute right-4 top-4 z-10 w-60 rounded-xl border border-white/15 bg-black/70 px-4 py-3 shadow-lg shadow-black/40 backdrop-blur-sm">
          <p className="text-[11px] font-medium uppercase tracking-[0.2em] text-emerald-300/80">
            Swarm-Optimized Calendar
          </p>
          <div className="mt-2 space-y-1.5 text-sm">
            <div className="flex justify-between">
              <span className="text-white/55">Revenue (cum.)</span>
              <span className="tabular-nums font-semibold text-white">{cumulative.revenue.toFixed(0)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/55">Carbon (cum.)</span>
              <span className="tabular-nums font-semibold text-white">
                {(cumulative.carbon / 1_000_000).toFixed(1)}M kg
              </span>
            </div>
            <div className="my-1 h-px bg-white/10" />
            <div className="flex justify-between">
              <span className="text-white/55">Total revenue</span>
              <span className="tabular-nums font-semibold text-emerald-300">
                {optimized.totalRevenue.toFixed(0)}
              </span>
            </div>
            {vs ? (
              <div className="flex justify-between">
                <span className="text-white/55">vs baseline</span>
                <span className="tabular-nums font-semibold text-emerald-300">
                  +{vs.revenueDeltaPct}% rev
                </span>
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="absolute left-4 top-4 z-10 flex flex-col gap-2">
        <div className="rounded-lg border border-white/15 bg-black/65 p-1.5 shadow-lg shadow-black/30 backdrop-blur-sm">
          <Button onClick={handleStart}>
            {flightRun === 0 ? <PlaneTakeoff data-icon="inline-start" /> : <RotateCcw data-icon="inline-start" />}
            {flightRun === 0 ? "Start season" : "Replay season"}
          </Button>
        </div>

        <div className="w-52 rounded-lg border border-white/15 bg-black/65 px-3 py-2.5 shadow-lg shadow-black/30 backdrop-blur-sm">
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
        </div>
      </div>
    </main>
  );
}
