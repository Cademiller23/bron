import { Marker } from "react-map-gl/maplibre";

import { cn } from "@/lib/utils";

export type CircuitMarkerState = "upcoming" | "current" | "visited";

type CircuitMarkerProps = {
  name: string;
  city: string;
  latitude: number;
  longitude: number;
  state: CircuitMarkerState;
};

export function CircuitMarker({
  name,
  city,
  latitude,
  longitude,
  state,
}: CircuitMarkerProps) {
  const isCurrent = state === "current";
  const isVisited = state === "visited";

  return (
    <Marker longitude={longitude} latitude={latitude} anchor="center">
      <div
        aria-label={`${name}, ${city}`}
        className="pointer-events-none relative flex items-center justify-center"
        title={`${name}, ${city}`}
      >
        {isCurrent ? (
          <span
            aria-hidden="true"
            className="absolute size-5 animate-ping rounded-full bg-red-500/40"
          />
        ) : null}

        <span
          aria-hidden="true"
          className={cn(
            "relative size-3.5 rounded-full border-2 shadow-md",
            isCurrent && "border-red-400 bg-red-500 shadow-red-500/50",
            isVisited && "border-emerald-400/80 bg-emerald-500/90 shadow-emerald-500/30",
            !isCurrent && !isVisited && "border-white/50 bg-slate-700/90 shadow-black/40",
          )}
        />

        {isCurrent ? (
          <span className="absolute left-1/2 top-[calc(100%+6px)] -translate-x-1/2 whitespace-nowrap rounded bg-black/75 px-1.5 py-0.5 text-[10px] font-medium text-white shadow-lg backdrop-blur-sm">
            {city}
          </span>
        ) : null}
      </div>
    </Marker>
  );
}
