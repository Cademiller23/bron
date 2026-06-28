import type { Metadata } from "next";

import { FlightMap } from "@/components/flight-map";

export const metadata: Metadata = {
  title: "Flight Map | Lebronsseiur",
  description: "F1 calendar flight map and Grand Prix route visualization.",
};

export default function FlightPage() {
  return <FlightMap />;
}
