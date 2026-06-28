export type Circuit = {
  key: string;
  name: string;
  city: string;
  country: string;
  grandPrix: string;
  raceDate: string;
  latitude: number;
  longitude: number;
};

export const circuits = [
  { key: "melbourne", name: "Albert Park", city: "Melbourne", country: "Australia", grandPrix: "Australian Grand Prix", raceDate: "2026-03-08", latitude: -37.8497, longitude: 144.968 },
  { key: "shanghai", name: "Shanghai Intl", city: "Shanghai", country: "China", grandPrix: "Chinese Grand Prix", raceDate: "2026-03-15", latitude: 31.3389, longitude: 121.22 },
  { key: "suzuka", name: "Suzuka", city: "Suzuka", country: "Japan", grandPrix: "Japanese Grand Prix", raceDate: "2026-03-29", latitude: 34.8431, longitude: 136.541 },
  { key: "sakhir", name: "Bahrain Intl", city: "Sakhir", country: "Bahrain", grandPrix: "Bahrain Grand Prix", raceDate: "2026-04-12", latitude: 26.0325, longitude: 50.5106 },
  { key: "jeddah", name: "Jeddah Corniche", city: "Jeddah", country: "Saudi Arabia", grandPrix: "Saudi Arabian Grand Prix", raceDate: "2026-04-19", latitude: 21.6319, longitude: 39.1044 },
  { key: "miami", name: "Miami Intl Autodrome", city: "Miami", country: "USA", grandPrix: "Miami Grand Prix", raceDate: "2026-05-03", latitude: 25.9581, longitude: -80.2389 },
  { key: "montreal", name: "Gilles Villeneuve", city: "Montreal", country: "Canada", grandPrix: "Canadian Grand Prix", raceDate: "2026-05-24", latitude: 45.5, longitude: -73.5228 },
  { key: "monaco", name: "Circuit de Monaco", city: "Monte Carlo", country: "Monaco", grandPrix: "Monaco Grand Prix", raceDate: "2026-06-07", latitude: 43.7347, longitude: 7.4206 },
  { key: "barcelona", name: "Catalunya", city: "Barcelona", country: "Spain", grandPrix: "Spanish Grand Prix", raceDate: "2026-06-14", latitude: 41.57, longitude: 2.2611 },
  { key: "spielberg", name: "Red Bull Ring", city: "Spielberg", country: "Austria", grandPrix: "Austrian Grand Prix", raceDate: "2026-06-28", latitude: 47.2197, longitude: 14.7647 },
  { key: "silverstone", name: "Silverstone", city: "Silverstone", country: "UK", grandPrix: "British Grand Prix", raceDate: "2026-07-05", latitude: 52.0786, longitude: -1.0169 },
  { key: "budapest", name: "Hungaroring", city: "Budapest", country: "Hungary", grandPrix: "Hungarian Grand Prix", raceDate: "2026-07-26", latitude: 47.5789, longitude: 19.2486 },
  { key: "spa", name: "Spa-Francorchamps", city: "Stavelot", country: "Belgium", grandPrix: "Belgian Grand Prix", raceDate: "2026-07-19", latitude: 50.4372, longitude: 5.9714 },
  { key: "zandvoort", name: "Zandvoort", city: "Zandvoort", country: "Netherlands", grandPrix: "Dutch Grand Prix", raceDate: "2026-08-23", latitude: 52.3888, longitude: 4.5409 },
  { key: "monza", name: "Monza", city: "Monza", country: "Italy", grandPrix: "Italian Grand Prix", raceDate: "2026-09-06", latitude: 45.6156, longitude: 9.2811 },
  { key: "madrid", name: "Madring (IFEMA)", city: "Madrid", country: "Spain", grandPrix: "Madrid Grand Prix", raceDate: "2026-09-13", latitude: 40.465, longitude: -3.616 },
  { key: "baku", name: "Baku City", city: "Baku", country: "Azerbaijan", grandPrix: "Azerbaijan Grand Prix", raceDate: "2026-09-26", latitude: 40.3725, longitude: 49.8533 },
  { key: "singapore", name: "Marina Bay", city: "Singapore", country: "Singapore", grandPrix: "Singapore Grand Prix", raceDate: "2026-10-11", latitude: 1.2914, longitude: 103.864 },
  { key: "austin", name: "Circuit of Americas", city: "Austin", country: "USA", grandPrix: "United States Grand Prix", raceDate: "2026-10-25", latitude: 30.1328, longitude: -97.6411 },
  { key: "mexico_city", name: "Hermanos Rodriguez", city: "Mexico City", country: "Mexico", grandPrix: "Mexico City Grand Prix", raceDate: "2026-11-01", latitude: 19.4042, longitude: -99.0907 },
  { key: "sao_paulo", name: "Interlagos", city: "Sao Paulo", country: "Brazil", grandPrix: "São Paulo Grand Prix", raceDate: "2026-11-08", latitude: -23.7036, longitude: -46.6997 },
  { key: "las_vegas", name: "Las Vegas Strip", city: "Las Vegas", country: "USA", grandPrix: "Las Vegas Grand Prix", raceDate: "2026-11-21", latitude: 36.1147, longitude: -115.1728 },
  { key: "losail", name: "Lusail Intl", city: "Lusail", country: "Qatar", grandPrix: "Qatar Grand Prix", raceDate: "2026-11-29", latitude: 25.49, longitude: 51.4542 },
  { key: "yas_marina", name: "Yas Marina", city: "Abu Dhabi", country: "UAE", grandPrix: "Abu Dhabi Grand Prix", raceDate: "2026-12-06", latitude: 24.4672, longitude: 54.6031 },
] as const satisfies Circuit[];

/** 2026 calendar order used by the current visualization (left panel). */
export const currentCalendarKeys = circuits.map((circuit) => circuit.key);

/** Proposed optimized calendar order (right panel). */
export const proposedCalendarKeys = [
  "sakhir",
  "jeddah",
  "baku",
  "losail",
  "yas_marina",
  "melbourne",
  "shanghai",
  "suzuka",
  "singapore",
  "monaco",
  "barcelona",
  "madrid",
  "spielberg",
  "silverstone",
  "budapest",
  "spa",
  "zandvoort",
  "monza",
  "miami",
  "montreal",
  "austin",
  "mexico_city",
  "sao_paulo",
  "las_vegas",
] as const satisfies readonly string[];

/** ISO-style season week per race for the proposed calendar. */
export const proposedCalendarWeeks: Record<(typeof proposedCalendarKeys)[number], number> = {
  sakhir: 8,
  jeddah: 10,
  baku: 12,
  losail: 14,
  yas_marina: 16,
  melbourne: 18,
  shanghai: 20,
  suzuka: 22,
  singapore: 24,
  monaco: 26,
  barcelona: 28,
  madrid: 30,
  spielberg: 33,
  silverstone: 35,
  budapest: 37,
  spa: 39,
  zandvoort: 40,
  monza: 41,
  miami: 43,
  montreal: 44,
  austin: 46,
  mexico_city: 47,
  sao_paulo: 49,
  las_vegas: 50,
};

export type CalendarStop = Circuit & { week: number };

export function isoWeekFromRaceDate(raceDate: string) {
  const date = new Date(`${raceDate}T12:00:00`);
  const utcDate = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
  const dayNumber = utcDate.getUTCDay() || 7;
  utcDate.setUTCDate(utcDate.getUTCDate() + 4 - dayNumber);
  const yearStart = new Date(Date.UTC(utcDate.getUTCFullYear(), 0, 1));
  return Math.ceil(((utcDate.getTime() - yearStart.getTime()) / 86_400_000 + 1) / 7);
}

export function getCircuitsInOrder(keys: readonly string[]) {
  return keys.map((key) => getCircuit(key));
}

export function buildCalendarStops(
  keys: readonly string[],
  weekByKey?: Partial<Record<string, number>>,
): CalendarStop[] {
  return keys.map((key) => {
    const circuit = getCircuit(key);
    return {
      ...circuit,
      week: weekByKey?.[key] ?? isoWeekFromRaceDate(circuit.raceDate),
    };
  });
}

export function formatRaceDate(raceDate: string) {
  return new Intl.DateTimeFormat("en-GB", {
    weekday: "long",
    day: "numeric",
    month: "long",
    year: "numeric",
  }).format(new Date(`${raceDate}T12:00:00`));
}

export function getCircuit(key: string) {
  const circuit = circuits.find((item) => item.key === key);

  if (!circuit) {
    throw new Error(`Unknown circuit: ${key}`);
  }

  return circuit;
}

export function haversineKm(a: Circuit, b: Circuit) {
  const toRadians = (value: number) => (value * Math.PI) / 180;
  const lat1 = toRadians(a.latitude);
  const lon1 = toRadians(a.longitude);
  const lat2 = toRadians(b.latitude);
  const lon2 = toRadians(b.longitude);
  const dlat = lat2 - lat1;
  const dlon = lon2 - lon1;
  const h =
    Math.sin(dlat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dlon / 2) ** 2;

  return 2 * 6371 * Math.asin(Math.sqrt(h));
}

export function carbonProxyKg(a: Circuit, b: Circuit) {
  return haversineKm(a, b) * 350 * 0.5;
}
