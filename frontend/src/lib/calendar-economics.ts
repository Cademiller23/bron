import { type CalendarStop, carbonProxyKg } from "@/lib/f1-circuits";

/** Placeholder per-race economics — replace via backend when wired up. */
export type RaceEconomics = {
  hostingCostUsd: number;
  ticketRevenueUsd: number;
};

/**
 * Stub ticket revenue + hosting cost per circuit (USD).
 * Scaled from internal optimizer tiers; swap for API values later.
 */
export const PLACEHOLDER_RACE_ECONOMICS: Record<string, RaceEconomics> = {
  monaco: { hostingCostUsd: 0, ticketRevenueUsd: 104_000_000 },
  silverstone: { hostingCostUsd: 12_500_000, ticketRevenueUsd: 99_750_000 },
  monza: { hostingCostUsd: 12_500_000, ticketRevenueUsd: 96_600_000 },
  las_vegas: { hostingCostUsd: 35_000_000, ticketRevenueUsd: 99_750_000 },
  singapore: { hostingCostUsd: 30_000_000, ticketRevenueUsd: 94_500_000 },
  austin: { hostingCostUsd: 22_500_000, ticketRevenueUsd: 89_250_000 },
  sao_paulo: { hostingCostUsd: 20_000_000, ticketRevenueUsd: 89_250_000 },
  suzuka: { hostingCostUsd: 20_000_000, ticketRevenueUsd: 86_100_000 },
  melbourne: { hostingCostUsd: 21_000_000, ticketRevenueUsd: 84_000_000 },
  spa: { hostingCostUsd: 14_000_000, ticketRevenueUsd: 84_000_000 },
  zandvoort: { hostingCostUsd: 16_000_000, ticketRevenueUsd: 81_900_000 },
  mexico_city: { hostingCostUsd: 22_500_000, ticketRevenueUsd: 84_000_000 },
  miami: { hostingCostUsd: 27_500_000, ticketRevenueUsd: 86_100_000 },
  montreal: { hostingCostUsd: 19_000_000, ticketRevenueUsd: 75_600_000 },
  barcelona: { hostingCostUsd: 15_000_000, ticketRevenueUsd: 71_400_000 },
  budapest: { hostingCostUsd: 17_500_000, ticketRevenueUsd: 68_250_000 },
  spielberg: { hostingCostUsd: 15_000_000, ticketRevenueUsd: 69_300_000 },
  madrid: { hostingCostUsd: 25_000_000, ticketRevenueUsd: 73_500_000 },
  baku: { hostingCostUsd: 27_500_000, ticketRevenueUsd: 63_000_000 },
  jeddah: { hostingCostUsd: 32_500_000, ticketRevenueUsd: 65_100_000 },
  yas_marina: { hostingCostUsd: 32_500_000, ticketRevenueUsd: 73_500_000 },
  shanghai: { hostingCostUsd: 25_000_000, ticketRevenueUsd: 60_900_000 },
  sakhir: { hostingCostUsd: 32_500_000, ticketRevenueUsd: 57_750_000 },
  losail: { hostingCostUsd: 30_000_000, ticketRevenueUsd: 54_600_000 },
};

/** Placeholder freight surcharge per kg CO2 for logistics cost rollup. */
const LOGISTICS_COST_PER_KG_CO2 = 2.8;

export type SeasonEconomicsTotals = {
  costUsd: number;
  carbonKg: number;
  revenueUsd: number;
  visitedRaces: number;
  completedLegs: number;
};

export function getRaceEconomics(circuitKey: string): RaceEconomics {
  return (
    PLACEHOLDER_RACE_ECONOMICS[circuitKey] ?? {
      hostingCostUsd: 15_000_000,
      ticketRevenueUsd: 65_000_000,
    }
  );
}

export function getEconomicsProgress(
  hasStarted: boolean,
  reachedStopIndex: number,
  activeLegIndex: number | null,
) {
  if (!hasStarted) {
    return { visitedRaces: 0, completedLegs: 0 };
  }

  return {
    visitedRaces: reachedStopIndex + 1,
    completedLegs: activeLegIndex ?? reachedStopIndex,
  };
}

export function computeSeasonTotals(
  stops: CalendarStop[],
  visitedRaces: number,
  completedLegs: number,
): SeasonEconomicsTotals {
  let costUsd = 0;
  let revenueUsd = 0;
  let carbonKg = 0;

  for (let index = 0; index < visitedRaces; index += 1) {
    const economics = getRaceEconomics(stops[index].key);
    costUsd += economics.hostingCostUsd;
    revenueUsd += economics.ticketRevenueUsd;
  }

  for (let index = 0; index < completedLegs; index += 1) {
    const legCarbon = carbonProxyKg(stops[index], stops[index + 1]);
    carbonKg += legCarbon;
    costUsd += legCarbon * LOGISTICS_COST_PER_KG_CO2;
  }

  return {
    costUsd,
    carbonKg,
    revenueUsd,
    visitedRaces,
    completedLegs,
  };
}

export function formatCompactUsd(value: number) {
  if (value >= 1_000_000_000) {
    return `$${(value / 1_000_000_000).toFixed(2)}B`;
  }

  if (value >= 1_000_000) {
    return `$${(value / 1_000_000).toFixed(1)}M`;
  }

  if (value >= 1_000) {
    return `$${(value / 1_000).toFixed(0)}K`;
  }

  return `$${value.toFixed(0)}`;
}

export function formatCarbonKg(value: number) {
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)}M kg`;
  }

  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}t`;
  }

  return `${Math.round(value)} kg`;
}
