"""
F1 2026 calendar-optimization dataset  (HARDENED).

What is real vs fabricated (the Q&A line):
  - Circuit locations and distances are REAL (lat/long + haversine).
  - Carbon is computed from real distances via a freight-emissions factor.
  - Weather windows, revenue, and hosting cost are realistic estimates,
    because the system is data-agnostic. The contribution is the
    optimization mechanism, not the calibration of these stubs.

This module exposes the OBJECTIVE COMPONENTS and the CONSTRAINT/VALIDITY
LAYER. It deliberately does NOT implement the fitness function. Combining
terms, applying weights, and the cost-of-thinking penalty is B2's job
(the scorekeeper). This module's job is: (a) measure the problem-side terms,
and (b) report every hard violation and soft penalty so B2 can compose them.
Keep that boundary clean.

------------------------------------------------------------------------
DECISION SPACE (hardened):
A calendar is an ORDERED list of (race, week), where week is 1-52.
  - Reordering changes carbon (travel between consecutive rounds).
  - The chosen week per race changes revenue and weather feasibility.
  - Week (not month) is the unit so the spacing + August-break + window
    constraints are enforceable. Month is derived from week.

WHY THIS IS HARD (three overlapping constraint families, by design):
  1. ROUTING       — consecutive-leg distance cap + no >2 long-haul legs in
                     a row. Conflicts with revenue (marquee races are
                     geographically scattered).
  2. SCHEDULING    — Mar-Dec season window, >=1 week spacing, mandated
                     August summer break, weather-hard months forbidden.
                     Conflicts with revenue-peak months and with each other.
  3. CLUSTERING    — geographic regions must appear as contiguous "swings"
                     (you cannot leave a region and come back). This is the
                     resilience/logistics constraint that forces a fourth
                     specialist and reads as enterprise-grade.

No single agent reasons cleanly across all three. That is the point:
topology (division of labor + an arbitrator) must beat a monolith here.
------------------------------------------------------------------------
"""

from math import radians, sin, cos, asin, sqrt
from itertools import pairwise

# ---------------------------------------------------------------------------
# 1. CIRCUITS  (REAL. coords accurate to ~0.01 deg, ample for demo distances)
# ---------------------------------------------------------------------------
# key -> (display_name, city, country, lat, lon)
CIRCUITS = {
    "melbourne":    ("Albert Park",          "Melbourne",     "Australia",   -37.8497, 144.9680),
    "shanghai":     ("Shanghai Intl",        "Shanghai",      "China",        31.3389, 121.2200),
    "suzuka":       ("Suzuka",               "Suzuka",        "Japan",        34.8431, 136.5410),
    "sakhir":       ("Bahrain Intl",         "Sakhir",        "Bahrain",      26.0325,  50.5106),
    "jeddah":       ("Jeddah Corniche",      "Jeddah",        "Saudi Arabia", 21.6319,  39.1044),
    "miami":        ("Miami Intl Autodrome", "Miami",         "USA",          25.9581, -80.2389),
    "montreal":     ("Gilles Villeneuve",    "Montreal",      "Canada",       45.5000, -73.5228),
    "monaco":       ("Circuit de Monaco",    "Monte Carlo",   "Monaco",       43.7347,   7.4206),
    "barcelona":    ("Catalunya",            "Barcelona",     "Spain",        41.5700,   2.2611),
    "spielberg":    ("Red Bull Ring",        "Spielberg",     "Austria",      47.2197,  14.7647),
    "silverstone":  ("Silverstone",          "Silverstone",   "UK",           52.0786,  -1.0169),
    "budapest":     ("Hungaroring",          "Budapest",      "Hungary",      47.5789,  19.2486),
    "spa":          ("Spa-Francorchamps",    "Stavelot",      "Belgium",      50.4372,   5.9714),
    "zandvoort":    ("Zandvoort",            "Zandvoort",     "Netherlands",  52.3888,   4.5409),
    "monza":        ("Monza",                "Monza",         "Italy",        45.6156,   9.2811),
    "madrid":       ("Madring (IFEMA)",      "Madrid",        "Spain",        40.4650,  -3.6160),
    "baku":         ("Baku City",            "Baku",          "Azerbaijan",   40.3725,  49.8533),
    "singapore":    ("Marina Bay",           "Singapore",     "Singapore",     1.2914, 103.8640),
    "austin":       ("Circuit of Americas",  "Austin",        "USA",          30.1328, -97.6411),
    "mexico_city":  ("Hermanos Rodriguez",   "Mexico City",   "Mexico",       19.4042, -99.0907),
    "sao_paulo":    ("Interlagos",           "Sao Paulo",     "Brazil",      -23.7036, -46.6997),
    "las_vegas":    ("Las Vegas Strip",      "Las Vegas",     "USA",          36.1147,-115.1728),
    "losail":       ("Lusail Intl",          "Lusail",        "Qatar",        25.4900,  51.4542),
    "yas_marina":   ("Yas Marina",           "Abu Dhabi",     "UAE",          24.4672,  54.6031),
}

RACES = list(CIRCUITS.keys())


# ---------------------------------------------------------------------------
# 2. DISTANCE + CARBON  (COMPUTE, do not source)
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0

# Air-freight tonne-km factor, kg CO2 per tonne-km (~0.5 typical for air cargo),
# times an assumed freight payload moved per round (logistics tonnage).
# Both are tunable stubs. Real distances make the relative carbon defensible.
KG_CO2_PER_TONNE_KM = 0.50
FREIGHT_TONNES_PER_ROUND = 1400.0  # paddock + cars + garages air-freighted


def haversine_km(a: str, b: str) -> float:
    """Great-circle distance in km between two circuit keys."""
    _, _, _, lat1, lon1 = CIRCUITS[a]
    _, _, _, lat2, lon2 = CIRCUITS[b]
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(h))


# Full N x N matrix. Any ordering is then an O(N) lookup.
DISTANCE_MATRIX = {
    a: {b: (0.0 if a == b else haversine_km(a, b)) for b in RACES}
    for a in RACES
}


def leg_carbon_kg(a: str, b: str) -> float:
    """CO2 (kg) of freighting the circus from race a to race b."""
    return DISTANCE_MATRIX[a][b] * FREIGHT_TONNES_PER_ROUND * KG_CO2_PER_TONNE_KM


# ---------------------------------------------------------------------------
# 3. WEATHER  (race x month feasibility. HAND-BUILT, defensible by climate)
# ---------------------------------------------------------------------------
# Months are 1-12. Anything not listed is "good".
#   hard  -> infeasible (heat / snow / monsoon). HARD CONSTRAINT VIOLATION.
#   soft  -> raceable but suboptimal (cold, shoulder season). Minor penalty.
WEATHER_HARD = {
    "sakhir":      {6, 7, 8},
    "jeddah":      {6, 7, 8, 9},
    "miami":       {8, 9},
    "montreal":    {11, 12, 1, 2, 3, 4},
    "spielberg":   {12, 1, 2},
    "las_vegas":   {6, 7, 8},
    "losail":      {6, 7, 8, 9},
    "yas_marina":  {6, 7, 8, 9},
}
WEATHER_SOFT = {
    "melbourne":   {6, 7, 8},
    "shanghai":    {12, 1, 2, 7, 8},
    "suzuka":      {1, 2, 8, 12},
    "sakhir":      {5, 9},
    "jeddah":      {5, 10},
    "miami":       {6, 7, 10},
    "montreal":    {5, 10},
    "monaco":      {12, 1, 2},
    "barcelona":   {12, 1},
    "spielberg":   {3, 11},
    "silverstone": {12, 1, 2},
    "budapest":    {12, 1, 2},
    "spa":         {12, 1, 2},
    "zandvoort":   {11, 12, 1, 2, 3},
    "monza":       {12, 1},
    "madrid":      {7, 8},
    "baku":        {7, 8, 12, 1},
    "singapore":   {11, 12},
    "austin":      {7, 8},
    "mexico_city": {6, 7, 8, 9},
    "sao_paulo":   {6, 7},
    "las_vegas":   {5, 9},
    "losail":      {5, 10},
    "yas_marina":  {5, 10},
}

# Penalty weight for the SOFT case. HARD is handled as a validity violation.
WEATHER_SOFT_PENALTY = 0.30   # fraction of that race's revenue lost as penalty


def weather_status(race: str, month: int) -> str:
    """Return 'good' | 'soft' | 'hard' for racing `race` in `month`."""
    if month in WEATHER_HARD.get(race, set()):
        return "hard"
    if month in WEATHER_SOFT.get(race, set()):
        return "soft"
    return "good"


# ---------------------------------------------------------------------------
# 4. REVENUE  (base tier x seasonal multiplier. FABRICATED, relative ordering)
# ---------------------------------------------------------------------------
REVENUE_BASE = {
    "monaco": 100, "silverstone": 95, "monza": 92, "las_vegas": 95,
    "singapore": 90, "austin": 85, "sao_paulo": 85, "suzuka": 82,
    "melbourne": 80, "spa": 80, "zandvoort": 78, "mexico_city": 80,
    "miami": 82, "montreal": 72, "barcelona": 68, "budapest": 65,
    "spielberg": 66, "madrid": 70, "baku": 60, "jeddah": 62,
    "yas_marina": 70, "shanghai": 58, "sakhir": 55, "losail": 52,
}

REVENUE_PEAK_MONTHS = {
    "melbourne": {3, 11},           "shanghai": {4, 10},
    "suzuka": {4, 10},              "sakhir": {3, 11},
    "jeddah": {3, 12},              "miami": {5},
    "montreal": {6, 9},            "monaco": {5},
    "barcelona": {6},              "spielberg": {7},
    "silverstone": {7},           "budapest": {7, 8},
    "spa": {7, 8},                "zandvoort": {8},
    "monza": {9},                 "madrid": {9},
    "baku": {9},                  "singapore": {9},
    "austin": {10, 11},           "mexico_city": {10, 11},
    "sao_paulo": {11},            "las_vegas": {11},
    "losail": {11, 12},           "yas_marina": {12},
}

PEAK_MULTIPLIER = 1.30
OFF_MULTIPLIER = 0.80   # racing a marquee venue well outside its peak window


def revenue(race: str, month: int) -> float:
    """Revenue for `race` in `month`, after seasonal and weather effects."""
    base = REVENUE_BASE[race]
    if month in REVENUE_PEAK_MONTHS.get(race, set()):
        mult = PEAK_MULTIPLIER
    elif weather_status(race, month) == "hard":
        mult = OFF_MULTIPLIER * 0.6   # nobody comes; doubles as a soft signal
    elif weather_status(race, month) == "soft":
        mult = OFF_MULTIPLIER
    else:
        mult = 1.0
    return base * mult


# ---------------------------------------------------------------------------
# 5. HOSTING COST  (FABRICATED tiers).  See WARNING below.
# ---------------------------------------------------------------------------
# WARNING: per-race constant. Under PURE PERMUTATION (no race dropping) this
# sums to the same value for every calendar and is INERT in the optimizer.
# Do NOT include it in the fitness sum unless you add race selection. It is
# kept here only for that future extension. DEFAULT: leave it out.
HOSTING_COST = {
    "monaco": 0,
    "monza": 25, "silverstone": 25, "spa": 28, "barcelona": 30,
    "zandvoort": 32, "spielberg": 30, "budapest": 35, "montreal": 38,
    "suzuka": 40, "melbourne": 42, "sao_paulo": 40, "mexico_city": 45,
    "madrid": 50, "austin": 45, "miami": 55, "shanghai": 50,
    "baku": 55, "singapore": 60, "jeddah": 65, "sakhir": 65,
    "losail": 60, "yas_marina": 65, "las_vegas": 70,
}


# ===========================================================================
#  CONSTRAINT LAYER  (the three hardening families)
#  Pure measurement: returns violations + penalties. B2 composes them.
# ===========================================================================

# --- week/month helpers ----------------------------------------------------
# A season is weeks 10..49 (early March .. early December). Weeks outside that
# window are out-of-season violations. Month is derived from week for the
# weather check, so scheduling and weather share one time axis.
SEASON_WEEK_START = 8    # ~late February (real F1 opener)
SEASON_WEEK_END = 50     # ~mid December (real F1 finale)
SUMMER_BREAK_WEEKS = {31, 32}   # mandated August shutdown: no race may fall here

# Approximate week -> month (ISO-ish; good enough for feasibility, not payroll)
def week_to_month(week: int) -> int:
    """Map ISO week (1-52) to a calendar month (1-12)."""
    # 52 weeks across 12 months ~ 4.345 weeks/month.
    m = int((week - 1) / 4.345) + 1
    return max(1, min(12, m))


# --- 5a. ROUTING constraint -------------------------------------------------
# Consecutive races cannot be absurdly far apart, and you cannot string more
# than MAX_CONSECUTIVE_LONGHAUL long-haul legs in a row (crew/freight limit).
MAX_LEG_KM = 11000.0           # a single consecutive leg may not exceed this
LONGHAUL_KM = 7000.0           # a leg at/above this counts as "long-haul"
MAX_CONSECUTIVE_LONGHAUL = 2   # at most this many long-haul legs back-to-back


def routing_violations(order):
    """
    order: list of race keys in calendar order.
    Returns list of violation dicts. Empty => routing-feasible.
    """
    v = []
    legs = list(pairwise(order))
    for a, b in legs:
        d = DISTANCE_MATRIX[a][b]
        if d > MAX_LEG_KM:
            v.append({"type": "leg_too_long", "from": a, "to": b,
                      "km": round(d, 1), "cap": MAX_LEG_KM})
    # streak of consecutive long-haul legs
    streak = 0
    for a, b in legs:
        if DISTANCE_MATRIX[a][b] >= LONGHAUL_KM:
            streak += 1
            if streak > MAX_CONSECUTIVE_LONGHAUL:
                v.append({"type": "too_many_longhaul_in_a_row",
                          "at_leg": (a, b), "streak": streak,
                          "max": MAX_CONSECUTIVE_LONGHAUL})
        else:
            streak = 0
    return v


# --- 5b. SCHEDULING constraint ----------------------------------------------
# Season window, >=1 week between races (no two in the same week), mandated
# summer break, and weather-hard months forbidden (promoted from penalty to
# hard violation here so an unraceable date can never win).
MIN_WEEK_GAP = 1   # at least this many weeks between any two consecutive races


def scheduling_violations(calendar):
    """
    calendar: list of (race, week).
    Returns list of violation dicts. Empty => schedule-feasible.
    """
    v = []
    weeks = [w for _, w in calendar]

    # out-of-season
    for r, w in calendar:
        if w < SEASON_WEEK_START or w > SEASON_WEEK_END:
            v.append({"type": "out_of_season", "race": r, "week": w,
                      "window": (SEASON_WEEK_START, SEASON_WEEK_END)})

    # summer break
    for r, w in calendar:
        if w in SUMMER_BREAK_WEEKS:
            v.append({"type": "races_in_summer_break", "race": r, "week": w})

    # duplicate / too-close weeks (sort by week, check gaps)
    by_week = sorted(calendar, key=lambda rw: rw[1])
    for (r1, w1), (r2, w2) in pairwise(by_week):
        if w2 - w1 < MIN_WEEK_GAP:
            v.append({"type": "weeks_too_close", "race_a": r1, "race_b": r2,
                      "week_a": w1, "week_b": w2, "min_gap": MIN_WEEK_GAP})

    # weather-hard months -> hard violation (uses derived month)
    for r, w in calendar:
        if weather_status(r, week_to_month(w)) == "hard":
            v.append({"type": "weather_infeasible", "race": r, "week": w,
                      "month": week_to_month(w)})
    return v


# --- 5c. CLUSTERING / resilience constraint --------------------------------
# Each geographic region must appear as ONE contiguous block in the calendar
# (a "swing"). You may not leave a region and return to it later. This is the
# logistics-resilience rule that forces a clustering specialist to emerge.
REGION = {
    "melbourne": "APAC", "shanghai": "APAC", "suzuka": "APAC", "singapore": "APAC",
    "sakhir": "MIDEAST", "jeddah": "MIDEAST", "losail": "MIDEAST", "yas_marina": "MIDEAST", "baku": "MIDEAST",
    "miami": "AMERICAS", "montreal": "AMERICAS", "austin": "AMERICAS",
    "mexico_city": "AMERICAS", "sao_paulo": "AMERICAS", "las_vegas": "AMERICAS",
    "monaco": "EUROPE", "barcelona": "EUROPE", "spielberg": "EUROPE",
    "silverstone": "EUROPE", "budapest": "EUROPE", "spa": "EUROPE",
    "zandvoort": "EUROPE", "monza": "EUROPE", "madrid": "EUROPE",
}


def clustering_violations(order):
    """
    order: list of race keys in calendar order.
    A region is violated if its races are not contiguous (the region's label
    appears, disappears, and reappears). Returns list of violation dicts.
    """
    v = []
    seen_blocks = {}   # region -> number of separate contiguous runs
    prev = None
    for r in order:
        reg = REGION[r]
        if reg != prev:
            seen_blocks[reg] = seen_blocks.get(reg, 0) + 1
        prev = reg
    for reg, blocks in seen_blocks.items():
        if blocks > 1:
            v.append({"type": "region_not_contiguous", "region": reg,
                      "separate_blocks": blocks})
    return v


# ===========================================================================
#  6. OBJECTIVE COMPONENTS + FULL VALIDITY  (helpers for B2; NOT fitness)
# ===========================================================================
def calendar_components(calendar, include_hosting=False):
    """
    Compute raw objective components for a calendar.

    `calendar`: ordered list of (race_key, week_int).
    Returns the problem-side terms only. B2 owns weighting + the penalty.
    """
    races = [r for r, _ in calendar]
    weeks = [w for _, w in calendar]
    months = [week_to_month(w) for w in weeks]

    total_carbon = sum(leg_carbon_kg(a, b) for a, b in pairwise(races))
    total_distance = sum(DISTANCE_MATRIX[a][b] for a, b in pairwise(races))
    total_revenue = sum(revenue(r, m) for r, m in zip(races, months))

    soft_penalty = sum(
        revenue(r, m) * WEATHER_SOFT_PENALTY
        for r, m in zip(races, months)
        if weather_status(r, m) == "soft"
    )

    out = {
        "total_distance_km": round(total_distance, 1),
        "total_carbon_kg": round(total_carbon, 1),
        "total_revenue": round(total_revenue, 1),
        "weather_soft_penalty": round(soft_penalty, 1),
        "n_races": len(calendar),
    }
    if include_hosting:
        out["total_hosting_cost"] = sum(HOSTING_COST[r] for r in races)
    return out


def calendar_validity(calendar):
    """
    THE function B2 calls to check feasibility. Runs all three constraint
    families and returns a single structured report.

    `calendar`: ordered list of (race_key, week_int).

    Returns:
      {
        "is_valid": bool,                  # True iff zero hard violations
        "n_violations": int,
        "routing":     [...],
        "scheduling":  [...],
        "clustering":  [...],
      }

    B2 rule: an invalid calendar must NEVER outscore a valid one. Map
    n_violations to a hard penalty (e.g. each violation subtracts a large
    constant, or is_valid=False forces score below any feasible score).
    """
    routing = routing_violations([r for r, _ in calendar])
    scheduling = scheduling_violations(calendar)
    clustering = clustering_violations([r for r, _ in calendar])
    total = len(routing) + len(scheduling) + len(clustering)
    return {
        "is_valid": total == 0,
        "n_violations": total,
        "routing": routing,
        "scheduling": scheduling,
        "clustering": clustering,
    }


def full_report(calendar, include_hosting=False):
    """Convenience: components + validity in one object (for B2 / telemetry)."""
    return {
        "components": calendar_components(calendar, include_hosting),
        "validity": calendar_validity(calendar),
    }


# ---------------------------------------------------------------------------
#  PROVEN-FEASIBLE BASELINE  (greedy region-swing placement; validity == 0)
#  Use as: a known-good start state, a regression anchor, or the "before"
#  the optimizer must beat. Verified zero violations across all 3 families.
# ---------------------------------------------------------------------------
FEASIBLE_BASELINE = [
    ("melbourne", 8), ("shanghai", 9), ("suzuka", 10), ("singapore", 11),
    ("sakhir", 12), ("jeddah", 13), ("losail", 14), ("yas_marina", 15), ("baku", 16),
    ("monaco", 17), ("barcelona", 18), ("spielberg", 19), ("silverstone", 20),
    ("spa", 21), ("zandvoort", 22), ("monza", 23), ("budapest", 24), ("madrid", 25),
    ("montreal", 26), ("miami", 27), ("austin", 28), ("mexico_city", 29),
    ("sao_paulo", 30), ("las_vegas", 45),
]


# ---------------------------------------------------------------------------
# DEMO / SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"circuits: {len(RACES)}  matrix: {len(DISTANCE_MATRIX)}x{len(RACES)}")
    print(f"max pair distance: {max(DISTANCE_MATRIX['melbourne'].values()):.0f} km")
    print(f"regions: {sorted(set(REGION.values()))}")
    print()

    # --- BAD calendar: arbitrary alpha order, everyone shoved into week 30 ---
    # This should rack up violations across ALL THREE families: weeks collide
    # (scheduling), region order is scrambled (clustering), and long legs jump
    # around the globe (routing). This is the "low score" start state.
    bad_order = sorted(RACES)
    bad = [(r, 30) for r in bad_order]
    bad_rep = full_report(bad)
    print("=== BAD calendar (alpha order, all week 30) ===")
    print("components:", bad_rep["components"])
    print(f"is_valid: {bad_rep['validity']['is_valid']}  "
          f"violations: {bad_rep['validity']['n_violations']} "
          f"(routing={len(bad_rep['validity']['routing'])}, "
          f"sched={len(bad_rep['validity']['scheduling'])}, "
          f"cluster={len(bad_rep['validity']['clustering'])})")
    print()

    # --- A HAND-CLUSTERED, spread-out attempt -------------------------------
    # Group by region into contiguous swings, spread across the season.
    # Demonstrates that a sensibly STRUCTURED calendar clears clustering and
    # most scheduling, leaving routing/revenue tension for the optimizer.
    # (Not optimal -- just feasibility-leaning, to show the conflict surface.)
    swing_order = (
        # APAC swing
        ["melbourne", "shanghai", "suzuka", "singapore"]
        # MIDEAST swing
        + ["sakhir", "jeddah", "losail", "yas_marina", "baku"]
        # EUROPE swing
        + ["monaco", "barcelona", "spielberg", "silverstone",
           "spa", "zandvoort", "monza", "budapest", "madrid"]
        # AMERICAS swing
        + ["montreal", "miami", "austin", "mexico_city", "sao_paulo", "las_vegas"]
    )
    # assign increasing weeks, skipping the summer break, ~1.75 wk apart
    weeks_seq = []
    w = float(SEASON_WEEK_START)
    for _ in swing_order:
        wi = int(round(w))
        while wi in SUMMER_BREAK_WEEKS:
            wi += 1
        weeks_seq.append(wi)
        w = wi + 1.75
    swing = list(zip(swing_order, weeks_seq))
    swing_rep = full_report(swing)
    print("=== SWING calendar (region-clustered, spread) ===")
    print("components:", swing_rep["components"])
    print(f"is_valid: {swing_rep['validity']['is_valid']}  "
          f"violations: {swing_rep['validity']['n_violations']} "
          f"(routing={len(swing_rep['validity']['routing'])}, "
          f"sched={len(swing_rep['validity']['scheduling'])}, "
          f"cluster={len(swing_rep['validity']['clustering'])})")
    if swing_rep["validity"]["routing"]:
        print("  routing detail:", swing_rep["validity"]["routing"])
    if swing_rep["validity"]["scheduling"]:
        print("  sched detail:", swing_rep["validity"]["scheduling"][:4], "...")
    print()
    print("Read: BAD violates all three families; SWING is far cleaner. The gap "
          "between them is the room your evolved org chart climbs through.")