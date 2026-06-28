"""F1 calendar scorer — wraps the f1_calendar validity/components layer into the
engine's ScoreBreakdown, so B5/B6/B8 consume an F1 solve identically to a
supply-chain solve.

THREE TIERS (matching scorer.py's contract exactly):
  * MALFORMED  — not a well-formed calendar (unknown/dup/missing races, bad week).
                 Scores _MALFORMED_FITNESS (-1e19): strictly below everything.
  * INFEASIBLE — well-formed but >=1 hard violation. Scores
                 _infeasible_fitness(penalty) in (-1e18, 0). Penalty grows with
                 BOTH count and per-family severity, so FEWER violations scores
                 strictly HIGHER (the monotonic gradient the loop climbs).
  * FEASIBLE   — zero violations. normalized_score in (0,1]; only tier that can
                 clear the 0.90 gate.

Calibration anchors (Wave F1-A, measured on this machine):
  REFERENCE_CARBON_KG = 45,080,226.9   WORST = 134,982,847.2   REVENUE_REF = 1,829.2
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from darwin.problem import f1_calendar as F1
from darwin.problem.schemas import ObjectiveWeights, ScoreBreakdown, Violation, ViolationType
from darwin.problem.scorer import (
    SCORER_VERSION,
    _INFEASIBLE_FLOOR,
    _MALFORMED_FITNESS,
    _MALFORMED_PENALTY,
    _infeasible_fitness,
    _now_iso,
)

REFERENCE_CARBON_KG: float = round(F1.calendar_components(F1.FEASIBLE_BASELINE)["total_carbon_kg"], 1)
_REVENUE_REF: float = round(F1.calendar_components(F1.FEASIBLE_BASELINE)["total_revenue"], 1)

_FAMILY_SEVERITY: Dict[str, float] = {"scheduling": 4.0, "clustering": 3.0, "routing": 2.0}
_BASE_PENALTY_PER_VIOLATION: float = 1.0

# --- REVENUE-DRIVEN feasible scoring (calibrated from measured feasible range) ---
# Carbon's feasible band is only ~4.4% (clustering pins the routing), so carbon is
# a light TIE-BREAKER. Revenue's feasible band is ~11.5% (week/peak-month placement)
# and is the real optimization lever. We map feasible revenue -> [0.75, 0.97] so the
# baseline lands ~0.75 (feasible but not winning) and only a swarm that lifts revenue
# toward the feasible ceiling clears the 0.90 gate. The gate is PROVABLY reachable:
# the smoke test asserts the best-feasible calendar scores >= 0.90.
_R_FLOOR: float = 1757.1     # worst feasible revenue (measured)
_R_BASELINE: float = 1829.2  # FEASIBLE_BASELINE revenue (the 0.75 anchor)
_R_CEILING: float = 2038.9   # best feasible revenue (measured) -> ~0.97
_SCORE_AT_BASELINE: float = 0.75
_SCORE_AT_CEILING: float = 0.97
# carbon tie-breaker: at most +/- this much, only to separate equal-revenue calendars
_CARBON_TIEBREAK: float = 0.02
_CARBON_REF_KG: float = 45080226.9  # baseline carbon; below this nudges score up slightly


def _is_well_formed(calendar: Any) -> Tuple[bool, str]:
    if not isinstance(calendar, (list, tuple)) or not calendar:
        return False, "calendar is empty or not a list"
    races = []
    for item in calendar:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False, f"entry is not a (race, week) pair: {item!r}"
        r, w = item
        if r not in F1.CIRCUITS:
            return False, f"unknown race key {r!r}"
        if not isinstance(w, int) or isinstance(w, bool):
            return False, f"week is not an int: {w!r}"
        # Structural sanity only: reject genuine garbage. The 8..50 SEASON window
        # is a SCHEDULING violation (out_of_season), owned by the dataset's
        # scheduling_violations — NOT a malformed-data condition. A week like 53
        # is a bad schedule the optimizer must climb away from, not garbage.
        if w < 1 or w > 60:
            return False, f"week {w} structurally out of range (1..60 sanity bound)"
        races.append(r)
    if len(races) != len(F1.RACES):
        return False, f"expected {len(F1.RACES)} races, got {len(races)}"
    if len(set(races)) != len(races):
        dupes = sorted({x for x in races if races.count(x) > 1})
        return False, f"duplicate races: {dupes}"
    missing = set(F1.RACES) - set(races)
    if missing:
        return False, f"missing races: {sorted(missing)}"
    return True, ""


def _violations_to_models(validity: Dict[str, Any]) -> List[Violation]:
    out: List[Violation] = []
    for family in ("routing", "scheduling", "clustering"):
        for v in validity.get(family, []):
            vtype = v.get("type", "violation")
            loc = (v.get("race") or v.get("region")
                   or (f"{v.get('from')}->{v.get('to')}" if v.get("from") else "")
                   or v.get("race_a") or "calendar")
            out.append(Violation(
                violation_type=ViolationType.CUSTOM_CONSTRAINT,
                location=str(loc),
                magnitude=_FAMILY_SEVERITY[family],
                message=f"[{family}] {vtype}: {v}",
            ))
    return out


def _feasible_normalized_score(components: Dict[str, Any]) -> Tuple[float, float, Dict[str, float]]:
    """For a FEASIBLE calendar: revenue is the primary driver, carbon a light
    tie-breaker. Linearly map revenue in [_R_BASELINE, _R_CEILING] onto
    [0.75, 0.97]; below baseline scales down toward _R_FLOOR. Carbon below the
    baseline reference nudges the score up by at most _CARBON_TIEBREAK."""
    revenue = float(components["total_revenue"])
    carbon = float(components["total_carbon_kg"])

    # revenue -> base score in [~0.70, 0.97]
    span = _R_CEILING - _R_BASELINE
    if revenue >= _R_BASELINE:
        frac = (revenue - _R_BASELINE) / span if span > 1e-9 else 0.0
        rev_score = _SCORE_AT_BASELINE + frac * (_SCORE_AT_CEILING - _SCORE_AT_BASELINE)
    else:
        # below baseline: scale from a floor of ~0.70 at _R_FLOOR up to 0.75 at baseline
        lo_span = _R_BASELINE - _R_FLOOR
        frac = (revenue - _R_FLOOR) / lo_span if lo_span > 1e-9 else 0.0
        rev_score = 0.70 + max(0.0, frac) * (_SCORE_AT_BASELINE - 0.70)
    rev_score = max(0.0, min(_SCORE_AT_CEILING, rev_score))

    # carbon tie-breaker: lower carbon than baseline ref adds a little; higher subtracts.
    carbon_ratio = _CARBON_REF_KG / carbon if carbon > 1e-9 else 1.0
    carbon_adj = max(-_CARBON_TIEBREAK, min(_CARBON_TIEBREAK, (carbon_ratio - 1.0)))

    normalized_score = max(0.0, min(1.0, rev_score + carbon_adj))
    weighted_objective = 1.0 - normalized_score
    return normalized_score, weighted_objective, {
        "revenue": revenue, "rev_score": rev_score,
        "carbon_kg": carbon, "carbon_adj": carbon_adj,
    }

def score_f1(calendar: Any, *, solution_id: str = "f1_calendar_solution",
             instance_id: str = "f1_2026_calendar",
             weights: Optional[ObjectiveWeights] = None) -> ScoreBreakdown:
    weights = weights or ObjectiveWeights.balanced()

    ok, why = _is_well_formed(calendar)
    if not ok:
        return ScoreBreakdown(
            solution_id=solution_id, instance_id=instance_id, feasible=False,
            violations=[Violation(violation_type=ViolationType.MALFORMED_SOLUTION,
                                  location="calendar", magnitude=0.0,
                                  message=f"malformed calendar: {why}")],
            raw_cost=0.0, raw_lead_time=0.0, raw_risk=0.0, weighted_objective=0.0,
            normalized_score=0.0, total_penalty=_MALFORMED_PENALTY,
            final_fitness=_MALFORMED_FITNESS, objective_weights=weights,
            scorer_version=SCORER_VERSION, computed_at=_now_iso(),
            diagnostics={"malformed": True, "reason": why},
        )

    calendar = [(r, int(w)) for r, w in calendar]
    components = F1.calendar_components(calendar)
    validity = F1.calendar_validity(calendar)
    carbon = float(components["total_carbon_kg"])
    raw_risk = float(components.get("weather_soft_penalty", 0.0))

    diagnostics: Dict[str, Any] = {
        "components": components, "n_violations": validity["n_violations"],
        "violations_by_family": {"routing": len(validity["routing"]),
                                 "scheduling": len(validity["scheduling"]),
                                 "clustering": len(validity["clustering"])},
        "reference_carbon_kg": REFERENCE_CARBON_KG, "revenue_ref": _REVENUE_REF,
    }

    if not validity["is_valid"]:
        violations = _violations_to_models(validity)
        total_penalty = _BASE_PENALTY_PER_VIOLATION * len(violations) + math.fsum(v.magnitude for v in violations)
        final_fitness = _infeasible_fitness(total_penalty)
        stored_penalty = total_penalty if math.isfinite(total_penalty) else _INFEASIBLE_FLOOR
        diagnostics["tier"] = "infeasible"
        return ScoreBreakdown(
            solution_id=solution_id, instance_id=instance_id, feasible=False,
            violations=violations, raw_cost=carbon, raw_lead_time=0.0, raw_risk=raw_risk,
            weighted_objective=0.0, normalized_score=0.0, total_penalty=stored_penalty,
            final_fitness=final_fitness, objective_weights=weights,
            scorer_version=SCORER_VERSION, computed_at=_now_iso(), diagnostics=diagnostics,
        )

    normalized_score, weighted_objective, parts = _feasible_normalized_score(components)
    diagnostics["tier"] = "feasible"; diagnostics.update(parts)
    return ScoreBreakdown(
        solution_id=solution_id, instance_id=instance_id, feasible=True, violations=[],
        raw_cost=carbon, raw_lead_time=0.0, raw_risk=raw_risk,
        weighted_objective=weighted_objective, normalized_score=normalized_score,
        total_penalty=0.0, final_fitness=normalized_score, objective_weights=weights,
        scorer_version=SCORER_VERSION, computed_at=_now_iso(), diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# Runner-signature adapter: matches runner.py line 146
#   self.scorer(instance, final_solution, weights)
# Inject via TeamRunner(scorer=score_f1_solution) in Wave F1-D. Decodes the
# calendar from the Solution (via the codec) and delegates to score_f1.
# ---------------------------------------------------------------------------
from darwin.problem.f1_codec import solution_to_calendar  # noqa: E402


def score_f1_solution(instance, solution, weights=None):
    """Decode the calendar out of the Solution and score it. Never raises."""
    try:
        calendar = solution_to_calendar(solution)
    except Exception:
        calendar = []  # -> score_f1's MALFORMED tier handles a garbage Solution
    return score_f1(
        calendar,
        solution_id=getattr(solution, "solution_id", "f1_calendar_solution"),
        instance_id=getattr(instance, "instance_id", "f1_2026_calendar"),
        weights=weights,
    )


if __name__ == "__main__":
    print(f"REFERENCE_CARBON_KG = {REFERENCE_CARBON_KG:,.1f}")
    print(f"_REVENUE_REF        = {_REVENUE_REF:,.1f}\n")

    base = score_f1(F1.FEASIBLE_BASELINE)
    print("=== FEASIBLE_BASELINE ===")
    print(f"  feasible={base.feasible}  fitness={base.final_fitness:.4f}  "
          f"norm={base.normalized_score:.4f}  violations={len(base.violations)}")

    bad = score_f1([(r, 30) for r in sorted(F1.RACES)])
    print("=== BAD (alpha, all wk30) ===")
    print(f"  feasible={bad.feasible}  fitness={bad.final_fitness:.4f}  "
          f"penalty={bad.total_penalty:.1f}  violations={len(bad.violations)}")

    mal = score_f1([("melbourne", 8)])
    print("=== MALFORMED (1 race) ===")
    print(f"  feasible={mal.feasible}  fitness={mal.final_fitness:.2e}")

    swing_order = (["melbourne","shanghai","suzuka","singapore"]
        + ["sakhir","jeddah","losail","yas_marina","baku"]
        + ["monaco","barcelona","spielberg","silverstone","spa","zandvoort","monza","budapest","madrid"]
        + ["montreal","miami","austin","mexico_city","sao_paulo","las_vegas"])
    # Spread 24 races across weeks 8..49 (fits the 1..52 calendar year and the
    # 8..50 season window), skipping the summer break. Compressed so the finale
    # never overflows past week 50 (the Wave F1-A fixture overflowed to wk55).
    n = len(swing_order)
    lo, hi = F1.SEASON_WEEK_START, F1.SEASON_WEEK_END - 1  # 8..49
    weeks_seq, used = [], set()
    for i in range(n):
        wi = lo + round(i * (hi - lo) / (n - 1))
        while wi in F1.SUMMER_BREAK_WEEKS or wi in used:
            wi += 1
        used.add(wi); weeks_seq.append(wi)
    sw = score_f1(list(zip(swing_order, weeks_seq)))
    print("=== SWING (region-clustered) ===")
    print(f"  feasible={sw.feasible}  fitness={sw.final_fitness:.4f}  violations={len(sw.violations)}")

    import json as _json
    BEST_FEASIBLE = [tuple(x) for x in _json.loads('[["baku", 8], ["sakhir", 10], ["losail", 12], ["yas_marina", 13], ["jeddah", 15], ["shanghai", 17], ["suzuka", 19], ["melbourne", 21], ["singapore", 23], ["barcelona", 24], ["zandvoort", 26], ["silverstone", 28], ["monza", 30], ["spielberg", 33], ["monaco", 34], ["spa", 35], ["madrid", 37], ["budapest", 39], ["mexico_city", 41], ["montreal", 43], ["austin", 45], ["las_vegas", 46], ["sao_paulo", 48], ["miami", 50]]')]
    bf = score_f1(BEST_FEASIBLE)
    print("=== BEST_FEASIBLE (measured ceiling) ===")
    print(f"  feasible={bf.feasible}  fitness={bf.final_fitness:.4f}  violations={len(bf.violations)}")

    print("\n=== INVARIANTS ===")
    # baseline must be FEASIBLE but ~0.75 (room to climb)
    assert base.feasible, "baseline must be feasible"
    assert 0.70 <= base.final_fitness <= 0.80, f"baseline should be ~0.75, got {base.final_fitness:.4f}"
    print(f"  [OK] baseline feasible at ~0.75 (fitness={base.final_fitness:.4f}) -- room to climb")
    # THE REACHABILITY GUARANTEE: the best feasible calendar must clear 0.90
    assert bf.feasible and bf.final_fitness >= 0.90, f"best-feasible must clear 0.90, got {bf.final_fitness:.4f}"
    print(f"  [OK] best-feasible clears the 0.90 gate (fitness={bf.final_fitness:.4f}) -- gate is REACHABLE")
    assert sw.final_fitness > bad.final_fitness, "fewer violations must score higher"
    print(f"  [OK] SWING ({len(sw.violations)} viol, {sw.final_fitness:.2f}) > BAD ({len(bad.violations)} viol, {bad.final_fitness:.2f})")
    assert bad.final_fitness > mal.final_fitness, "infeasible must beat malformed"
    print(f"  [OK] BAD ({bad.final_fitness:.2f}) > MALFORMED ({mal.final_fitness:.2e})")
    assert mal.final_fitness == _MALFORMED_FITNESS, "malformed hits the dedicated floor"
    print(f"  [OK] MALFORMED at dedicated floor ({mal.final_fitness:.2e})")
    print("\nALL INVARIANTS HOLD — scorer is monotonic and tiered correctly.")
