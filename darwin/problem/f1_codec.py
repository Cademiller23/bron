"""F1 calendar <-> Solution codec (pure leaf functions; touch nothing).

A race calendar IS a tour plus a time assignment, so it rides inside the engine's
existing ``Solution`` with NO new output model:

  * ``Route.node_sequence``  -> the ordered race keys (the calendar order).
  * ``flows``                -> one FlowAssignment per race carrying its week:
                                arc_id = f"{race}@w{week}", quantity = float(week).

This means F1 agents emit a normal ``FullSolutionOutput`` (a ``Solution``), so the
closed ``OutputUnion``, ``worker.py``, and ``test_outputs.py`` are untouched. The
runner's attribute-based ``_extract_solution`` (getattr(output, "solution")) works
unchanged, and the injected F1 scorer decodes the calendar back out of the Solution.
"""

from typing import List, Tuple

from darwin.problem.schemas import FlowAssignment, Route, Solution

_VEHICLE_ID = "f1_circus"  # the single "vehicle" = the travelling F1 paddock


def calendar_to_solution(
    calendar: List[Tuple[str, int]],
    *,
    solution_id: str = "f1_calendar_solution",
    instance_id: str = "f1_2026_calendar",
    produced_by: str = "unknown",
) -> Solution:
    """Pack an ordered [(race, week)] calendar into a Solution.

    Order -> Route.node_sequence; weeks -> flows (arc_id 'race@wWEEK', qty=week).
    """
    order = [r for r, _ in calendar]
    flows = [
        FlowAssignment(arc_id=f"{r}@w{int(w)}", quantity=float(int(w)))
        for r, w in calendar
    ]
    route = Route(vehicle_id=_VEHICLE_ID, node_sequence=list(order))
    return Solution(
        solution_id=solution_id,
        instance_id=instance_id,
        flows=flows,
        routes=[route],
        produced_by=produced_by,
    )


def solution_to_calendar(solution: Solution) -> List[Tuple[str, int]]:
    """Decode a Solution back into an ordered [(race, week)] calendar.

    Order comes from the Route.node_sequence (authoritative for ORDER); the week
    for each race is read from the matching flow arc_id 'race@wWEEK'. Robust to a
    missing route (falls back to flow order) and to malformed arc_ids (week 0,
    which the F1 scorer will then flag as an out-of-season violation, not a crash).
    """
    # week lookup from flows
    week_of = {}
    for fa in solution.flows or []:
        aid = fa.arc_id
        if "@w" in aid:
            race, _, wk = aid.partition("@w")
            try:
                week_of[race] = int(round(float(wk)))
            except ValueError:
                week_of[race] = int(round(fa.quantity))
        else:
            # tolerate a bare race id: use the quantity as the week
            week_of[aid] = int(round(fa.quantity))

    # order: prefer the route's node_sequence; else fall back to flow order
    order: List[str] = []
    if solution.routes:
        order = list(solution.routes[0].node_sequence)
    if not order:
        order = [fa.arc_id.partition("@w")[0] for fa in (solution.flows or [])]

    return [(race, week_of.get(race, 0)) for race in order]
