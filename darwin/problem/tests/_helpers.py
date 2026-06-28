"""Shared test helpers and Hypothesis strategies."""

from typing import List

from hypothesis import strategies as st

from darwin.problem.schemas import FlowAssignment, ProblemInstance, Solution


def make_solution(instance: ProblemInstance, flows_by_arc: dict, **kwargs) -> Solution:
    """Convenience builder for a flow solution from an ``{arc_id: qty}`` dict."""
    return Solution(
        solution_id=kwargs.pop("solution_id", "test-sol"),
        instance_id=instance.instance_id,
        flows=[FlowAssignment(arc_id=a, quantity=q) for a, q in flows_by_arc.items()],
        **kwargs,
    )


def random_flow_solution_strategy(instance: ProblemInstance):
    """A Hypothesis strategy producing arbitrary (mostly infeasible) flow
    solutions over the arcs of ``instance`` — used for invariant property tests."""
    arc_ids = [a.arc_id for a in instance.arcs]

    @st.composite
    def _strategy(draw):
        flows = []
        for arc_id in arc_ids:
            if draw(st.booleans()):
                qty = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False))
                flows.append(FlowAssignment(arc_id=arc_id, quantity=qty))
        return Solution(solution_id="prop-sol", instance_id=instance.instance_id, flows=flows)

    return _strategy()


@st.composite
def feasible_transportation_solution(draw):
    """A Hypothesis strategy that emits *guaranteed-feasible* solutions for the
    golden transportation instance (S1/S2 supply 10; D1 demand 8, D2 demand 7).

    Serving D1 entirely from S1 (a ∈ [8, 10]) and D2 entirely from S2
    (b ∈ [7, 10]) satisfies every constraint, so the feasible branch of the
    invariant property tests is reliably exercised.
    """
    from darwin.problem.fixtures import golden_transportation

    inst = golden_transportation()
    a = draw(st.floats(min_value=8.0, max_value=10.0))
    b = draw(st.floats(min_value=7.0, max_value=10.0))
    return make_solution(inst, {"S1-D1": a, "S2-D2": b}, solution_id="feasible-prop")
