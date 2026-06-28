"""§8.4 Oracle tests — ground truth, and the loop between oracle and scorer."""

import math

import pytest

from darwin.problem import fixtures as fx
from darwin.problem import oracle
from darwin.problem.scorer import score
from darwin.problem.schemas import (
    Arc,
    KnownOptimum,
    Node,
    NodeType,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
)

_BACKENDS = ["auto", "pure"]  # ortools (auto) + pure-Python fallback


@pytest.mark.parametrize("backend", _BACKENDS)
def test_transportation_optimum(backend):
    r = oracle.solve_optimum(fx.golden_transportation(), backend=backend)
    assert r.status == oracle.STATUS_OPTIMAL
    assert math.isclose(r.objective_value, 23.0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_facility_optimum(backend):
    r = oracle.solve_optimum(fx.golden_facility_location(), backend=backend)
    assert r.status == oracle.STATUS_OPTIMAL
    assert math.isclose(r.objective_value, 130.0)
    assert r.solution.open_facilities and len(r.solution.open_facilities) == 1


def test_vrp_optimum():
    r = oracle.solve_optimum(fx.golden_vrp())
    assert r.status == oracle.STATUS_OPTIMAL
    assert math.isclose(r.objective_value, fx.VRP_OPTIMUM)


@pytest.mark.parametrize(
    "inst",
    [fx.golden_transportation(), fx.golden_facility_location(), fx.golden_vrp()],
)
def test_oracle_solution_scores_feasible_and_optimal(inst):
    """Closes the loop: the oracle's own solution must score feasible with
    normalized_score == 1.0 under the scorer."""
    r = oracle.solve_optimum(inst)
    sb = score(inst, r.solution)
    assert sb.feasible
    assert math.isclose(sb.normalized_score, 1.0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_infeasible_instance_reported_not_wrong_number(backend):
    inst = ProblemInstance(
        instance_id="infeasible", source="fixture", problem_class=ProblemClass.TRANSPORTATION,
        nodes=[Node(node_id="S1", node_type=NodeType.SOURCE, supply=10.0),
               Node(node_id="D1", node_type=NodeType.SINK, demand=10.0)],
        arcs=[Arc(arc_id="a", from_node="S1", to_node="D1", unit_cost=1.0, capacity=5.0)],  # can't deliver 10
    )
    r = oracle.solve_optimum(inst, backend=backend)
    assert r.status == oracle.STATUS_INFEASIBLE
    assert r.objective_value is None


def test_ortools_and_pure_backends_agree():
    for inst in (fx.golden_transportation(), fx.golden_facility_location()):
        a = oracle.solve_optimum(inst, backend="ortools")
        b = oracle.solve_optimum(inst, backend="pure")
        assert a.status == b.status == oracle.STATUS_OPTIMAL
        assert math.isclose(a.objective_value, b.objective_value)


def test_label_verification_accepts_correct_label():
    agrees, labeled, solver_value, status = oracle.verify_label(fx.golden_transportation())
    assert agrees is True
    assert math.isclose(labeled, 23.0) and math.isclose(solver_value, 23.0)


def test_label_verification_detects_wrong_label():
    inst = fx.golden_transportation().model_copy(
        update={"known_optimum": KnownOptimum(objective_value=999.0, source=OptimumSource.BENCHMARK_LABEL)}
    )
    agrees, labeled, solver_value, status = oracle.verify_label(inst)
    assert agrees is False
    assert math.isclose(labeled, 999.0)
    assert math.isclose(solver_value, 23.0)  # the oracle's true value surfaces the mislabel
