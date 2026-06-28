"""§8.6 Generator tests — fresh, feasible, optimum-attached, reproducible."""

import pytest

from darwin.problem import fixtures as fx
from darwin.problem import oracle
from darwin.problem.generator import _signature, generate_instance
from darwin.problem.scorer import score
from darwin.problem.schemas import OptimumSource, ProblemClass, ProblemInstance

_CLASSES = [
    ProblemClass.TRANSPORTATION,
    ProblemClass.TRANSSHIPMENT,
    ProblemClass.FACILITY_LOCATION,
    ProblemClass.VEHICLE_ROUTING,
]


@pytest.mark.parametrize("pclass", _CLASSES)
def test_generated_instance_passes_schema_validation(pclass):
    inst = generate_instance(seed=42, problem_class=pclass)
    # round-trips cleanly through the validators
    assert ProblemInstance(**inst.model_dump()) == inst


@pytest.mark.parametrize("pclass", _CLASSES)
def test_generated_instance_is_feasible_with_attached_verified_optimum(pclass):
    inst = generate_instance(seed=42, problem_class=pclass)
    assert inst.known_optimum is not None
    assert inst.known_optimum.verified is True
    assert inst.known_optimum.source == OptimumSource.SOLVER_VERIFIED
    result = oracle.solve_optimum(inst)
    assert result.status == oracle.STATUS_OPTIMAL
    # the attached optimum matches a fresh oracle solve, and its solution scores 1.0
    import math

    assert math.isclose(inst.known_optimum.objective_value, result.objective_value, rel_tol=1e-6, abs_tol=1e-6)
    sb = score(inst, result.solution)
    assert sb.feasible and math.isclose(sb.normalized_score, 1.0)


@pytest.mark.parametrize("pclass", _CLASSES)
def test_same_seed_reproducible_different_seed_differs(pclass):
    a = generate_instance(seed=7, problem_class=pclass)
    b = generate_instance(seed=7, problem_class=pclass)
    c = generate_instance(seed=8, problem_class=pclass)
    assert a.model_dump() == b.model_dump()
    assert a.model_dump() != c.model_dump()


def test_generated_not_numerically_identical_to_preloaded():
    first = generate_instance(seed=11, problem_class=ProblemClass.TRANSPORTATION)
    # asking again with the same seed but declaring `first` as preloaded forces
    # the duplicate guard to produce a numerically different instance.
    second = generate_instance(seed=11, problem_class=ProblemClass.TRANSPORTATION, existing=[first])
    assert _signature(second) != _signature(first)


def test_generator_avoids_collision_with_golden_fixtures():
    inst = generate_instance(
        seed=3,
        problem_class=ProblemClass.TRANSPORTATION,
        existing=[fx.golden_transportation()],
    )
    assert _signature(inst) != _signature(fx.golden_transportation())
