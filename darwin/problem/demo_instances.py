"""Curated, oracle-verified demo instances for the live Darwin demo (acceptance §10).

Provides 5–8 instances spanning difficulty so that — per the build spec — some
are expected to clear the rearrangement threshold (B6) on their own and some are
hard enough to force escalation. Every instance's optimum is *independently
verified by the oracle* before it is trusted as a demo denominator (the
IndustryOR mislabelling guard).

``expected_to_clear_threshold`` is set heuristically from difficulty here
(EASY → clears, HARD → escalates, MEDIUM → unknown). These are **provisional
staging tags**: they become ground truth only once B5/B6 (the rearrangement loop
and threshold gate) exist to confirm them offline.
"""

import os
from typing import List

import darwin.problem as _pkg
from darwin.problem import oracle
from darwin.problem.fixtures import (
    golden_facility_location,
    golden_transportation,
    golden_vrp,
)
from darwin.problem.generator import generate_instance
from darwin.problem.loader import load_instance
from darwin.problem.schemas import (
    Difficulty,
    KnownOptimum,
    OptimumSource,
    ProblemClass,
    ProblemInstance,
)

_DATA = os.path.join(os.path.dirname(_pkg.__file__), "data")

_STAGE_BY_DIFFICULTY = {
    Difficulty.EASY: True,  # rearrangement alone is expected to clear the threshold
    Difficulty.MEDIUM: None,  # provisional / to be confirmed by B5
    Difficulty.HARD: False,  # expected to force escalation
}


def _verify_and_attach(instance: ProblemInstance) -> ProblemInstance:
    """Independently verify the optimum with the oracle and attach it as
    SOLVER_VERIFIED, plus a provisional threshold-staging tag."""
    result = oracle.solve_optimum(instance)
    if result.status != oracle.STATUS_OPTIMAL or result.objective_value is None:
        raise RuntimeError(f"demo instance {instance.instance_id} is not solvable: {result.status}")

    known = KnownOptimum(
        objective_value=round(result.objective_value, 9),
        source=OptimumSource.SOLVER_VERIFIED,
        verified=True,
        solver_used=f"oracle:{result.backend}",
    )
    meta = instance.metadata.model_copy(
        update={"expected_to_clear_threshold": _STAGE_BY_DIFFICULTY[instance.metadata.difficulty]}
    )
    return instance.model_copy(update={"known_optimum": known, "metadata": meta})


def curated_demo_instances() -> List[ProblemInstance]:
    """Return the curated, oracle-verified demo suite (8 instances)."""
    raw = [
        golden_transportation(),  # EASY transportation
        golden_facility_location(),  # MEDIUM facility location
        golden_vrp(),  # MEDIUM vehicle routing
        generate_instance(101, ProblemClass.TRANSPORTATION),  # EASY generated
        generate_instance(202, ProblemClass.TRANSSHIPMENT),  # MEDIUM generated
        generate_instance(303, ProblemClass.FACILITY_LOCATION, {"num_optional": 4, "num_sinks": 4}),  # HARD generated
        load_instance("industryor", os.path.join(_DATA, "industryor_sample.json")),  # EASY benchmark
        load_instance("mamo", os.path.join(_DATA, "mamo_sample.json")),  # MEDIUM benchmark
    ]
    return [_verify_and_attach(i) for i in raw]


def verification_report() -> List[dict]:
    """Per-instance verification summary for the demo dashboard / acceptance check."""
    report = []
    for inst in curated_demo_instances():
        result = oracle.solve_optimum(inst)
        report.append(
            {
                "instance_id": inst.instance_id,
                "problem_class": inst.problem_class.value,
                "difficulty": inst.metadata.difficulty.value,
                "labeled_optimum": inst.known_optimum.objective_value,
                "solver_optimum": result.objective_value,
                "verified": inst.known_optimum.verified,
                "expected_to_clear_threshold": inst.metadata.expected_to_clear_threshold,
                "backend": result.backend,
            }
        )
    return report
