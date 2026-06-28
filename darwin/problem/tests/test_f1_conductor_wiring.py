"""F1-into-Conductor wiring: the instance + runner-signature scorer fit together.

The architect is F1-aware via problem_class, and the runner scores F1 through the
injected adapter (decode calendar from the Solution -> score_f1). These are the two
seams the Conductor relies on; here we prove them with no network.
"""

from darwin.problem.f1_calendar import FEASIBLE_BASELINE
from darwin.problem.f1_codec import calendar_to_solution
from darwin.problem.f1_problem import build_f1_instance
from darwin.problem.f1_scorer import score_f1_solution
from darwin.problem.schemas import ProblemClass, ScoreBreakdown


def test_f1_instance_is_valid_and_flagged_for_the_architect():
    inst = build_f1_instance()
    assert inst.problem_class == ProblemClass.F1_CALENDAR
    assert inst.instance_id == "f1_2026_calendar"
    assert inst.known_optimum.verified  # scoring is anchored


def test_runner_signature_scorer_scores_a_feasible_baseline():
    inst = build_f1_instance()
    solution = calendar_to_solution(FEASIBLE_BASELINE, instance_id=inst.instance_id)
    # Exactly how TeamRunner calls it: scorer(instance, solution, weights)
    breakdown = score_f1_solution(inst, solution, None)
    assert isinstance(breakdown, ScoreBreakdown)
    assert breakdown.feasible is True
    assert breakdown.normalized_score > 0.0


def test_runner_signature_scorer_flags_a_garbage_solution_without_raising():
    inst = build_f1_instance()
    from darwin.problem.schemas import Solution

    junk = Solution(solution_id="junk", instance_id=inst.instance_id, flows=[], routes=[],
                    produced_by="test")
    breakdown = score_f1_solution(inst, junk, None)
    assert breakdown.feasible is False  # malformed/empty -> infeasible, never a crash
