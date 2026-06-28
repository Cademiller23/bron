"""Acceptance §10 — the curated demo suite is loaded and oracle-verified."""

import math

from darwin.problem import oracle
from darwin.problem.demo_instances import curated_demo_instances
from darwin.problem.scorer import score
from darwin.problem.schemas import Difficulty


def test_demo_suite_has_five_to_eight_instances():
    suite = curated_demo_instances()
    assert 5 <= len(suite) <= 8


def test_every_demo_instance_is_oracle_verified_and_scores_one():
    for inst in curated_demo_instances():
        assert inst.known_optimum is not None
        assert inst.known_optimum.verified is True
        result = oracle.solve_optimum(inst)
        assert result.status == oracle.STATUS_OPTIMAL
        assert math.isclose(inst.known_optimum.objective_value, result.objective_value, rel_tol=1e-6, abs_tol=1e-6)
        sb = score(inst, result.solution)
        assert sb.feasible and math.isclose(sb.normalized_score, 1.0)


def test_demo_suite_spans_difficulty():
    difficulties = {inst.metadata.difficulty for inst in curated_demo_instances()}
    assert {Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD} <= difficulties


def test_demo_suite_has_threshold_clearing_and_escalating_instances():
    stages = [inst.metadata.expected_to_clear_threshold for inst in curated_demo_instances()]
    assert any(s is True for s in stages)  # some clear by rearrangement alone
    assert any(s is False for s in stages)  # some force escalation
