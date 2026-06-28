"""§12.4 GenomeEvaluation tests — always a real fitness + a score_breakdown."""

import pytest

from darwin.problem import score
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import FlowAssignment, ObjectiveWeights, Solution
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import ArbiterTier


def _evaluation(**over):
    inst = golden_transportation()
    sol = Solution(solution_id="s", instance_id=inst.instance_id,
                   flows=[FlowAssignment(arc_id="S1-D1", quantity=8.0), FlowAssignment(arc_id="S2-D2", quantity=7.0)])
    sb = score(inst, sol, ObjectiveWeights.cost_only())
    kwargs = dict(
        genome_id="g1", version=1, instance_id=inst.instance_id, completed=True,
        final_solution=sol, score_breakdown=sb, fitness=sb.final_fitness,
        normalized_score=sb.normalized_score, cleared_threshold=sb.normalized_score >= 0.90,
        arbiter_tier_used=ArbiterTier.PRIMARY,
    )
    kwargs.update(over)
    return GenomeEvaluation(**kwargs)


def test_evaluation_carries_real_fitness_and_breakdown():
    ev = _evaluation()
    assert isinstance(ev.fitness, float)
    assert ev.score_breakdown is not None
    assert ev.cleared_threshold is True


def test_evaluation_is_frozen_and_strict():
    from pydantic import ValidationError

    ev = _evaluation()
    with pytest.raises(ValidationError):
        ev.fitness = 0.0
    with pytest.raises(ValidationError):
        GenomeEvaluation(**{**_evaluation().model_dump(), "bogus": 1})


def test_error_field_optional_defaults_none():
    assert _evaluation().error is None
    assert _evaluation(error="boundary fired").error == "boundary fired"
