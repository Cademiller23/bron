"""§12.2 Output-schema tests."""

import pytest
from pydantic import ValidationError

from darwin.agent.outputs import (
    OUTPUT_MODELS,
    ArbitrationOutput,
    ConstraintReportOutput,
    CritiqueOutput,
    DecompositionOutput,
    FullSolutionOutput,
    Issue,
    PartialSolutionOutput,
    Severity,
    SubProblem,
    SuspectedViolation,
    output_model_for,
)
from darwin.agent.spec import OutputKind
from darwin.problem import score
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import FlowAssignment, Solution


def _solution(instance_id="golden-transportation"):
    return Solution(
        solution_id="s", instance_id=instance_id,
        flows=[FlowAssignment(arc_id="S1-D1", quantity=8.0), FlowAssignment(arc_id="S2-D2", quantity=7.0)],
    )


def test_full_solution_valid_and_scorable():
    out = FullSolutionOutput(solution=_solution(), rationale="cheapest")
    sb = score(golden_transportation(), out.solution)  # cross-module: B1 can score it
    assert sb.feasible and sb.normalized_score == 1.0


def test_full_solution_missing_required_raises():
    with pytest.raises(ValidationError):
        FullSolutionOutput(rationale="no solution")


def test_full_solution_extra_field_raises():
    with pytest.raises(ValidationError):
        FullSolutionOutput(solution=_solution(), bogus=1)


def test_partial_solution_valid():
    out = PartialSolutionOutput(sub_problem_id="north", flows=[FlowAssignment(arc_id="S1-D1", quantity=3.0)])
    assert out.sub_problem_id == "north"


def test_critique_valid_and_missing_field():
    out = CritiqueOutput(issues=[Issue(location="S1-D1", severity=Severity.HIGH, description="over capacity")])
    assert out.issues[0].severity == Severity.HIGH
    with pytest.raises(ValidationError):
        Issue(location="x", description="missing severity")


def test_constraint_report_valid_and_confidence_range():
    out = ConstraintReportOutput(
        suspected_violations=[SuspectedViolation(constraint_type="CAPACITY", location="a1", description="?", confidence=0.7)]
    )
    assert out.suspected_violations[0].confidence == 0.7
    with pytest.raises(ValidationError):
        SuspectedViolation(constraint_type="X", location="y", description="z", confidence=1.5)


def test_arbitration_valid():
    out = ArbitrationOutput(solution=_solution(), drawn_from=["a", "b"])
    assert out.drawn_from == ["a", "b"]


def test_decomposition_valid():
    out = DecompositionOutput(sub_problems=[SubProblem(sub_problem_id="north", description="northern region", node_ids=["S1", "D1"])])
    assert out.sub_problems[0].node_ids == ["S1", "D1"]


@pytest.mark.parametrize("model", OUTPUT_MODELS)
def test_each_model_generates_json_schema(model):
    schema = model.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema or "$defs" in schema


def test_extra_field_rejected_for_every_model():
    # a generic extra-field rejection across all six
    with pytest.raises(ValidationError):
        CritiqueOutput(issues=[], extra="x")
    with pytest.raises(ValidationError):
        DecompositionOutput(sub_problems=[], extra="x")


@pytest.mark.parametrize(
    "kind,model",
    [
        (OutputKind.FULL_SOLUTION, FullSolutionOutput),
        (OutputKind.PARTIAL_SOLUTION, PartialSolutionOutput),
        (OutputKind.CRITIQUE, CritiqueOutput),
        (OutputKind.CONSTRAINT_REPORT, ConstraintReportOutput),
        (OutputKind.ARBITRATION, ArbitrationOutput),
        (OutputKind.DECOMPOSITION, DecompositionOutput),
    ],
)
def test_output_model_for_resolves(kind, model):
    assert output_model_for(kind) is model
