"""A9 Architect schema tests."""

import pytest
from pydantic import ValidationError

from darwin.agent.spec import InputKind, OutputKind
from darwin.architect.schemas import (
    AgentSpecDraft,
    ArchitectTeamDesign,
    EdgeDraft,
    ProblemAnalysis,
)
from darwin.team.genome import EdgeType


def _draft(**over):
    kw = dict(role_name="cost minimizer", role_description="do the thing",
              input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION,
              model_id="gemini-3.5-flash")
    kw.update(over)
    return AgentSpecDraft(**kw)


def test_problem_analysis_constructs_and_round_trips():
    pa = ProblemAnalysis(problem_class="TRANSPORTATION", dominant_objectives=["cost"], suggested_part_count=3)
    assert ProblemAnalysis.model_validate(pa.model_dump()) == pa
    assert pa.suggested_part_count == 3


def test_problem_analysis_rejects_part_count_below_one():
    with pytest.raises(ValidationError):
        ProblemAnalysis(problem_class="X", suggested_part_count=0)


def test_agent_spec_draft_valid_and_empty_role_rejected():
    d = _draft()
    assert d.role_name == "cost minimizer"
    with pytest.raises(ValidationError):
        _draft(role_name="")
    with pytest.raises(ValidationError):
        _draft(role_description="")


def test_agent_spec_draft_invalid_contract_rejected():
    with pytest.raises(ValidationError):
        _draft(input_contract="NOT_A_KIND")
    with pytest.raises(ValidationError):
        _draft(output_contract="NOPE")


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        _draft(bogus=1)
    with pytest.raises(ValidationError):
        EdgeDraft(from_role_name="a", to_role_name="b", edge_type=EdgeType.FEEDS_ARBITER, extra="x")


def test_team_design_requires_at_least_one_agent():
    with pytest.raises(ValidationError):
        ArchitectTeamDesign(
            analysis=ProblemAnalysis(problem_class="X"), agents=[], arbiter_role_name="a"
        )


def test_team_design_round_trips():
    design = ArchitectTeamDesign(
        analysis=ProblemAnalysis(problem_class="TRANSPORTATION"),
        agents=[_draft(), _draft(role_name="arbitrator", output_contract=OutputKind.ARBITRATION,
                                 input_contract=InputKind.SIBLING_OUTPUTS)],
        edges=[EdgeDraft(from_role_name="cost minimizer", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER)],
        arbiter_role_name="arbitrator",
    )
    assert ArchitectTeamDesign.model_validate(design.model_dump()) == design
    assert ArchitectTeamDesign.model_validate_json(design.model_dump_json()) == design
