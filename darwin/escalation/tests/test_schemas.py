"""B6 schemas — frozen, extra-forbid, defaults, enum surface."""

import pytest
from pydantic import ValidationError

from darwin.escalation.schemas import (
    CorpusEntry,
    EscalationMethod,
    EscalationResult,
    GapDescription,
    ScoredCorpusEntry,
    SolveBudget,
    SolveResult,
    SolveStatus,
    WeakDimension,
)
from darwin.escalation.fixtures import base_genome, risk_specialist_spec, simple_gap
from darwin.escalation.fixtures import evaluation_with


def test_gap_description_defaults_and_frozen():
    gap = GapDescription(capability_needed="cut cost")
    assert gap.weak_dimensions == [] and gap.severity == 0.0
    with pytest.raises(ValidationError):
        GapDescription(capability_needed="x", severity=-0.1)  # severity ge 0
    with pytest.raises(ValidationError):
        gap.capability_needed = "mutate"  # frozen


def test_gap_extra_forbid():
    with pytest.raises(ValidationError):
        GapDescription(capability_needed="x", bogus=1)


def test_enums_surface():
    assert {m.value for m in EscalationMethod} == {"CORPUS", "CURATED", "NONE_AVAILABLE"}
    assert {s.value for s in SolveStatus} == {"SEALED", "EXHAUSTED"}
    assert {d.value for d in WeakDimension} == {"COST", "LEAD_TIME", "RESILIENCE", "FEASIBILITY"}


def test_corpus_entry_roundtrip():
    e = CorpusEntry(entry_id="e1", agent_spec=risk_specialist_spec(), role_name="r",
                    role_description="d", role_description_embedding=[0.1, 0.2])
    dumped = e.model_dump(mode="json")
    again = CorpusEntry.model_validate(dumped)
    assert again.entry_id == "e1" and again.agent_spec.role_name == "disruption_risk_modeler"


def test_scored_corpus_entry():
    e = CorpusEntry(entry_id="e1", agent_spec=risk_specialist_spec(), role_name="r", role_description="d")
    s = ScoredCorpusEntry(entry=e, similarity=0.8, combined_score=1.2)
    assert s.similarity == 0.8 and s.combined_score == 1.2


def test_solve_budget_defaults_and_bounds():
    b = SolveBudget()
    assert b.max_escalations >= 0 and b.max_team_size >= 1
    with pytest.raises(ValidationError):
        SolveBudget(max_wall_clock_seconds=0)  # gt 0
    with pytest.raises(ValidationError):
        SolveBudget(max_team_size=0)  # ge 1


def test_escalation_result_none_available_holds_no_genome():
    r = EscalationResult(method=EscalationMethod.NONE_AVAILABLE, gap=simple_gap())
    assert r.genome is None and r.added_spec is None


def test_solve_result_carries_genome_and_eval():
    g = base_genome()
    ev = evaluation_with(g, fitness=0.95, normalized=0.95)
    res = SolveResult(instance_id="i", final_genome=g, final_evaluation=ev,
                      cleared_threshold=True, status=SolveStatus.SEALED)
    assert res.final_genome is g and res.final_evaluation.fitness == 0.95
    with pytest.raises(ValidationError):
        res.status = SolveStatus.EXHAUSTED  # frozen
