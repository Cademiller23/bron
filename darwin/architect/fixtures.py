"""Canned Architect designs + test doubles (no network)."""

from typing import List, Optional, Union

from darwin.agent.client import ModelResponse, ProviderError, Usage
from darwin.agent.fixtures import scripted_client
from darwin.agent.registry import reset_default_registry
from darwin.agent.spec import InputKind, OutputKind, ThinkingLevel
from darwin.architect.architect import Architect
from darwin.architect.schemas import (
    AgentSpecDraft,
    ArchitectTeamDesign,
    CuratedAgentDraft,
    EdgeDraft,
    ProblemAnalysis,
)
from darwin.constants import ARCHITECT_MODEL_ID, FAST_MODEL_ID
from darwin.team.fixtures import FakeMongoCollection
from darwin.team.genome import EdgeType
from darwin.team.store import GenomeStore


def analysis_fixture() -> ProblemAnalysis:
    return ProblemAnalysis(
        problem_class="TRANSPORTATION", dominant_objectives=["cost"],
        binding_constraints=["capacity"], difficulty_estimate="EASY", suggested_part_count=3,
        rationale="small cost-dominated transportation problem",
    )


def _draft(role, desc, ic, oc, model=FAST_MODEL_ID, tl=ThinkingLevel.MEDIUM) -> AgentSpecDraft:
    return AgentSpecDraft(role_name=role, role_description=desc, input_contract=ic, output_contract=oc,
                          model_id=model, thinking_level=tl, responsibility=role, why_this_model="fit")


def valid_design() -> ArchitectTeamDesign:
    return ArchitectTeamDesign(
        analysis=analysis_fixture(),
        agents=[
            _draft("cost minimizer", "Reason directly; produce a low-cost feasible Solution. Never call a solver.",
                   InputKind.FULL_PROBLEM, OutputKind.FULL_SOLUTION),
            _draft("risk analyst", "Reason directly; produce a resilient feasible Solution. Never call a solver.",
                   InputKind.FULL_PROBLEM, OutputKind.FULL_SOLUTION),
            _draft("capacity auditor", "Audit proposals for capacity feasibility.",
                   InputKind.SIBLING_OUTPUTS, OutputKind.CRITIQUE),
            _draft("arbitrator", "Synthesize the final Solution from siblings. Reason directly; never call a solver.",
                   InputKind.SIBLING_OUTPUTS, OutputKind.ARBITRATION, model=ARCHITECT_MODEL_ID, tl=ThinkingLevel.HIGH),
        ],
        edges=[
            EdgeDraft(from_role_name="cost minimizer", to_role_name="capacity auditor", edge_type=EdgeType.CHECKS),
            EdgeDraft(from_role_name="cost minimizer", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER),
            EdgeDraft(from_role_name="risk analyst", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER),
            EdgeDraft(from_role_name="capacity auditor", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER),
        ],
        arbiter_role_name="arbitrator",
        design_rationale="parallel proposers -> auditor -> arbitrator",
    )


def design_missing_arbiter() -> ArchitectTeamDesign:
    d = valid_design()
    return d.model_copy(update={"arbiter_role_name": "nonexistent_role"})


def curated_agent_fixture() -> CuratedAgentDraft:
    return CuratedAgentDraft(
        agent=_draft("disruption risk modeler", "Reason directly about disruption risk; produce a resilient Solution. Never call a solver.",
                     InputKind.FULL_PROBLEM, OutputKind.FULL_SOLUTION),
        # wire to the arbiter by its ROLE name (assembly resolves role -> agent_id)
        edges=[EdgeDraft(from_role_name="disruption risk modeler", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER)],
        rationale="adds the missing resilience capability",
    )


# ---------------------------------------------------------------------------
# Model-response builders + scripted Architect
# ---------------------------------------------------------------------------
def model_response(payload) -> ModelResponse:
    """A canned structured response carrying a pydantic model as parsed JSON."""
    text = payload.model_dump_json()
    return ModelResponse(
        raw_text=text, parsed=payload.model_dump(mode="json"),
        usage=Usage(tokens_in=200, tokens_out=300), latency_ms=1500.0,
        model_id=ARCHITECT_MODEL_ID, finish_reason="stop",
    )


def make_architect(script: List[Union[ModelResponse, BaseException]], *, with_store: bool = True):
    """Build (architect, store, client) wired with a scripted frontier model."""
    reset_default_registry()
    client = scripted_client(script)
    store = GenomeStore(FakeMongoCollection()) if with_store else None
    architect = Architect(client, store=store)
    return architect, store, client
