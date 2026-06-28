"""A9 Assembly tests — design -> genome, deterministic."""

import pytest

from darwin.agent.registry import reset_default_registry
from darwin.agent.spec import InputKind, OutputKind
from darwin.architect import fixtures as AF
from darwin.architect.assembly import AssemblyError, _slugify, assemble
from darwin.architect.schemas import AgentSpecDraft, ArchitectTeamDesign, EdgeDraft, ProblemAnalysis
from darwin.team.genome import EdgeType
from darwin.team.validation import validate


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


def test_assemble_produces_valid_genome():
    g = assemble(AF.valid_design(), "golden-transportation")
    assert len(g.agents) == 4
    assert g.arbiter_id == "arbitrator"
    assert validate(g).valid
    # specs embedded with slugified role_names + architect provenance
    arb = g.node_index["arbitrator"]
    assert arb.spec.role_name == "arbitrator"
    assert arb.spec.created_by == "architect"
    assert arb.spec.model_id == "gemini-3.1-pro"
    # edges resolved from role names to agent_ids
    assert any(e.from_agent_id == "cost_minimizer" and e.to_agent_id == "arbitrator" for e in g.edges)
    # layout positions assigned
    assert all(n.layout is not None for n in g.agents)


def test_slugify():
    assert _slugify("Cost Minimizer!") == "cost_minimizer"
    assert _slugify("disruption-risk modeler") == "disruption_risk_modeler"  # all non-alnum -> _
    assert _slugify("   ") == "agent"
    # every slug is a legal B2 role_name
    import re
    for name in ("Cost Minimizer!", "disruption-risk modeler", "123 go", "a/b\\c"):
        assert re.fullmatch(r"[a-z0-9]+(?:[_-][a-z0-9]+)*", _slugify(name))


def test_duplicate_looking_role_names_resolve_distinctly():
    # two distinct originals that slugify to the same slug
    design = ArchitectTeamDesign(
        analysis=ProblemAnalysis(problem_class="TRANSPORTATION"),
        agents=[
            AgentSpecDraft(role_name="Cost Minimizer", role_description="d1",
                           input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION, model_id="gemini-3.5-flash"),
            AgentSpecDraft(role_name="cost minimizer", role_description="d2",
                           input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION, model_id="gemini-3.5-flash"),
            AgentSpecDraft(role_name="arbitrator", role_description="merge",
                           input_contract=InputKind.SIBLING_OUTPUTS, output_contract=OutputKind.ARBITRATION, model_id="gemini-3.5-flash"),
        ],
        edges=[
            EdgeDraft(from_role_name="Cost Minimizer", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER),
            EdgeDraft(from_role_name="cost minimizer", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER),
        ],
        arbiter_role_name="arbitrator",
    )
    g = assemble(design, "i")
    ids = {n.agent_id for n in g.agents}
    assert "cost_minimizer" in ids and "cost_minimizer-2" in ids  # distinct ids
    # each original role name resolved to its OWN agent_id in the edges
    targets = {e.from_agent_id for e in g.edges if e.to_agent_id == "arbitrator"}
    assert targets == {"cost_minimizer", "cost_minimizer-2"}
    # role_names are ALSO unique (no two agents share a role)
    role_names = [n.spec.role_name for n in g.agents]
    assert len(role_names) == len(set(role_names))


def test_assembly_is_deterministic():
    design = AF.valid_design()
    g1 = assemble(design, "golden-transportation")
    reset_default_registry()
    g2 = assemble(design, "golden-transportation")
    # same structure twice (ids, edges, arbiter, models) — the module's namesake property
    assert [a.agent_id for a in g1.agents] == [a.agent_id for a in g2.agents]
    assert g1.arbiter_id == g2.arbiter_id
    assert {(e.from_agent_id, e.to_agent_id, e.edge_type.value) for e in g1.edges} == \
           {(e.from_agent_id, e.to_agent_id, e.edge_type.value) for e in g2.edges}
    assert {(a.agent_id, a.spec.model_id) for a in g1.agents} == {(a.agent_id, a.spec.model_id) for a in g2.agents}


def test_unresolvable_arbiter_raises():
    with pytest.raises(AssemblyError):
        assemble(AF.design_missing_arbiter(), "i")


def test_unresolvable_edge_raises():
    design = AF.valid_design().model_copy(
        update={"edges": [EdgeDraft(from_role_name="ghost", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER)]}
    )
    with pytest.raises(AssemblyError):
        assemble(design, "i")


def test_invalid_model_id_in_draft_raises_at_assembly():
    design = AF.valid_design().model_copy(
        update={"agents": [d.model_copy(update={"model_id": "not-a-model"}) if d.role_name == "cost minimizer" else d
                           for d in AF.valid_design().agents]}
    )
    with pytest.raises(Exception):  # AgentSpec validation rejects the bad model_id
        assemble(design, "i")
