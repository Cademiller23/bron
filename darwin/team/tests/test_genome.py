"""§12.4 Genome tests — construction, referential integrity, round-trip."""

import pytest
from pydantic import ValidationError

from darwin.agent.registry import reset_default_registry
from darwin.agent.spec import AgentSpec, InputKind, OutputKind
from darwin.team.genome import (
    AgentNode,
    Edge,
    EdgeType,
    GenomeStatus,
    MutationType,
    TeamGenome,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


def _node(agent_id, oc=OutputKind.FULL_SOLUTION, ic=InputKind.FULL_PROBLEM):
    return AgentNode(agent_id=agent_id, spec=AgentSpec(
        agent_id=agent_id, role_name="role_x", role_description="x", input_contract=ic, output_contract=oc))


def _genome(**over):
    kwargs = dict(
        instance_id="golden-transportation",
        agents=[_node("p1"), _node("a1", OutputKind.ARBITRATION, InputKind.SIBLING_OUTPUTS)],
        edges=[Edge(from_agent_id="p1", to_agent_id="a1", edge_type=EdgeType.FEEDS_ARBITER)],
        arbiter_id="a1",
    )
    kwargs.update(over)
    return TeamGenome.create(**kwargs)


def test_create_defaults_and_initial_history():
    g = _genome()
    assert g.version == 1
    assert g.status == GenomeStatus.DRAFT
    assert len(g.history) == 1
    assert g.history[0].mutation_type == MutationType.INITIAL_CURATION
    assert g.history[0].from_version == 0 and g.history[0].to_version == 1


def test_arbiter_must_be_a_node():
    with pytest.raises(ValidationError):
        _genome(arbiter_id="ghost")


def test_edges_must_reference_existing_nodes():
    with pytest.raises(ValidationError):
        _genome(edges=[Edge(from_agent_id="p1", to_agent_id="ghost", edge_type=EdgeType.FEEDS_ARBITER)])


def test_duplicate_agent_ids_rejected():
    with pytest.raises(ValidationError):
        TeamGenome.create(instance_id="i", agents=[_node("p1"), _node("p1")], edges=[], arbiter_id="p1")


def test_node_id_must_match_embedded_spec_id():
    spec = AgentSpec(agent_id="REAL", role_name="r", role_description="d",
                     input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION)
    with pytest.raises(ValidationError):
        AgentNode(agent_id="DIFFERENT", spec=spec)  # node id != embedded spec id


def test_frozen_and_extra_forbidden():
    g = _genome()
    with pytest.raises(ValidationError):
        g.version = 99
    with pytest.raises(ValidationError):
        TeamGenome.model_validate({**g.model_dump(), "bogus": 1})


def test_round_trip_identical_including_embedded_specs_and_history():
    g = _genome()
    restored = TeamGenome.model_validate(g.model_dump())
    assert restored == g
    assert restored.agents[0].spec == g.agents[0].spec
    assert restored.history == g.history
    # JSON round-trip too (the form the store persists)
    assert TeamGenome.model_validate(g.model_dump(mode="json")) == g


def test_convenience_views():
    g = _genome()
    assert g.proposer_ids() == ["p1"]
    assert g.arbiter_feeder_ids() == ["p1"]
    assert [e.from_agent_id for e in g.upstream_edges("a1")] == ["p1"]
    assert g.downstream_edges("p1")[0].to_agent_id == "a1"
