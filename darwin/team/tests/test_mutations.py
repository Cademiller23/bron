"""Tests for the typed mutation operations (the genome-edit vocabulary)."""

import pytest

from darwin.agent.registry import (
    CapabilityTier,
    ModelEntry,
    Provider,
    default_registry,
    reset_default_registry,
)
from darwin.agent.spec import AgentSpec, InputKind, OutputKind
from darwin.team import fixtures as F
from darwin.team import mutations as M
from darwin.team.genome import AgentNode, Edge, EdgeType, MutationActor, MutationType


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


def test_add_edge_appends_and_records():
    g = F.proposer_checker_arbiter_genome()
    new = Edge(from_agent_id="p2", to_agent_id="chk", edge_type=EdgeType.CHECKS)
    set_ops, record = M.add_edge(new)(g)
    assert len(set_ops["edges"]) == len(g.edges) + 1
    assert record.mutation_type == MutationType.REARRANGE_EDGE
    assert record.from_version == g.version and record.to_version == g.version + 1


def test_swap_model_valid_and_invalid():
    default_registry().register(
        ModelEntry(model_id="fast-cheap", provider=Provider.OPENAI_COMPAT, endpoint="https://x/v1",
                   capability_tier=CapabilityTier.CHEAP)
    )
    g = F.proposer_checker_arbiter_genome()
    set_ops, record = M.swap_model("p1", "fast-cheap")(g)
    p1_doc = next(a for a in set_ops["agents"] if a["agent_id"] == "p1")
    assert p1_doc["spec"]["model_id"] == "fast-cheap"
    assert record.mutation_type == MutationType.SWAP_MODEL

    with pytest.raises(Exception):  # bad model rejected by AgentSpec re-validation
        M.swap_model("p1", "ghost-model")(g)


def test_retarget_arbiter():
    g = F.proposer_checker_arbiter_genome()
    # make p1 a valid terminal arbiter candidate first by removing its non-arbiter edges
    # (here we just verify the op produces the right set_ops/record shape)
    set_ops, record = M.retarget_arbiter("p1")(g)
    assert set_ops["arbiter_id"] == "p1"
    assert record.mutation_type == MutationType.RETARGET_ARBITER


def test_add_agent_grows_graph():
    g = F.proposer_checker_arbiter_genome()
    node = AgentNode(agent_id="risk", spec=AgentSpec(
        agent_id="risk", role_name="resilience_analyst", role_description="assess disruption risk",
        input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION))
    edges = [Edge(from_agent_id="risk", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER)]
    set_ops, record = M.add_agent(node, edges, mutation_type=MutationType.ADD_CURATED_AGENT,
                                  actor=MutationActor.ESCALATION)(g)
    assert len(set_ops["agents"]) == len(g.agents) + 1
    assert len(set_ops["edges"]) == len(g.edges) + 1
    assert record.mutation_type == MutationType.ADD_CURATED_AGENT
    assert record.actor == MutationActor.ESCALATION


def test_add_agent_rejects_wrong_mutation_type():
    node = AgentNode(agent_id="x", spec=AgentSpec(
        agent_id="x", role_name="r", role_description="d",
        input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION))
    with pytest.raises(ValueError):
        M.add_agent(node, [], mutation_type=MutationType.REARRANGE_EDGE)


def test_remove_agent_and_its_edges():
    g = F.proposer_checker_arbiter_genome()
    set_ops, record = M.remove_agent("p2")(g)
    assert all(a["agent_id"] != "p2" for a in set_ops["agents"])
    assert all("p2" not in (e["from_agent_id"], e["to_agent_id"]) for e in set_ops["edges"])
    assert record.mutation_type == MutationType.REMOVE_AGENT


def test_cannot_remove_arbiter():
    g = F.proposer_checker_arbiter_genome()
    with pytest.raises(ValueError):
        M.remove_agent("arb")(g)


def test_remove_unknown_agent_raises():
    g = F.proposer_checker_arbiter_genome()
    with pytest.raises(KeyError):
        M.remove_agent("nope")(g)


async def test_mutation_applied_end_to_end_via_store():
    default_registry().register(
        ModelEntry(model_id="fast-cheap", provider=Provider.OPENAI_COMPAT, endpoint="https://x/v1")
    )
    g = F.proposer_checker_arbiter_genome()
    store = F.new_store()
    await store.save_new(g)
    result = await store.retry_mutate(g.genome_id, M.swap_model("p1", "fast-cheap"))
    assert result.version == 2
    p1 = next(a for a in result.agents if a.agent_id == "p1")
    assert p1.spec.model_id == "fast-cheap"
    assert result.history[-1].mutation_type == MutationType.SWAP_MODEL
