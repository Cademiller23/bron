"""B8 Operator tests — valid candidates, agent-set-invariant, correct mutation."""

import random

import pytest

from darwin.agent.registry import default_registry, reset_default_registry
from darwin.rearrange import operators as O
from darwin.team import fixtures as TF
from darwin.team.genome import MutationType
from darwin.team.validation import validate


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


def _genome():
    return TF.proposer_checker_arbiter_genome()


def _first_valid(op, max_seed=40):
    reg = default_registry()
    for seed in range(max_seed):
        c = op(_genome(), random.Random(seed), reg)
        if c is not None:
            return c
    return None


@pytest.mark.parametrize(
    "op,expected_mut",
    [
        (O.reassign_model, MutationType.SWAP_MODEL),
        (O.reorder_pipeline, MutationType.REARRANGE_EDGE),
        (O.swap_arbiter, MutationType.RETARGET_ARBITER),
    ],
)
def test_operator_produces_valid_agent_set_invariant_candidate(op, expected_mut):
    c = _first_valid(op)
    assert c is not None, "operator never produced a valid candidate"
    assert validate(c.genome, default_registry()).valid
    base = {a.agent_id for a in _genome().agents}
    assert {a.agent_id for a in c.genome.agents} == base  # agent set invariant
    assert c.mutation_type == expected_mut


def test_reassign_model_changes_only_one_model():
    c = _first_valid(O.reassign_model)
    base = _genome()
    base_models = {a.agent_id: a.spec.model_id for a in base.agents}
    changed = [a.agent_id for a in c.genome.agents if a.spec.model_id != base_models[a.agent_id]]
    assert len(changed) == 1  # exactly one agent's model changed
    # edges + arbiter unchanged
    assert {(e.from_agent_id, e.to_agent_id) for e in c.genome.edges} == {(e.from_agent_id, e.to_agent_id) for e in base.edges}
    assert c.genome.arbiter_id == base.arbiter_id


def test_swap_arbiter_changes_the_arbiter():
    c = _first_valid(O.swap_arbiter)
    assert c.genome.arbiter_id != _genome().arbiter_id
    # the new arbiter is terminal (no outgoing edges)
    assert not c.genome.downstream_edges(c.genome.arbiter_id)


def test_reorder_changes_an_edge_only():
    c = _first_valid(O.reorder_pipeline)
    base = _genome()
    assert c.genome.arbiter_id == base.arbiter_id
    assert {a.spec.model_id for a in c.genome.agents} == {a.spec.model_id for a in base.agents}
    # the edge set differs (something was rewired)
    assert {(e.from_agent_id, e.to_agent_id, e.edge_type.value) for e in c.genome.edges} != \
           {(e.from_agent_id, e.to_agent_id, e.edge_type.value) for e in base.edges}


def test_operator_derive_is_commit_compatible():
    # the candidate's derive_fn yields a (set_ops, record) usable by store.mutate
    c = _first_valid(O.reassign_model)
    set_ops, record = c.derive_fn(_genome())
    assert "agents" in set_ops
    assert record.actor.value == "REARRANGER"


def _redirect_friendly_genome():
    """A genome on which redirect_edge can succeed: a redundant CHECKS edge into a
    FULL_PROBLEM proposer (which ignores it), so retargeting it stays valid."""
    from darwin.agent.spec import AgentSpec, InputKind, OutputKind
    from darwin.team.genome import AgentNode, Edge, EdgeType, TeamGenome

    def node(aid, oc, ic):
        return AgentNode(agent_id=aid, spec=AgentSpec(agent_id=aid, role_name=aid, role_description="d",
                                                      input_contract=ic, output_contract=oc))
    agents = [node("p1", OutputKind.FULL_SOLUTION, InputKind.FULL_PROBLEM),
              node("p2", OutputKind.FULL_SOLUTION, InputKind.FULL_PROBLEM),
              node("p3", OutputKind.FULL_SOLUTION, InputKind.FULL_PROBLEM),
              node("arb", OutputKind.ARBITRATION, InputKind.SIBLING_OUTPUTS)]
    edges = [Edge(from_agent_id="p1", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER),
             Edge(from_agent_id="p2", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER),
             Edge(from_agent_id="p3", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER),
             Edge(from_agent_id="p1", to_agent_id="p2", edge_type=EdgeType.CHECKS)]  # redundant -> redirectable
    return TeamGenome.create(instance_id="i", agents=agents, edges=edges, arbiter_id="arb")


def test_redirect_edge_produces_valid_candidate_when_applicable():
    reg = default_registry()
    got = None
    for seed in range(40):
        c = O.redirect_edge(_redirect_friendly_genome(), random.Random(seed), reg)
        if c is not None:
            got = c
            assert validate(c.genome, reg).valid
            assert {a.agent_id for a in c.genome.agents} == {"p1", "p2", "p3", "arb"}  # agent set invariant
    assert got is not None  # redirect_edge DOES produce valid candidates here


@pytest.mark.parametrize("op", [O.reassign_model, O.reorder_pipeline, O.swap_arbiter])
def test_operators_never_produce_duplicate_edges(op):
    reg = default_registry()
    for seed in range(40):
        c = op(_genome(), random.Random(seed), reg)
        if c is None:
            continue
        keys = [(e.from_agent_id, e.to_agent_id, e.edge_type.value) for e in c.genome.edges]
        assert len(keys) == len(set(keys))  # no duplicate edges
