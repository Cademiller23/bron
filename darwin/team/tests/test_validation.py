"""§12.4 / §3.1 Validation tests — reject malformed teams before they run."""

import pytest

from darwin.agent.registry import reset_default_registry
from darwin.team import fixtures as F
from darwin.team.genome import Edge, EdgeType, TeamGenome
from darwin.team.validation import validate


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


def _with_edges(genome, edges):
    return TeamGenome.model_validate({**genome.model_dump(), "edges": [e.model_dump() for e in edges]})


def test_valid_proposer_checker_arbiter_passes():
    assert validate(F.proposer_checker_arbiter_genome()).valid


def test_one_agent_genome_passes():
    assert validate(F.one_agent_genome()).valid


def test_cycle_is_rejected():
    g = F.proposer_checker_arbiter_genome()
    bad = _with_edges(g, list(g.edges) + [Edge(from_agent_id="arb", to_agent_id="p1", edge_type=EdgeType.PASSES_PROPOSAL)])
    result = validate(bad)
    assert not result.valid


def test_arbiter_with_downstream_edge_is_rejected():
    g = F.proposer_checker_arbiter_genome()
    bad = _with_edges(g, list(g.edges) + [Edge(from_agent_id="arb", to_agent_id="chk", edge_type=EdgeType.PASSES_PROPOSAL)])
    result = validate(bad)
    assert not result.valid
    assert any("terminal" in e for e in result.errors)


def test_orphan_agent_is_rejected():
    # p2 feeds nobody -> never reaches the arbiter
    g = F.proposer_checker_arbiter_genome()
    edges = [e for e in g.edges if e.from_agent_id != "p2"]
    bad = _with_edges(g, edges)
    result = validate(bad)
    assert not result.valid
    assert any("orphan" in e for e in result.errors)


def test_duplicate_edge_is_rejected():
    # a duplicate (from, to, type) would make the runner feed the arbiter twice
    g = F.proposer_checker_arbiter_genome()
    dup = _with_edges(g, list(g.edges) + [Edge(from_agent_id="p2", to_agent_id="arb", edge_type=EdgeType.FEEDS_ARBITER)])
    result = validate(dup)
    assert not result.valid
    assert any("duplicate edge" in e for e in result.errors)


def test_feeds_arbiter_must_terminate_at_arbiter():
    g = F.proposer_checker_arbiter_genome()
    edges = [Edge(from_agent_id="p1", to_agent_id="chk", edge_type=EdgeType.FEEDS_ARBITER)] + [
        e for e in g.edges if not (e.from_agent_id == "p1" and e.to_agent_id == "chk")
    ]
    result = validate(_with_edges(g, edges))
    assert not result.valid
    assert any("FEEDS_ARBITER" in e for e in result.errors)


def test_sibling_outputs_agent_without_upstream_is_rejected():
    # an arbiter expecting SIBLING_OUTPUTS but with no feeders
    g = F.proposer_checker_arbiter_genome()
    bad = _with_edges(g, [])  # no edges at all
    result = validate(bad)
    assert not result.valid


def test_model_not_in_registry_is_rejected():
    from darwin.agent.registry import ModelEntry, ModelRegistry, Provider

    g = F.proposer_checker_arbiter_genome()
    empty_registry = ModelRegistry({})  # knows no models
    result = validate(g, registry=empty_registry)
    assert not result.valid
    assert any("not in the registry" in e for e in result.errors)


def test_validation_result_raise_if_invalid():
    result = validate(F.proposer_checker_arbiter_genome())
    result.raise_if_invalid()  # valid -> no raise
    bad = validate(_with_edges(F.proposer_checker_arbiter_genome(), []))
    with pytest.raises(ValueError):
        bad.raise_if_invalid()
