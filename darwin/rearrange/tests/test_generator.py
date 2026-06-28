"""B8 Generator tests — K distinct, valid candidates; agent-set invariant; hints."""

import random

import pytest

from darwin.agent.registry import default_registry, reset_default_registry
from darwin.rearrange.generator import generate_candidates
from darwin.rearrange.operators import signature
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


def test_produces_k_distinct_valid_candidates():
    reg = default_registry()
    g = _genome()
    cands = generate_candidates(g, k=5, rng=random.Random(7), registry=reg)
    assert len(cands) == 5
    sigs = {c.signature for c in cands}
    assert len(sigs) == 5  # all distinct
    assert signature(g) not in sigs  # none identical to the baseline
    base = {a.agent_id for a in g.agents}
    for c in cands:
        assert validate(c.genome, reg).valid
        assert {a.agent_id for a in c.genome.agents} == base  # agent set invariant


def test_returns_fewer_when_search_space_is_small():
    # tiny one-agent genome: only reassign_model applies, with 2 models -> 1 candidate
    reg = default_registry()
    g = TF.one_agent_genome()
    cands = generate_candidates(g, k=5, rng=random.Random(1), registry=reg, max_tries=200)
    assert 1 <= len(cands) <= 5
    assert len({c.signature for c in cands}) == len(cands)


def test_hints_bias_operator_selection():
    reg = default_registry()
    g = _genome()
    cands = generate_candidates(g, k=6, rng=random.Random(3), registry=reg,
                                hints={"reassign_model": 50.0})
    # heavy bias toward reassign_model -> the majority are SWAP_MODEL
    swaps = sum(1 for c in cands if c.mutation_type == MutationType.SWAP_MODEL)
    assert swaps >= len(cands) // 2
