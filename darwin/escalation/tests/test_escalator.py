"""Escalator — corpus-first then curate, valid wiring, degrade, optimistic commit."""

from darwin.agent.spec import InputKind, OutputKind
from darwin.escalation.escalator import Escalator
from darwin.escalation.schemas import EscalationMethod
from darwin.escalation.fixtures import (
    MockArchitect,
    base_genome,
    capacity_violation,
    cost_specialist_spec,
    evaluation_with,
    make_corpus,
    risk_specialist_spec,
    spec,
)
from darwin.team.genome import EdgeType
from darwin.team.validation import validate
from darwin.team.fixtures import new_store


class FakeInstance:
    def __init__(self, pc="transportation", iid="golden-transportation"):
        self.instance_id = iid
        self.problem_class = type("PC", (), {"value": pc})()


def _risk_eval(genome):
    return evaluation_with(genome, feasible=True, fitness=0.7, normalized=0.7, raw_risk=0.7)


def _cost_eval(genome):
    return evaluation_with(genome, feasible=True, fitness=0.7, normalized=0.7, raw_risk=0.0)


async def test_corpus_reuse_when_match_exists():
    corpus = make_corpus()
    await corpus.promote(risk_specialist_spec(), 0.3, "transportation", "i0")
    genome = base_genome()
    esc = Escalator(corpus, MockArchitect(cost_specialist_spec()), store=None)
    res = await esc.escalate(genome, FakeInstance(), None, _risk_eval(genome))
    assert res.method == EscalationMethod.CORPUS
    assert res.added_spec.role_name == "disruption_risk_modeler"
    assert res.corpus_entry_id is not None
    assert len(res.genome.agents) == len(genome.agents) + 1
    # the new agent feeds the arbiter
    assert any(e.from_agent_id == res.added_agent_id and e.to_agent_id == genome.arbiter_id
               for e in res.genome.edges)


async def test_curate_when_corpus_empty():
    corpus = make_corpus()  # empty
    arch = MockArchitect(cost_specialist_spec())
    genome = base_genome()
    esc = Escalator(corpus, arch, store=None)
    res = await esc.escalate(genome, FakeInstance(), None, _cost_eval(genome))
    assert res.method == EscalationMethod.CURATED
    assert arch.calls == 1
    assert res.added_spec.role_name == "cost_reduction_specialist"
    assert len(res.genome.agents) == len(genome.agents) + 1


async def test_curation_failure_degrades_to_none_available():
    corpus = make_corpus()
    esc = Escalator(corpus, MockArchitect(cost_specialist_spec(), fail=True), store=None)
    genome = base_genome()
    res = await esc.escalate(genome, FakeInstance(), None, _cost_eval(genome))
    assert res.method == EscalationMethod.NONE_AVAILABLE
    assert res.genome is None


async def test_duplicate_role_curated_is_rejected():
    corpus = make_corpus()
    genome = base_genome()
    existing_role = genome.agents[0].spec.role_name  # already on the team
    dup = spec(existing_role, "duplicate role agent")
    esc = Escalator(corpus, MockArchitect(dup), store=None)
    res = await esc.escalate(genome, FakeInstance(), None, _cost_eval(genome))
    assert res.method == EscalationMethod.NONE_AVAILABLE


async def test_corpus_candidate_with_existing_role_is_skipped():
    corpus = make_corpus()
    genome = base_genome()
    # promote an agent whose role collides with one already on the team
    await corpus.promote(spec(genome.agents[0].spec.role_name, "minimize total cost cheaper"),
                         0.3, "transportation", "i0")
    arch = MockArchitect(cost_specialist_spec())
    esc = Escalator(corpus, arch, store=None)
    res = await esc.escalate(genome, FakeInstance(), None, _cost_eval(genome))
    # corpus candidate skipped (dup role) -> falls through to curation
    assert res.method == EscalationMethod.CURATED
    assert arch.calls == 1


async def test_commit_persists_through_store():
    store = new_store()
    genome = base_genome()
    await store.save_new(genome)
    corpus = make_corpus()
    await corpus.promote(risk_specialist_spec(), 0.3, "transportation", "i0")
    esc = Escalator(corpus, MockArchitect(cost_specialist_spec()), store=store)
    res = await esc.escalate(genome, FakeInstance(), None, _risk_eval(genome))
    assert res.method == EscalationMethod.CORPUS
    reloaded = await store.load(genome.genome_id)
    assert reloaded.version == genome.version + 1
    assert len(reloaded.agents) == len(genome.agents) + 1
    assert reloaded.history[-1].mutation_type.value == "ADD_AGENT_FROM_CORPUS"


def test_wire_problem_plus_draft_uses_passes_proposal():
    # regression: PROBLEM_PLUS_DRAFT must be fed by PASSES_PROPOSAL (not CHECKS),
    # even when the agent's OUTPUT is check-shaped (CONSTRAINT_REPORT/CRITIQUE).
    esc = Escalator(make_corpus(), MockArchitect(cost_specialist_spec()), store=None)
    genome = base_genome()
    auditor = spec("draft_auditor", "rebalance flows to respect capacity limits and resolve overflow throughput",
                   oc=OutputKind.CONSTRAINT_REPORT, ic=InputKind.PROBLEM_PLUS_DRAFT)
    edges = esc._wire(auditor, genome)
    feeder = [e for e in edges if e.to_agent_id == "draft_auditor"]
    assert feeder and feeder[0].edge_type == EdgeType.PASSES_PROPOSAL


def test_wire_sibling_outputs_checker_uses_checks():
    # unchanged behavior: a SIBLING_OUTPUTS checker still gets a CHECKS feeder
    esc = Escalator(make_corpus(), MockArchitect(cost_specialist_spec()), store=None)
    checker = spec("sib_checker", "audit sibling proposals for constraint violations",
                   oc=OutputKind.CRITIQUE, ic=InputKind.SIBLING_OUTPUTS)
    edges = esc._wire(checker, base_genome())
    feeder = [e for e in edges if e.to_agent_id == "sib_checker"]
    assert feeder and feeder[0].edge_type == EdgeType.CHECKS


async def test_corpus_reuse_of_draft_auditing_checker_succeeds():
    # regression end-to-end: a promoted PROBLEM_PLUS_DRAFT+CONSTRAINT_REPORT agent
    # must be REUSABLE (previously _wire produced an invalid genome -> skipped).
    corpus = make_corpus()
    auditor = spec("draft_auditor", "rebalance flows to respect capacity limits and resolve overflow throughput",
                   oc=OutputKind.CONSTRAINT_REPORT, ic=InputKind.PROBLEM_PLUS_DRAFT)
    await corpus.promote(auditor, 0.3, "transportation", "i0")
    genome = base_genome()
    esc = Escalator(corpus, MockArchitect(cost_specialist_spec()), store=None)
    ev = evaluation_with(genome, feasible=False, fitness=-3.0, violations=[capacity_violation()])
    res = await esc.escalate(genome, FakeInstance(), None, ev)
    assert res.method == EscalationMethod.CORPUS
    assert res.added_spec.role_name == "draft_auditor"
    assert validate(res.genome, None).valid
    fed = [e for e in res.genome.edges if e.to_agent_id == res.added_agent_id]
    assert any(e.edge_type == EdgeType.PASSES_PROPOSAL for e in fed)


async def test_failed_commit_degrades_to_none_for_that_candidate():
    class BrokenStore:
        async def retry_mutate(self, *a, **k):
            raise RuntimeError("commit failed")

    corpus = make_corpus()
    await corpus.promote(risk_specialist_spec(), 0.3, "transportation", "i0")
    genome = base_genome()
    # broken store -> corpus commit fails -> falls through to curation (store also broken -> none)
    esc = Escalator(corpus, MockArchitect(cost_specialist_spec()), store=BrokenStore())
    res = await esc.escalate(genome, FakeInstance(), None, _risk_eval(genome))
    assert res.method == EscalationMethod.NONE_AVAILABLE
