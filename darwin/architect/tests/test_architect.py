"""A9 Architect tests — the largest suite (happy / repair / safe-default / ...)."""

import pytest

from darwin.agent.client import ErrorCategory, ProviderError
from darwin.agent.registry import default_registry, reset_default_registry
from darwin.architect import fixtures as AF
from darwin.architect.prompts import DIRECT_DECISION_REMINDER, SYSTEM_PROMPT
from darwin.problem.fixtures import golden_transportation
from darwin.problem.schemas import ObjectiveWeights
from darwin.team import fixtures as TF
from darwin.team.genome import EdgeType
from darwin.team.inference_gate import InferenceGate
from darwin.team.runner import TeamRunner
from darwin.team.validation import validate

INSTANCE = golden_transportation()
WEIGHTS = ObjectiveWeights.cost_only()


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


async def test_happy_path_designs_valid_persisted_genome():
    arch, store, _ = AF.make_architect([AF.model_response(AF.analysis_fixture()), AF.model_response(AF.valid_design())])
    genome = await arch.design_initial_team(INSTANCE, WEIGHTS)
    assert validate(genome).valid
    assert len(genome.agents) == 4
    persisted = await store.load(genome.genome_id)
    assert persisted is not None and persisted.version == 1
    assert genome.history[0].mutation_type.value == "INITIAL_CURATION"


async def test_repair_path_recovers_from_invalid_design():
    arch, _store, client = AF.make_architect([
        AF.model_response(AF.analysis_fixture()),
        AF.model_response(AF.design_missing_arbiter()),  # invalid
        AF.model_response(AF.valid_design()),  # repaired
    ])
    genome = await arch.design_initial_team(INSTANCE, WEIGHTS)
    assert validate(genome).valid and len(genome.agents) == 4


async def test_repair_exhausted_falls_back_to_safe_default(caplog):
    import logging

    arch, _store, _ = AF.make_architect(
        [AF.model_response(AF.analysis_fixture())] + [AF.model_response(AF.design_missing_arbiter())] * 5
    )
    with caplog.at_level(logging.WARNING, logger="darwin.architect"):
        genome = await arch.design_initial_team(INSTANCE, WEIGHTS)
    assert validate(genome).valid
    assert {a.agent_id for a in genome.agents} == {"cost_minimizer", "arbitrator"}  # the minimal safe team
    assert any("safe default" in r.message for r in caplog.records)


async def test_safe_default_is_runnable_through_b3():
    # the fallback team must actually run end-to-end (degraded, never dead)
    arch, _store, _ = AF.make_architect([ProviderError(ErrorCategory.AUTH, "401")] * 8)
    genome = await arch.design_initial_team(INSTANCE, WEIGHTS)
    scripts = {
        "cost_minimizer": TF.agent_result("cost_minimizer", TF.full_solution_output(TF.optimal_solution())),
        "arbitrator": TF.agent_result("arbitrator", TF.arbitration_output(TF.optimal_solution())),
    }
    runner = TeamRunner(model_client=None, telemetry=TF.telemetry_sink(), inference_gate=InferenceGate(4),
                        store=None, worker_factory=TF.scripted_worker_factory(scripts))
    ev = await runner.evaluate(genome, INSTANCE, WEIGHTS)
    assert ev.error is None and ev.cleared_threshold is True


async def test_never_crashes_on_total_model_failure():
    arch, _store, _ = AF.make_architect([ProviderError(ErrorCategory.OTHER, "boom")] * 8)
    genome = await arch.design_initial_team(INSTANCE, WEIGHTS)  # must not raise
    assert validate(genome).valid


async def test_safe_default_never_raises_with_unregistered_fast_model():
    # the last line of defense must not itself raise when fast_model_id is bogus
    from darwin.agent.fixtures import scripted_client
    from darwin.architect.architect import Architect

    client = scripted_client([ProviderError(ErrorCategory.AUTH, "x")] * 8)
    arch = Architect(client, store=None, fast_model_id="not-registered")
    genome = await arch.design_initial_team(INSTANCE, WEIGHTS)  # must not raise
    assert validate(genome).valid
    assert all(default_registry().contains(n.spec.model_id) for n in genome.agents)


async def test_curate_never_raises_with_unregistered_fast_model():
    from darwin.agent.fixtures import scripted_client
    from darwin.architect.architect import Architect

    genome, ev = await _infeasible_eval()
    arch = Architect(scripted_client([ProviderError(ErrorCategory.AUTH, "x")] * 8),
                     store=None, fast_model_id="not-registered")
    spec, edges = await arch.curate_agent_for_gap(genome, INSTANCE, ev)  # must not raise
    assert default_registry().contains(spec.model_id)


async def test_model_assignment_is_registry_legal():
    arch, _store, _ = AF.make_architect([AF.model_response(AF.analysis_fixture()), AF.model_response(AF.valid_design())])
    genome = await arch.design_initial_team(INSTANCE, WEIGHTS)
    for node in genome.agents:
        assert default_registry().contains(node.spec.model_id)


async def test_out_of_catalog_model_triggers_repair():
    bad = AF.valid_design().model_copy(
        update={"agents": [d.model_copy(update={"model_id": "ghost"}) if d.role_name == "cost minimizer" else d
                           for d in AF.valid_design().agents]}
    )
    arch, _store, _ = AF.make_architect([AF.model_response(AF.analysis_fixture()), AF.model_response(bad),
                                         AF.model_response(AF.valid_design())])
    genome = await arch.design_initial_team(INSTANCE, WEIGHTS)
    assert validate(genome).valid  # the bad model_id was repaired away


def test_direct_decision_reminder_in_system_prompt():
    assert DIRECT_DECISION_REMINDER in SYSTEM_PROMPT
    assert "never write or call a solver" in DIRECT_DECISION_REMINDER.lower()
    assert "you do not solve the problem" in SYSTEM_PROMPT.lower()


async def test_safe_default_role_descriptions_carry_direct_decision_language():
    arch, _store, _ = AF.make_architect([ProviderError(ErrorCategory.AUTH, "x")] * 8)
    genome = await arch.design_initial_team(INSTANCE, WEIGHTS)
    for node in genome.agents:
        assert "solver" in node.spec.role_description.lower()  # "never call a solver"


# ---------------------------------------------------------------------------
# curate_agent_for_gap
# ---------------------------------------------------------------------------
async def _infeasible_eval():
    genome = TF.proposer_checker_arbiter_genome()
    runner = TeamRunner(model_client=None, telemetry=TF.telemetry_sink(), inference_gate=InferenceGate(4), store=None,
                        worker_factory=TF.scripted_worker_factory({
                            "p1": TF.agent_result("p1", None, success=False, error="x"),
                            "p2": TF.agent_result("p2", None, success=False, error="x"),
                            "chk": TF.agent_result("chk", TF.critique_output()),
                            "arb": TF.agent_result("arb", None, success=False, error="x")}))
    return genome, await runner.evaluate(genome, INSTANCE, WEIGHTS)


async def test_curate_returns_targeted_single_agent_keeping_genome_valid():
    genome, ev = await _infeasible_eval()
    arch, _store, _ = AF.make_architect([AF.model_response(AF.curated_agent_fixture())])
    spec, edges = await arch.curate_agent_for_gap(genome, INSTANCE, ev)
    # exactly one new agent, registry-legal, not a duplicate of an existing role
    assert spec.model_id in default_registry().all_ids()
    assert spec.agent_id not in {a.agent_id for a in genome.agents}
    assert spec.role_name not in {a.spec.role_name for a in genome.agents}  # no duplicate role
    # the suggested wiring keeps the genome valid AND actually connects the new agent
    candidate = arch._with_added_agent(genome, spec, edges)
    assert validate(candidate).valid
    assert any(spec.agent_id in (e.from_agent_id, e.to_agent_id) for e in edges)


async def test_curate_does_not_duplicate_a_role_even_if_model_repeats_one():
    genome, ev = await _infeasible_eval()
    # the model authors a new agent whose role slugifies to an EXISTING role ("arbitrator")
    from darwin.architect.schemas import AgentSpecDraft, CuratedAgentDraft, EdgeDraft
    from darwin.agent.spec import InputKind, OutputKind
    dup_role = CuratedAgentDraft(
        agent=AgentSpecDraft(role_name="arbitrator", role_description="reason directly; never call a solver",
                             input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION,
                             model_id="gemini-3.5-flash"),
        edges=[EdgeDraft(from_role_name="arbitrator", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER)])
    arch, _store, _ = AF.make_architect([AF.model_response(dup_role)])
    spec, edges = await arch.curate_agent_for_gap(genome, INSTANCE, ev)
    assert spec.role_name not in {a.spec.role_name for a in genome.agents}  # uniquified, not a duplicate
    assert validate(arch._with_added_agent(genome, spec, edges)).valid


async def test_curate_role_collision_with_existing_role_name_is_uniquified():
    # the fixture genome has role_name 'cost_minimizer' on agent_id 'p1' (role != id).
    # A curated agent whose role slugifies to 'cost_minimizer' must NOT reuse that role.
    from darwin.architect.schemas import AgentSpecDraft, CuratedAgentDraft, EdgeDraft
    from darwin.agent.spec import InputKind, OutputKind

    genome, ev = await _infeasible_eval()
    collide = CuratedAgentDraft(
        agent=AgentSpecDraft(role_name="cost_minimizer", role_description="reason directly; never call a solver",
                             input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION,
                             model_id="gemini-3.5-flash"),
        edges=[EdgeDraft(from_role_name="cost_minimizer", to_role_name="arbitrator", edge_type=EdgeType.FEEDS_ARBITER)])
    arch, _store, _ = AF.make_architect([AF.model_response(collide)])
    spec, edges = await arch.curate_agent_for_gap(genome, INSTANCE, ev)
    existing_roles = {a.spec.role_name for a in genome.agents}
    assert spec.role_name not in existing_roles  # role_name de-duplicated against existing roles
    assert spec.agent_id not in {a.agent_id for a in genome.agents}
    candidate = arch._with_added_agent(genome, spec, edges)
    # no two agents in the resulting genome share a role_name
    roles = [n.spec.role_name for n in candidate.agents]
    assert len(roles) == len(set(roles))
    assert validate(candidate).valid


async def test_curate_diagnosis_from_high_risk():
    from darwin.team.evaluation import GenomeEvaluation
    from darwin.team.genome import ArbiterTier
    from darwin.problem import score
    from darwin.problem.schemas import FlowAssignment, Solution

    # each sink fed entirely by one source -> max concentration + worst-case unmet
    # => high raw_risk, feasible, no violations -> the RISK branch must fire.
    sol = Solution(solution_id="risky", instance_id=INSTANCE.instance_id,
                   flows=[FlowAssignment(arc_id="S1-D1", quantity=8.0), FlowAssignment(arc_id="S2-D2", quantity=7.0)])
    sb = score(INSTANCE, sol, WEIGHTS)
    assert sb.feasible and sb.raw_risk > 0.45 and not sb.violations  # premise check
    ev = GenomeEvaluation(genome_id="g", version=1, instance_id=INSTANCE.instance_id, completed=True,
                          final_solution=sol, score_breakdown=sb, fitness=sb.final_fitness,
                          normalized_score=sb.normalized_score, cleared_threshold=False,
                          arbiter_tier_used=ArbiterTier.PRIMARY)
    arch, _store, _ = AF.make_architect([])
    diagnosis = arch.diagnose(ev)
    assert "disruption-risk modeler" in diagnosis  # pins the risk branch specifically


async def test_curate_never_crashes_falls_back_to_heuristic():
    genome, ev = await _infeasible_eval()
    arch, _store, _ = AF.make_architect([ProviderError(ErrorCategory.OTHER, "boom")] * 8)
    spec, edges = await arch.curate_agent_for_gap(genome, INSTANCE, ev)  # must not raise
    candidate = arch._with_added_agent(genome, spec, edges)
    assert validate(candidate).valid
    assert edges and edges[0].to_agent_id == genome.arbiter_id
