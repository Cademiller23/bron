"""The F1 seed team — a hand-built 4-specialist genome (routing / scheduling /
clustering proposers -> arbiter), each F1-aware via its role_description.

Built on the architect's _safe_default pattern (AgentSpec.model_validate with
registry context, AgentNode/Edge/EdgeType.FEEDS_ARBITER, TeamGenome.create). The
rearrangement loop then mutates THIS seed (swap models, rewire), so the org chart
genuinely evolves -- we just start F1-shaped instead of transport-shaped.

Each role_description carries the EXACT calendar encoding so the agent emits a
scorable Solution, and names the objective (revenue-peak placement + feasibility).
"""

from darwin.agent.registry import default_registry
from darwin.agent.spec import AgentSpec, InputKind, OutputKind, ThinkingLevel
from darwin.constants import DEFAULT_MODEL_ID
from darwin.problem import f1_calendar as F1
from darwin.team.genome import AgentNode, Edge, EdgeType, MutationActor, TeamGenome

# The 24 race keys + the encoding contract, embedded in every proposer's prompt.
_RACES = ", ".join(F1.RACES)
_REGIONS = {}
for _r in F1.RACES:
    _REGIONS.setdefault(F1.REGION[_r], []).append(_r)
_REGION_TEXT = "; ".join(f"{reg}: {', '.join(rs)}" for reg, rs in _REGIONS.items())

_ENCODING = (
    "OUTPUT FORMAT (critical): emit a FullSolutionOutput whose `solution` is a Solution where:\n"
    "  - solution.routes = [ a single Route with vehicle_id='f1_circus' and node_sequence = "
    "your ORDERED list of ALL 24 race keys (each exactly once) ],\n"
    "  - solution.flows = [ one FlowAssignment per race, arc_id='RACE@wWEEK', quantity=WEEK ] "
    "where WEEK is an integer in 8..50 (the season window).\n"
    "  - solution.solution_id and solution.instance_id are non-empty strings.\n"
    f"The 24 races: {_RACES}.\n"
    "Use EVERY race exactly once. Weeks must be >=1 apart; avoid weeks 31-32 (August break)."
)

_OBJECTIVE = (
    "OBJECTIVE: produce a FEASIBLE calendar (the hard part -- only ~1% of orderings are feasible) "
    "that MAXIMIZES total revenue. Revenue is highest when each race is placed in its peak month. "
    "Carbon (travel between consecutive races) is a secondary tie-breaker."
)

_FEASIBILITY = (
    "THREE HARD CONSTRAINT FAMILIES (all must be satisfied):\n"
    "  1. ROUTING: consecutive races can't exceed ~11000 km; no >2 long-haul (>=7000km) legs in a row.\n"
    "  2. SCHEDULING: season is weeks 8..50; >=1 week between races; NO races in weeks 31-32 "
    "(August break); avoid weather-impossible months (heat/monsoon/snow per venue).\n"
    "  3. CLUSTERING: each geographic region must appear as ONE contiguous block (a 'swing') -- "
    "you cannot leave a region and return. "
    f"Regions: {_REGION_TEXT}."
)


def _spec(agent_id, role_name, role_desc, *, output, inp, thinking=ThinkingLevel.MEDIUM, model=None):
    return AgentSpec.model_validate(
        {
            "agent_id": agent_id, "role_name": role_name,
            "role_description": role_desc,
            "input_contract": inp.value, "output_contract": output.value,
            "model_id": model or DEFAULT_MODEL_ID, "thinking_level": thinking.value,
            "created_by": "human_seed", "spec_version": "1.0.0",
        },
        context={"registry": default_registry()},
    )


def build_f1_seed_team(instance_id: str = "f1_2026_calendar") -> TeamGenome:
    """A 4-agent F1 team: routing + scheduling + clustering proposers -> arbiter."""
    routing = _spec(
        "routing_specialist", "routing_specialist",
        "You are the ROUTING specialist for an F1 calendar. " + _OBJECTIVE + "\n\n" + _FEASIBILITY
        + "\n\nYOUR FOCUS: order the races to keep consecutive-leg distances low and avoid >2 long-haul "
        "legs in a row, WITHIN region swings. Then place weeks to hit revenue peaks. " + _ENCODING,
        output=OutputKind.FULL_SOLUTION, inp=InputKind.FULL_PROBLEM,
    )
    scheduling = _spec(
        "scheduling_specialist", "scheduling_specialist",
        "You are the SCHEDULING specialist for an F1 calendar. " + _OBJECTIVE + "\n\n" + _FEASIBILITY
        + "\n\nYOUR FOCUS: choose each race's WEEK so it lands in its revenue-peak month and a weather-OK "
        "month, respects >=1 week spacing, the 8..50 window, and skips weeks 31-32. " + _ENCODING,
        output=OutputKind.FULL_SOLUTION, inp=InputKind.FULL_PROBLEM,
    )
    clustering = _spec(
        "clustering_specialist", "clustering_specialist",
        "You are the CLUSTERING/logistics specialist for an F1 calendar. " + _OBJECTIVE + "\n\n" + _FEASIBILITY
        + "\n\nYOUR FOCUS: group races into contiguous regional swings (APAC, MIDEAST, EUROPE, AMERICAS) "
        "so no region is split, then order swings to minimize inter-region travel. " + _ENCODING,
        output=OutputKind.FULL_SOLUTION, inp=InputKind.FULL_PROBLEM,
    )
    arbiter = _spec(
        "arbitrator", "arbitrator",
        "You are the ARBITRATOR. You receive calendar proposals from a routing, a scheduling, and a "
        "clustering specialist. Synthesize ONE final calendar that is FEASIBLE across all three families "
        "AND maximizes revenue. Reconcile conflicts (e.g. a revenue-peak week that breaks a region swing). "
        + _ENCODING,
        output=OutputKind.ARBITRATION, inp=InputKind.SIBLING_OUTPUTS, thinking=ThinkingLevel.HIGH,
    )

    nodes = [
        AgentNode(agent_id="routing_specialist", spec=routing),
        AgentNode(agent_id="scheduling_specialist", spec=scheduling),
        AgentNode(agent_id="clustering_specialist", spec=clustering),
        AgentNode(agent_id="arbitrator", spec=arbiter),
    ]
    edges = [
        Edge(from_agent_id="routing_specialist", to_agent_id="arbitrator", edge_type=EdgeType.FEEDS_ARBITER),
        Edge(from_agent_id="scheduling_specialist", to_agent_id="arbitrator", edge_type=EdgeType.FEEDS_ARBITER),
        Edge(from_agent_id="clustering_specialist", to_agent_id="arbitrator", edge_type=EdgeType.FEEDS_ARBITER),
    ]
    return TeamGenome.create(
        instance_id=instance_id, agents=nodes, edges=edges, arbiter_id="arbitrator",
        actor=MutationActor.ARCHITECT,
        description="F1 seed team: routing + scheduling + clustering -> arbiter",
    )
