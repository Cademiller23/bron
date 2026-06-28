"""The F1 seed team — 4 specialists (routing/scheduling/clustering -> arbiter).

CALIBRATION (proven by the bounded-thinking test):
  - thinking_level = LOW (budget 1024): all 24 races in ~9s. NEVER medium(-1, unbounded)
    or high(24576) -- both blow the 30s timeout.
  - max_output_tokens = 4096.
  - SLIM FORMAT: calendar rides in flows ONLY (no redundant route). arc_id=race,
    quantity=week, ORDER=flow order. ~30% smaller, robust against truncation.
"""

from darwin.agent.registry import default_registry
from darwin.agent.spec import AgentSpec, InputKind, OutputKind, ThinkingLevel
from darwin.constants import DEFAULT_MODEL_ID
from darwin.problem import f1_calendar as F1
from darwin.team.genome import AgentNode, Edge, EdgeType, MutationActor, TeamGenome

_F1_MAX_TOKENS = 4096
_F1_THINKING = ThinkingLevel.LOW

_RACES = ", ".join(F1.RACES)
_REGIONS = {}
for _r in F1.RACES:
    _REGIONS.setdefault(F1.REGION[_r], []).append(_r)
_REGION_TEXT = "; ".join(f"{reg}: {', '.join(rs)}" for reg, rs in _REGIONS.items())

_ENCODING = (
    "OUTPUT FORMAT (return ONLY JSON, no prose). Emit a FullSolutionOutput whose solution is:\n"
    '  {"solution_id":"f1_cal","instance_id":"f1_2026_calendar",\n'
    '   "flows":[{"arc_id":"<race_key>","quantity":<week_int>}, ... 24 entries ...]}\n'
    "RULES:\n"
    "  - One flow per race, ALL 24 races exactly once, in CALENDAR ORDER (first flow = first race).\n"
    "  - quantity = week number (integer 8..50).\n"
    "  - Do NOT include a routes field. Order is the flow order. Keep it compact.\n"
    f"The 24 race keys: {_RACES}."
)
_OBJECTIVE = (
    "OBJECTIVE: produce a FEASIBLE calendar that MAXIMIZES total revenue (each race in its peak "
    "month). Only ~1% of orderings are feasible, so feasibility is the hard part. Carbon is a "
    "secondary tie-breaker."
)
_FEASIBILITY = (
    "THREE HARD CONSTRAINT FAMILIES (all must hold):\n"
    "  1. ROUTING: consecutive races <~11000 km apart; no >2 long-haul (>=7000km) legs in a row.\n"
    "  2. SCHEDULING: weeks 8..50; >=1 week between races; NO races in weeks 31-32; avoid "
    "weather-impossible months per venue.\n"
    "  3. CLUSTERING: each region is ONE contiguous block (a swing); never leave and return. "
    f"Regions: {_REGION_TEXT}."
)


def _spec(agent_id, role_name, role_desc, *, output, inp, thinking=_F1_THINKING):
    return AgentSpec.model_validate(
        {
            "agent_id": agent_id, "role_name": role_name, "role_description": role_desc,
            "input_contract": inp.value, "output_contract": output.value,
            "model_id": DEFAULT_MODEL_ID, "thinking_level": thinking.value,
            "max_output_tokens": _F1_MAX_TOKENS,
            "created_by": "human_seed", "spec_version": "1.0.0",
        },
        context={"registry": default_registry()},
    )


def build_f1_seed_team(instance_id: str = "f1_2026_calendar") -> TeamGenome:
    routing = _spec(
        "routing_specialist", "routing_specialist",
        "You are the ROUTING specialist. " + _OBJECTIVE + "\n\n" + _FEASIBILITY
        + "\n\nYOUR FOCUS: order races to keep consecutive-leg distance low within region swings, "
        "then pick weeks for revenue peaks. " + _ENCODING,
        output=OutputKind.FULL_SOLUTION, inp=InputKind.FULL_PROBLEM,
    )
    scheduling = _spec(
        "scheduling_specialist", "scheduling_specialist",
        "You are the SCHEDULING specialist. " + _OBJECTIVE + "\n\n" + _FEASIBILITY
        + "\n\nYOUR FOCUS: choose each race's WEEK for its revenue-peak + weather-OK month, "
        ">=1 week spacing, 8..50 window, skip weeks 31-32. " + _ENCODING,
        output=OutputKind.FULL_SOLUTION, inp=InputKind.FULL_PROBLEM,
    )
    clustering = _spec(
        "clustering_specialist", "clustering_specialist",
        "You are the CLUSTERING/logistics specialist. " + _OBJECTIVE + "\n\n" + _FEASIBILITY
        + "\n\nYOUR FOCUS: group races into contiguous regional swings (APAC, MIDEAST, EUROPE, "
        "AMERICAS), no region split, order swings to cut inter-region travel. " + _ENCODING,
        output=OutputKind.FULL_SOLUTION, inp=InputKind.FULL_PROBLEM,
    )
    arbiter = _spec(
        "arbitrator", "arbitrator",
        "You are the ARBITRATOR. You receive calendar proposals from routing, scheduling, and "
        "clustering specialists. Synthesize ONE final calendar that is FEASIBLE across all three "
        "families AND maximizes revenue. Reconcile conflicts. " + _ENCODING,
        output=OutputKind.ARBITRATION, inp=InputKind.SIBLING_OUTPUTS,
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
        description="F1 seed team: routing+scheduling+clustering -> arbiter (LOW thinking, slim format)",
    )
