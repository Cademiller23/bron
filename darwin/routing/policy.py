"""The routing policy — the principled warm-start: role-kind → tier → model.

This is the *first pass* the Architect's reasoning encodes ("this role is
mechanical → FAST; this is judgment-heavy → FRONTIER") and the bias B5's
model-aware operators follow. It is a warm start, not a verdict — the cost/
latency-penalized search (``efficiency.py``) then verifies and refines the guess
for the specific problem. Warm-starting from a good guess is why the model
search converges fast instead of exploring from scratch.

``AgentSpec`` carries no explicit ``role_kind``; we derive it deterministically
from the agent's ``output_contract`` (and whether it is the genome's arbiter).
"""

from typing import Iterable, List, Optional

from darwin.agent.registry import CapabilityTier
from darwin.agent.spec import AgentSpec, OutputKind
from darwin.routing.fleet import FAST, FRONTIER, MID, FleetModel, get_fleet

# Role kinds (strings, aligned with B6's GapDescription.suggested_role_kind).
ARBITER = "arbiter"
CHECKER = "checker"
PROPOSER = "proposer"
SPECIALIST = "specialist"
DECOMPOSER = "decomposer"
ARCHITECT = "architect"

# Role kind → preferred capability tier (the warm-start rule, §4).
_TIER_BY_ROLE = {
    CHECKER: FAST,        # mechanical / bounded work — a cheap fast model suffices
    PROPOSER: MID,        # real reasoning, but not the hardest
    SPECIALIST: MID,      # objective specialists (cost / lead-time / resilience)
    DECOMPOSER: MID,
    ARBITER: FRONTIER,    # resolves conflicting proposals into the final answer
    ARCHITECT: FRONTIER,  # designing teams is the hardest, rarest reasoning (B4)
}

# Tier fallback order when the preferred tier has no available model.
_FALLBACK = {
    FAST: [FAST, MID, FRONTIER],
    MID: [MID, FAST, FRONTIER],
    FRONTIER: [FRONTIER, MID, FAST],
}


def classify_role(spec: AgentSpec, *, is_arbiter: bool = False) -> str:
    """Derive the role kind from the spec's output contract + arbiter position."""
    if is_arbiter or spec.output_contract == OutputKind.ARBITRATION:
        return ARBITER
    if spec.output_contract in (OutputKind.CRITIQUE, OutputKind.CONSTRAINT_REPORT):
        return CHECKER
    if spec.output_contract == OutputKind.DECOMPOSITION:
        return DECOMPOSER
    return PROPOSER  # FULL_SOLUTION / PARTIAL_SOLUTION


def classify_role_in_genome(genome, agent_id: str) -> str:
    """Classify an agent within a genome (knows which one is the arbiter)."""
    node = next((a for a in genome.agents if a.agent_id == agent_id), None)
    if node is None:
        raise KeyError(f"agent {agent_id!r} not in genome")
    return classify_role(node.spec, is_arbiter=(agent_id == genome.arbiter_id))


def suggest_tier(role_kind: str) -> CapabilityTier:
    """The preferred tier for a role kind (defaults to MID for unknown kinds)."""
    return _TIER_BY_ROLE.get(role_kind, MID)


def _cost_key(m: FleetModel):
    # cheapest first: total token price, then latency, then id (deterministic)
    return (m.est_cost_per_1k_in + m.est_cost_per_1k_out, m.est_latency_ms, m.model_id)


def cheapest_in_tier(
    tier: CapabilityTier,
    fleet: Optional[List[FleetModel]] = None,
    *,
    allowed_ids: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """The cheapest model in a tier (optionally restricted to ``allowed_ids``,
    e.g. the ids actually present in a registry). ``None`` if the tier is empty."""
    models = fleet if fleet is not None else get_fleet()
    allow = set(allowed_ids) if allowed_ids is not None else None
    pool = [m for m in models if m.tier == tier and (allow is None or m.model_id in allow)]
    if not pool:
        return None
    return min(pool, key=_cost_key).model_id


def suggest_model(
    role_kind: str,
    fleet: Optional[List[FleetModel]] = None,
    *,
    allowed_ids: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """The warm-start model for a role kind: the cheapest model in its preferred
    tier, walking the fallback chain if that tier is empty. ``None`` only if the
    fleet (restricted to ``allowed_ids``) has no usable model at all."""
    models = fleet if fleet is not None else get_fleet()
    preferred = suggest_tier(role_kind)
    for tier in _FALLBACK[preferred]:
        pick = cheapest_in_tier(tier, models, allowed_ids=allowed_ids)
        if pick is not None:
            return pick
    return None


def suggest_model_for_spec(
    spec: AgentSpec,
    *,
    is_arbiter: bool = False,
    fleet: Optional[List[FleetModel]] = None,
    allowed_ids: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Convenience: classify the spec then suggest a model."""
    return suggest_model(classify_role(spec, is_arbiter=is_arbiter), fleet, allowed_ids=allowed_ids)


def warm_start_genotype(
    genome, *, fleet: Optional[List[FleetModel]] = None, allowed_ids: Optional[Iterable[str]] = None
) -> dict:
    """The policy's first-pass model assignment for a whole genome:
    ``{agent_id -> suggested_model_id}`` (skips agents with no usable suggestion).
    A warm start the efficiency search then refines."""
    out = {}
    for node in genome.agents:
        pick = suggest_model_for_spec(
            node.spec, is_arbiter=(node.agent_id == genome.arbiter_id), fleet=fleet, allowed_ids=allowed_ids
        )
        if pick is not None:
            out[node.agent_id] = pick
    return out
