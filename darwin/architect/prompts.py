"""The Architect's meta-prompts.

The system prompt is the most important text in B4. It establishes, every time:
the **meta-role** (you design agents, you do not solve), the **model catalog** +
assignment guidance, the **schema contract**, and the **direct-decision
reminder** the authored agents must carry.
"""

import json
from typing import Any

from darwin.agent.spec import InputKind, OutputKind, ThinkingLevel
from darwin.architect.schemas import ArchitectTeamDesign, CuratedAgentDraft, ProblemAnalysis
from darwin.problem.schemas import ObjectiveWeights, ProblemInstance
from darwin.team.evaluation import GenomeEvaluation
from darwin.team.genome import EdgeType, TeamGenome

# The mandatory direct-decision reminder propagated into every authored agent.
DIRECT_DECISION_REMINDER = (
    "Each agent you author will, when it runs, produce the actual "
    "allocation/routing decision DIRECTLY — it must NEVER write or call a solver, "
    "and never emit code. Author role descriptions that instruct direct "
    "constraint-reasoning over the problem, not code generation."
)

SYSTEM_PROMPT = (
    "You are the Architect. Your ONLY job is to DESIGN a team of agents that will "
    "solve a supply-chain optimization problem. You do NOT solve the problem "
    "yourself — you never output flows, routes, or a solution. You output a TEAM "
    "DESIGN as JSON conforming to the provided schema.\n\n"
    "You invent the roles from scratch based on THIS problem — there is no fixed "
    "menu. Supply-chain optimization tends to need: cost optimization, "
    "feasibility/capacity checking, lead-time reduction, disruption-risk / "
    "resilience modeling, and a final ARBITRATOR that resolves the multi-objective "
    "tradeoff into one Solution. A sensible default wiring is: parallel "
    "objective-specialist proposers -> a feasibility checker -> an arbitrator that "
    "synthesizes the final Solution. But YOU decide the actual roles, the number of "
    "parts, and the wiring from this problem's analysis. Guidance shapes; it does "
    "not enumerate.\n\n"
    "Every agent MUST have: a role_name, a role_description, an input_contract, an "
    "output_contract, and a model_id chosen from the catalog. Assign each agent the "
    "CHEAPEST model that can do its job well — mechanical/checking roles get FAST "
    "models; the arbitrator and any deep-reasoning role get a FRONTIER model.\n\n"
    + DIRECT_DECISION_REMINDER
    + "\n\nReturn ONLY JSON. No prose, no markdown."
)


def _enum_values(enum_cls) -> str:
    return ", ".join(e.value for e in enum_cls)


def build_model_catalog(registry: Any) -> str:
    lines = ["AVAILABLE MODELS (model_id | tier | est_cost_per_1k_out | est_latency_ms):"]
    for model_id in registry.all_ids():
        e = registry.get(model_id)
        lines.append(
            f"  - {e.model_id} | {e.capability_tier.value} | {e.est_cost_per_1k_out} | {e.est_latency_ms}"
        )
    return "\n".join(lines)


def _instance_summary(instance: ProblemInstance, weights: ObjectiveWeights) -> str:
    return json.dumps(
        {
            "problem_class": instance.problem_class.value,
            "num_nodes": len(instance.nodes),
            "num_arcs": len(instance.arcs),
            "num_sources": len(instance.sources()),
            "num_sinks": len(instance.sinks()),
            "total_supply": instance.total_supply(),
            "total_demand": instance.total_demand(),
            "num_additional_constraints": len(instance.additional_constraints),
            "has_known_optimum": instance.known_optimum is not None,
            "objective_weights": {
                "cost": weights.cost_weight,
                "lead_time": weights.lead_time_weight,
                "risk": weights.risk_weight,
            },
            "difficulty": instance.metadata.difficulty.value,
        },
        indent=2,
    )


def _schema_block(model_cls) -> str:
    return (
        f"Return JSON conforming to this schema:\n{json.dumps(model_cls.model_json_schema())}\n"
        f"Valid input_contract values: {_enum_values(InputKind)}.\n"
        f"Valid output_contract values: {_enum_values(OutputKind)}.\n"
        f"Valid edge_type values: {_enum_values(EdgeType)}.\n"
        f"Valid thinking_level values: {_enum_values(ThinkingLevel)}."
    )


def build_analysis_prompt(instance: ProblemInstance, weights: ObjectiveWeights) -> str:
    return (
        "Analyze this supply-chain optimization problem and decide how to decompose "
        "it. Identify the dominant objectives (from the weights and structure), the "
        "binding-looking constraints, the difficulty, and how many parts to split it "
        "into.\n\nPROBLEM:\n"
        + _instance_summary(instance, weights)
        + "\n\n"
        + _schema_block(ProblemAnalysis)
    )


def build_design_prompt(analysis: ProblemAnalysis, registry: Any) -> str:
    return (
        "Design the team for this problem. Author each agent from scratch (role, "
        "description, contracts, model). Wire them (edges by role name) and name the "
        "arbitrator. Reflect the analysis: weight resilience-heavy problems toward a "
        "stronger risk specialist, capacity-tight problems toward a capacity auditor, "
        "and so on.\n\nANALYSIS:\n"
        + analysis.model_dump_json(indent=2)
        + "\n\n"
        + build_model_catalog(registry)
        + "\n\n"
        + _schema_block(ArchitectTeamDesign)
    )


def build_gap_prompt(
    genome: TeamGenome, evaluation: GenomeEvaluation, diagnosis: str, registry: Any
) -> str:
    existing_roles = [
        {"role_name": n.spec.role_name, "output_contract": n.spec.output_contract.value}
        for n in genome.agents
    ]
    violations = [
        {"type": v.violation_type.value, "location": v.location, "magnitude": v.magnitude}
        for v in evaluation.score_breakdown.violations
    ]
    return (
        "This team scored below the 90% threshold. Author EXACTLY ONE new agent that "
        "supplies the missing capability to push it past 90%, and say which existing "
        "agents it should connect to (edges by role name). Do NOT duplicate an "
        "existing role.\n\n"
        f"DIAGNOSIS (from the deterministic scorer): {diagnosis}\n"
        f"normalized_score: {evaluation.normalized_score}\n"
        f"raw_cost: {evaluation.score_breakdown.raw_cost} | raw_risk: {evaluation.score_breakdown.raw_risk} | "
        f"raw_lead_time: {evaluation.score_breakdown.raw_lead_time}\n"
        f"violations: {json.dumps(violations)}\n"
        f"existing_roles: {json.dumps(existing_roles)}\n\n"
        + build_model_catalog(registry)
        + "\n\n"
        + _schema_block(CuratedAgentDraft)
    )


def repair_prompt(previous_user: str, error: str) -> str:
    return (
        previous_user
        + f"\n\nYOUR PREVIOUS OUTPUT FAILED VALIDATION: {error}\n"
        "Return a corrected design as JSON only. Fix exactly what failed."
    )
