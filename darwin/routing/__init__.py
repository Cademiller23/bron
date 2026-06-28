"""Darwin Phase B7 — Multi-Model Routing & the Model Registry (the model-aware layer).

B7 does not rebuild B2's model-agnostic client or B3's model gene — it weaves a
thin "model-aware" layer through the phases already built:

* ``fleet`` — the curated ~5-model fleet (FAST=CHEAP / MID / FRONTIER) behind one
  interface; adding a model is a registry entry, not code.
* ``policy`` — the role→tier warm-start the Architect's first pass encodes.
* ``efficiency`` — the cost/latency-penalized SELECTION fitness + the guarded
  lexicographic comparator (efficiency can never sacrifice clearing the 0.90 gate).
* ``gene`` — ``model_id`` as a first-class evolvable gene + model-aware operators.
* ``observability`` — per-model economics (the Modular MAX story made visible).

Frozen handoff surfaces: ``get_fleet`` / ``install_fleet`` / ``profile``,
``suggest_model``, ``efficiency_adjusted_fitness`` + ``compare`` + the strategies,
the ``MODEL_AWARE_OPERATORS``, and ``aggregate`` / ``ModelEconomics``.
"""

from darwin.routing.efficiency import (
    DEFAULT_PARAMS,
    EfficiencyParams,
    EfficiencyStrategy,
    RawFitnessStrategy,
    compare,
    efficiency_adjusted_fitness,
    improves,
)
from darwin.routing.fleet import (
    CURATED_FLEET,
    FAST,
    FRONTIER,
    MID,
    FleetModel,
    by_tier,
    fleet_registry,
    get_fleet,
    install_fleet,
    profile,
)
from darwin.routing.gene import (
    MODEL_AWARE_OPERATORS,
    downgrade_mechanical,
    genotype,
    model_of,
    rebalance_genotype,
    retier_to_policy,
    swap_to_tier,
    upgrade_critical,
)
from darwin.routing.observability import ModelEconomics, aggregate, aggregate_sink
from darwin.routing.policy import (
    classify_role,
    classify_role_in_genome,
    suggest_model,
    suggest_tier,
    warm_start_genotype,
)

__all__ = [
    # fleet
    "FleetModel", "CURATED_FLEET", "get_fleet", "profile", "by_tier", "install_fleet",
    "fleet_registry", "FAST", "MID", "FRONTIER",
    # policy
    "classify_role", "classify_role_in_genome", "suggest_tier", "suggest_model", "warm_start_genotype",
    # efficiency
    "EfficiencyParams", "DEFAULT_PARAMS", "efficiency_adjusted_fitness", "compare", "improves",
    "EfficiencyStrategy", "RawFitnessStrategy",
    # gene
    "genotype", "model_of", "swap_to_tier", "downgrade_mechanical", "upgrade_critical",
    "retier_to_policy", "rebalance_genotype", "MODEL_AWARE_OPERATORS",
    # observability
    "aggregate", "aggregate_sink", "ModelEconomics",
]
