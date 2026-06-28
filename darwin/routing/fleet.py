"""The curated model fleet — confirmed-live models across three tiers, one interface.

B2 built the ``ModelRegistry`` (model_id → provider/endpoint/profile) and the
model-agnostic ``ModelClient``. B7 does NOT rebuild any of that — it *populates*
the registry with a small, curated, sponsor-aligned fleet and carries the extra
operational metadata (api_key_env, default thinking level, hf id) that the
frozen ``ModelEntry`` doesn't hold, in a ``FleetModel`` descriptor that maps
down to a ``ModelEntry``.

Tiers reuse B2's ``CapabilityTier`` (``CHEAP`` is the spec's "FAST" tier):
  * **FAST** (CHEAP)  — the workhorses that run the bulk of mechanical/checking/
    proposer calls: ``llama3.3-70b-instruct`` served by DigitalOcean serverless
    inference (OpenAI-compatible, the load-bearing tier) plus a cheap native
    ``gemini-3.1-flash-lite`` for fast mechanical auditing.
  * **MID**           — ``gemini-3.5-flash`` (near-Pro quality at Flash cost).
  * **FRONTIER**      — ``gemini-3.1-pro-preview``: the Architect + arbitrator only.

Every model_id below has been verified to return a real response on the project's
own credentials (the DigitalOcean key for the OpenAI-compatible workhorse, the
Gemini key for the native tiers). No unconfirmed id is ever wired — a wrong id
silently falls back to a safe-default heuristic team, which would fake the demo.

The honesty constraint (§10): we curate a handful of models and demo on that
tractable tier; the method *scales* to a catalog of thousands because adding a
model is a single registry entry — every backend already speaks one of two
interfaces (native Gemini or OpenAI-compatible). We never claim to search
thousands live.
"""

import os
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from darwin.agent.registry import CapabilityTier, ModelEntry, ModelRegistry, Provider
from darwin.agent.spec import ThinkingLevel
from darwin.constants import DEFAULT_MODEL_ID

# Readability alias: the spec's "FAST" tier is B2's CHEAP capability tier.
FAST = CapabilityTier.CHEAP
MID = CapabilityTier.MID
FRONTIER = CapabilityTier.FRONTIER

# The DigitalOcean serverless-inference endpoint (OpenAI-compatible). Deploy-time
# overridable via DIGITALOCEAN_MODEL_BASE_URL; falls back to the documented URL.
_DO_BASE_URL = os.environ.get(
    "DIGITALOCEAN_MODEL_BASE_URL", "https://inference.do-ai.run/v1"
).rstrip("/") or "https://inference.do-ai.run/v1"


class FleetModel(BaseModel):
    """One curated model — a B2 ``ModelEntry`` plus B7 operational metadata.

    ``ModelEntry`` is frozen and intentionally minimal; the extra fields here
    (display name, api_key_env, default_thinking_level, hf id, notes) are the
    routing/observability/deploy metadata B7 owns. ``to_registry_entry`` projects
    this down to the exact ``ModelEntry`` the registry and client dispatch on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    provider: Provider
    tier: CapabilityTier
    endpoint: str = ""  # base_url for OPENAI_COMPAT; "" for native Gemini
    api_key_env: str = ""  # the env var holding the key
    est_cost_per_1k_in: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    est_cost_per_1k_out: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    est_latency_ms: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)
    supports_native_schema: bool = False
    default_thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM
    hf_model_id: str = ""  # the HuggingFace id the OpenAI-compat host serves (provenance + "served" badge)
    notes: str = ""

    def to_registry_entry(self) -> ModelEntry:
        return ModelEntry(
            model_id=self.model_id,
            provider=self.provider,
            endpoint=self.endpoint,
            api_key_env=self.api_key_env,  # per-model key so multi-provider auth works
            capability_tier=self.tier,
            est_cost_per_1k_in=self.est_cost_per_1k_in,
            est_cost_per_1k_out=self.est_cost_per_1k_out,
            est_latency_ms=self.est_latency_ms,
            supports_native_schema=self.supports_native_schema,
        )


# ---------------------------------------------------------------------------
# The curated fleet — concrete, sponsor-aligned, one entry per model.
# Endpoints are deploy-time config (env-overridable); prices/latencies are the
# routing/penalty profile. Adding more of DigitalOcean's 69-model catalog = just
# appending entries.
# ---------------------------------------------------------------------------
CURATED_FLEET: List[FleetModel] = [
    # The load-bearing workhorse: Llama 3.3 70B served by DigitalOcean serverless
    # inference (OpenAI-compatible). The bulk of mechanical and proposer calls land
    # here — DigitalOcean's throughput paces the whole swarm. Cheapest FAST model,
    # so the policy warm-start and the model-aware operators converge onto it.
    FleetModel(
        model_id="llama3.3-70b-instruct",
        display_name="Llama 3.3 70B (DigitalOcean)",
        provider=Provider.OPENAI_COMPAT,
        tier=FAST,
        endpoint=_DO_BASE_URL,
        api_key_env="DIGITAL_OCEAN_API_KEY",
        est_cost_per_1k_in=0.00002,
        est_cost_per_1k_out=0.00004,
        est_latency_ms=350.0,
        supports_native_schema=True,
        default_thinking_level=ThinkingLevel.MINIMAL,
        hf_model_id="meta-llama/Llama-3.3-70B-Instruct",
        notes="DigitalOcean-served; load-bearing FAST tier — confirmed live on the project key.",
    ),
    # A second FAST model: the cheap native Gemini flash-lite for fast mechanical
    # auditing — demonstrates a multi-provider FAST tier (DigitalOcean + native).
    # Priced just above the DO workhorse so the workhorse stays the cheapest FAST.
    FleetModel(
        model_id="gemini-3.1-flash-lite",
        display_name="Gemini 3.1 Flash-Lite",
        provider=Provider.GEMINI,
        tier=FAST,
        endpoint="",
        api_key_env="GEMINI_API_KEY",
        est_cost_per_1k_in=0.0001,
        est_cost_per_1k_out=0.0004,
        est_latency_ms=600.0,
        supports_native_schema=True,
        default_thinking_level=ThinkingLevel.LOW,
        notes="Native Gemini FAST tier — confirmed live (GA, non-preview).",
    ),
    # Near-Pro agentic/coding quality at Flash speed/cost — the strong default for
    # proposers and objective specialists. (Already in B2's seed registry.)
    FleetModel(
        model_id=DEFAULT_MODEL_ID,  # "gemini-3.5-flash"
        display_name="Gemini 3.5 Flash",
        provider=Provider.GEMINI,
        tier=MID,
        endpoint="",
        api_key_env="GEMINI_API_KEY",
        est_cost_per_1k_in=0.00015,
        est_cost_per_1k_out=0.0006,
        est_latency_ms=900.0,
        supports_native_schema=True,
        default_thinking_level=ThinkingLevel.MEDIUM,
        notes="MID default for proposers/specialists — confirmed live.",
    ),
    # The rare, expensive, top-reasoning model: the arbitrator and the Architect.
    # NOTE: the GA id is `-preview`; bare `gemini-3.1-pro` 404s on the API.
    FleetModel(
        model_id="gemini-3.1-pro-preview",
        display_name="Gemini 3.1 Pro",
        provider=Provider.GEMINI,
        tier=FRONTIER,
        endpoint="",
        api_key_env="GEMINI_API_KEY",
        est_cost_per_1k_in=0.00125,
        est_cost_per_1k_out=0.005,
        est_latency_ms=2600.0,
        supports_native_schema=True,
        default_thinking_level=ThinkingLevel.HIGH,
        notes="FRONTIER — arbitrator + Architect only. Confirmed live (must use -preview).",
    ),
]


class FleetError(ValueError):
    """Raised when the fleet is malformed at load (fail fast, loudly)."""


def _validate_fleet(models: List[FleetModel]) -> None:
    if not models:
        raise FleetError("the fleet is empty")
    ids = [m.model_id for m in models]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise FleetError(f"duplicate model_id(s) in the fleet: {dupes}")
    for m in models:
        if m.provider == Provider.OPENAI_COMPAT and not m.endpoint.strip():
            raise FleetError(f"OPENAI_COMPAT model {m.model_id!r} must declare an endpoint")
        if m.provider == Provider.OPENAI_COMPAT and not (
            m.endpoint.startswith("http://") or m.endpoint.startswith("https://")
        ):
            raise FleetError(f"model {m.model_id!r} endpoint must be a well-formed http(s) URL")
    tiers = {m.tier for m in models}
    for required in (FAST, MID, FRONTIER):
        if required not in tiers:
            raise FleetError(f"the fleet must have at least one model in tier {required.value}")


_validate_fleet(CURATED_FLEET)  # fail fast at import if the table is malformed
_BY_ID: Dict[str, FleetModel] = {m.model_id: m for m in CURATED_FLEET}


def get_fleet() -> List[FleetModel]:
    """The curated fleet (a fresh list; the entries are frozen)."""
    return list(CURATED_FLEET)


def profile(model_id: str) -> FleetModel:
    """The full ``FleetModel`` for a model_id, or a clear error if out-of-catalog."""
    try:
        return _BY_ID[model_id]
    except KeyError:
        raise KeyError(
            f"model_id {model_id!r} is not in the curated fleet; known: {sorted(_BY_ID)}"
        ) from None


def by_tier(tier: CapabilityTier) -> List[str]:
    """The model_ids in a tier (stable order = fleet order)."""
    return [m.model_id for m in CURATED_FLEET if m.tier == tier]


def install_fleet(registry: Optional[ModelRegistry] = None) -> ModelRegistry:
    """Register every fleet model into ``registry`` (the process-wide default if
    omitted) and return it. Idempotent — re-registering replaces the entry."""
    from darwin.agent.registry import default_registry

    reg = registry if registry is not None else default_registry()
    for m in CURATED_FLEET:
        reg.register(m.to_registry_entry())
    return reg


def fleet_registry() -> ModelRegistry:
    """A FRESH registry containing exactly the curated fleet (for tests / a clean
    routing context that doesn't depend on the process-wide seed)."""
    return install_fleet(ModelRegistry())
