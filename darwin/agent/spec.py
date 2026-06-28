"""The AgentSpec data model — the contract the Architect (B4) fills.

The WorkerAgent is generic; what varies per agent is this spec. B4's Architect
authors these as structured data, on the fly, per problem — which is what makes
"we never hand-define agent types" real. B2 defines and validates the contract
so a malformed spec is rejected loudly at construction, keeping the
self-curation production-shaped rather than chaotic.
"""

import re
from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from darwin.constants import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_MODEL_ID

# A permissive slug: lowercase, starts alnum, words joined by _ or -.
_SLUG_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")


class InputKind(str, Enum):
    """What an agent expects to receive."""

    FULL_PROBLEM = "FULL_PROBLEM"
    SUB_PROBLEM = "SUB_PROBLEM"  # a region / slice of the problem
    SIBLING_OUTPUTS = "SIBLING_OUTPUTS"  # other agents' outputs to critique or merge
    PROBLEM_PLUS_DRAFT = "PROBLEM_PLUS_DRAFT"  # the problem plus a candidate to improve


class OutputKind(str, Enum):
    """Which structured shape an agent must produce (see ``outputs.py``)."""

    FULL_SOLUTION = "FULL_SOLUTION"
    PARTIAL_SOLUTION = "PARTIAL_SOLUTION"
    CRITIQUE = "CRITIQUE"
    CONSTRAINT_REPORT = "CONSTRAINT_REPORT"
    ARBITRATION = "ARBITRATION"
    DECOMPOSITION = "DECOMPOSITION"


class ThinkingLevel(str, Enum):
    """Per-agent reasoning dial (the model-aware knob, even within one model)."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class AgentSpec(BaseModel):
    """The frozen, validated job description for one worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    agent_id: str = Field(min_length=1)  # unique within a team
    role_name: str = Field(min_length=1)  # e.g. "cost_minimizer" — coined by the Architect
    role_description: str = Field(min_length=1)  # the system-prompt-level "job"
    input_contract: InputKind
    output_contract: OutputKind
    model_id: str = DEFAULT_MODEL_ID  # looked up in the ModelRegistry
    thinking_level: ThinkingLevel = ThinkingLevel.MEDIUM
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, gt=0)
    tool_names: List[str] = Field(default_factory=list)  # empty for the simple atom
    created_by: str = "architect"  # or "human_seed"
    spec_version: str = "1.0.0"

    @field_validator("role_name")
    @classmethod
    def _role_name_is_slug(cls, v: str) -> str:
        if not _SLUG_RE.match(v):
            raise ValueError(
                f"role_name must be a slug (lowercase, _/- separated), got {v!r}"
            )
        return v

    @field_validator("role_description")
    @classmethod
    def _role_description_nonblank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("role_description must not be blank")
        return v

    @model_validator(mode="after")
    def _model_id_in_registry(self, info: ValidationInfo) -> "AgentSpec":
        # The registry is resolved from validation context if provided
        # (AgentSpec.model_validate(data, context={"registry": reg})), else the
        # process-wide default registry — so B7 can extend the fleet centrally.
        registry = None
        if info.context:
            registry = info.context.get("registry")
        if registry is None:
            from darwin.agent.registry import default_registry

            registry = default_registry()
        if not registry.contains(self.model_id):
            raise ValueError(
                f"model_id {self.model_id!r} is not in the registry "
                f"(known: {registry.all_ids()}); fail fast at spec construction"
            )
        return self
