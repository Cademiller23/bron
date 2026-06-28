"""§12.1 AgentSpec tests."""

import pytest
from pydantic import ValidationError

from darwin.agent.registry import (
    CapabilityTier,
    ModelEntry,
    ModelRegistry,
    Provider,
    reset_default_registry,
)
from darwin.agent.spec import AgentSpec, InputKind, OutputKind, ThinkingLevel
from darwin.constants import DEFAULT_MODEL_ID


@pytest.fixture(autouse=True)
def _clean_registry():
    reset_default_registry()
    yield
    reset_default_registry()


def _spec(**overrides) -> AgentSpec:
    kwargs = dict(
        agent_id="a1",
        role_name="cost_minimizer",
        role_description="Minimize total cost.",
        input_contract=InputKind.FULL_PROBLEM,
        output_contract=OutputKind.FULL_SOLUTION,
    )
    kwargs.update(overrides)
    return AgentSpec(**kwargs)


def test_valid_spec_and_defaults():
    s = _spec()
    assert s.model_id == DEFAULT_MODEL_ID
    assert s.thinking_level == ThinkingLevel.MEDIUM
    assert s.max_output_tokens > 0
    assert s.tool_names == []
    assert s.created_by == "architect"


def test_empty_role_name_raises():
    with pytest.raises(ValidationError):
        _spec(role_name="")


def test_non_slug_role_name_raises():
    with pytest.raises(ValidationError):
        _spec(role_name="Cost Minimizer!")


def test_empty_role_description_raises():
    with pytest.raises(ValidationError):
        _spec(role_description="")


def test_blank_role_description_raises():
    with pytest.raises(ValidationError):
        _spec(role_description="   ")


def test_unknown_model_id_raises_at_construction():
    with pytest.raises(ValidationError):
        _spec(model_id="not-registered")


def test_invalid_thinking_level_raises():
    with pytest.raises(ValidationError):
        _spec(thinking_level="ultra")


def test_non_positive_max_output_tokens_raises():
    with pytest.raises(ValidationError):
        _spec(max_output_tokens=0)
    with pytest.raises(ValidationError):
        _spec(max_output_tokens=-5)


def test_frozen():
    s = _spec()
    with pytest.raises(ValidationError):
        s.role_name = "other"


def test_extra_field_forbidden():
    with pytest.raises(ValidationError):
        _spec(unexpected="x")


def test_round_trip_serialize_deserialize():
    s = _spec(thinking_level=ThinkingLevel.HIGH, max_output_tokens=4096, tool_names=["search"])
    restored = AgentSpec.model_validate(s.model_dump())
    assert restored == s
    assert AgentSpec.model_validate_json(s.model_dump_json()) == s


def test_validation_against_injected_registry_context():
    reg = ModelRegistry(
        {
            "fleet-max": ModelEntry(
                model_id="fleet-max", provider=Provider.OPENAI_COMPAT,
                endpoint="https://example/v1", capability_tier=CapabilityTier.FRONTIER,
            )
        }
    )
    # default registry does not know "fleet-max"
    with pytest.raises(ValidationError):
        _spec(model_id="fleet-max")
    # but with the fleet registry in context it validates
    data = dict(
        agent_id="a1", role_name="r", role_description="d",
        input_contract=InputKind.FULL_PROBLEM, output_contract=OutputKind.FULL_SOLUTION,
        model_id="fleet-max",
    )
    spec = AgentSpec.model_validate(data, context={"registry": reg})
    assert spec.model_id == "fleet-max"
