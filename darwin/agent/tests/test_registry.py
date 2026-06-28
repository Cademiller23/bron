"""§12.3 Registry tests."""

import math

import pytest

from darwin.agent.registry import (
    CapabilityTier,
    ModelEntry,
    ModelRegistry,
    Provider,
    default_registry,
    reset_default_registry,
)
from darwin.constants import DEFAULT_MODEL_ID


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


def test_known_lookup_returns_entry():
    entry = default_registry().get(DEFAULT_MODEL_ID)
    assert entry.model_id == DEFAULT_MODEL_ID
    assert entry.provider == Provider.GEMINI
    assert entry.supports_native_schema is True


def test_unknown_lookup_raises_clear_error():
    with pytest.raises(KeyError) as exc:
        default_registry().get("nope")
    assert "unknown model_id" in str(exc.value)


def test_contains_and_all_ids():
    reg = default_registry()
    assert reg.contains(DEFAULT_MODEL_ID)
    assert not reg.contains("nope")
    assert DEFAULT_MODEL_ID in reg.all_ids()


def test_cost_estimation_uses_rates():
    reg = ModelRegistry(
        {
            "m": ModelEntry(
                model_id="m", provider=Provider.OPENAI_COMPAT, endpoint="https://x/v1",
                est_cost_per_1k_in=0.01, est_cost_per_1k_out=0.03,
            )
        }
    )
    # 2000 in @ 0.01/1k + 1000 out @ 0.03/1k = 0.02 + 0.03 = 0.05
    assert math.isclose(reg.estimate_cost("m", 2000, 1000), 0.05)


def test_degraded_flag_round_trips_via_registry():
    reg = default_registry()
    assert reg.is_degraded(DEFAULT_MODEL_ID) is False
    reg.set_degraded(DEFAULT_MODEL_ID, True)
    assert reg.is_degraded(DEFAULT_MODEL_ID) is True
    reg.set_degraded(DEFAULT_MODEL_ID, False)
    assert reg.is_degraded(DEFAULT_MODEL_ID) is False


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_model_entry_rejects_non_finite_cost(value):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelEntry(model_id="m", provider=Provider.GEMINI, est_cost_per_1k_in=value)


def test_register_adds_fleet_entry():
    reg = default_registry()
    reg.register(
        ModelEntry(model_id="fleet-max", provider=Provider.OPENAI_COMPAT, endpoint="https://do/v1",
                   capability_tier=CapabilityTier.FRONTIER)
    )
    assert reg.contains("fleet-max")
    assert reg.get("fleet-max").provider == Provider.OPENAI_COMPAT
