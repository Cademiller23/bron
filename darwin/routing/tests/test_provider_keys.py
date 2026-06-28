"""Per-model API key resolution — MAX and MiniMax authenticate with different keys
behind the one OpenAI-compatible adapter (B7 fleet metadata reaches the client)."""

from darwin.agent.providers.openai_compat import OpenAICompatProvider
from darwin.agent.registry import ModelEntry, Provider
from darwin.routing.fleet import profile


def _ok_response():
    return 200, {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}], "usage": {}}


async def _auth_for(entry, *, default_env, env):
    captured = {}

    async def fake_post(url, headers, body):
        captured["auth"] = headers["Authorization"]
        return _ok_response()

    prov = OpenAICompatProvider(post_json=fake_post, api_key_env=default_env)
    await prov.raw_complete(entry, "sys", "user", {}, "low", 128)
    return captured["auth"]


async def test_per_entry_api_key_env_overrides_adapter_default(monkeypatch):
    monkeypatch.setenv("DIGITAL_OCEAN_API_KEY", "secret-do")
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    entry = ModelEntry(model_id="llama3.3-70b-instruct", provider=Provider.OPENAI_COMPAT,
                       endpoint="https://inference.do-ai.run/v1", api_key_env="DIGITAL_OCEAN_API_KEY")
    auth = await _auth_for(entry, default_env="OPENAI_API_KEY", env="DIGITAL_OCEAN_API_KEY")
    assert auth == "Bearer secret-do"  # the model's own key, not the adapter default


async def test_two_fleet_models_use_two_different_keys(monkeypatch):
    monkeypatch.setenv("DIGITAL_OCEAN_API_KEY", "do-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    do_entry = profile("llama3.3-70b-instruct").to_registry_entry()
    gem_entry = profile("gemini-3.5-flash").to_registry_entry()
    assert await _auth_for(do_entry, default_env="OPENAI_API_KEY", env="DIGITAL_OCEAN_API_KEY") == "Bearer do-key"
    assert await _auth_for(gem_entry, default_env="OPENAI_API_KEY", env="GEMINI_API_KEY") == "Bearer gem-key"


async def test_entry_without_api_key_env_falls_back_to_adapter_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "default-key")
    entry = ModelEntry(model_id="x", provider=Provider.OPENAI_COMPAT, endpoint="https://x/v1")  # no api_key_env
    auth = await _auth_for(entry, default_env="OPENAI_API_KEY", env="OPENAI_API_KEY")
    assert auth == "Bearer default-key"
