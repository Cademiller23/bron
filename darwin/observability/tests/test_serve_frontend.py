"""End-to-end pipe test: demo Conductor -> emitter/bus -> translator -> sends.

No socket, no network, no API key. Drives a full run through the real assembly the
serve entrypoint uses (demo stack) and asserts the captured stream is a valid
frontend run: run_started ... a climbing curve that grows an agent and clears the
gate ... run_complete.
"""

import os

import pytest

from darwin.observability.serve_frontend import replay_session, run_one_session


@pytest.fixture(autouse=True)
def _force_demo(monkeypatch):
    monkeypatch.setenv("DARWIN_DEMO", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield


async def test_full_demo_run_produces_a_valid_frontend_stream():
    sent = []

    async def capture(doc):
        sent.append(doc)

    await run_one_session(capture, pace_s=0.0)

    types = [e["type"] for e in sent]
    assert types[0] == "run_started"
    assert types[-1] == "run_complete"
    assert "agent_created" in types  # seed agents and/or a grown one
    assert "generation_committed" in types

    started = sent[0]
    assert started["threshold"] == pytest.approx(0.90)
    assert started["problemLabel"]

    # The curve must grow an agent and break the ceiling: a generation below the
    # threshold, then one at/above it (the rearrange -> grow -> clear arc).
    gens = [e["iteration"] for e in sent if e["type"] == "generation_committed"]
    assert len(gens) >= 2
    fitnesses = [g["fitness"] for g in gens]
    assert min(fitnesses) < 0.90 <= max(fitnesses)

    # Every committed generation carries a drawable org chart.
    for g in gens:
        assert g["nodes"] and any(n["terminal"] for n in g["nodes"])  # the arbiter is terminal

    # A grown agent appeared (origin grown or corpus), not only seeds.
    origins = {e["agent"]["origin"] for e in sent if e["type"] == "agent_created"}
    assert origins & {"grown", "corpus"}

    final = sent[-1]
    assert final["bestFitness"] >= 0.90


async def test_record_then_replay_is_byte_identical(tmp_path):
    rec = tmp_path / "run.jsonl"

    live = []
    async def capture_live(doc):
        live.append(doc)

    await run_one_session(capture_live, pace_s=0.0, record_path=str(rec))
    assert rec.exists() and live[-1]["type"] == "run_complete"

    replayed = []
    async def capture_replay(doc):
        replayed.append(doc)

    n = await replay_session(capture_replay, str(rec), pace_s=0.0)
    assert n == len(live)
    assert replayed == live  # replay reproduces the recorded run exactly
