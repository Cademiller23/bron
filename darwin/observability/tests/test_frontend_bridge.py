"""Tests for the frontend bridge translator (darwin events -> face RunEvents).

Pure and offline: feed darwin (event_type, payload, description) tuples and assert
the emitted frontend RunEvent dicts match the face's `agent-run.ts` contract.
"""

from darwin.observability.frontend_bridge import EventTranslator, _convert_graph


def _graph():
    return {
        "arbiter_id": "arb",
        "nodes": [
            {"id": "p1", "model": "gemma", "role": "cost_minimizer", "description": "cost minimizer", "is_arbiter": False},
            {"id": "arb", "model": "gemini-3.5-flash", "role": "arbiter", "description": "arbiter", "is_arbiter": True},
        ],
        "edges": [{"from": "p1", "to": "arb", "edge_type": "FEEDS_ARBITER", "primary": True}],
    }


def test_convert_graph_marks_arbiter_as_terminal_output():
    g = _convert_graph(_graph())
    p1 = next(n for n in g["nodes"] if n["id"] == "p1")
    arb = next(n for n in g["nodes"] if n["id"] == "arb")
    assert p1["kind"] == "agent" and p1["terminal"] is False
    assert arb["kind"] == "output" and arb["terminal"] is True
    assert g["edges"] == [{"from": "p1", "to": "arb", "primary": True}]


def test_run_started_sets_problem_label_and_threshold():
    t = EventTranslator()
    out = t.translate("RUN_STARTED", {"instance_id": "f1-2026", "problem_class": "F1_CALENDAR",
                                      "threshold": 0.9}, "solve started")
    assert out[0] == {"type": "run_started", "problemLabel": "F1_CALENDAR · f1-2026",
                      "threshold": 0.9, "corpus": []}
    assert out[1]["type"] == "phase"


def test_team_designed_announces_seed_agents_then_candidate():
    t = EventTranslator()
    t.translate("RUN_STARTED", {"threshold": 0.9}, "")
    out = t.translate("TEAM_DESIGNED", {"graph": _graph()}, "designed")
    created = [e for e in out if e["type"] == "agent_created"]
    assert {e["agent"]["id"] for e in created} == {"p1", "arb"}
    assert all(e["agent"]["origin"] == "seed" for e in created)
    candidate = next(e for e in out if e["type"] == "candidate")
    assert candidate["label"] == "Initial team"
    assert len(candidate["nodes"]) == 2 and len(candidate["edges"]) == 1


def test_genome_evaluated_emits_scored_then_committed_with_graph():
    t = EventTranslator()
    t.translate("RUN_STARTED", {"threshold": 0.9}, "")
    out = t.translate("GENOME_EVALUATED",
                      {"normalized_score": 0.62, "fitness": 0.62, "graph": _graph()}, "evaluated")
    assert out[0] == {"type": "candidate_scored", "fitness": 0.62}
    gen = out[1]
    assert gen["type"] == "generation_committed"
    assert gen["iteration"]["id"] == 1
    assert gen["iteration"]["fitness"] == 0.62
    assert len(gen["iteration"]["nodes"]) == 2


def test_generation_counter_increments_and_uses_normalized_not_raw():
    t = EventTranslator()
    t.translate("RUN_STARTED", {"threshold": 0.9}, "")
    a = t.translate("GENOME_EVALUATED", {"normalized_score": 0.5, "fitness": -1e9, "graph": _graph()}, "")
    b = t.translate("GENOME_EVALUATED", {"normalized_score": 0.78, "fitness": 0.78, "graph": _graph()}, "")
    assert a[1]["iteration"]["id"] == 1 and a[0]["fitness"] == 0.5  # normalized, not the raw -1e9
    assert b[1]["iteration"]["id"] == 2 and b[1]["iteration"]["fitness"] == 0.78


def test_escalation_creates_agent_and_logs_gap():
    t = EventTranslator()
    out = t.translate("ESCALATION_CURATED",
                      {"role": "risk_balancer", "model": "gemini-3.1-pro-preview",
                       "added_agent_id": "risk", "origin": "grown",
                       "gap": "multi-supplier risk"}, "escalation")
    kinds = [e["type"] for e in out]
    assert "agent_created" in kinds
    created = next(e for e in out if e["type"] == "agent_created")
    assert created["agent"] == {"id": "risk", "role": "risk_balancer",
                                "model": "gemini-3.1-pro-preview", "origin": "grown"}
    assert any(e["type"] == "log" and "risk" in e["text"] for e in out)


def test_run_sealed_emits_run_complete_with_best_fitness():
    t = EventTranslator()
    t.translate("RUN_STARTED", {"threshold": 0.9}, "")
    t.translate("GENOME_EVALUATED", {"normalized_score": 0.78, "graph": _graph()}, "")
    out = t.translate("RUN_SEALED", {"normalized_score": 0.78}, "sealed")
    complete = next(e for e in out if e["type"] == "run_complete")
    assert complete["bestFitness"] == 0.78


def test_model_panel_update_is_dropped():
    t = EventTranslator()
    assert t.translate("MODEL_PANEL_UPDATE", {"genotype": {}}, "") == []


def test_unknown_event_falls_back_to_log():
    t = EventTranslator()
    out = t.translate("SOME_FUTURE_EVENT", {}, "a thing happened")
    assert out == [{"type": "log", "text": "a thing happened", "kind": "info"}]
