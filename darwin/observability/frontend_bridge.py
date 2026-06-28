"""The frontend bridge — translate darwin events into the face's RunEvent shape.

The TypeScript face (``f1-optimization-visualization``) renders entirely from its
own ``RunEvent`` union (see ``src/lib/agent-run.ts``): ``run_started`` / ``phase``
/ ``log`` / ``candidate`` / ``candidate_scored`` / ``generation_committed`` /
``agent_created`` / ``run_complete``. The Python brain emits a *different*,
richer envelope (``RunEvent{event_type, payload, sequence_number, …}`` in
``events.py``). This module is the one-way adapter between them.

:class:`EventTranslator` is a **pure, stateful translator**: feed it each darwin
event (``event_type``, ``payload``, ``description``) in order and it returns the
list of frontend events to forward. It does no I/O, so it is fully unit-testable
offline. The serve entrypoint (``serve_frontend.py``) wires it to the live bus and
a WebSocket; the frontend's ``createSocketRun`` consumes the result unchanged.

Design choices that keep the face's contract intact:
* The face's curve and 0.90 gate are in [0, 1], so generation/candidate fitness is
  the brain's ``normalized_score`` (quality Q), never the raw signed ``fitness``.
* Seed agents have no ``ESCALATION_*`` event, so the first ``TEAM_DESIGNED`` graph
  is replayed as ``agent_created`` (origin ``seed``) to populate the corpus panel.
* Each ``GENOME_EVALUATED`` becomes the candidate→scored→committed beat the face
  animates, carrying the genome's org chart (nodes + edges) from the enriched
  payload (``conductor._graph``).
"""

from typing import Any, Dict, List, Optional


def _node(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Darwin graph node -> frontend GenomeNode."""
    is_arbiter = bool(raw.get("is_arbiter"))
    return {
        "id": raw["id"],
        "model": raw.get("model", ""),
        "description": raw.get("description") or raw.get("role", ""),
        "kind": "output" if is_arbiter else "agent",
        "terminal": is_arbiter,
    }


def _edge(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Darwin graph edge -> frontend GenomeEdge."""
    return {"from": raw["from"], "to": raw["to"], "primary": bool(raw.get("primary"))}


def _convert_graph(graph: Optional[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    if not graph:
        return {"nodes": [], "edges": []}
    return {
        "nodes": [_node(n) for n in graph.get("nodes", [])],
        "edges": [_edge(e) for e in graph.get("edges", [])],
    }


def _label(payload: Dict[str, Any]) -> str:
    pclass = payload.get("problem_class")
    iid = payload.get("instance_id", "instance")
    return f"{pclass} · {iid}" if pclass else str(iid)


class EventTranslator:
    """Stateful, pure mapper from darwin events to frontend RunEvents.

    Call :meth:`translate` once per darwin event, in ``sequence_number`` order.
    Returns the (possibly empty) list of frontend event dicts to send.
    """

    def __init__(self) -> None:
        self._threshold: float = 0.90
        self._best_fitness: float = 0.0
        self._generation: int = 0
        self._last_graph: Dict[str, List[Dict[str, Any]]] = {"nodes": [], "edges": []}
        self._known_agents: set = set()  # agent_ids already announced to the corpus

    def translate(
        self, event_type: str, payload: Optional[Dict[str, Any]] = None, description: str = ""
    ) -> List[Dict[str, Any]]:
        payload = payload or {}
        et = event_type.value if hasattr(event_type, "value") else str(event_type)
        handler = getattr(self, f"_on_{et.lower()}", None)
        if handler is None:
            return [self._log(description, "info")] if description else []
        return handler(payload, description)

    # -- helpers -------------------------------------------------------------
    def _log(self, text: str, kind: str = "info") -> Dict[str, Any]:
        return {"type": "log", "text": text, "kind": kind}

    def _agent_created(self, agent_id: str, role: str, model: str, origin: str) -> Dict[str, Any]:
        self._known_agents.add(agent_id)
        return {"type": "agent_created",
                "agent": {"id": agent_id, "role": role, "model": model, "origin": origin}}

    # -- per-event handlers (named _on_<event_type lowercased>) --------------
    def _on_run_started(self, payload, description) -> List[Dict[str, Any]]:
        self._threshold = float(payload.get("threshold", 0.90))
        self._best_fitness = 0.0
        self._generation = 0
        self._known_agents = set()
        return [
            {"type": "run_started", "problemLabel": _label(payload),
             "threshold": self._threshold, "corpus": []},
            {"type": "phase", "text": description or "Decomposing problem"},
        ]

    def _on_grounding_done(self, payload, description) -> List[Dict[str, Any]]:
        return [self._log(description or "real data grounded", "info")]

    def _on_team_designed(self, payload, description) -> List[Dict[str, Any]]:
        graph = _convert_graph(payload.get("graph"))
        self._last_graph = graph
        out: List[Dict[str, Any]] = []
        # Seed agents have no ESCALATION_* event; announce them so the corpus fills.
        for n in graph["nodes"]:
            if n["id"] not in self._known_agents:
                out.append(self._agent_created(n["id"], n["description"], n["model"], "seed"))
        out.append({"type": "candidate", "label": "Initial team",
                    "nodes": graph["nodes"], "edges": graph["edges"]})
        out.append({"type": "phase", "text": description or "Architect designed the team"})
        return out

    def _on_genome_evaluated(self, payload, description) -> List[Dict[str, Any]]:
        graph = _convert_graph(payload.get("graph")) if payload.get("graph") else self._last_graph
        self._last_graph = graph
        fitness = float(payload.get("normalized_score", 0.0))
        self._best_fitness = max(self._best_fitness, fitness)
        self._generation += 1
        return [
            {"type": "candidate_scored", "fitness": fitness},
            {"type": "generation_committed", "iteration": {
                "id": self._generation,
                "name": f"Generation {self._generation}",
                "description": description or "team evaluated",
                "fitness": fitness,
                "nodes": graph["nodes"], "edges": graph["edges"],
            }},
        ]

    def _on_threshold_check(self, payload, description) -> List[Dict[str, Any]]:
        score = float(payload.get("normalized_score", 0.0))
        cleared = bool(payload.get("cleared"))
        kind = "success" if cleared else "score"
        mark = "✓ above" if cleared else "below"
        return [self._log(f"Score {score:.2f} — {mark} threshold {self._threshold:.2f}", kind)]

    def _on_rearrange_adopted(self, payload, description) -> List[Dict[str, Any]]:
        return [{"type": "phase", "text": description or "Rearranging topology"}]

    def _on_escalation_corpus_hit(self, payload, description) -> List[Dict[str, Any]]:
        return self._on_escalation(payload, description)

    def _on_escalation_curated(self, payload, description) -> List[Dict[str, Any]]:
        return self._on_escalation(payload, description)

    def _on_escalation(self, payload, description) -> List[Dict[str, Any]]:
        role = payload.get("role", "agent")
        model = payload.get("model", "")
        origin = payload.get("origin", "grown")
        agent_id = payload.get("added_agent_id", role)
        gap = payload.get("gap")
        out: List[Dict[str, Any]] = []
        if gap:
            out.append(self._log(f"Missing capability: {gap}", "corpus"))
        out.append(self._agent_created(agent_id, role, model, origin))
        out.append(self._log(f"Created agent: {role} → {model or 'model'}", "corpus"))
        return out

    def _on_agent_rolled_back(self, payload, description) -> List[Dict[str, Any]]:
        return [self._log(description or "rolled back unhelpful agent", "corpus")]

    def _on_model_panel_update(self, payload, description) -> List[Dict[str, Any]]:
        return []  # the face's RunEvent union has no model panel; nothing to send

    def _on_scorer_retuned(self, payload, description) -> List[Dict[str, Any]]:
        return [self._log(description or "scorer retuned its criteria", "info")]

    def _on_run_sealed(self, payload, description) -> List[Dict[str, Any]]:
        return self._terminal(payload)

    def _on_run_exhausted(self, payload, description) -> List[Dict[str, Any]]:
        return self._terminal(payload)

    def _terminal(self, payload) -> List[Dict[str, Any]]:
        best = float(payload.get("normalized_score", self._best_fitness))
        self._best_fitness = max(self._best_fitness, best)
        return [
            {"type": "phase", "text": "Converged — selected best team"},
            {"type": "run_complete", "bestFitness": self._best_fitness},
        ]
