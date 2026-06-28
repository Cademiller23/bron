"""The event stream schema — the narrative of a solve.

Clean separation of concerns: the *domain state* (genomes, agent_invocations,
agent_corpus — what IS, authoritative) versus the *event log* (RunEvents — what
HAPPENED, in order — the narrative the screen animates and replay reads). Events
reference domain objects by id + version; the domain collections stay
authoritative.

A ``RunEvent`` is one structured, ordered moment. ``sequence_number`` is monotonic
per run, so the stream has a single, replayable truth — no second guess.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class RunEventType(str, Enum):
    """The ordered narrative of a solve (one enum value per meaningful moment)."""

    RUN_STARTED = "RUN_STARTED"
    GROUNDING_DONE = "GROUNDING_DONE"  # B9 — real (e.g. F1) data loaded
    TEAM_DESIGNED = "TEAM_DESIGNED"  # B4
    GENOME_EVALUATED = "GENOME_EVALUATED"  # B3
    REARRANGE_CANDIDATE_SCORED = "REARRANGE_CANDIDATE_SCORED"  # B5
    REARRANGE_ADOPTED = "REARRANGE_ADOPTED"  # B5
    THRESHOLD_CHECK = "THRESHOLD_CHECK"  # B6
    ESCALATION_CORPUS_HIT = "ESCALATION_CORPUS_HIT"  # B6 — reused a proven agent
    ESCALATION_CURATED = "ESCALATION_CURATED"  # B6 — authored a new agent
    AGENT_ROLLED_BACK = "AGENT_ROLLED_BACK"  # B6 — unhelpful growth reverted
    MODEL_PANEL_UPDATE = "MODEL_PANEL_UPDATE"  # B7 — per-model call/cost breakdown
    SCORER_RETUNED = "SCORER_RETUNED"  # B8 — the self-improving scorer tuned weights
    RUN_SEALED = "RUN_SEALED"  # cleared the 0.90 gate
    RUN_EXHAUSTED = "RUN_EXHAUSTED"  # budget ran out; best-so-far


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunEvent(BaseModel):
    """One ordered moment in a run. Frozen; the durable, replayable unit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex)
    run_id: str = Field(min_length=1)
    sequence_number: int = Field(ge=0)  # monotonic per run — the replay order key
    timestamp: str = Field(default_factory=_now_iso)
    event_type: RunEventType
    description: str = ""  # human/voice-narration line
    payload: Dict[str, Any] = Field(default_factory=dict)  # data + domain refs (id/version)

    def to_doc(self) -> Dict[str, Any]:
        """Mongo document form (``_id`` = event_id)."""
        doc = self.model_dump(mode="json")
        doc["_id"] = doc.pop("event_id")
        return doc

    @classmethod
    def from_doc(cls, doc: Dict[str, Any]) -> "RunEvent":
        data = dict(doc)
        if "event_id" not in data and "_id" in data:
            data["event_id"] = data.pop("_id")
        else:
            data.pop("_id", None)
        return cls.model_validate(data)
