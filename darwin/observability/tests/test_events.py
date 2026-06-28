"""RunEvent schema — round-trip, frozen, the event-type narrative."""

import pytest
from pydantic import ValidationError

from darwin.observability.events import RunEvent, RunEventType


def test_event_roundtrips_through_doc():
    e = RunEvent(run_id="r1", sequence_number=3, event_type=RunEventType.TEAM_DESIGNED,
                 description="designed", payload={"genome_id": "g", "version": 2})
    doc = e.to_doc()
    assert doc["_id"] == e.event_id and "event_id" not in doc
    back = RunEvent.from_doc(doc)
    assert back == e


def test_model_dump_json_roundtrip():
    e = RunEvent(run_id="r1", sequence_number=0, event_type=RunEventType.RUN_STARTED)
    assert RunEvent.model_validate(e.model_dump()) == e


def test_frozen_and_extra_forbid():
    e = RunEvent(run_id="r", sequence_number=1, event_type=RunEventType.RUN_SEALED)
    with pytest.raises(ValidationError):
        e.sequence_number = 2
    with pytest.raises(ValidationError):
        RunEvent(run_id="r", sequence_number=1, event_type=RunEventType.RUN_SEALED, bogus=1)


def test_validation_bounds():
    with pytest.raises(ValidationError):
        RunEvent(run_id="", sequence_number=0, event_type=RunEventType.RUN_STARTED)  # run_id min_length
    with pytest.raises(ValidationError):
        RunEvent(run_id="r", sequence_number=-1, event_type=RunEventType.RUN_STARTED)  # ge 0


def test_event_id_and_timestamp_autofilled():
    e = RunEvent(run_id="r", sequence_number=0, event_type=RunEventType.RUN_STARTED)
    assert e.event_id and e.timestamp
    e2 = RunEvent(run_id="r", sequence_number=1, event_type=RunEventType.RUN_STARTED)
    assert e.event_id != e2.event_id  # unique


def test_event_type_narrative_complete():
    names = {t.value for t in RunEventType}
    for required in ("RUN_STARTED", "TEAM_DESIGNED", "GENOME_EVALUATED", "REARRANGE_ADOPTED",
                     "THRESHOLD_CHECK", "ESCALATION_CORPUS_HIT", "ESCALATION_CURATED", "AGENT_ROLLED_BACK",
                     "MODEL_PANEL_UPDATE", "SCORER_RETUNED", "RUN_SEALED", "RUN_EXHAUSTED"):
        assert required in names


def test_from_doc_without_id_key():
    # a doc that already uses event_id (not _id) also validates
    e = RunEvent(run_id="r", sequence_number=0, event_type=RunEventType.RUN_STARTED)
    assert RunEvent.from_doc(e.model_dump(mode="json")) == e
