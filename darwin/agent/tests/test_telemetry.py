"""§12.6 Telemetry tests — write paths and fire-and-forget failure-safety."""

import logging

import pytest

from darwin.agent import fixtures as F
from darwin.agent.registry import reset_default_registry
from darwin.agent.telemetry import InMemoryTelemetrySink, MongoTelemetrySink
from darwin.agent.worker import WorkerAgent


@pytest.fixture(autouse=True)
def _clean():
    reset_default_registry()
    yield
    reset_default_registry()


class FakeAsyncCollection:
    """A minimal stand-in for a motor AsyncIOMotorCollection."""

    def __init__(self, fail: bool = False):
        self.docs = []
        self.fail = fail

    async def insert_one(self, doc):
        if self.fail:
            raise RuntimeError("mongo is down")
        self.docs.append(dict(doc))
        return type("R", (), {"inserted_id": len(self.docs)})()

    async def find_one(self, query=None):
        for doc in self.docs:
            if all(doc.get(k) == v for k, v in (query or {}).items()):
                return doc
        return None


async def _run_with_sink(script, sink):
    spec = F.make_spec()
    client = F.scripted_client(script)
    worker = WorkerAgent(spec, client, sink)
    return await worker.run(F.make_input())


async def test_successful_run_writes_document_with_required_fields():
    inv, corpus = FakeAsyncCollection(), FakeAsyncCollection()
    sink = MongoTelemetrySink(inv, corpus)
    result = await _run_with_sink([F.native_schema_response()], sink)
    assert result.success is True
    assert len(inv.docs) == 1
    doc = inv.docs[0]
    for key in ("invocation_id", "agent_id", "role_name", "model_id", "thinking_level",
                "instance_id", "input_kind", "output_kind", "success", "num_repairs",
                "latency_ms", "tokens_in", "tokens_out", "est_cost", "raw_output",
                "validated", "error", "scorer_fitness", "scorer_version", "created_at"):
        assert key in doc, key
    assert doc["success"] is True and doc["validated"] is True


async def test_failed_run_writes_document_with_success_false_and_error():
    inv = FakeAsyncCollection()
    sink = MongoTelemetrySink(inv, FakeAsyncCollection())
    result = await _run_with_sink([F.malformed_json_response()], sink)
    assert result.success is False
    assert len(inv.docs) == 1
    assert inv.docs[0]["success"] is False
    assert inv.docs[0]["error"]


async def test_mongo_write_failure_degrades_to_local_log(caplog):
    sink = MongoTelemetrySink(FakeAsyncCollection(fail=True), FakeAsyncCollection())
    with caplog.at_level(logging.WARNING, logger="darwin.agent.telemetry"):
        result = await _run_with_sink([F.native_schema_response()], sink)  # must NOT raise
    assert result.success is True  # the run completed despite the telemetry failure
    assert any("degraded to local log" in r.message for r in caplog.records)


async def test_corpus_write_and_read_back():
    corpus = FakeAsyncCollection()
    sink = MongoTelemetrySink(FakeAsyncCollection(), corpus)
    spec = F.make_spec()
    await sink.save_corpus_spec(
        {"spec": spec.model_dump(), "role_name": spec.role_name, "role_description": spec.role_description}
    )
    assert len(corpus.docs) == 1
    fetched = await corpus.find_one({"role_name": spec.role_name})
    assert fetched is not None and fetched["role_description"] == spec.role_description


async def test_corpus_failure_is_also_safe(caplog):
    sink = MongoTelemetrySink(FakeAsyncCollection(), FakeAsyncCollection(fail=True))
    with caplog.at_level(logging.WARNING, logger="darwin.agent.telemetry"):
        await sink.save_corpus_spec({"role_name": "x"})  # must not raise
    assert any("save_corpus_spec failed" in r.message for r in caplog.records)


async def test_in_memory_sink_captures():
    sink = InMemoryTelemetrySink()
    await _run_with_sink([F.native_schema_response()], sink)
    assert len(sink.invocations) == 1
