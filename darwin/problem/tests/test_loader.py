"""§8.5 Loader tests — adapters, errors, provenance, and caching."""

import os

import pytest

import darwin.problem as problem_pkg
from darwin.problem import loader
from darwin.problem.schemas import OptimumSource, ProblemClass

DATA = os.path.join(os.path.dirname(problem_pkg.__file__), "data")


@pytest.fixture(autouse=True)
def _clear_cache():
    loader.clear_cache()
    yield
    loader.clear_cache()


def test_industryor_adapter_golden():
    inst = loader.load_instance("industryor", os.path.join(DATA, "industryor_sample.json"))
    assert inst.instance_id == "industryor-tp-0001"
    assert inst.source == "industryor"
    assert inst.problem_class == ProblemClass.TRANSPORTATION
    assert inst.metadata.num_nodes == 4 and inst.metadata.num_arcs == 4
    assert inst.known_optimum.objective_value == 23.0
    assert inst.known_optimum.source == OptimumSource.BENCHMARK_LABEL
    assert inst.known_optimum.verified is False


def test_mamo_adapter_golden():
    inst = loader.load_instance("mamo", os.path.join(DATA, "mamo_sample.json"))
    assert inst.instance_id == "mamo-ts-easy-01"
    assert inst.source == "mamo"
    assert inst.problem_class == ProblemClass.TRANSSHIPMENT
    assert inst.known_optimum.objective_value == 16.0
    assert inst.known_optimum.source == OptimumSource.SOLVER_VERIFIED
    assert inst.transshipments()[0].node_id == "T"


def test_cvrplib_adapter_golden():
    inst = loader.load_instance("cvrplib", os.path.join(DATA, "cvrplib_sample.vrp"))
    assert inst.source == "cvrplib"
    assert inst.problem_class == ProblemClass.VEHICLE_ROUTING
    assert len(inst.sinks()) == 3  # 3 customers
    assert len(inst.sources()) == 1  # depot
    assert inst.additional_constraints[0].parameters["vehicle_capacity"] == 10.0
    assert inst.known_optimum.objective_value == 54.0  # from COMMENT, labelled only
    # coordinates carried through
    depot = inst.sources()[0]
    assert depot.coordinates == (0.0, 0.0)


def test_generated_adapter_passthrough():
    from darwin.problem.generator import generate_instance

    gen = generate_instance(seed=1, problem_class=ProblemClass.TRANSPORTATION)
    reloaded = loader.load_instance("generated", gen)
    assert reloaded.instance_id == gen.instance_id
    assert reloaded == gen


def test_unknown_source_raises():
    with pytest.raises(ValueError):
        loader.load_instance("does-not-exist", {})


def test_malformed_raw_raises_clearly():
    # arc references an unknown node -> schema validator rejects at load
    bad = {
        "id": "bad", "difficulty": "EASY", "problem_class": "TRANSPORTATION",
        "nodes": [{"id": "S1", "type": "SOURCE", "supply": 10.0}],
        "arcs": [{"id": "a", "from": "S1", "to": "GHOST", "cost": 1.0}],
    }
    with pytest.raises(Exception):
        loader.load_instance("industryor", bad)


def test_not_json_raises():
    with pytest.raises(Exception):
        loader.load_instance("industryor", "this is not json {")


def test_caching_returns_identical_object():
    path = os.path.join(DATA, "industryor_sample.json")
    first = loader.load_instance("industryor", path)
    second = loader.load_instance("industryor", path)
    assert first is second  # served from cache


def test_cache_can_be_bypassed_and_cleared():
    path = os.path.join(DATA, "industryor_sample.json")
    first = loader.load_instance("industryor", path)
    fresh = loader.load_instance("industryor", path, use_cache=False)
    assert fresh is not first
    assert fresh == first
    loader.clear_cache()
    after_clear = loader.load_instance("industryor", path)
    assert after_clear is not first
