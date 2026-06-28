"""Tests for parsing.extract_json / try_parse_json — the fallback ladder rung."""

import json

import pytest

from darwin.agent.parsing import extract_json, try_parse_json


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ("```json\n{\"a\": 2}\n```", {"a": 2}),
        ("```\n{\"a\": 3}\n```", {"a": 3}),
        ("Here you go:\n{\"a\": 4}\nThanks!", {"a": 4}),
        ('{"note": "use {curly} and [brackets]", "x": 1}', {"note": "use {curly} and [brackets]", "x": 1}),
        ('{"q": "she said \\"hi\\" {", "y": 2}', {"q": 'she said "hi" {', "y": 2}),
        ("[{\"a\": 1}, {\"b\": 2}]", [{"a": 1}, {"b": 2}]),
        ('Top picks [A, B]. {"k": 1}', {"k": 1}),  # junk balanced bracket before the real object
        ('{"a":1} and also {"b":2}', {"a": 1}),  # first balanced object wins
    ],
)
def test_extract_json_recovers(text, expected):
    span = extract_json(text)
    assert span is not None
    assert json.loads(span) == expected


def test_extract_json_picks_first_valid_fence():
    text = "```\nnot json at all\n```\n```json\n{\"x\": 2}\n```"
    assert json.loads(extract_json(text)) == {"x": 2}


@pytest.mark.parametrize("text", ["", "just words", "{not: valid", "{\"a\": [1,2,3", "[[[unbalanced"])
def test_extract_json_returns_none_on_garbage(text):
    assert extract_json(text) is None


def test_deeply_nested_does_not_raise():
    # Contract is "never raises" (graceful degradation). On Py3.12 the parser may
    # accept deeper nesting than 3.9 without RecursionError; we assert no-raise and
    # that a balanced-but-pathological input yields a value or None — never a crash.
    try_parse_json("[" * 5000)            # must not raise
    extract_json("[" * 5000)              # must not raise
    out = try_parse_json("[" * 5000 + "]" * 5000)  # must not raise
    assert out is None or isinstance(out, list)


def test_large_unbalanced_input_is_time_bounded():
    import time

    big = "{" * 60000  # would be O(n^2) (~minutes) without the early-stop + scan cap
    start = time.perf_counter()
    assert extract_json(big) is None
    assert time.perf_counter() - start < 2.0
    # the bound must not break recovery of a real object after junk brackets
    assert json.loads(extract_json('[junk] {"k": 1}')) == {"k": 1}


def test_try_parse_json_basic():
    assert try_parse_json('{"a": 1}') == {"a": 1}
    assert try_parse_json("[1, 2]") == [1, 2]
    assert try_parse_json("nonsense") is None
    assert try_parse_json("") is None
