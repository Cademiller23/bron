"""Robust JSON extraction — the fallback for non-schema providers and the
occasional stray-prose response (§6 step 5).

Strips markdown fences and locates the outermost balanced JSON object/array,
respecting string literals and escapes so braces inside strings don't confuse
the scan.
"""

import json
import re
from typing import Any, Optional

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Bound the work / recursion exposure on pathological (e.g. runaway) model output.
_MAX_PARSE_CHARS = 200_000
# Cap how many opening-bracket start positions the balanced scan will try, so a
# pathological all-brackets payload can't drive O(n^2) work (32 starts x O(n) =
# O(n)). The real JSON object in any sane (even prose-wrapped) response sits
# within the first few openers.
_MAX_SCAN_STARTS = 64


def try_parse_json(text: str) -> Optional[Any]:
    """``json.loads`` the whole text, returning ``None`` on failure.

    Catches ``RecursionError`` too: deeply-nested JSON makes ``json.loads``
    recurse past the interpreter limit, and a runaway/adversarial model can emit
    exactly that — it must degrade to ``None``, never escape.
    """
    if not text:
        return None
    try:
        return json.loads(text)
    except (ValueError, TypeError, RecursionError):
        return None


def extract_json(text: str) -> Optional[str]:
    """Return a balanced JSON object/array substring that parses, or ``None``.

    Tries, in order: the whole string; the contents of each ``` fence (direct
    parse or a balanced scan within the fence); then a balanced bracket scan over
    the original text, attempting every opening bracket (so junk brackets in
    prose before the real object don't defeat it).
    """
    if not text:
        return None
    text = text.strip()[:_MAX_PARSE_CHARS]

    if try_parse_json(text) is not None:
        return text

    for match in _FENCE_RE.finditer(text):
        inner = match.group(1).strip()
        if try_parse_json(inner) is not None:
            return inner
        span = _balanced_span(inner)
        if span is not None:
            return span

    return _balanced_span(text)


def _balanced_span(text: str) -> Optional[str]:
    """Try a balanced-bracket scan from each opening bracket; return the first
    span that parses as JSON.

    Bounded by ``_MAX_SCAN_STARTS`` opening positions so a pathological
    all-brackets payload can't drive O(n^2) work — while still finding valid JSON
    that follows junk/unbalanced brackets in prose.
    """
    starts = 0
    for start, ch in enumerate(text):
        if ch not in "{[":
            continue
        span = _scan_from(text, start, ch)
        if span is not None:
            return span
        starts += 1
        if starts >= _MAX_SCAN_STARTS:
            return None
    return None


def _scan_from(text: str, start: int, opener: str) -> Optional[str]:
    """Return the balanced span from ``start`` if it parses as JSON, else
    ``None`` (whether it balanced-but-wasn't-JSON or never closed)."""
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                return candidate if try_parse_json(candidate) is not None else None
    return None
