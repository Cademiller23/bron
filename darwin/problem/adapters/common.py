"""Shared helpers for source adapters."""

import json
import os
from typing import Any, Dict


def read_text(raw: Any) -> str:
    """Return raw text from a path, an open file, or a string blob."""
    if hasattr(raw, "read"):
        return raw.read()
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8")
    if isinstance(raw, str) and os.path.exists(raw):
        with open(raw, "r", encoding="utf-8") as fh:
            return fh.read()
    if isinstance(raw, os.PathLike):
        with open(raw, "r", encoding="utf-8") as fh:
            return fh.read()
    if isinstance(raw, str):
        return raw  # already-inlined text blob
    raise TypeError(f"cannot read raw input of type {type(raw)!r}")


def as_dict(raw: Any) -> Dict[str, Any]:
    """Return a dict from a path to JSON, a JSON string, or an already-parsed dict."""
    if isinstance(raw, dict):
        return raw
    text = read_text(raw)
    return json.loads(text)
