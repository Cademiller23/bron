"""The top-level loader: raw data → canonical, validated ``ProblemInstance``.

``load_instance(source, raw_path_or_blob)`` dispatches to the right adapter,
which parses the source's native format, maps it onto the canonical schema, and
attaches provenance + any labelled optimum. The schema's validators then run
automatically on construction — so a malformed source file is caught *at load*.

A small in-memory cache makes repeated demo runs instant and deterministic:
loading the same file twice returns the *same object* (identity-equal).
"""

import os
from typing import Any, Callable, Dict, Hashable

from darwin.problem.adapters import cvrplib, generated, industryor, mamo
from darwin.problem.schemas import ProblemInstance

_ADAPTERS: Dict[str, Callable[[Any], ProblemInstance]] = {
    "industryor": industryor.parse,
    "mamo": mamo.parse,
    "cvrplib": cvrplib.parse,
    "generated": generated.parse,
}

# instance cache keyed by (source, cache_key) -> ProblemInstance
_CACHE: Dict[Hashable, ProblemInstance] = {}


def available_sources() -> "list[str]":
    return sorted(_ADAPTERS)


def clear_cache() -> None:
    _CACHE.clear()


def _cache_key(source: str, raw: Any) -> Hashable:
    if isinstance(raw, str) and os.path.exists(raw):
        return (source, "path", os.path.abspath(raw))
    if isinstance(raw, os.PathLike):
        return (source, "path", os.path.abspath(os.fspath(raw)))
    if isinstance(raw, ProblemInstance):
        return (source, "instance", raw.instance_id)
    if isinstance(raw, (str, bytes)):
        return (source, "blob", hash(raw))
    return None  # unhashable / un-keyable input => skip caching


def load_instance(source: str, raw_path_or_blob: Any, use_cache: bool = True) -> ProblemInstance:
    """Load and canonicalize one problem instance from ``source``.

    ``source`` is one of :func:`available_sources`. ``raw_path_or_blob`` may be a
    file path, a JSON/text blob, a parsed dict, or (for ``generated``) a
    :class:`ProblemInstance`.
    """
    key = source.lower()
    if key not in _ADAPTERS:
        raise ValueError(
            f"unknown source {source!r}; expected one of {available_sources()}"
        )

    cache_key = _cache_key(key, raw_path_or_blob) if use_cache else None
    if cache_key is not None and cache_key in _CACHE:
        return _CACHE[cache_key]

    instance = _ADAPTERS[key](raw_path_or_blob)

    if cache_key is not None:
        _CACHE[cache_key] = instance
    return instance
