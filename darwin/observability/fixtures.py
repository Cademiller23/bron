"""B8 test doubles — an in-memory async collection for run_events / scorer_versions."""

import copy
from typing import Any, Dict, List, Optional


def _sort_key(v: Any):
    """Type-safe ordering that mimics Mongo's cross-type sort (never raises on a
    mixed-type column, e.g. a corrupt non-int sequence_number alongside ints)."""
    if v is None:
        return (0, 0)
    if isinstance(v, bool):
        return (1, int(v))
    if isinstance(v, (int, float)):
        return (1, v)
    if isinstance(v, str):
        return (2, v)
    return (3, str(v))


def _cmp(dv: Any, ov: Any, op: str) -> bool:
    """A single range comparison that mirrors Mongo's type-bracketing: a value of
    an incomparable type (e.g. a corrupt string ``sequence_number`` against an int
    filter) simply does NOT match, rather than raising a TypeError."""
    if dv is None:
        return False  # null never matches a numeric range filter (Mongo-ish)
    try:
        if op == "$gt":
            return dv > ov
        if op == "$gte":
            return dv >= ov
        if op == "$lt":
            return dv < ov
        if op == "$lte":
            return dv <= ov
    except TypeError:  # incomparable types -> exclude (do not crash the query)
        return False
    return False


def _match(doc: Dict[str, Any], filt: Dict[str, Any]) -> bool:
    for k, v in filt.items():
        dv = doc.get(k)
        if isinstance(v, dict):  # operator form, e.g. {"$gt": 3}
            for op, ov in v.items():
                if not _cmp(dv, ov, op):
                    return False
        elif dv != v:
            return False
    return True


class _Cursor:
    def __init__(self, docs: List[Dict[str, Any]]) -> None:
        self._docs = docs
        self._sort: Optional[tuple] = None

    def sort(self, key: str, direction: int = 1) -> "_Cursor":
        self._sort = (key, direction)
        return self

    async def to_list(self, length: Optional[int] = None) -> List[Dict[str, Any]]:
        docs = list(self._docs)
        if self._sort:
            k, d = self._sort
            docs.sort(key=lambda x: _sort_key(x.get(k)), reverse=(d == -1))
        out = docs if length is None else docs[:length]
        return [copy.deepcopy(x) for x in out]


class FakeEventCollection:
    """Minimal async Mongo-collection stand-in: insert_one + find().sort().to_list()."""

    def __init__(self) -> None:
        self.docs: List[Dict[str, Any]] = []

    async def insert_one(self, doc: Dict[str, Any]) -> Any:
        self.docs.append(copy.deepcopy(doc))
        return type("R", (), {"inserted_id": doc.get("_id")})()

    def find(self, filt: Optional[Dict[str, Any]] = None) -> _Cursor:
        sel = [d for d in self.docs if _match(d, filt or {})]
        return _Cursor(sel)

    async def count_documents(self, filt: Optional[Dict[str, Any]] = None) -> int:
        return len([d for d in self.docs if _match(d, filt or {})])

    async def create_index(self, *args, **kwargs) -> None:  # pragma: no cover
        return None
