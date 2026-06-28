"""Adapter: generated instances → canonical ``ProblemInstance``.

A trivial pass-through for instances the generator already produced in canonical
form. Accepts a ready :class:`ProblemInstance`, its ``model_dump()`` dict, or a
JSON blob/path of the same, and re-validates it through the schema.
"""

from typing import Any

from darwin.problem.adapters.common import as_dict
from darwin.problem.schemas import ProblemInstance


def parse(raw: Any) -> ProblemInstance:
    if isinstance(raw, ProblemInstance):
        # Re-validate to honour the "everything passes through the validators"
        # contract, even for already-canonical objects.
        return ProblemInstance(**raw.model_dump())
    return ProblemInstance(**as_dict(raw))
