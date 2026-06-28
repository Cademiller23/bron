"""Embeddings for corpus semantic search.

The corpus indexes each agent's ``role_description`` and queries with the
``GapDescription.capability_needed``; the SAME embedder must do both (consistency
is required for meaningful similarity).

``KeywordEmbedder`` is a deterministic, dependency-free bag-of-keywords embedder
over a supply-chain vocabulary — it gives real, testable cosine similarity
without a network. For production semantic quality it is a config swap for
``VoyageEmbedder`` (MongoDB's native Atlas Vector Search option) or a Gemini
embedding model; the corpus is agnostic to which embedder it is given.
"""

import math
import re
from typing import List, Protocol

# A compact supply-chain vocabulary. Synonyms map to the same dimension so that
# semantically-related descriptions land near each other.
_VOCAB = {
    "cost": "cost", "cheap": "cost", "cheaper": "cost", "minimize": "cost", "minimization": "cost",
    "price": "cost", "expensive": "cost", "budget": "cost",
    "risk": "risk", "disruption": "risk", "resilience": "risk", "resilient": "risk", "diversify": "risk",
    "diversifying": "risk", "sourcing": "risk", "failure": "risk", "robust": "risk", "single": "risk",
    "demand": "demand", "unmet": "demand", "coverage": "demand", "customer": "demand", "satisfy": "demand",
    "feasible": "feasible", "feasibility": "feasible", "constraint": "feasible", "valid": "feasible",
    "capacity": "capacity", "rebalance": "capacity", "overflow": "capacity", "limit": "capacity",
    "throughput": "capacity",
    "lead": "lead", "time": "lead", "expedite": "lead", "slow": "lead", "delivery": "lead", "fast": "lead",
    "route": "routing", "routing": "routing", "allocation": "routing", "flow": "routing", "ship": "routing",
    "arbitrate": "arbiter", "arbitrator": "arbiter", "synthesize": "arbiter", "merge": "arbiter",
    "check": "check", "checker": "check", "audit": "check", "auditor": "check", "verify": "check",
}
_DIMS = sorted(set(_VOCAB.values()))
_DIM_INDEX = {d: i for i, d in enumerate(_DIMS)}
_TOKEN_RE = re.compile(r"[a-z]+")


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


class KeywordEmbedder:
    """Deterministic bag-of-keywords embedder (offline, real cosine similarity)."""

    dimension = len(_DIMS)

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * len(_DIMS)
        for tok in _TOKEN_RE.findall((text or "").lower()):
            dim = _VOCAB.get(tok)
            if dim is not None:
                vec[_DIM_INDEX[dim]] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        return [x / norm for x in vec] if norm > 0 else vec


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class VoyageEmbedder:  # pragma: no cover - requires VOYAGE_API_KEY / network
    """Production embedder via Voyage AI (Atlas Vector Search's native option)."""

    def __init__(self, model: str = "voyage-3", api_key_env: str = "VOYAGE_API_KEY") -> None:
        self.model = model
        self.api_key_env = api_key_env
        self._client = None

    def embed(self, text: str) -> List[float]:
        import os

        import voyageai

        if self._client is None:
            self._client = voyageai.Client(api_key=os.environ.get(self.api_key_env))
        return self._client.embed([text], model=self.model, input_type="document").embeddings[0]
