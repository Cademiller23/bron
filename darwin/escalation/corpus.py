"""The agent corpus — Darwin's genuine "gets better across problems" mechanism.

Every agent the Architect curates that proves useful is written back here with a
performance record and a semantic embedding. On later problems the system
searches this memory first and reuses a proven agent (cheap, fast) before
inventing a new one. Cold, the corpus is empty and the system always curates;
warm, it increasingly reuses — that compounding is the strongest MongoDB story.

Search prefers Atlas ``$vectorSearch`` and falls back to a brute-force cosine
scan (works at demo scale and when no vector index exists). Every operation is
wrapped so an Atlas/embedding failure degrades to empty / no-op — the escalator
simply proceeds to curation, never crashing the solve.
"""

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from darwin.constants import CORPUS_SEARCH_K, CORPUS_SIM_THRESHOLD
from darwin.escalation.embedding import Embedder, KeywordEmbedder, cosine_similarity
from darwin.escalation.schemas import CorpusEntry, GapDescription, ScoredCorpusEntry

logger = logging.getLogger("darwin.escalation.corpus")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentCorpus:
    COLLECTION = "agent_corpus"
    VECTOR_INDEX = "role_description_vector_index"

    def __init__(self, collection: Any, embedder: Optional[Embedder] = None, *, sim_threshold: float = CORPUS_SIM_THRESHOLD):
        self._col = collection
        self._embedder = embedder or KeywordEmbedder()
        self._sim_threshold = sim_threshold

    @classmethod
    def from_uri(cls, uri: str, db_name: str = "darwin", embedder: Optional[Embedder] = None) -> "AgentCorpus":  # pragma: no cover
        from motor.motor_asyncio import AsyncIOMotorClient

        return cls(AsyncIOMotorClient(uri)[db_name][cls.COLLECTION], embedder=embedder)

    # =======================================================================
    # Search (degrade to empty -> escalator falls through to curation)
    # =======================================================================
    async def search(
        self, gap: GapDescription, k: int = CORPUS_SEARCH_K, problem_class: Optional[str] = None
    ) -> List[ScoredCorpusEntry]:
        try:
            query_vec = self._embedder.embed(gap.capability_needed)
            docs = await self._fetch(query_vec, k)
        except Exception as exc:  # noqa: BLE001 - corpus failure must never crash the solve
            logger.warning("corpus search failed (degrading to empty): %s", exc)
            return []

        pc = problem_class or gap.problem_class
        scored: List[ScoredCorpusEntry] = []
        for doc in docs:
            # The ENTIRE per-row body is guarded: a malformed embedding, a
            # non-numeric stat, or a corrupt sub-document must skip just that row,
            # never poison the whole search (the math runs on raw Mongo data).
            try:
                emb = doc.get("role_description_embedding") or []
                sim = cosine_similarity(query_vec, emb)
                if sim < self._sim_threshold:
                    continue
                avg = float(doc.get("avg_fitness_contribution", 0.0) or 0.0)
                reused = int(doc.get("times_reused", 0) or 0)
                # favour BOTH relevance and a proven track record (clamp the reuse
                # count so a corrupt negative value can't blow up math.log)
                combined = sim * (1.0 + max(0.0, avg)) * math.log(max(0, reused) + 2)
                # a proven match for a compatible class ranks higher (not a hard
                # filter, so a strong cross-class match can still be reused)
                helped = doc.get("helped_problem_classes") or []
                if pc and helped and pc in helped:
                    combined *= 1.25
                # a non-finite score (e.g. inf/nan avg in a corrupt row — a valid
                # float that wouldn't raise) must not sort to the top and starve
                # legitimate agents: drop the row from ranking.
                if not math.isfinite(combined) or not math.isfinite(sim):
                    continue
                entry = self._to_entry(doc)
            except Exception:  # noqa: BLE001 - skip the bad row, keep the good matches
                continue
            scored.append(ScoredCorpusEntry(entry=entry, similarity=sim, combined_score=combined))

        scored.sort(key=lambda s: -s.combined_score)
        return scored[:k]

    async def _fetch(self, query_vec: List[float], k: int) -> List[dict]:
        # Prefer Atlas $vectorSearch; fall back to a brute-force scan.
        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.VECTOR_INDEX,
                        "path": "role_description_embedding",
                        "queryVector": query_vec,
                        "numCandidates": max(50, k * 10),
                        "limit": k,
                    }
                }
            ]
            cursor = self._col.aggregate(pipeline)
            return await cursor.to_list(length=k)
        except Exception:
            cursor = self._col.find({})
            return await cursor.to_list(length=None)

    # =======================================================================
    # Promote (write-back) + update_stats — the compounding
    # =======================================================================
    async def promote(
        self, agent_spec, fitness_contribution: float, problem_class: str, origin_instance_id: str
    ) -> bool:
        try:
            emb = self._embedder.embed(agent_spec.role_description)
            now = _now_iso()
            existing = await self._col.find_one({"role_name": agent_spec.role_name})
            if existing:
                n = int(existing.get("success_count", 0)) + int(existing.get("failure_count", 0))
                new_avg = (float(existing.get("avg_fitness_contribution", 0.0)) * n + fitness_contribution) / (n + 1)
                await self._col.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$set": {
                            "avg_fitness_contribution": new_avg, "last_used_at": now,
                            "role_description_embedding": emb,
                            "agent_spec": agent_spec.model_dump(mode="json"),
                            "role_description": agent_spec.role_description,
                        },
                        "$inc": {"times_reused": 1, "success_count": 1},
                        "$addToSet": {"helped_problem_classes": problem_class},
                    },
                )
            else:
                entry = CorpusEntry(
                    entry_id=uuid.uuid4().hex, agent_spec=agent_spec, role_name=agent_spec.role_name,
                    role_description=agent_spec.role_description, role_description_embedding=emb,
                    helped_problem_classes=[problem_class], avg_fitness_contribution=fitness_contribution,
                    times_reused=1, success_count=1, failure_count=0, created_at=now, last_used_at=now,
                    origin_instance_id=origin_instance_id,
                )
                doc = entry.model_dump(mode="json")
                doc["_id"] = doc.pop("entry_id")
                await self._col.insert_one(doc)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("corpus promote failed (degraded): %s", exc)
            return False

    async def update_stats(self, entry_id: str, fitness_contribution: float, succeeded: bool) -> bool:
        try:
            existing = await self._col.find_one({"_id": entry_id})
            if existing is None:
                return False
            n = int(existing.get("success_count", 0)) + int(existing.get("failure_count", 0))
            new_avg = (float(existing.get("avg_fitness_contribution", 0.0)) * n + fitness_contribution) / (n + 1)
            inc = {"times_reused": 1, "success_count": 1} if succeeded else {"failure_count": 1}
            await self._col.update_one(
                {"_id": entry_id},
                {"$set": {"avg_fitness_contribution": new_avg, "last_used_at": _now_iso()}, "$inc": inc},
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("corpus update_stats failed (degraded): %s", exc)
            return False

    @staticmethod
    def _to_entry(doc: dict) -> CorpusEntry:
        data = dict(doc)
        data["entry_id"] = data.pop("_id")
        return CorpusEntry.model_validate(data)
