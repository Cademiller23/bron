"""Persistence & the atomic mutation primitive — optimistic locking.

Genomes are mutated **in place** in MongoDB. Every "evolve the genome" path is a
single atomic ``find_one_and_update({_id, version}, {$set, $inc:{version:1},
$push:{history}})``, so two concurrent writers holding the same ``expected_version``
can never both win — the version filter matches for exactly one; the loser sees
``None`` and reloads/retries. This optimistic concurrency control gives the
no-lost-updates / no-torn-writes guarantee immutability used to, while mutating in
place, and keeps the lineage (``history``) atomic with the version bump.

The store takes an injected collection (a motor ``AsyncIOMotorCollection`` or any
object exposing the same async ``insert_one`` / ``find_one`` /
``find_one_and_update`` surface), so it is fully unit-testable without a server.
"""

import asyncio
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from pymongo import ReturnDocument

from darwin.constants import MUTATE_MAX_ATTEMPTS
from darwin.team.genome import MutationRecord, TeamGenome

# derive_fn(current_genome) -> (set_ops, optional MutationRecord)
DeriveFn = Callable[[TeamGenome], Tuple[Dict[str, Any], Optional[MutationRecord]]]


class OptimisticLockError(RuntimeError):
    """Raised by ``retry_mutate`` when contention can't be resolved in budget."""


class GenomeStore:
    COLLECTION = "genomes"

    def __init__(self, collection: Any, registry: Any = None) -> None:
        self._col = collection
        # The registry embedded specs are re-validated against on load. Defaults to
        # the process-wide singleton (which B7 extends with the hosted fleet); pass
        # an explicit one for isolated / custom-fleet deployments.
        self._registry = registry

    @classmethod
    def from_uri(cls, uri: str, db_name: str = "darwin", registry: Any = None) -> "GenomeStore":  # pragma: no cover
        from motor.motor_asyncio import AsyncIOMotorClient

        return cls(AsyncIOMotorClient(uri)[db_name][cls.COLLECTION], registry=registry)

    # -- (de)serialization: the model's genome_id maps to Mongo's _id ---------
    @staticmethod
    def _to_doc(genome: TeamGenome) -> Dict[str, Any]:
        doc = genome.model_dump(mode="json")  # JSON-safe (enums -> strings) for BSON
        doc["_id"] = doc.pop("genome_id")
        return doc

    @staticmethod
    def _from_doc(doc: Dict[str, Any], registry: Any = None) -> TeamGenome:
        data = dict(doc)
        data["genome_id"] = data.pop("_id")
        if registry is not None:  # propagates to the nested AgentSpec validators
            return TeamGenome.model_validate(data, context={"registry": registry})
        return TeamGenome.model_validate(data)

    async def ensure_indexes(self) -> None:  # pragma: no cover - requires a server
        await self._col.create_index("instance_id")
        await self._col.create_index("status")
        await self._col.create_index("updated_at")

    async def save_new(self, genome: TeamGenome) -> TeamGenome:
        await self._col.insert_one(self._to_doc(genome))
        return genome

    async def load(self, genome_id: str) -> Optional[TeamGenome]:
        doc = await self._col.find_one({"_id": genome_id})
        return self._from_doc(doc, self._registry) if doc else None

    async def mutate(
        self,
        genome_id: str,
        expected_version: int,
        set_ops: Dict[str, Any],
        mutation_record: Optional[MutationRecord] = None,
    ) -> Optional[TeamGenome]:
        """The atomic optimistic-locking primitive.

        Returns the after-document rebuilt into a fresh ``TeamGenome`` on success,
        or ``None`` if a concurrent writer already advanced the version (the
        caller must reload and retry).
        """
        update: Dict[str, Any] = {
            "$set": {**set_ops, "updated_at": _now_iso()},
            "$inc": {"version": 1},
        }
        if mutation_record is not None:
            update["$push"] = {"history": mutation_record.model_dump(mode="json")}
        doc = await self._col.find_one_and_update(
            {"_id": genome_id, "version": expected_version},
            update,
            return_document=ReturnDocument.AFTER,
        )
        return self._from_doc(doc, self._registry) if doc else None

    async def retry_mutate(
        self,
        genome_id: str,
        derive_fn: DeriveFn,
        max_attempts: int = MUTATE_MAX_ATTEMPTS,
        sleep: Optional[Callable[[float], Awaitable[None]]] = None,
    ) -> TeamGenome:
        """Load → derive update → mutate; on a version conflict, reload and retry
        up to ``max_attempts`` with a small backoff. Conflict-safe by construction."""
        sleeper = sleep or asyncio.sleep
        last: Optional[TeamGenome] = None
        for attempt in range(max_attempts):
            current = await self.load(genome_id)
            if current is None:
                raise KeyError(f"genome {genome_id!r} not found")
            set_ops, record = derive_fn(current)
            result = await self.mutate(genome_id, current.version, set_ops, record)
            if result is not None:
                return result
            last = current
            await sleeper(0.01 * (attempt + 1))
        raise OptimisticLockError(
            f"could not mutate {genome_id!r} after {max_attempts} attempts "
            f"(last seen version {last.version if last else '?'})"
        )


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
