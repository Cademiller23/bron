"""The candidate generator — K distinct, valid rearrangement candidates.

Programmatic (no LLM): sample the rearrangement operators (optionally biased by
``hints``) to produce K distinct, valid candidates. De-duplicates by structural
signature (a candidate identical to the baseline is excluded), validates each
(discarding/resampling invalid ones), and guarantees the agent set is unchanged.
"""

import random
from typing import Any, Dict, List, Optional

from darwin.agent.registry import default_registry
from darwin.constants import REARRANGE_K
from darwin.rearrange.operators import ALL_OPERATORS, CandidateRearrangement, signature
from darwin.team.genome import TeamGenome


def generate_candidates(
    genome: TeamGenome,
    k: int = REARRANGE_K,
    *,
    rng: Optional[random.Random] = None,
    hints: Optional[Dict[str, float]] = None,
    registry: Any = None,
    max_tries: Optional[int] = None,
) -> List[CandidateRearrangement]:
    rng = rng or random.Random()
    registry = registry or default_registry()
    operators = list(ALL_OPERATORS)
    weights = [max(0.01, (hints or {}).get(op.__name__, 1.0)) for op in operators]

    candidates: List[CandidateRearrangement] = []
    seen = {signature(genome)}  # never return a candidate identical to the baseline
    tries = 0
    budget = max_tries if max_tries is not None else k * 20

    while len(candidates) < k and tries < budget:
        tries += 1
        op = rng.choices(operators, weights=weights, k=1)[0]
        candidate = op(genome, rng, registry)
        if candidate is None or candidate.signature in seen:
            continue
        seen.add(candidate.signature)
        candidates.append(candidate)

    return candidates
