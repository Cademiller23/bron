"""The self-improving scorer — the second-order loop (the airtight-recursion answer).

The primary scorer (B1) is deterministic math and stays that way. This meta-loop
tunes only B1's ``ObjectiveWeights`` when they stop *predicting true optimality*:

1. **Calibrate** — rank the seen solutions by the scorer's goodness (a function of
   the weights) and by their TRUE goodness (distance-to-optimum from B1's OR-Tools
   oracle / known optima), and take the Spearman rank correlation. High = the
   weights rank solutions like the truth; degraded = they mis-weight the objectives.
2. **Re-tune** — when correlation degrades, gradient-free search over the
   ``ObjectiveWeights`` simplex for the weights that best correlate with the true
   optimum, adopt them, and bump ``scorer_version`` (B1 stamps every score with it).
   Bounded in frequency to avoid thrashing/overfitting.

**The firewall (non-negotiable):** the meta-loop is anchored to verifiable ground
truth (the oracle), NEVER an LLM. The number the system optimizes stays
deterministic; only the *weights* move, toward agreement with the oracle. That is
what keeps the recursion legitimate — not "GPT grading GPT." The persisted
``scorer_versions`` history yields the second money-shot curve: the scorer's
predictive validity *improving* — the system getting better at knowing what
"better" means.
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple

from darwin.problem.schemas import ObjectiveWeights

logger = logging.getLogger("darwin.observability.self_improving_scorer")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class CalibrationSample:
    """One seen solution: its NORMALIZED objective components (each ~[0,1], lower
    is better) plus its TRUE goodness from the oracle (higher is better — e.g.
    ``-distance_to_optimum``). The components feed the scorer's prediction; the
    truth is the oracle anchor the weights are tuned against."""

    cost: float
    lead_time: float
    risk: float
    true_quality: float


def predicted_goodness(weights: ObjectiveWeights, s: CalibrationSample) -> float:
    """The scorer's goodness for a sample under ``weights`` — the negative weighted
    objective (lower objective ⇒ better ⇒ higher goodness)."""
    return -(weights.cost_weight * s.cost + weights.lead_time_weight * s.lead_time
             + weights.risk_weight * s.risk)


# ---------------------------------------------------------------------------
# Spearman rank correlation (pure, ties handled via average ranks)
# ---------------------------------------------------------------------------
def _average_ranks(xs: Sequence[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank for the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman's rho ∈ [-1, 1]; 0.0 for degenerate input (too few points, no spread)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((r - mx) ** 2 for r in rx)
    vy = sum((r - my) ** 2 for r in ry)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    rho = cov / math.sqrt(vx * vy)
    return max(-1.0, min(1.0, rho))


def correlation(samples: Sequence[CalibrationSample], weights: ObjectiveWeights) -> float:
    """How well ``weights`` rank the samples vs the oracle truth (Spearman rho)."""
    if len(samples) < 2:
        return 0.0
    predicted = [predicted_goodness(weights, s) for s in samples]
    true = [s.true_quality for s in samples]
    return spearman(predicted, true)


# ---------------------------------------------------------------------------
# Gradient-free re-tune over the ObjectiveWeights simplex
# ---------------------------------------------------------------------------
def _simplex_grid(steps: int) -> List[Tuple[float, float, float]]:
    pts: List[Tuple[float, float, float]] = []
    for a in range(steps + 1):
        for b in range(steps + 1 - a):
            c = steps - a - b
            pts.append((a / steps, b / steps, c / steps))
    return pts


def retune(samples: Sequence[CalibrationSample], *, grid_steps: int = 8) -> Tuple[ObjectiveWeights, float]:
    """Search the weight simplex (cost+lead+risk normalized) for the weights that
    maximize correlation with the oracle truth. Returns (best_weights, best_corr).
    Deterministic: the cost_only baseline is retained on ties (strict ``>``
    replacement), and among grid points the first in iteration order wins (the grid
    is enumerated risk-weighted first)."""
    best_w = ObjectiveWeights.cost_only()
    best_c = correlation(samples, best_w)
    for wc, wl, wr in _simplex_grid(grid_steps):
        if wc + wl + wr <= 0:
            continue
        w = ObjectiveWeights(cost_weight=wc, lead_time_weight=wl, risk_weight=wr)
        c = correlation(samples, w)
        if c > best_c:
            best_c, best_w = c, w
    return best_w, best_c


def bump_version(version: str) -> str:
    """``"1.0.0"`` -> ``"1.0.1"`` (increment the patch component)."""
    parts = version.split(".")
    try:
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)
    except (ValueError, IndexError):
        return version + ".1"


@dataclass(frozen=True)
class RetuneResult:
    retuned: bool
    weights: ObjectiveWeights
    correlation_before: float
    correlation_after: float
    scorer_version: str
    reason: str


class SelfImprovingScorer:
    """The bounded second-order loop. Holds the current ``ObjectiveWeights`` +
    ``scorer_version``; ``maybe_retune`` calibrates and, only when degraded, tunes."""

    def __init__(
        self, weights: Optional[ObjectiveWeights] = None, *, store: Any = None, emitter: Any = None,
        scorer_version: str = "1.0.0", retune_threshold: float = 0.8, min_improvement: float = 0.02,
        min_samples: int = 8, grid_steps: int = 8, max_retunes: Optional[int] = None,
    ) -> None:
        self.weights = weights or ObjectiveWeights.cost_only()
        self.scorer_version = scorer_version
        self._store = store
        self._emitter = emitter
        self._retune_threshold = retune_threshold
        self._min_improvement = min_improvement
        self._min_samples = min_samples
        self._grid_steps = grid_steps
        self._max_retunes = max_retunes
        self.retune_count = 0

    def calibrate(self, samples: Sequence[CalibrationSample]) -> float:
        """The current weights' predictive validity (Spearman vs the oracle)."""
        return correlation(samples, self.weights)

    async def maybe_retune(self, samples: Sequence[CalibrationSample]) -> RetuneResult:
        """Calibrate; if the weights have stopped predicting true optimality and a
        better-correlated set exists, adopt it, bump the version, persist, and emit
        SCORER_RETUNED. Bounded: only fires when degraded and under ``max_retunes``."""
        before = self.calibrate(samples)
        if len(samples) < self._min_samples:
            return self._noop(before, "insufficient samples")
        if before >= self._retune_threshold:
            return self._noop(before, "well-calibrated")
        if self._max_retunes is not None and self.retune_count >= self._max_retunes:
            return self._noop(before, "retune budget exhausted")

        best_w, best_c = retune(samples, grid_steps=self._grid_steps)
        if best_c <= before + self._min_improvement:
            return self._noop(before, "no materially better weights found")

        # adopt
        self.weights = best_w
        self.scorer_version = bump_version(self.scorer_version)
        self.retune_count += 1
        record = {
            "scorer_version": self.scorer_version,
            "weights": {"cost": best_w.cost_weight, "lead_time": best_w.lead_time_weight,
                        "risk": best_w.risk_weight},
            "correlation": best_c, "correlation_before": before, "num_samples": len(samples),
            "timestamp": _now_iso(), "reason": "correlation degraded; re-tuned to the oracle",
        }
        if self._store is not None:
            try:
                await self._store.save_scorer_version(record)
            except Exception as exc:  # noqa: BLE001 - persistence is best-effort, like the emit
                logger.warning("scorer_versions save failed (ignored): %s", exc)
        if self._emitter is not None:
            try:
                await self._emitter.emit("SCORER_RETUNED", record,
                                         description=f"scorer re-tuned -> {self.scorer_version} "
                                                     f"(correlation {before:.2f} -> {best_c:.2f})")
            except Exception as exc:  # noqa: BLE001 - narration must never break the loop
                logger.debug("SCORER_RETUNED emit failed (ignored): %s", exc)
        return RetuneResult(True, best_w, before, best_c, self.scorer_version,
                            "correlation degraded; re-tuned to the oracle")

    def _noop(self, before: float, reason: str) -> RetuneResult:
        return RetuneResult(False, self.weights, before, before, self.scorer_version, reason)
