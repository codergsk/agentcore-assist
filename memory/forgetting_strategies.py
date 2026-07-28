"""
Forgetting strategies for agentic memory systems.

Each strategy exposes a single method:
    select_for_removal(items, n) -> list[MemoryItem]

which returns up to `n` items that should be evicted/down-weighted.
"""
from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod
from typing import Sequence

from .memory_item import MemoryItem


class ForgettingStrategy(ABC):
    """Base class for all forgetting strategies."""

    @abstractmethod
    def select_for_removal(
        self, items: Sequence[MemoryItem], n: int = 1
    ) -> list[MemoryItem]:
        """Return up to n items ranked for removal (most forgettable first)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# 1. Ebbinghaus Decay — forgetting curve: R = e^(-t/S)
# ---------------------------------------------------------------------------

class EbbinghausForgetting(ForgettingStrategy):
    """
    Selects items with the lowest current retention score.

    The retention fraction R = exp(-t / S) where
      t = time since last access (days)
      S = memory stability, boosted on each retrieval (spacing effect)

    Items with R below `threshold` are candidates for removal.
    """

    def __init__(self, threshold: float = 0.2):
        self.threshold = threshold

    def select_for_removal(
        self, items: Sequence[MemoryItem], n: int = 1
    ) -> list[MemoryItem]:
        now = time.time()
        scored = sorted(items, key=lambda m: m.retention(now))
        candidates = [m for m in scored if m.retention(now) < self.threshold]
        return candidates[:n] if candidates else scored[:n]


# ---------------------------------------------------------------------------
# 2. LRU (Least Recently Used)
# ---------------------------------------------------------------------------

class LRUForgetting(ForgettingStrategy):
    """
    Classic cache eviction: remove the item that was accessed least recently.
    Ignores salience; purely time-of-last-access based.
    """

    def select_for_removal(
        self, items: Sequence[MemoryItem], n: int = 1
    ) -> list[MemoryItem]:
        sorted_items = sorted(items, key=lambda m: m.last_accessed_at)
        return sorted_items[:n]


# ---------------------------------------------------------------------------
# 3. Salience-weighted forgetting
# ---------------------------------------------------------------------------

class SalienceForgetting(ForgettingStrategy):
    """
    Combines salience and recency into a single eviction score.

    score = salience * retention(t)

    The item with the lowest score is most forgettable: it is both
    unimportant and stale.
    """

    def select_for_removal(
        self, items: Sequence[MemoryItem], n: int = 1
    ) -> list[MemoryItem]:
        now = time.time()

        def score(m: MemoryItem) -> float:
            return m.salience * m.retention(now)

        return sorted(items, key=score)[:n]


# ---------------------------------------------------------------------------
# 4. Capacity-based (fixed-size store — evict to make room)
# ---------------------------------------------------------------------------

class CapacityForgetting(ForgettingStrategy):
    """
    Hard capacity limit.  When the store exceeds `max_items`, evict the
    lowest-scoring items according to a combined salience × retention score.
    This mirrors what production vector-store systems do with a fixed index.
    """

    def __init__(self, max_items: int = 100):
        if max_items < 1:
            raise ValueError("max_items must be >= 1")
        self.max_items = max_items

    def items_to_evict(self, current_size: int) -> int:
        return max(0, current_size - self.max_items)

    def select_for_removal(
        self, items: Sequence[MemoryItem], n: int = 1
    ) -> list[MemoryItem]:
        now = time.time()

        def score(m: MemoryItem) -> float:
            return m.salience * m.retention(now)

        return sorted(items, key=score)[:n]
