"""
MemoryStore: a typed, strategy-driven in-memory store.

Supports add / retrieve / forget / consolidate operations.
Thread-safe via a simple RLock.
"""
from __future__ import annotations

import threading
import time
from typing import Callable, Sequence

from .memory_item import MemoryItem, MemoryType
from .forgetting_strategies import ForgettingStrategy, SalienceForgetting


class MemoryStore:
    def __init__(
        self,
        strategy: ForgettingStrategy | None = None,
        max_size: int = 1000,
    ):
        self._items: dict[str, MemoryItem] = {}
        self.strategy = strategy or SalienceForgetting()
        self.max_size = max_size
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, item: MemoryItem) -> MemoryItem:
        with self._lock:
            self._items[item.id] = item
            self._enforce_capacity()
        return item

    def update(self, item_id: str, **kwargs) -> MemoryItem | None:
        with self._lock:
            item = self._items.get(item_id)
            if item is None:
                return None
            for k, v in kwargs.items():
                if hasattr(item, k):
                    setattr(item, k, v)
            return item

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, item_id: str, record_access: bool = True) -> MemoryItem | None:
        with self._lock:
            item = self._items.get(item_id)
            if item and record_access:
                item.touch()
            return item

    def search(
        self,
        query_fn: Callable[[MemoryItem], bool],
        memory_type: MemoryType | None = None,
        limit: int = 10,
        record_access: bool = True,
    ) -> list[MemoryItem]:
        with self._lock:
            results = [
                m for m in self._items.values()
                if (memory_type is None or m.memory_type == memory_type)
                and query_fn(m)
            ]
            results.sort(key=lambda m: m.salience, reverse=True)
            if record_access:
                for m in results[:limit]:
                    m.touch()
            return results[:limit]

    def all_items(self) -> list[MemoryItem]:
        with self._lock:
            return list(self._items.values())

    @property
    def size(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------
    # Forgetting
    # ------------------------------------------------------------------

    def forget(self, n: int = 1) -> list[MemoryItem]:
        """Evict n items selected by the active strategy."""
        with self._lock:
            items = list(self._items.values())
            to_remove = self.strategy.select_for_removal(items, n)
            for m in to_remove:
                self._items.pop(m.id, None)
            return to_remove

    def forget_item(self, item_id: str) -> MemoryItem | None:
        with self._lock:
            return self._items.pop(item_id, None)

    def prune_below_retention(self, threshold: float = 0.1) -> list[MemoryItem]:
        """Remove all items whose Ebbinghaus retention falls below threshold."""
        now = time.time()
        with self._lock:
            to_remove = [
                m for m in self._items.values()
                if m.retention(now) < threshold
            ]
            for m in to_remove:
                self._items.pop(m.id, None)
            return to_remove

    # ------------------------------------------------------------------
    # Consolidation (working → long-term promotion)
    # ------------------------------------------------------------------

    def consolidate(self, salience_threshold: float = 0.5) -> list[MemoryItem]:
        """
        Promote high-salience WORKING memories to EPISODIC,
        and discard low-salience WORKING memories.
        Returns the promoted items.
        """
        promoted = []
        discarded = []
        with self._lock:
            for m in list(self._items.values()):
                if m.memory_type != MemoryType.WORKING:
                    continue
                if m.salience >= salience_threshold:
                    m.memory_type = MemoryType.EPISODIC
                    promoted.append(m)
                else:
                    discarded.append(m)
            for m in discarded:
                self._items.pop(m.id, None)
        return promoted

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enforce_capacity(self) -> None:
        while len(self._items) > self.max_size:
            items = list(self._items.values())
            to_remove = self.strategy.select_for_removal(items, n=1)
            for m in to_remove:
                self._items.pop(m.id, None)

    def stats(self) -> dict:
        now = time.time()
        items = list(self._items.values())
        if not items:
            return {"size": 0}
        retentions = [m.retention(now) for m in items]
        return {
            "size": len(items),
            "avg_retention": sum(retentions) / len(retentions),
            "min_retention": min(retentions),
            "max_retention": max(retentions),
            "avg_salience": sum(m.salience for m in items) / len(items),
            "type_counts": {
                t.value: sum(1 for m in items if m.memory_type == t)
                for t in MemoryType
            },
        }
