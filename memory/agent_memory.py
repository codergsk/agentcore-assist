"""
AgentMemorySystem: high-level facade used by an agent at runtime.

Provides:
  - remember()      add a new memory
  - recall()        query memories
  - reflect()       trigger consolidation + pruning
  - forget_cycle()  run the active forgetting strategy
  - snapshot()      full stats across all tiers
"""
from __future__ import annotations

import time
from typing import Any

from .memory_item import MemoryItem, MemoryType
from .memory_store import MemoryStore
from .forgetting_strategies import (
    ForgettingStrategy,
    EbbinghausForgetting,
    LRUForgetting,
    SalienceForgetting,
    CapacityForgetting,
)


class AgentMemorySystem:
    """
    Three-tier agentic memory:
      working   – short-lived scratch-pad (small capacity)
      episodic  – events & experiences
      semantic  – facts & general knowledge

    Each tier can use a different forgetting strategy.
    """

    def __init__(
        self,
        working_capacity: int = 10,
        episodic_capacity: int = 200,
        semantic_capacity: int = 500,
        working_strategy: ForgettingStrategy | None = None,
        episodic_strategy: ForgettingStrategy | None = None,
        semantic_strategy: ForgettingStrategy | None = None,
    ):
        self.working = MemoryStore(
            strategy=working_strategy or LRUForgetting(),
            max_size=working_capacity,
        )
        self.episodic = MemoryStore(
            strategy=episodic_strategy or EbbinghausForgetting(threshold=0.2),
            max_size=episodic_capacity,
        )
        self.semantic = MemoryStore(
            strategy=semantic_strategy or SalienceForgetting(),
            max_size=semantic_capacity,
        )

        self._tier_map: dict[MemoryType, MemoryStore] = {
            MemoryType.WORKING: self.working,
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.PROCEDURAL: self.semantic,  # procedural lives in semantic
        }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def remember(
        self,
        content: Any,
        memory_type: MemoryType = MemoryType.WORKING,
        salience: float = 0.5,
        tags: list[str] | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            content=content,
            memory_type=memory_type,
            salience=salience,
            tags=tags or [],
        )
        store = self._tier_map[memory_type]
        return store.add(item)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        memory_type: MemoryType | None = None,
        limit: int = 5,
    ) -> list[MemoryItem]:
        """
        Simple substring search.  In production this would be a vector search.
        """
        q = query.lower()

        def match(m: MemoryItem) -> bool:
            return q in str(m.content).lower() or any(q in t.lower() for t in m.tags)

        stores = (
            [self._tier_map[memory_type]]
            if memory_type
            else [self.working, self.episodic, self.semantic]
        )

        results: list[MemoryItem] = []
        for store in stores:
            results.extend(store.search(match, limit=limit))

        results.sort(key=lambda m: m.salience, reverse=True)
        return results[:limit]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def forget_cycle(self, n_per_tier: int = 1) -> dict[str, list[MemoryItem]]:
        """Run each tier's forgetting strategy, removing n_per_tier items each."""
        return {
            "working": self.working.forget(n_per_tier),
            "episodic": self.episodic.forget(n_per_tier),
            "semantic": self.semantic.forget(n_per_tier),
        }

    def reflect(
        self,
        consolidation_threshold: float = 0.5,
        retention_prune_threshold: float = 0.1,
    ) -> dict:
        """
        Consolidation + pruning pass (run periodically, e.g. end of conversation).

        1. Promote high-salience WORKING memories → EPISODIC.
        2. Discard low-salience WORKING memories.
        3. Prune items from all tiers with very low Ebbinghaus retention.
        """
        promoted = self.working.consolidate(consolidation_threshold)
        # Move promoted items into the episodic store
        for item in promoted:
            self.episodic.add(item)

        pruned = {
            "working": self.working.prune_below_retention(retention_prune_threshold),
            "episodic": self.episodic.prune_below_retention(retention_prune_threshold),
            "semantic": self.semantic.prune_below_retention(retention_prune_threshold),
        }

        return {
            "consolidated": len(promoted),
            "pruned": {k: len(v) for k, v in pruned.items()},
        }

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "working": self.working.stats(),
            "episodic": self.episodic.stats(),
            "semantic": self.semantic.stats(),
        }
