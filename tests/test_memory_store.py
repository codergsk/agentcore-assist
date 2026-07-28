"""Tests for MemoryStore."""
import time
import pytest
from memory.memory_item import MemoryItem, MemoryType
from memory.memory_store import MemoryStore
from memory.forgetting_strategies import LRUForgetting, SalienceForgetting, ForgettingStrategy


def fresh_store(strategy: ForgettingStrategy | None = None, max_size: int = 100) -> MemoryStore:
    return MemoryStore(strategy=strategy or LRUForgetting(), max_size=max_size)


def make_item(content: str = "x", salience: float = 0.5, memory_type: MemoryType = MemoryType.EPISODIC) -> MemoryItem:
    return MemoryItem(content, memory_type=memory_type, salience=salience)


class TestAdd:
    def test_add_increases_size(self) -> None:
        store = fresh_store()
        store.add(make_item("a"))
        assert store.size == 1

    def test_add_returns_item(self) -> None:
        store = fresh_store()
        item = make_item("b")
        returned = store.add(item)
        assert returned is item

    def test_capacity_enforced(self) -> None:
        store = fresh_store(max_size=3)
        for i in range(5):
            store.add(make_item(f"item{i}"))
        assert store.size == 3


class TestGet:
    def test_get_returns_item(self) -> None:
        store = fresh_store()
        item = store.add(make_item("hello"))
        assert store.get(item.id) is item

    def test_get_missing_returns_none(self) -> None:
        store = fresh_store()
        assert store.get("nonexistent") is None

    def test_get_records_access(self) -> None:
        store = fresh_store()
        item = store.add(make_item("touch me"))
        count_before = item.access_count
        store.get(item.id)
        assert item.access_count == count_before + 1

    def test_get_no_record_access(self) -> None:
        store = fresh_store()
        item = store.add(make_item("no touch"))
        store.get(item.id, record_access=False)
        assert item.access_count == 0


class TestSearch:
    def test_search_by_content(self) -> None:
        store = fresh_store()
        store.add(make_item("AWS S3 pricing"))
        store.add(make_item("Lambda functions"))
        store.add(make_item("S3 lifecycle rules"))
        results = store.search(lambda m: "s3" in str(m.content).lower())
        assert len(results) == 2

    def test_search_respects_limit(self) -> None:
        store = fresh_store()
        for i in range(10):
            store.add(make_item(f"item{i}"))
        results = store.search(lambda m: True, limit=3)
        assert len(results) == 3

    def test_search_by_type(self) -> None:
        store = fresh_store()
        store.add(make_item("episodic", memory_type=MemoryType.EPISODIC))
        store.add(make_item("semantic", memory_type=MemoryType.SEMANTIC))
        results = store.search(lambda m: True, memory_type=MemoryType.SEMANTIC)
        assert all(m.memory_type == MemoryType.SEMANTIC for m in results)


class TestForget:
    def test_forget_removes_items(self) -> None:
        store = fresh_store()
        for i in range(5):
            store.add(make_item(f"item{i}"))
        removed = store.forget(n=2)
        assert len(removed) == 2
        assert store.size == 3

    def test_forget_item_by_id(self) -> None:
        store = fresh_store()
        item = store.add(make_item("to remove"))
        store.forget_item(item.id)
        assert store.get(item.id) is None

    def test_forget_item_missing_returns_none(self) -> None:
        store = fresh_store()
        assert store.forget_item("fake-id") is None

    def test_prune_below_retention(self) -> None:
        store = fresh_store()
        fresh = store.add(make_item("fresh"))
        stale = store.add(make_item("stale"))
        stale.last_accessed_at = time.time() - 999 * 86_400
        removed = store.prune_below_retention(threshold=0.5)
        assert stale in removed
        assert fresh not in removed


class TestConsolidate:
    def test_promotes_high_salience_working(self) -> None:
        store = fresh_store()
        high = store.add(MemoryItem("important", memory_type=MemoryType.WORKING, salience=0.9))
        low = store.add(MemoryItem("trivial", memory_type=MemoryType.WORKING, salience=0.2))
        promoted = store.consolidate(salience_threshold=0.5)
        assert high in promoted
        assert high.memory_type == MemoryType.EPISODIC

    def test_discards_low_salience_working(self) -> None:
        store = fresh_store()
        low = store.add(MemoryItem("trivial", memory_type=MemoryType.WORKING, salience=0.1))
        store.consolidate(salience_threshold=0.5)
        assert store.get(low.id) is None

    def test_non_working_untouched(self) -> None:
        store = fresh_store()
        ep = store.add(MemoryItem("event", memory_type=MemoryType.EPISODIC, salience=0.9))
        store.consolidate()
        assert ep.memory_type == MemoryType.EPISODIC


class TestStats:
    def test_empty_stats(self) -> None:
        store = fresh_store()
        stats = store.stats()
        assert stats["size"] == 0

    def test_stats_keys(self) -> None:
        store = fresh_store()
        store.add(make_item("x"))
        stats = store.stats()
        for key in ("size", "avg_retention", "min_retention", "max_retention",
                    "avg_salience", "type_counts"):
            assert key in stats

    def test_type_counts(self) -> None:
        store = fresh_store()
        store.add(MemoryItem("ep", memory_type=MemoryType.EPISODIC))
        store.add(MemoryItem("sem", memory_type=MemoryType.SEMANTIC))
        store.add(MemoryItem("sem2", memory_type=MemoryType.SEMANTIC))
        stats = store.stats()
        assert stats["type_counts"]["episodic"] == 1
        assert stats["type_counts"]["semantic"] == 2
