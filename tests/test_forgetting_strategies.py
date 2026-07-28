"""Tests for the four forgetting strategies."""
import time
import pytest
from memory.memory_item import MemoryItem, MemoryType
from memory.forgetting_strategies import (
    EbbinghausForgetting,
    LRUForgetting,
    SalienceForgetting,
    CapacityForgetting,
)


def make_item(content="x", salience=0.5, days_ago=0.0):
    m = MemoryItem(content, salience=salience)
    if days_ago:
        m.last_accessed_at = time.time() - days_ago * 86_400
    return m


# ──────────────────────────────────────────────────────────────────────────────
# EbbinghausForgetting
# ──────────────────────────────────────────────────────────────────────────────

class TestEbbinghausForgetting:
    def test_selects_stalest_item(self):
        strat = EbbinghausForgetting(threshold=0.5)
        fresh = make_item("fresh", days_ago=0)
        stale = make_item("stale", days_ago=5)
        removed = strat.select_for_removal([fresh, stale], n=1)
        assert removed[0] is stale

    def test_returns_n_items(self):
        strat = EbbinghausForgetting()
        items = [make_item(f"item{i}", days_ago=i) for i in range(5)]
        removed = strat.select_for_removal(items, n=3)
        assert len(removed) == 3

    def test_empty_list(self):
        strat = EbbinghausForgetting()
        assert strat.select_for_removal([], n=1) == []

    def test_all_fresh_still_returns_something(self):
        strat = EbbinghausForgetting(threshold=0.99)
        items = [make_item(f"i{i}") for i in range(3)]
        removed = strat.select_for_removal(items, n=1)
        assert len(removed) == 1  # falls back to lowest retention

    def test_threshold_filters(self):
        strat = EbbinghausForgetting(threshold=0.5)
        very_stale = make_item("very_stale", days_ago=100)
        fresh = make_item("fresh", days_ago=0)
        removed = strat.select_for_removal([very_stale, fresh], n=1)
        assert removed[0] is very_stale


# ──────────────────────────────────────────────────────────────────────────────
# LRUForgetting
# ──────────────────────────────────────────────────────────────────────────────

class TestLRUForgetting:
    def test_evicts_oldest(self):
        strat = LRUForgetting()
        now = time.time()
        old = make_item("old")
        old.last_accessed_at = now - 1000
        new = make_item("new")
        new.last_accessed_at = now
        removed = strat.select_for_removal([new, old], n=1)
        assert removed[0] is old

    def test_returns_n_items(self):
        strat = LRUForgetting()
        items = [make_item(f"i{i}") for i in range(10)]
        removed = strat.select_for_removal(items, n=4)
        assert len(removed) == 4

    def test_ignores_salience(self):
        strat = LRUForgetting()
        now = time.time()
        high_salience_old = make_item("critical", salience=1.0)
        high_salience_old.last_accessed_at = now - 9999
        low_salience_new = make_item("trivial", salience=0.01)
        low_salience_new.last_accessed_at = now
        removed = strat.select_for_removal([high_salience_old, low_salience_new], n=1)
        assert removed[0] is high_salience_old


# ──────────────────────────────────────────────────────────────────────────────
# SalienceForgetting
# ──────────────────────────────────────────────────────────────────────────────

class TestSalienceForgetting:
    def test_low_salience_stale_evicted_first(self):
        strat = SalienceForgetting()
        important_fresh = make_item("important", salience=0.9, days_ago=0)
        trivial_stale = make_item("trivial", salience=0.1, days_ago=5)
        removed = strat.select_for_removal([important_fresh, trivial_stale], n=1)
        assert removed[0] is trivial_stale

    def test_high_salience_protected(self):
        strat = SalienceForgetting()
        items = [make_item(f"item{i}", salience=0.1 * i, days_ago=1) for i in range(1, 6)]
        removed = strat.select_for_removal(items, n=2)
        # highest salience items should NOT be in removed
        for m in removed:
            assert m.salience < 0.5

    def test_score_combines_salience_and_retention(self):
        strat = SalienceForgetting()
        now = time.time()
        # medium salience, very stale vs low salience, fresh
        m_stale = make_item("medium_stale", salience=0.5, days_ago=30)
        m_fresh = make_item("low_fresh", salience=0.3, days_ago=0)
        removed = strat.select_for_removal([m_stale, m_fresh], n=1)
        # stale medium should score lower than fresh low-salience
        assert removed[0] is m_stale


# ──────────────────────────────────────────────────────────────────────────────
# CapacityForgetting
# ──────────────────────────────────────────────────────────────────────────────

class TestCapacityForgetting:
    def test_items_to_evict(self):
        strat = CapacityForgetting(max_items=5)
        assert strat.items_to_evict(7) == 2
        assert strat.items_to_evict(5) == 0
        assert strat.items_to_evict(3) == 0

    def test_evicts_lowest_scored(self):
        strat = CapacityForgetting(max_items=3)
        items = [
            make_item("keep1", salience=0.9),
            make_item("keep2", salience=0.8),
            make_item("evict", salience=0.1, days_ago=5),
        ]
        removed = strat.select_for_removal(items, n=1)
        assert removed[0].content == "evict"

    def test_invalid_max_items(self):
        with pytest.raises(ValueError):
            CapacityForgetting(max_items=0)

    def test_n_larger_than_list(self):
        strat = CapacityForgetting(max_items=10)
        items = [make_item(f"i{i}") for i in range(3)]
        removed = strat.select_for_removal(items, n=10)
        assert len(removed) == 3  # capped at list length
