"""Tests for MemoryItem."""
import math
import time
import pytest
from memory.memory_item import MemoryItem, MemoryType


def test_defaults():
    m = MemoryItem("hello")
    assert m.memory_type == MemoryType.EPISODIC
    assert m.salience == 1.0
    assert m.access_count == 0
    assert m.id  # non-empty UUID


def test_unique_ids():
    ids = {MemoryItem("x").id for _ in range(100)}
    assert len(ids) == 100


def test_touch_increments_access_count():
    m = MemoryItem("test")
    m.touch()
    m.touch()
    assert m.access_count == 2


def test_touch_updates_last_accessed_at():
    m = MemoryItem("test")
    before = time.time()
    m.touch()
    assert m.last_accessed_at >= before


def test_retention_fresh_is_near_one():
    m = MemoryItem("fresh")
    assert m.retention() > 0.99


def test_retention_decreases_over_time():
    m = MemoryItem("old")
    now = time.time()
    # simulate 5 days ago
    ret_5d = m.retention(now + 5 * 86_400)
    ret_10d = m.retention(now + 10 * 86_400)
    assert ret_5d < 1.0
    assert ret_10d < ret_5d


def test_retention_formula():
    m = MemoryItem("formula check")
    # stability starts at 1 day
    elapsed_days = 1.0
    now = m.last_accessed_at + elapsed_days * 86_400
    expected = math.exp(-elapsed_days / m._stability)
    assert abs(m.retention(now) - expected) < 1e-9


def test_touch_raises_stability():
    m = MemoryItem("spacing")
    s_before = m._stability
    m.touch()
    assert m._stability > s_before


def test_stability_cap():
    m = MemoryItem("cap test")
    for _ in range(1000):
        m.touch()
    assert m._stability <= 365.0


def test_age_seconds():
    m = MemoryItem("age")
    time.sleep(0.01)
    assert m.age_seconds() >= 0.01


def test_memory_types():
    for mt in MemoryType:
        m = MemoryItem("x", memory_type=mt)
        assert m.memory_type == mt


def test_tags():
    m = MemoryItem("tagged", tags=["aws", "s3"])
    assert "aws" in m.tags
