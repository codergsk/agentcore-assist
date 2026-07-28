"""Tests for AgentMemorySystem."""
import time
import pytest
from memory.memory_item import MemoryType
from memory.agent_memory import AgentMemorySystem


def make_system(**kwargs):
    return AgentMemorySystem(
        working_capacity=kwargs.get("working_capacity", 5),
        episodic_capacity=kwargs.get("episodic_capacity", 20),
        semantic_capacity=kwargs.get("semantic_capacity", 50),
    )


class TestRemember:
    def test_adds_to_correct_tier(self):
        sys = make_system()
        sys.remember("event", MemoryType.EPISODIC)
        sys.remember("fact", MemoryType.SEMANTIC)
        sys.remember("scratch", MemoryType.WORKING)
        assert sys.episodic.size == 1
        assert sys.semantic.size == 1
        assert sys.working.size == 1

    def test_returns_memory_item(self):
        sys = make_system()
        item = sys.remember("hello", MemoryType.WORKING)
        assert item.content == "hello"

    def test_salience_assigned(self):
        sys = make_system()
        item = sys.remember("hi", salience=0.75)
        assert item.salience == 0.75

    def test_tags_assigned(self):
        sys = make_system()
        item = sys.remember("tagged", tags=["aws", "s3"])
        assert "aws" in item.tags

    def test_working_capacity_enforced(self):
        sys = make_system(working_capacity=3)
        for i in range(6):
            sys.remember(f"w{i}", MemoryType.WORKING)
        assert sys.working.size == 3


class TestRecall:
    def test_finds_by_content_substring(self):
        sys = make_system()
        sys.remember("AWS S3 pricing", MemoryType.SEMANTIC, salience=0.9, tags=[])
        sys.remember("Lambda timeout config", MemoryType.SEMANTIC, salience=0.8)
        results = sys.recall("s3")
        assert any("S3" in str(m.content) for m in results)

    def test_finds_by_tag(self):
        sys = make_system()
        sys.remember("EC2 instance types", MemoryType.SEMANTIC, tags=["ec2", "aws"])
        results = sys.recall("ec2")
        assert len(results) > 0

    def test_respects_limit(self):
        sys = make_system()
        for i in range(10):
            sys.remember(f"fact about topic {i}", MemoryType.SEMANTIC, salience=0.5)
        results = sys.recall("fact", limit=3)
        assert len(results) <= 3

    def test_cross_tier_search(self):
        sys = make_system()
        sys.remember("episodic pricing event", MemoryType.EPISODIC)
        sys.remember("semantic pricing fact", MemoryType.SEMANTIC)
        results = sys.recall("pricing")
        contents = [str(m.content) for m in results]
        assert any("episodic" in c for c in contents)
        assert any("semantic" in c for c in contents)

    def test_no_results_for_unknown_query(self):
        sys = make_system()
        sys.remember("something unrelated", MemoryType.SEMANTIC)
        results = sys.recall("xyzzy")
        assert results == []

    def test_filter_by_memory_type(self):
        sys = make_system()
        sys.remember("pricing fact", MemoryType.SEMANTIC, salience=0.9)
        sys.remember("pricing event", MemoryType.EPISODIC, salience=0.9)
        results = sys.recall("pricing", memory_type=MemoryType.SEMANTIC)
        assert all(m.memory_type == MemoryType.SEMANTIC for m in results)


class TestForgetCycle:
    def test_returns_dict_with_tiers(self):
        sys = make_system()
        sys.remember("a", MemoryType.EPISODIC)
        sys.remember("b", MemoryType.SEMANTIC)
        sys.remember("c", MemoryType.WORKING)
        result = sys.forget_cycle(n_per_tier=1)
        assert set(result.keys()) == {"working", "episodic", "semantic"}

    def test_removes_items_from_store(self):
        sys = make_system()
        for _ in range(3):
            sys.remember("episodic item", MemoryType.EPISODIC)
        before = sys.episodic.size
        sys.forget_cycle(n_per_tier=1)
        assert sys.episodic.size == before - 1


class TestReflect:
    def test_consolidates_high_salience_working(self):
        sys = make_system()
        sys.remember("important working item", MemoryType.WORKING, salience=0.9)
        sys.remember("trivial scratch", MemoryType.WORKING, salience=0.1)
        result = sys.reflect(consolidation_threshold=0.5)
        assert result["consolidated"] >= 1

    def test_discards_low_salience_working(self):
        sys = make_system()
        sys.remember("trivial A", MemoryType.WORKING, salience=0.1)
        sys.remember("trivial B", MemoryType.WORKING, salience=0.1)
        sys.reflect(consolidation_threshold=0.5)
        assert sys.working.size == 0

    def test_prune_very_stale(self):
        sys = make_system()
        item = sys.remember("ancient fact", MemoryType.SEMANTIC)
        item.last_accessed_at = time.time() - 999 * 86_400  # ancient
        result = sys.reflect(retention_prune_threshold=0.5)
        total_pruned = sum(result["pruned"].values())
        assert total_pruned >= 1

    def test_reflect_result_structure(self):
        sys = make_system()
        result = sys.reflect()
        assert "consolidated" in result
        assert "pruned" in result
        assert set(result["pruned"].keys()) == {"working", "episodic", "semantic"}


class TestSnapshot:
    def test_snapshot_has_three_tiers(self):
        sys = make_system()
        snap = sys.snapshot()
        assert set(snap.keys()) == {"working", "episodic", "semantic"}

    def test_snapshot_size_accurate(self):
        sys = make_system()
        sys.remember("ep", MemoryType.EPISODIC)
        sys.remember("sem", MemoryType.SEMANTIC)
        snap = sys.snapshot()
        assert snap["episodic"]["size"] == 1
        assert snap["semantic"]["size"] == 1
        assert snap["working"]["size"] == 0
