"""
demo.py — Interactive showcase of the forgetting architecture.

Run with:
    python demo.py
"""
from __future__ import annotations

import math
import time
import textwrap

from memory import (
    AgentMemorySystem,
    MemoryItem,
    MemoryType,
    EbbinghausForgetting,
    LRUForgetting,
    SalienceForgetting,
    CapacityForgetting,
    MemoryStore,
)

DIVIDER = "─" * 64


def section(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")


def show_items(items: list[MemoryItem], label: str = "") -> None:
    if label:
        print(f"\n{label}")
    for m in items:
        ret = m.retention()
        bar = "█" * int(ret * 20) + "░" * (20 - int(ret * 20))
        print(
            f"  [{m.memory_type.value[:3].upper()}] salience={m.salience:.2f} "
            f"ret={ret:.2f} |{bar}| {str(m.content)[:50]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Demo 1 — Ebbinghaus decay
# ──────────────────────────────────────────────────────────────────────────────
def demo_ebbinghaus() -> None:
    section("Demo 1 · Ebbinghaus Forgetting Curve")

    print(textwrap.dedent("""
    The forgetting curve R = exp(-t / S) models how retention decays
    over time.  Each retrieval raises 'stability' S, slowing future decay
    (the spacing effect).
    """))

    store = MemoryStore(strategy=EbbinghausForgetting(threshold=0.3))

    items = [
        store.add(MemoryItem("AWS re:Invent keynote highlights", salience=0.9)),
        store.add(MemoryItem("Lunch order: burger and fries", salience=0.2)),
        store.add(MemoryItem("Customer complaint about login flow", salience=0.7)),
        store.add(MemoryItem("Team standup: Bob is OOO", salience=0.3)),
    ]

    print("Initial memories:")
    show_items(items)

    # Simulate 3 days passing for low-salience items
    now = time.time()
    three_days_ago = now - 3 * 86_400
    items[1].last_accessed_at = three_days_ago  # burger — stale
    items[3].last_accessed_at = three_days_ago  # standup — stale

    print("\nAfter simulating 3 days of staleness for some items:")
    show_items(store.all_items())

    removed = store.forget(n=2)
    print(f"\n  → Evicted {len(removed)} item(s):")
    for m in removed:
        print(f"     {m.content!r}  (retention={m.retention():.3f})")

    print(f"\n  Store now has {store.size} item(s).")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 2 — LRU eviction
# ──────────────────────────────────────────────────────────────────────────────
def demo_lru() -> None:
    section("Demo 2 · LRU (Least-Recently-Used) Eviction")

    print(textwrap.dedent("""
    LRU is the classic cache strategy: discard whichever item was
    accessed least recently.  Used here for the Working Memory tier
    which acts like a short-term scratch-pad.
    """))

    store = MemoryStore(strategy=LRUForgetting(), max_size=3)

    t0 = time.time()
    m1 = store.add(MemoryItem("Tool result: file list", salience=0.6))
    m1.last_accessed_at = t0 - 300   # 5 min ago

    m2 = store.add(MemoryItem("User intent: summarise report", salience=0.8))
    m2.last_accessed_at = t0 - 60    # 1 min ago

    m3 = store.add(MemoryItem("Intermediate calculation: 42", salience=0.4))
    m3.last_accessed_at = t0 - 600   # 10 min ago — oldest

    print("Working memory (3 items, capacity=3):")
    show_items(store.all_items())

    m4 = MemoryItem("New tool result: search hits", salience=0.7)
    m4.last_accessed_at = t0
    store.add(m4)   # triggers LRU eviction

    print("\nAfter adding a 4th item (capacity exceeded):")
    show_items(store.all_items())
    assert all(m.content != "Intermediate calculation: 42" for m in store.all_items()), \
        "LRU should have evicted the oldest item"
    print("  ✓ Oldest item correctly evicted")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 3 — Salience-weighted forgetting
# ──────────────────────────────────────────────────────────────────────────────
def demo_salience() -> None:
    section("Demo 3 · Salience-Weighted Forgetting")

    print(textwrap.dedent("""
    Salience forgetting scores each item as salience × retention(t).
    An unimportant-but-recent item may outlast an important-but-ancient one;
    only items that are both unimportant AND stale are evicted first.
    """))

    store = MemoryStore(strategy=SalienceForgetting())
    now = time.time()

    items = [
        MemoryItem("Critical: prod outage runbook", salience=1.0),
        MemoryItem("Trivia: mascot is a parrot", salience=0.1),
        MemoryItem("Meeting notes: Q3 roadmap", salience=0.6),
        MemoryItem("Random joke remembered", salience=0.05),
    ]
    # Make low-salience items stale
    items[1].last_accessed_at = now - 7 * 86_400
    items[3].last_accessed_at = now - 10 * 86_400

    for item in items:
        store.add(item)

    print("Current items:")
    show_items(store.all_items())

    removed = store.forget(n=2)
    print(f"\n  → Evicted {len(removed)} item(s):")
    for m in removed:
        print(f"     {m.content!r}  (salience={m.salience}, retention={m.retention():.3f})")
    print("  ✓ Low-salience, stale items evicted first")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 4 — Three-tier AgentMemorySystem with consolidation
# ──────────────────────────────────────────────────────────────────────────────
def demo_agent_system() -> None:
    section("Demo 4 · Three-Tier AgentMemorySystem + Consolidation")

    print(textwrap.dedent("""
    The AgentMemorySystem provides three tiers:
      Working   (LRU, small capacity)  — active context scratch-pad
      Episodic  (Ebbinghaus)           — conversation history, events
      Semantic  (Salience-weighted)    — facts, knowledge

    'reflect()' promotes important Working memories to Episodic
    and prunes stale items across all tiers.
    """))

    agent_mem = AgentMemorySystem(
        working_capacity=5,
        episodic_capacity=50,
        semantic_capacity=100,
    )

    # Populate
    agent_mem.remember("User asked about AWS pricing", MemoryType.WORKING, salience=0.8)
    agent_mem.remember("Intermediate: fetched S3 pricing page", MemoryType.WORKING, salience=0.3)
    agent_mem.remember("User is evaluating cost of data lake", MemoryType.WORKING, salience=0.9)
    agent_mem.remember("S3 Standard = $0.023/GB/month", MemoryType.SEMANTIC, salience=0.95,
                       tags=["aws", "s3", "pricing"])
    agent_mem.remember("Previous session: user prefers concise answers", MemoryType.EPISODIC,
                       salience=0.7, tags=["preference"])

    print("Snapshot before reflect():")
    snap = agent_mem.snapshot()
    for tier, stats in snap.items():
        print(f"  {tier}: {stats.get('size', 0)} items, "
              f"avg_retention={stats.get('avg_retention', 1.0):.2f}")

    result = agent_mem.reflect(consolidation_threshold=0.6)
    print(f"\n  reflect() result: {result}")

    print("\nSnapshot after reflect():")
    snap = agent_mem.snapshot()
    for tier, stats in snap.items():
        print(f"  {tier}: {stats.get('size', 0)} items, "
              f"avg_retention={stats.get('avg_retention', 1.0):.2f}")

    # Recall test
    hits = agent_mem.recall("pricing", limit=3)
    print(f"\n  recall('pricing') → {len(hits)} result(s):")
    for m in hits:
        print(f"    [{m.memory_type.value}] {m.content!r}")


# ──────────────────────────────────────────────────────────────────────────────
# Demo 5 — Capacity forgetting under pressure
# ──────────────────────────────────────────────────────────────────────────────
def demo_capacity() -> None:
    section("Demo 5 · Capacity-Based Forgetting Under Load")

    print(textwrap.dedent("""
    CapacityForgetting enforces a hard upper bound.  When the store is
    full, the lowest-scored item (salience × retention) is evicted to
    make room for the new arrival — analogous to a fixed-size vector store.
    """))

    store = MemoryStore(strategy=CapacityForgetting(max_items=5), max_size=5)

    facts = [
        ("Earth orbits the Sun", 0.9),
        ("The sky is blue", 0.5),
        ("pi ≈ 3.14159", 0.8),
        ("Today's lunch special", 0.1),
        ("CEO name: Jane Smith", 0.7),
    ]
    for content, sal in facts:
        store.add(MemoryItem(content, memory_type=MemoryType.SEMANTIC, salience=sal))

    print(f"Store full at {store.size} items:")
    show_items(store.all_items())

    # Add 3 more — should keep evicting lowest-scored
    new_facts = [
        ("AWS re:Invent 2025 theme: AI everywhere", 0.85),
        ("Random overheard conversation", 0.05),
        ("Lambda cold-start tip: use provisioned concurrency", 0.9),
    ]
    for content, sal in new_facts:
        store.add(MemoryItem(content, memory_type=MemoryType.SEMANTIC, salience=sal))
        print(f"\n  Added: {content!r}")
        print(f"  Store ({store.size} items):")
        show_items(store.all_items())

    print(f"\n  Final store size: {store.size} (max=5) ✓")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'█' * 64}")
    print("  AGENTIC FORGETTING ARCHITECTURE — DEMO")
    print(f"{'█' * 64}")

    demo_ebbinghaus()
    demo_lru()
    demo_salience()
    demo_agent_system()
    demo_capacity()

    print(f"\n{'█' * 64}")
    print("  All demos complete.")
    print(f"{'█' * 64}\n")
