# Agentic Forgetting Architecture

A Python implementation of biologically-inspired and engineering-practical **forgetting strategies** for agentic memory systems — the mechanisms that decide *what an AI agent should stop remembering, and when*.

---

## Why Forgetting Matters

Long-running agents accumulate memories.  Without forgetting:

- **Context windows overflow** — too many memories compete for limited token budget.
- **Signal drowns in noise** — every trivial fact has equal weight to critical knowledge.
- **Stale facts mislead** — outdated information is never evicted, even when superseded.

Forgetting is not a failure mode; it is a first-class feature of robust memory design.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     AgentMemorySystem                        │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │   Working    │  │   Episodic   │  │     Semantic     │  │
│  │   Memory     │  │   Memory     │  │     Memory       │  │
│  │  (LRU, n=10) │  │ (Ebbinghaus) │  │ (Salience-wtd)  │  │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘  │
│         │                 │                   │             │
│         └────────────┬────┘                   │             │
│                      │ reflect()              │             │
│               consolidate &                   │             │
│               prune low-ret                   │             │
└─────────────────────────────────────────────────────────────┘
```

### Three Memory Tiers

| Tier | Purpose | Default Strategy | Typical Capacity |
|------|---------|-----------------|-----------------|
| **Working** | Active context scratch-pad | LRU | 10 items |
| **Episodic** | Conversation history, events | Ebbinghaus decay | 200 items |
| **Semantic** | Facts, general knowledge | Salience-weighted | 500 items |

---

## Forgetting Strategies

### 1. Ebbinghaus Decay (`EbbinghausForgetting`)

Models the psychological **forgetting curve**:

```
R(t) = exp( -t / S )
```

Where:
- `R` = retention fraction (0 → 1)
- `t` = time since last access (days)
- `S` = memory stability (days); increases on each retrieval via the **spacing effect**

Each time a memory is retrieved, its stability `S` grows by ×1.5, capped at 365 days.  
Items whose retention falls below a configurable threshold become eviction candidates.

```python
from memory import EbbinghausForgetting, MemoryStore

store = MemoryStore(strategy=EbbinghausForgetting(threshold=0.2))
```

**Best for:** Episodic memories — conversation turns, tool results, recent events — that naturally fade unless revisited.

---

### 2. LRU Eviction (`LRUForgetting`)

Classic **Least-Recently-Used** cache eviction.  The item accessed furthest in the past is evicted first.  No salience or retention scoring — purely time-of-last-access.

```python
from memory import LRUForgetting, MemoryStore

store = MemoryStore(strategy=LRUForgetting(), max_size=10)
```

**Best for:** Working memory — a fixed-size scratch-pad where recency is the only signal that matters.

---

### 3. Salience-Weighted Forgetting (`SalienceForgetting`)

Combines importance and recency into a single eviction score:

```
score(m) = salience(m) × retention(m, t)
```

An unimportant-but-recent item can outlast an important-but-very-stale one.  Only items that are simultaneously *low-salience* and *stale* are evicted first.

```python
from memory import SalienceForgetting, MemoryStore

store = MemoryStore(strategy=SalienceForgetting())
```

**Best for:** Mixed stores where importance metadata is available (e.g. "this fact came from a verified source → salience=0.9").

---

### 4. Capacity-Based Forgetting (`CapacityForgetting`)

Enforces a **hard upper bound** on store size.  When the store is full and a new item arrives, the lowest-scored item (salience × retention) is evicted.  Analogous to a fixed-size vector index.

```python
from memory import CapacityForgetting, MemoryStore

store = MemoryStore(strategy=CapacityForgetting(max_items=100), max_size=100)
```

**Best for:** Production deployments with fixed vector store budgets (e.g. Pinecone pod size, OpenSearch domain).

---

## Consolidation & the `reflect()` Cycle

At conversation end (or periodically), `AgentMemorySystem.reflect()` runs a two-phase maintenance pass:

1. **Consolidation** — High-salience Working memories are promoted to Episodic storage; low-salience Working memories are discarded.  Mirrors the biological sleep-consolidation process.

2. **Retention pruning** — Items across all tiers whose Ebbinghaus retention falls below a threshold (default 0.1) are removed.

```python
agent_mem = AgentMemorySystem()
# ... agent runs ...
result = agent_mem.reflect(consolidation_threshold=0.6, retention_prune_threshold=0.1)
# result = {'consolidated': 3, 'pruned': {'working': 1, 'episodic': 0, 'semantic': 0}}
```

---

## Quick Start

```python
from memory import AgentMemorySystem, MemoryType

agent = AgentMemorySystem()

# Store memories across tiers
agent.remember("User asked about AWS S3 pricing", MemoryType.WORKING, salience=0.8)
agent.remember("S3 Standard = $0.023/GB/month", MemoryType.SEMANTIC, salience=0.95,
               tags=["aws", "s3", "pricing"])
agent.remember("User prefers concise bullet-point answers", MemoryType.EPISODIC,
               salience=0.7, tags=["preference"])

# Recall by keyword (substring search; swap for vector search in production)
results = agent.recall("pricing", limit=5)

# End-of-turn maintenance
agent.reflect()

# Observability
print(agent.snapshot())
```

---

## Project Structure

```
agentcore-assist/
├── memory/
│   ├── __init__.py               Public API surface
│   ├── memory_item.py            MemoryItem dataclass + Ebbinghaus maths
│   ├── forgetting_strategies.py  Four eviction strategies
│   ├── memory_store.py           Thread-safe tier store
│   └── agent_memory.py           Three-tier AgentMemorySystem facade
├── tests/
│   ├── test_memory_item.py       13 unit tests
│   ├── test_forgetting_strategies.py  15 tests across all strategies
│   ├── test_memory_store.py      22 tests for store operations
│   └── test_agent_memory.py      16 integration tests
├── demo.py                       Five runnable demo scenarios
├── requirements.txt
└── README.md
```

---

## Running the Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
# 66 tests, all passing
```

## Running the Demo

```bash
python3 demo.py
```

Five demonstrations run sequentially:
1. Ebbinghaus decay with simulated time passage
2. LRU eviction in a working-memory scratch-pad
3. Salience-weighted eviction protecting critical knowledge
4. Three-tier system with consolidation
5. Capacity-based eviction under continuous load

---

## Design Decisions

**Why no vector embeddings?**  
`recall()` uses substring search so the package has zero external dependencies.  In production, replace the `query_fn` with a cosine-similarity call against any embedding store.

**Why three tiers?**  
Working / Episodic / Semantic maps to the standard cognitive architecture taxonomy.  Each tier has different volatility: working is ephemeral (seconds–minutes), episodic fades over days, semantic knowledge is semi-permanent.

**Why is salience caller-supplied?**  
The agent (or its orchestrator) has context the memory system doesn't: whether a tool call succeeded, whether the user explicitly flagged something as important, or whether a fact came from a verified source.  Externalising salience keeps the memory system strategy-agnostic.

**Thread safety**  
Each `MemoryStore` uses an `RLock`.  The `AgentMemorySystem` does not add a cross-tier lock; callers that need atomic cross-tier reads should coordinate externally.
