from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(Enum):
    EPISODIC = "episodic"      # specific events / experiences
    SEMANTIC = "semantic"      # general facts / knowledge
    PROCEDURAL = "procedural"  # how-to / skills
    WORKING = "working"        # in-context scratch-pad


@dataclass
class MemoryItem:
    content: Any
    memory_type: MemoryType = MemoryType.EPISODIC
    salience: float = 1.0          # 0.0 – 1.0, importance score
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    tags: list[str] = field(default_factory=list)

    # Ebbinghaus parameters
    _stability: float = field(default=1.0, repr=False)  # memory stability (days)

    def touch(self) -> None:
        """Record an access; also strengthens the memory trace."""
        self.last_accessed_at = time.time()
        self.access_count += 1
        # Each retrieval increases stability (spacing effect)
        self._stability = min(self._stability * 1.5, 365.0)

    def retention(self, now: float | None = None) -> float:
        """
        Ebbinghaus retention fraction R = e^(-t / S)
        t = elapsed time in days, S = stability.
        Returns a value in [0, 1].
        """
        now = now or time.time()
        elapsed_days = (now - self.last_accessed_at) / 86_400
        import math
        return math.exp(-elapsed_days / self._stability)

    def age_seconds(self, now: float | None = None) -> float:
        now = now or time.time()
        return now - self.created_at

    def __repr__(self) -> str:
        snippet = str(self.content)[:60]
        return (
            f"MemoryItem(id={self.id[:8]}, type={self.memory_type.value}, "
            f"salience={self.salience:.2f}, accesses={self.access_count}, "
            f"content={snippet!r})"
        )
