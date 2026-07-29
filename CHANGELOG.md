# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-07-28

### Added

- `MemoryItem` dataclass with Ebbinghaus retention model and spacing effect.
- `MemoryStore` with thread-safe add/get/search/forget/consolidate operations.
- Four forgetting strategies: Ebbinghaus decay, LRU, Salience-weighted, and Capacity-based.
- `AgentMemorySystem` three-tier facade (working, episodic, semantic) with reflect/recall.
- Interactive `demo.py` showcasing all forgetting strategies.
- Unit test suite covering all modules (66 tests).
