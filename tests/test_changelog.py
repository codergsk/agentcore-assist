"""Tests for CHANGELOG.md presence and structure."""
import os
import pytest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANGELOG_PATH = os.path.join(REPO_ROOT, "CHANGELOG.md")


def test_changelog_exists():
    assert os.path.isfile(CHANGELOG_PATH), "CHANGELOG.md must exist at repo root"


def test_changelog_has_heading():
    with open(CHANGELOG_PATH) as f:
        content = f.read()
    assert content.startswith("# Changelog")


def test_changelog_has_initial_release():
    with open(CHANGELOG_PATH) as f:
        content = f.read()
    assert "## [0.1.0] - 2026-07-28" in content


def test_changelog_has_added_section():
    with open(CHANGELOG_PATH) as f:
        content = f.read()
    assert "### Added" in content


def test_changelog_mentions_key_components():
    with open(CHANGELOG_PATH) as f:
        content = f.read()
    assert "MemoryItem" in content
    assert "MemoryStore" in content
    assert "AgentMemorySystem" in content
    assert "forgetting strategies" in content.lower() or "Forgetting" in content
