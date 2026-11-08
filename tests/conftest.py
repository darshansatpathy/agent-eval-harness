from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from evalgate import golden, kb
from evalgate.agent import AgentConfig, build

ROOT = Path(__file__).resolve().parents[1]


class WordJudge:
    """A stand-in judge for fast tests: a claim is supported when every content word occurs in
    some context. The NLI judge is tested separately (marker `judge`)."""

    name = "word-judge"

    def entailment(self, premises: Sequence[str], hypothesis: str) -> float:
        words = set(kb.tokens(hypothesis))
        return max((1.0 if words <= set(kb.tokens(p)) else 0.0 for p in premises), default=0.0)


@pytest.fixture(scope="session")
def chunks() -> list[kb.Chunk]:
    return kb.load(ROOT / "kb")


@pytest.fixture(scope="session")
def gold(chunks) -> golden.GoldenSet:
    return golden.load(ROOT / "golden" / "support.yaml", {c.doc for c in chunks})


@pytest.fixture
def agent_for():
    def make(**changes):
        return build(AgentConfig(**changes), ROOT)

    return make
