"""Golden sets: loading, validation and a content hash that ties results to the set they ran on."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Expect = Literal["answer", "abstain", "refuse"]


class ToolSpec(BaseModel):
    name: str
    args: dict[str, Any]


class Case(BaseModel):
    id: str
    question: str
    category: str
    expect: Expect = "answer"
    must_include: list[list[str]] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    tool: ToolSpec | None = None
    tags: list[str] = Field(default_factory=list)

    @property
    def critical(self) -> bool:
        return "critical" in self.tags

    @model_validator(mode="after")
    def _consistent(self) -> Case:
        if self.expect != "answer" and (self.must_include or self.citations or self.tool):
            raise ValueError(f"{self.id}: an abstention or refusal case cannot require content, citations or tools")
        if self.expect == "answer" and not (self.must_include or self.tool):
            raise ValueError(f"{self.id}: an answer case must state what the answer contains")
        return self


class GoldenSet(BaseModel):
    version: int
    cases: list[Case]

    @model_validator(mode="after")
    def _unique(self) -> GoldenSet:
        ids = [c.id for c in self.cases]
        dup = {i for i in ids if ids.count(i) > 1}
        if dup:
            raise ValueError(f"duplicate case ids: {sorted(dup)}")
        return self

    def digest(self) -> str:
        """Changes when any case changes, so results are never compared across different sets."""
        canon = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:16]


def load(path: Path, known_docs: set[str] | None = None) -> GoldenSet:
    g = GoldenSet.model_validate(yaml.safe_load(path.read_text()))
    if known_docs is not None:
        for c in g.cases:
            missing = set(c.citations) - known_docs
            if missing:
                raise ValueError(f"{c.id}: cites unknown documents {sorted(missing)}")
    return g
