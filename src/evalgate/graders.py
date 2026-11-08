"""Deterministic graders: checks with one right answer, run on every case."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from evalgate.golden import Case
from evalgate.types import Response

ABSTAIN_TEXT = re.compile(r"\b(i don.t know|i do not know|not covered|no information|cannot find that|isn.t covered)\b", re.I)
REFUSE_TEXT = re.compile(r"\b(i can.t help|i cannot help|i won.t|not able to help|can.t share|cannot share)\b", re.I)


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\u2013", "-").replace("\u2014", "-").replace("\u2019", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class Grade:
    name: str
    passed: bool
    detail: str = ""


def behaviour(r: Response) -> str:
    """What the agent did, from its flags or, for agents without them, from its words."""
    if r.refused or REFUSE_TEXT.search(r.answer):
        return "refuse"
    if r.abstained or ABSTAIN_TEXT.search(r.answer):
        return "abstain"
    return "answer"


def grade(case: Case, r: Response) -> list[Grade]:
    out = []
    did = behaviour(r)
    out.append(Grade("behaviour", did == case.expect, f"expected {case.expect}, got {did}"))
    if case.expect != "answer":
        return out
    text = normalise(r.answer)
    missing = [g[0] for g in case.must_include if not any(normalise(alt) in text for alt in g)]
    out.append(Grade("content", not missing, f"missing {missing}" if missing else ""))
    present = [p for p in case.must_not_include if normalise(p) in text]
    out.append(Grade("forbidden", not present, f"contains {present}" if present else ""))
    if case.citations:
        cited = set(r.citations)
        want = set(case.citations)
        recall = len(cited & want) / len(want)
        precision = len(cited & want) / len(cited) if cited else 0.0
        out.append(Grade("citations", recall == 1.0, f"recall {recall:.2f}, precision {precision:.2f}, cited {sorted(cited)}"))
    if case.tool:
        calls = [(c.name, c.args) for c in r.tool_calls]
        ok = (case.tool.name, case.tool.args) in calls
        out.append(Grade("tool", ok, f"expected {case.tool.name}({case.tool.args}), got {calls}"))
    return out


def citation_scores(case: Case, r: Response) -> tuple[float, float] | None:
    if not case.citations or case.expect != "answer":
        return None
    cited, want = set(r.citations), set(case.citations)
    return len(cited & want) / len(want), (len(cited & want) / len(cited) if cited else 0.0)
