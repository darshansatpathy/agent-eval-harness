"""Judge calibration on statements whose labels are known by construction.

Supported statements are sentences of the knowledge base, judged against their own paragraph.
Unsupported statements are the same sentences altered in the ways answers go wrong: a number
changed, a negation added or removed, one plan swapped for another, or a true sentence from a
different article (unrelated to the paragraph it is judged against). The judge's agreement with
these labels (Cohen's kappa) decides whether it may gate at all, and the threshold that maximises
it is the one used.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field

from evalgate.judge import Judge
from evalgate.kb import Chunk, sentences

NUMBER = re.compile(r"(?<![\w-])(\$?)(\d+(?:\.\d+)?)(%?)")
PLANS = {"Free": "Business", "Plus": "Business", "Business": "Plus"}
NEGATIONS = [(" cannot ", " can "), (" can ", " cannot "), (" not ", " "), (" is ", " is not "), (" are ", " are not ")]


@dataclass
class Statement:
    premise: str
    hypothesis: str
    supported: bool
    kind: str  # original | number | negation | entity | unrelated


def _number(s: str, rng: random.Random) -> str | None:
    found = list(NUMBER.finditer(s))
    if not found:
        return None
    m = rng.choice(found)
    value = float(m.group(2))
    new = value * rng.choice([2, 3]) + rng.choice([1, 5]) if value < 1000 else value + 1000
    text = f"{new:g}" if "." in m.group(2) else str(int(new))
    return s[: m.start(2)] + text + s[m.end(2) :]


def _negate(s: str) -> str | None:
    padded = f" {s} "
    for a, b in NEGATIONS:
        if a in padded:
            return padded.replace(a, b, 1).strip()
    return None


def _entity(s: str) -> str | None:
    for a, b in PLANS.items():
        if re.search(rf"\b{a}\b", s):
            return re.sub(rf"\b{a}\b", b, s, count=1)
    return None


def build(chunks: list[Chunk], seed: int = 3, per_kind: int = 40) -> list[Statement]:
    rng = random.Random(seed)
    pool = [(c, s) for c in chunks for s in sentences(c.text) if len(s.split()) >= 6 and "SYSTEM NOTE" not in s]
    rng.shuffle(pool)
    out: list[Statement] = []
    counts: Counter[str] = Counter()
    kinds = [("entity", _entity), ("number", _number), ("negation", _negate)]
    for i, (c, s) in enumerate(pool):
        if counts["original"] < per_kind * 2:
            out.append(Statement(c.text, s, True, "original"))
            counts["original"] += 1
        for kind, fn in kinds[i % 3 :] + kinds[: i % 3]:  # rotate, so every kind gets its share
            if counts[kind] >= per_kind:
                continue
            changed = fn(s, rng) if kind == "number" else fn(s)  # type: ignore[operator]
            if changed and changed != s:
                out.append(Statement(c.text, changed, False, kind))
                counts[kind] += 1
                break
    others = [(c, s) for c, s in pool]
    while counts["unrelated"] < per_kind:
        (c1, _), (c2, s2) = rng.sample(others, 2)
        if c1.doc != c2.doc:
            out.append(Statement(c1.text, s2, False, "unrelated"))
            counts["unrelated"] += 1
    return out


def kappa(truth: list[bool], pred: list[bool]) -> float:
    n = len(truth)
    agree = sum(t == p for t, p in zip(truth, pred, strict=True)) / n
    pt, pp = sum(truth) / n, sum(pred) / n
    expected = pt * pp + (1 - pt) * (1 - pp)
    return (agree - expected) / (1 - expected) if expected < 1 else 1.0


@dataclass
class Calibration:
    judge: str
    threshold: float
    kappa: float
    accuracy: float
    statements: int
    detection: dict[str, float] = field(default_factory=dict)  # share of each kind judged correctly
    sweep: list[tuple[float, float]] = field(default_factory=list)


def calibrate(judge: Judge, statements: list[Statement]) -> Calibration:
    scores = [judge.entailment([s.premise], s.hypothesis) for s in statements]
    truth = [s.supported for s in statements]
    sweep = []
    for t in [x / 20 for x in range(2, 20)]:
        sweep.append((t, kappa(truth, [x >= t for x in scores])))
    best_t, best_k = max(sweep, key=lambda x: (x[1], -abs(x[0] - 0.5)))
    pred = [x >= best_t for x in scores]
    detection = {}
    for kind in sorted({s.kind for s in statements}):
        idx = [i for i, s in enumerate(statements) if s.kind == kind]
        detection[kind] = sum(pred[i] == truth[i] for i in idx) / len(idx)
    acc = sum(p == t for p, t in zip(pred, truth, strict=True)) / len(truth)
    return Calibration(judge.name, best_t, best_k, acc, len(statements), detection, sweep)
