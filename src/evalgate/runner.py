"""Running an agent over a golden set: deterministic grades, the judge's verdicts, metrics."""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from evalgate import __version__, graders
from evalgate.golden import GoldenSet
from evalgate.judge import Judge, faithfulness
from evalgate.types import Agent


def run(agent: Agent, golden: GoldenSet, judge: Judge | None, threshold: float, config: dict[str, Any], repeats: int = 1) -> dict[str, Any]:
    """Each case runs `repeats` times; it passes only if every repetition passes (pass^k), so a
    flaky agent cannot pass on a lucky draw."""
    cases = []
    for case in golden.cases:
        reps = []
        for _ in range(repeats):
            r = agent.respond(case.question)
            grades = graders.grade(case, r)
            verdict = None
            if judge is not None and case.expect == "answer" and graders.behaviour(r) == "answer":
                verdict = faithfulness(judge, r.answer, r.contexts, threshold)
            ok = all(g.passed for g in grades) and (verdict is None or verdict.faithful)
            reps.append((r, grades, verdict, ok))
        r, grades, verdict, _ = reps[0]
        cites = graders.citation_scores(case, r)
        cases.append(
            {
                "id": case.id,
                "category": case.category,
                "critical": case.critical,
                "expect": case.expect,
                "passed": all(x[3] for x in reps),
                "pass_rate": sum(x[3] for x in reps) / repeats,
                "behaviour": graders.behaviour(r),
                "grades": [asdict(g) for g in grades],
                "faithful": None if verdict is None else verdict.faithful,
                "unsupported": [] if verdict is None else verdict.unsupported,
                "citation_recall": None if cites is None else cites[0],
                "citation_precision": None if cites is None else cites[1],
                "answer": r.answer,
                "citations": r.citations,
                "tool_calls": [{"name": t.name, "args": t.args} for t in r.tool_calls],
                "latency_ms": round(float(np.mean([x[0].latency_ms for x in reps])), 3),
                "cost_usd": float(sum(x[0].cost_usd for x in reps)),
            }
        )
    return {
        "meta": {
            "agent": agent.name,
            "config": config,
            "golden_digest": golden.digest(),
            "golden_version": golden.version,
            "judge": None if judge is None else judge.name,
            "threshold": threshold,
            "repeats": repeats,
            "harness": __version__,
            "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        },
        "cases": cases,
        "metrics": metrics(cases),
    }


def metrics(cases: list[dict[str, Any]]) -> dict[str, float]:
    def share(rows: list[dict[str, Any]], key: str = "passed") -> float:
        return float(np.mean([bool(r[key]) for r in rows])) if rows else float("nan")

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        by_cat[c["category"]].append(c)
    answered = [c for c in cases if c["faithful"] is not None]
    cited = [c for c in cases if c["citation_recall"] is not None]
    tools = [c for c in cases if any(g["name"] == "tool" for g in c["grades"])]
    out = {
        "pass_rate": share(cases),
        "critical_pass_rate": share([c for c in cases if c["critical"]]),
        "faithfulness": share(answered, "faithful") if answered else float("nan"),
        "citation_recall": float(np.mean([c["citation_recall"] for c in cited])) if cited else float("nan"),
        "citation_precision": float(np.mean([c["citation_precision"] for c in cited])) if cited else float("nan"),
        "abstention_accuracy": share([c for c in cases if c["expect"] == "abstain"]),
        "refusal_accuracy": share([c for c in cases if c["expect"] == "refuse"]),
        "tool_accuracy": float(np.mean([next(g["passed"] for g in c["grades"] if g["name"] == "tool") for c in tools]))
        if tools
        else float("nan"),
        "latency_p95_ms": float(np.percentile([c["latency_ms"] for c in cases], 95)),
        "cost_usd": float(sum(c["cost_usd"] for c in cases)),
    }
    out |= {f"pass_rate:{k}": share(v) for k, v in sorted(by_cat.items())}
    return out


def save(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=False) + "\n")


def load(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data
