"""The CI gate: compare a candidate run with the baseline and decide whether it may merge.

A candidate is blocked when any of these holds:

1. it was graded on a different golden set than the baseline (the comparison would be meaningless);
2. its judge has not been calibrated, or agrees with known labels less than the policy requires;
3. a case tagged critical fails;
4. a guarded metric drops by more than its tolerance;
5. it fails significantly more cases than the baseline, by an exact one-sided McNemar test on the
   cases that changed outcome.

Rules 4 and 5 complement each other: a tolerance catches a large drop on a small golden set
where no test could be significant, and the test catches many small slips that each stay inside
a tolerance. The decision lists every case that changed, with what changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, Field


class Policy(BaseModel):
    alpha: float = 0.05
    min_judge_kappa: float = 0.6
    tolerances: dict[str, float] = Field(
        default_factory=lambda: {
            "pass_rate": 0.03,
            "faithfulness": 0.02,
            "citation_recall": 0.05,
            "abstention_accuracy": 0.0,
            "refusal_accuracy": 0.0,
            "tool_accuracy": 0.0,
        }
    )
    max_latency_p95_increase: float = 0.5  # fraction

    @classmethod
    def from_file(cls, path: Path) -> Policy:
        return cls.model_validate(yaml.safe_load(path.read_text()) or {})


@dataclass
class Decision:
    blocked: bool
    reasons: list[str]
    regressed: list[dict[str, Any]] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)
    metrics: list[tuple[str, float, float, float | None]] = field(default_factory=list)  # name, baseline, candidate, tolerance
    mcnemar_p: float = 1.0
    pass_rate_ci: tuple[float, float] = (0.0, 0.0)


def mcnemar_one_sided(worse: int, better: int) -> float:
    """P(at least `worse` regressions among the changed cases if both runs were equally good)."""
    n = worse + better
    if n == 0:
        return 1.0
    return float(sum(math.comb(n, k) for k in range(worse, n + 1)) / 2**n)


def bootstrap_diff(base: list[bool], cand: list[bool], seed: int = 0, n: int = 4000) -> tuple[float, float]:
    """95% interval for the change in pass rate, resampling cases (the runs are paired)."""
    rng = np.random.default_rng(seed)
    b, c = np.asarray(base, dtype=float), np.asarray(cand, dtype=float)
    idx = rng.integers(0, len(b), (n, len(b)))
    diffs = c[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _why(case: dict[str, Any]) -> str:
    bad = [f"{g['name']}: {g['detail']}" for g in case["grades"] if not g["passed"]]
    if case.get("faithful") is False:
        bad.append("judge: unsupported claim: " + "; ".join(s[:80] for s in case["unsupported"]))
    return "; ".join(bad) or "failed"


def decide(baseline: dict[str, Any], candidate: dict[str, Any], policy: Policy, calibration: dict[str, Any] | None) -> Decision:
    reasons: list[str] = []
    if baseline["meta"]["golden_digest"] != candidate["meta"]["golden_digest"]:
        return Decision(
            True, ["the candidate ran on a different golden set than the baseline: regenerate the baseline with `evg baseline`"]
        )
    if candidate["meta"]["judge"] is not None:
        if calibration is None or calibration.get("judge") != candidate["meta"]["judge"]:
            reasons.append(f"judge {candidate['meta']['judge']} has no calibration")
        elif calibration["kappa"] < policy.min_judge_kappa:
            reasons.append(f"judge agreement kappa {calibration['kappa']:.2f} is below {policy.min_judge_kappa}")
    base = {c["id"]: c for c in baseline["cases"]}
    cand = {c["id"]: c for c in candidate["cases"]}
    common = [i for i in base if i in cand]
    regressed = [cand[i] for i in common if base[i]["passed"] and not cand[i]["passed"]]
    fixed = [i for i in common if not base[i]["passed"] and cand[i]["passed"]]
    for c in candidate["cases"]:
        if c["critical"] and not c["passed"]:
            reasons.append(f"critical case {c['id']} fails: {_why(c)}")
    rows: list[tuple[str, float, float, float | None]] = []
    bm, cm = baseline["metrics"], candidate["metrics"]
    for name, tol in policy.tolerances.items():
        b, c = bm.get(name, float("nan")), cm.get(name, float("nan"))
        rows.append((name, b, c, tol))
        if not (math.isnan(b) or math.isnan(c)) and b - c > tol + 1e-9:
            reasons.append(f"{name} fell from {b:.3f} to {c:.3f} (tolerance {tol:.3f})")
    b, c = bm["latency_p95_ms"], cm["latency_p95_ms"]
    rows.append(("latency_p95_ms", b, c, None))
    if b > 0 and c > b * (1 + policy.max_latency_p95_increase) and c - b > 50:
        reasons.append(f"p95 latency rose from {b:.0f} ms to {c:.0f} ms")
    p = mcnemar_one_sided(len(regressed), len(fixed))
    if len(regressed) > len(fixed) and p < policy.alpha:
        reasons.append(f"{len(regressed)} cases regressed and {len(fixed)} improved (McNemar p = {p:.3f})")
    ci = bootstrap_diff([base[i]["passed"] for i in common], [cand[i]["passed"] for i in common])
    return Decision(
        bool(reasons), reasons, [{"id": r["id"], "why": _why(r), "critical": r["critical"]} for r in regressed], fixed, rows, p, ci
    )


def markdown(d: Decision, baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    head = "## Agent evaluation: **blocked**" if d.blocked else "## Agent evaluation: passed"
    lines = [
        head,
        "",
        f"Candidate `{candidate['meta']['agent']}` against baseline `{baseline['meta']['agent']}` on golden set "
        f"`{candidate['meta']['golden_digest']}` ({len(candidate['cases'])} cases).",
        "",
    ]
    if d.reasons:
        lines += ["**Why it is blocked:**", ""] + [f"- {r}" for r in d.reasons] + [""]
    lines += ["| Metric | Baseline | Candidate | Change | Tolerance |", "|---|---|---|---|---|"]
    for name, b, c, tol in d.metrics:
        change = c - b
        tol_s = "" if tol is None else f"{tol:.3f}"
        digits = 0 if name.endswith("_ms") else 3
        lines.append(f"| {name} | {b:.{digits}f} | {c:.{digits}f} | {change:+.3f} | {tol_s} |")
    lo, hi = d.pass_rate_ci
    lines += ["", f"Pass rate change 95% interval: {lo:+.3f} to {hi:+.3f}. McNemar one-sided p = {d.mcnemar_p:.3f}.", ""]
    if d.regressed:
        lines += (
            ["**Regressed cases:**", ""]
            + [f"- `{r['id']}`{' (critical)' if r['critical'] else ''}: {r['why']}" for r in d.regressed]
            + [""]
        )
    if d.fixed:
        lines += ["**Fixed cases:** " + ", ".join(f"`{i}`" for i in d.fixed), ""]
    return "\n".join(lines)
