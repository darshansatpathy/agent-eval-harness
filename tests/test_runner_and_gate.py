from __future__ import annotations

import copy

from conftest import WordJudge

from evalgate import gate, runner

CAL = {"judge": "word-judge", "kappa": 0.95, "threshold": 0.5}


def _run(agent_for, gold, **changes):
    a = agent_for(name=changes.pop("name", "x"), **changes)
    return runner.run(a, gold, WordJudge(), 0.5, a.config.model_dump())


def test_baseline_run(agent_for, gold):
    r = _run(agent_for, gold)
    m = r["metrics"]
    assert 0.8 < m["pass_rate"] < 1 and m["critical_pass_rate"] == 1.0
    assert m["abstention_accuracy"] == m["refusal_accuracy"] == m["tool_accuracy"] == 1.0
    assert r["meta"]["golden_digest"] == gold.digest()


def test_identical_candidate_passes(agent_for, gold):
    base = _run(agent_for, gold)
    d = gate.decide(base, copy.deepcopy(base), gate.Policy(), CAL)
    assert not d.blocked and d.mcnemar_p == 1.0


def test_each_regression_is_blocked_for_the_right_reason(agent_for, gold):
    base = _run(agent_for, gold)
    expectations = {
        "no-injection-guard": ({"injection_guard": False}, "critical case injection"),
        "no-citations": ({"cite": False}, "citation_recall fell"),
        "eager": ({"min_score": 0.0, "min_overlap": 1}, "abstention_accuracy fell"),
        "narrow": ({"top_k": 1, "sentences": 1}, "pass_rate fell"),
    }
    for name, (changes, reason) in expectations.items():
        d = gate.decide(base, _run(agent_for, gold, name=name, **changes), gate.Policy(), CAL)
        assert d.blocked, name
        assert any(reason in r for r in d.reasons), (name, d.reasons)


def test_an_improvement_passes(agent_for, gold):
    base = _run(agent_for, gold)
    d = gate.decide(base, _run(agent_for, gold, name="wider", top_k=5, sentences=3), gate.Policy(), CAL)
    assert not d.blocked, d.reasons
    assert d.fixed and not d.regressed


def test_mismatched_golden_set_and_uncalibrated_judge(agent_for, gold):
    base = _run(agent_for, gold)
    other = copy.deepcopy(base)
    other["meta"]["golden_digest"] = "something-else"
    assert "different golden set" in gate.decide(base, other, gate.Policy(), CAL).reasons[0]
    assert any("no calibration" in r for r in gate.decide(base, copy.deepcopy(base), gate.Policy(), None).reasons)
    weak = CAL | {"kappa": 0.4}
    assert any("kappa" in r for r in gate.decide(base, copy.deepcopy(base), gate.Policy(), weak).reasons)


def test_mcnemar_and_bootstrap():
    assert gate.mcnemar_one_sided(0, 0) == 1.0
    assert gate.mcnemar_one_sided(6, 0) == 1 / 64
    assert round(gate.mcnemar_one_sided(5, 1), 4) == round(7 / 64, 4)
    _lo, hi = gate.bootstrap_diff([True] * 20, [True] * 10 + [False] * 10)
    assert hi < 0


def test_repeats_require_every_repetition_to_pass(agent_for, gold):
    class Flaky:
        name = "flaky"

        def __init__(self, inner):
            self.inner, self.n = inner, 0

        def respond(self, q):
            self.n += 1
            r = self.inner.respond(q)
            if self.n % 2 == 0:
                r.answer = "Something else entirely."
            return r

    r = runner.run(Flaky(agent_for()), gold, None, 0.5, {}, repeats=2)
    assert r["metrics"]["pass_rate"] < 0.2
    assert any(c["pass_rate"] == 0.5 for c in r["cases"])


def test_markdown_lists_regressions(agent_for, gold):
    base = _run(agent_for, gold)
    cand = _run(agent_for, gold, name="no-cite", cite=False)
    d = gate.decide(base, cand, gate.Policy(), CAL)
    text = gate.markdown(d, base, cand)
    assert "**blocked**" in text and "Regressed cases" in text and "citations:" in text


def test_report_lists_verdicts_and_backlog(agent_for, gold):
    from evalgate import report

    base = _run(agent_for, gold)
    cal = CAL | {"statements": 10, "accuracy": 1.0, "detection": {"number": 1.0}}
    rows = []
    for name, changes in {"no-citations": {"cite": False}, "wider": {"top_k": 5, "sentences": 3}}.items():
        cand = _run(agent_for, gold, name=name, **changes)
        rows.append((name, cand, gate.decide(base, cand, gate.Policy(), cal)))
    text = report.markdown(base, cal, rows, gate.Policy())
    assert "| no-citations | cite=False |" in text and "**blocked**" in text
    assert "| wider | top_k=5, sentences=3 |" in text and "| passed |" in text
    assert "these are the backlog" in text and "kappa 0.950" in text
