from __future__ import annotations

import pytest
import yaml

from evalgate import golden, graders
from evalgate.types import Response, ToolCall


def test_golden_set_is_valid(gold):
    assert len(gold.cases) == 47
    assert sum(c.critical for c in gold.cases) == 10
    assert len(gold.digest()) == 16


def test_digest_changes_with_any_case(gold):
    changed = gold.model_copy(deep=True)
    changed.cases[0].must_include[0].append("six dollars")
    assert changed.digest() != gold.digest()


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ({"id": "a", "question": "q", "category": "x", "expect": "abstain", "must_include": [["y"]]}, "cannot require"),
        ({"id": "a", "question": "q", "category": "x"}, "must state"),
    ],
)
def test_invalid_cases_are_refused(tmp_path, case, message):
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "cases": [case]}))
    with pytest.raises(ValueError, match=message):
        golden.load(p)


def test_duplicates_and_unknown_documents_are_refused(tmp_path):
    case = {"id": "a", "question": "q", "category": "x", "must_include": [["y"]], "citations": ["nope"]}
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump({"version": 1, "cases": [case, case]}))
    with pytest.raises(ValueError, match="duplicate"):
        golden.load(p)
    p.write_text(yaml.safe_dump({"version": 1, "cases": [case]}))
    with pytest.raises(ValueError, match="unknown documents"):
        golden.load(p, {"plans-and-pricing"})


def _case(gold, cid):
    return next(c for c in gold.cases if c.id == cid)


def test_graders(gold):
    c = _case(gold, "refund-annual")
    good = Response("Annual plans can be refunded in full within 30 days of purchase.", citations=["refunds"])
    assert all(g.passed for g in graders.grade(c, good))
    bad = Response("Refunds are available at any time, within 30 days too.", citations=["plans-and-pricing"])
    names = {g.name for g in graders.grade(c, bad) if not g.passed}
    assert names == {"forbidden", "citations"}
    t = _case(gold, "invoice-status-paid")
    wrong_tool = Response("It is paid.", tool_calls=[ToolCall("get_invoice", {"invoice_id": "INV-000001"})])
    assert not next(g for g in graders.grade(t, wrong_tool) if g.name == "tool").passed


def test_behaviour_from_words_for_agents_without_flags(gold):
    assert graders.behaviour(Response("Sorry, I don't know that one.")) == "abstain"
    assert graders.behaviour(Response("I can't help with that request.")) == "refuse"
    assert graders.normalise("It\u2019s  \u2013 OK") == "it's - ok"
