"""The NLI judge itself. Downloads the model once (about 280 MB); run with -m judge."""

from __future__ import annotations

import pytest

from evalgate import calibration
from evalgate.judge import NLIJudge, faithfulness

pytestmark = pytest.mark.judge


@pytest.fixture(scope="module")
def judge():
    return NLIJudge()


def test_sentence_inside_a_paragraph_is_entailed(judge):
    para = "If a payment fails, Nimbus Notes retries the card after 3 days and again after 7 days. The account moves to read-only mode 14 days later."
    assert judge.entailment([para], "Nimbus Notes retries the card after 3 days.") > 0.9
    assert judge.entailment([para], "Nimbus Notes retries the card after 9 days.") < 0.1


def test_faithfulness_flags_the_unsupported_sentence(judge):
    ctx = ["Monthly plans are not refunded. Annual plans can be refunded within 30 days."]
    v = faithfulness(judge, "Annual plans can be refunded within 30 days. Monthly plans are refunded too. (Sources: refunds)", ctx, 0.9)
    assert not v.faithful and v.unsupported == ["Monthly plans are refunded too."]


def test_calibration_meets_the_policy(judge, chunks):
    cal = calibration.calibrate(judge, calibration.build(chunks))
    assert cal.kappa > 0.85 and cal.detection["number"] > 0.9
