from __future__ import annotations

import json

import httpx
from conftest import WordJudge

from evalgate import calibration, llm
from evalgate.agent import Tools


def test_statements_have_every_kind(chunks):
    st = calibration.build(chunks)
    kinds = {s.kind for s in st}
    assert kinds == {"original", "number", "negation", "entity", "unrelated"}
    assert all(s.supported == (s.kind == "original") for s in st)
    assert all(s.hypothesis != s.premise for s in st if s.kind == "unrelated")


def test_kappa():
    assert calibration.kappa([True, False] * 5, [True, False] * 5) == 1.0
    assert abs(calibration.kappa([True, False] * 5, [False, True] * 5) + 1) < 1e-9


def test_calibration_picks_a_threshold(chunks):
    cal = calibration.calibrate(WordJudge(), calibration.build(chunks))
    assert 0 < cal.threshold < 1 and cal.statements > 150 and cal.sweep


def _transport(replies):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=replies.pop(0))

    return httpx.MockTransport(handler), seen


def test_openai_compatible_agent_with_a_tool_round(chunks, tmp_path):
    tool_turn = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "t1", "type": "function", "function": {"name": "get_invoice", "arguments": '{"invoice_id": "INV-204118"}'}}
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 400, "completion_tokens": 20},
    }
    final = {
        "choices": [{"message": {"role": "assistant", "content": "Invoice INV-204118 is paid. (Sources: billing-cycle)"}}],
        "usage": {"prompt_tokens": 450, "completion_tokens": 15},
    }
    transport, seen = _transport([tool_turn, final])
    tools = Tools({"INV-204118": {"status": "paid"}}, {})
    agent = llm.OpenAICompatibleAgent(llm.Chat("http://llm", "m", "k", transport=transport), chunks, tools, price_in=3, price_out=15)
    r = agent.respond("What is the status of invoice INV-204118?")
    assert r.tool_calls[0].result == {"status": "paid"} and r.citations == ["billing-cycle"]
    assert r.input_tokens == 850 and abs(r.cost_usd - (850 * 3 + 35 * 15) / 1e6) < 1e-12
    assert seen[1]["messages"][-1]["role"] == "tool"
    assert seen[0]["temperature"] == 0 and "excerpts" in seen[0]["messages"][1]["content"]


def test_openai_compatible_judge():
    transport, _ = _transport([{"choices": [{"message": {"content": "YES"}}]}, {"choices": [{"message": {"content": "No."}}]}])
    j = llm.OpenAICompatibleJudge(llm.Chat("http://llm", "judge-model", "k", transport=transport))
    assert j.entailment(["a"], "a") == 1.0 and j.entailment(["a"], "b") == 0.0
    assert j.name == "llm:judge-model"
