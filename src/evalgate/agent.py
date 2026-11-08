"""The reference support agent the harness is demonstrated on.

It answers questions about a product from its knowledge base, calls tools for account and invoice
lookups, abstains when the knowledge base has no answer, and refuses requests it must not serve.
It is extractive and deterministic (BM25 retrieval, sentence selection), so the harness's
behaviour on it is reproducible in CI; `llm.OpenAICompatibleAgent` is the same contract backed by a
language model.

Every behaviour that matters is a configuration switch, so a regression can be introduced on
purpose and the gate shown to catch it.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from evalgate import kb
from evalgate.types import Response, ToolCall

INVOICE = re.compile(r"\bINV-\d{6}\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
INJECTION = re.compile(r"ignore (all |your )?(previous|prior) instructions|system note for ai|you are now|disregard the above", re.I)
REFUSE_PATTERNS = [
    re.compile(r"\bpassword\b.*\b[\w.+-]+@", re.I),
    re.compile(r"\b(notes|data|files) of (the )?(user|customer|account)\b", re.I),
    re.compile(r"\b(poem|song|story|joke|essay)\b", re.I),
]
ABSTAIN = "I don't know. The help centre does not cover that."
REFUSE = "I can't help with that. I can answer questions about Nimbus Notes and look up your own invoices and account."


class AgentConfig(BaseModel):
    name: str = "reference"
    top_k: int = 3
    sentences: int = 2
    min_score: float = 4.0  # BM25 score below which the agent abstains
    min_overlap: int = 2  # content words a chosen sentence must share with the question
    cite: bool = True
    use_tools: bool = True
    injection_guard: bool = True
    refusal_guard: bool = True

    @classmethod
    def from_file(cls, path: Path) -> AgentConfig:
        return cls.model_validate(yaml.safe_load(path.read_text()))


@dataclass
class Tools:
    invoices: dict[str, dict[str, Any]]
    accounts: dict[str, dict[str, Any]]

    @classmethod
    def from_file(cls, path: Path) -> Tools:
        doc = yaml.safe_load(path.read_text())
        return cls(doc["invoices"], doc["accounts"])

    def get_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        return self.invoices.get(invoice_id)

    def get_account(self, email: str) -> dict[str, Any] | None:
        return self.accounts.get(email.lower())


class ReferenceAgent:
    def __init__(self, config: AgentConfig, chunks: list[kb.Chunk], tools: Tools) -> None:
        self.config, self.chunks, self.tools = config, chunks, tools
        self.name = config.name
        self.bm25 = BM25Okapi([kb.tokens(c.title + " " + c.text) for c in chunks])

    def respond(self, question: str) -> Response:
        start = time.perf_counter()
        r = self._respond(question)
        r.latency_ms = (time.perf_counter() - start) * 1000
        return r

    def _respond(self, question: str) -> Response:
        c = self.config
        if c.refusal_guard and any(p.search(question) for p in REFUSE_PATTERNS):
            return Response(REFUSE, refused=True)
        if c.use_tools:
            if m := INVOICE.search(question):
                return self._invoice(m.group(0))
            if m := EMAIL.search(question):
                return self._account(m.group(0), question)
        q = kb.tokens(question)
        scores = self.bm25.get_scores(q)
        ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[: c.top_k]
        if not ranked or scores[ranked[0]] < c.min_score:
            return Response(ABSTAIN, abstained=True)
        qset = set(q)
        needed = min(c.min_overlap, max(1, (len(qset) + 1) // 2))  # short questions share fewer words
        candidates = []
        for rank, i in enumerate(ranked):
            for j, s in enumerate(kb.sentences(self.chunks[i].text)):
                if c.injection_guard and INJECTION.search(s):
                    continue  # retrieved text is data, never instructions
                overlap = len(qset & set(kb.tokens(s)))
                if overlap >= needed:
                    candidates.append((overlap + float(scores[i]) / 10 - 0.3 * rank, rank, j, i, s))
        picked = sorted(candidates, key=lambda x: -x[0])[: c.sentences]
        if not picked:
            return Response(ABSTAIN, abstained=True)
        picked.sort(key=lambda x: (x[1], x[2]))  # reading order
        docs = list(dict.fromkeys(self.chunks[x[3]].doc for x in picked))
        answer = " ".join(x[4] for x in picked)
        if c.cite:
            answer += " (Sources: " + ", ".join(docs) + ")"
        return Response(answer, citations=docs if c.cite else [], contexts=[self.chunks[i].text for i in ranked])

    def _invoice(self, invoice_id: str) -> Response:
        rec = self.tools.get_invoice(invoice_id)
        call = ToolCall("get_invoice", {"invoice_id": invoice_id}, rec)
        if rec is None:
            text = f"I could not find invoice {invoice_id}. Please check the number on your billing email."
            return Response(
                text, tool_calls=[call], contexts=[f"get_invoice found no invoice {invoice_id}. Invoice {invoice_id} could not be found."]
            )
        fact = f"Invoice {invoice_id} for {rec['amount']} ({rec['plan']}, {rec['date']}) is {rec['status']}."
        return Response(fact, tool_calls=[call], contexts=[fact])

    def _account(self, email: str, question: str) -> Response:
        rec = self.tools.get_account(email)
        call = ToolCall("get_account", {"email": email}, rec)
        if rec is None:
            return Response(
                f"I could not find an account for {email}.", tool_calls=[call], contexts=[f"An account for {email} could not be found."]
            )
        if "device" in question.lower():
            fact = f"The account {email} has {rec['devices']} devices."
        else:
            fact = f"The account {email} is on the {rec['plan']} plan, billed {rec['billing']}."
        return Response(fact, tool_calls=[call], contexts=[fact])


def build(config: AgentConfig, root: Path) -> ReferenceAgent:
    return ReferenceAgent(config, kb.load(root / "kb"), Tools.from_file(root / "fixtures" / "records.yaml"))
