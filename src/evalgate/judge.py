"""The judge: a separate model that checks whether each claim in an answer is supported.

The judge is a natural-language-inference cross-encoder, not the model that wrote the answer, so
the agent cannot grade itself. Every sentence of an answer is checked against the agent's
contexts (the retrieved text and tool results); a sentence is supported when the judge's
entailment probability for some context passes the threshold. An answer is faithful when all its
sentences are supported. The threshold, and whether the judge can be trusted at all, come from
calibration on statements with known labels (`calibration.py`).

`Judge` is a protocol: an LLM judge behind an OpenAI-compatible endpoint (`llm.py`) fits it too.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from evalgate.kb import sentences

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-xsmall"
SOURCES = re.compile(r"\s*\(Sources?:[^)]*\)\s*$")


class Judge(Protocol):
    name: str

    def entailment(self, premises: Sequence[str], hypothesis: str) -> float: ...


class NLIJudge:
    def __init__(self, model: str = DEFAULT_MODEL, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = model
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail = labels["entailment"]

    @lru_cache(maxsize=4096)  # noqa: B019 - the judge lives as long as the run
    def _score(self, premise: str, hypothesis: str) -> float:
        with self._torch.no_grad():
            enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
            probs = self._torch.softmax(self.model(**enc).logits, dim=-1)[0]
        return float(probs[self.entail])

    def entailment(self, premises: Sequence[str], hypothesis: str) -> float:
        """Best support for the claim from any sentence of the contexts, or two adjacent ones.
        Small NLI models judge short premises reliably and long ones poorly (a sentence is
        not entailed by a paragraph that contains it word for word), so contexts are split."""
        windows: list[str] = []
        for p in premises:
            parts = sentences(p) or [p]
            windows += parts + [f"{a} {b}" for a, b in itertools.pairwise(parts)]
        return max((self._score(w, hypothesis) for w in dict.fromkeys(windows)), default=0.0)


@dataclass
class Verdict:
    faithful: bool
    supported: int
    sentences: int
    weakest: float
    unsupported: list[str]


def faithfulness(judge: Judge, answer: str, contexts: Sequence[str], threshold: float) -> Verdict:
    claims = sentences(SOURCES.sub("", answer))
    if not claims:
        return Verdict(True, 0, 0, 1.0, [])
    scores = [judge.entailment(contexts, c) for c in claims]
    bad = [c for c, s in zip(claims, scores, strict=True) if s < threshold]
    return Verdict(not bad, len(claims) - len(bad), len(claims), min(scores), bad)
