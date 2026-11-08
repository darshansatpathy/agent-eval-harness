# Changelog

## 0.1.0

- Golden set of 47 cases in six categories, ten critical, validated on load and pinned by digest.
- Deterministic graders: behaviour, required and forbidden content, citations, tool calls; pass^k with `--repeats`.
- NLI judge for faithfulness, scored over sentence windows; calibration on 185 labelled statements (kappa 0.978).
- Gate: critical cases, tolerances, latency, one-sided McNemar test, bootstrap interval; Markdown verdict for the CI job summary.
- Reference agent with BM25 retrieval, tools, refusal and injection guards; OpenAI-compatible agent and judge adapters.
- `evg calibrate`, `run`, `gate` and `variants`; evaluation report over five candidate changes.
