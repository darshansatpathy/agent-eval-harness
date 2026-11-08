# 2. Deterministic graders first, the judge only for faithfulness

Status: accepted

## Context

Model judges are slow, have their own error rate, and drift when the judge changes. Most of what a support agent must get right is checkable without one: the behaviour (answer, abstain, refuse), required facts, forbidden phrases, citations and tool calls.

## Decision

- Each case is graded by deterministic checks: behaviour, `must_include` (any-of groups), `must_not_include`, citation recall, and expected tool calls with arguments.
- The judge answers one question only: is every sentence of the answer supported by the retrieved context? Unsupported sentences are listed in the run, so a failure is explainable.
- A case passes only when every grader and the judge agree. With `--repeats k`, a case passes only if all k runs pass (pass^k), which catches flaky behaviour.

## Consequences

Content checks are literal, so a correct paraphrase can fail. Cases use any-of groups for the common paraphrases, and a failing case is reviewed rather than loosened blindly.
