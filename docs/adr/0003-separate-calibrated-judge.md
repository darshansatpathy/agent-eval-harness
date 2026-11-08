# 3. A separate judge, calibrated before it is trusted

Status: accepted

## Context

A judge that shares a model with the agent shares its blind spots, and an uncalibrated judge's threshold is a guess. The gate's verdicts are only as good as the judge's.

## Decision

- The default judge is a small NLI cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`), unrelated to any agent model, deterministic on CPU and free to run in CI. An OpenAI-compatible endpoint can be used instead (`--judge llm:<base url>#<model>`).
- Each answer sentence is scored against single context sentences and adjacent pairs, and the maximum is taken. NLI models trained on short premises lose entailment on whole paragraphs.
- `evg calibrate` builds statements with known labels from the knowledge base: originals, and copies with a changed number, a negation, a swapped entity, or an unrelated premise. It sweeps the threshold, records Cohen's kappa and per-kind detection in `judge-calibration.json`.
- The gate refuses to run with a judge whose calibration is missing, belongs to a different judge, or has kappa below `min_judge_kappa` (0.6).

## Consequences

Calibration statements are synthetic, so they test the judge on the errors we chose to generate. Swapped entities are the weakest kind and are reported separately.
