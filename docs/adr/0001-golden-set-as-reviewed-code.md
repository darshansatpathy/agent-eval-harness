# 1. The golden set is code: reviewed, versioned and pinned by digest

Status: accepted

## Context

An evaluation is only as good as its cases, and the easiest way to make a regression disappear is to edit the case that caught it. A gate that compares runs over different case sets compares nothing.

## Decision

- Cases live in `golden/support.yaml`, validated with pydantic on load: unique ids, a stated expected behaviour, content checks that agree with it (an `abstain` case cannot require content), and citations that name documents that exist.
- Every run records the golden set's digest. The gate refuses to compare a candidate with a baseline run over a different digest.
- Changing a case means regenerating `baselines/main.json` in the same pull request, so the reviewer sees the case change and the baseline change together.
- Cases are tagged `critical` where one failure must block on its own (injection, refusals, wrong account data).

## Consequences

The baseline file churns whenever the set grows. That is the point: growth is visible in review.
