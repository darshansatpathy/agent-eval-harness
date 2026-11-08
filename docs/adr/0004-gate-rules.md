# 4. The gate: hard rules first, then tolerances, then a significance test

Status: accepted

## Context

A pass-rate threshold alone either blocks noise or misses real damage: on 47 cases one case is about 2%, and a change that breaks the injection guard can leave the pass rate almost unchanged.

## Decision

`evg gate` blocks a candidate when any of these holds, and lists every reason:

1. It was run over a different golden set, or the judge is uncalibrated.
2. Any `critical` case fails.
3. A metric falls by more than its tolerance in `gate.yaml` (pass rate 3 points, faithfulness 2, citation recall 5; abstention, refusal and tool accuracy may not fall at all).
4. p95 latency rises by more than 50%.
5. More cases regressed than were fixed, and a one-sided McNemar test on the discordant pairs is significant at alpha 0.05.

A bootstrap interval for the pass-rate change is reported alongside. Regressed and fixed cases are listed with the reason for each regression.

## Consequences

Tolerances are policy, not statistics, and live in a reviewed file. Moving the baseline is an explicit act (`make baseline` in a pull request), never a side effect of a green run.
