# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # for the judge
make install
make check        # ruff, mypy, fast tests
make judge-test   # tests that load the NLI judge (downloads about 280 MB once)
make gate         # your agent against the baseline, exactly as CI runs it
```

## Changing the agent

Change `configs/agent.yaml` or the code, then `make gate`. If the gate blocks, the output lists each reason and every regressed case with why it failed. Fix the cause; do not loosen `gate.yaml` in the same pull request.

## Changing the golden set

- One case per behaviour, with an id that says what it checks.
- `must_include` is a list of any-of groups: every group must match, any phrase in a group will do. Add the common paraphrases.
- Tag a case `critical` only when one failure should block a release on its own.
- Regenerate the baseline in the same pull request: `make baseline`. The reviewer sees the cases and the baseline move together.

## Changing the judge

Run `evg calibrate --judge <model>` and commit `judge-calibration.json`. The gate refuses a judge below kappa 0.6 and one whose calibration belongs to another model.

## Commits

Small, focused commits with messages in the imperative. Update `CHANGELOG.md` for anything a user would notice.
