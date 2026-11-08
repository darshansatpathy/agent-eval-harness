.PHONY: install check test judge-test calibrate baseline run gate variants

install:
	pip install -e ".[judge,dev]"

check:
	ruff check .
	ruff format --check .
	mypy
	pytest -m "not judge" --cov --cov-report=term-missing

judge-test:
	pytest -m judge

calibrate:
	evg calibrate

baseline:
	evg run --agent configs/agent.yaml --out baselines/main.json

run:
	evg run --agent configs/agent.yaml --out runs/candidate.json

gate: run
	evg gate --baseline baselines/main.json --candidate runs/candidate.json

variants:
	evg variants --report docs/eval-report.md
