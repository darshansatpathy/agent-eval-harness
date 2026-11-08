from __future__ import annotations

import json
import os

import pytest
from conftest import ROOT

from evalgate.cli import main


@pytest.fixture(autouse=True)
def _cwd():
    old = os.getcwd()
    os.chdir(ROOT)
    yield
    os.chdir(old)


def test_run_and_gate_without_the_judge(tmp_path, capsys):
    base, cand = tmp_path / "b.json", tmp_path / "c.json"
    assert main(["run", "--judge", "none", "--out", str(base)]) == 0
    assert main(["run", "--judge", "none", "--agent", "configs/variants/no-citations.yaml", "--out", str(cand)]) == 0
    assert main(["gate", "--baseline", str(base), "--candidate", str(base), "--summary", str(tmp_path / "s.md")]) == 0
    assert main(["gate", "--baseline", str(base), "--candidate", str(cand)]) == 1
    assert "citation_recall fell" in capsys.readouterr().out
    assert json.loads(base.read_text())["meta"]["judge"] is None


def test_committed_baseline_matches_the_golden_set():
    from evalgate import golden, runner

    base = runner.load(ROOT / "baselines" / "main.json")
    assert base["meta"]["golden_digest"] == golden.load(ROOT / "golden" / "support.yaml").digest()
    cal = json.loads((ROOT / "judge-calibration.json").read_text())
    assert cal["judge"] == base["meta"]["judge"] and cal["kappa"] >= 0.6
