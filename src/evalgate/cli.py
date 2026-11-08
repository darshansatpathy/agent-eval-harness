"""evg: run an agent over a golden set, calibrate the judge, and gate a candidate against a baseline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

from evalgate import __version__

ROOT = Path(".")


def _agent(path: Path):  # type: ignore[no-untyped-def]
    doc = yaml.safe_load(path.read_text())
    if doc.get("kind") == "openai-compatible":
        from evalgate import kb, llm
        from evalgate.agent import Tools

        chat = llm.Chat(doc["base_url"], doc["model"])
        agent = llm.OpenAICompatibleAgent(
            chat,
            kb.load(ROOT / "kb"),
            Tools.from_file(ROOT / "fixtures" / "records.yaml"),
            doc.get("top_k", 4),
            doc.get("price_in", 0),
            doc.get("price_out", 0),
        )
        return agent, doc
    from evalgate.agent import AgentConfig, build

    config = AgentConfig.model_validate(doc)
    return build(config, ROOT), config.model_dump()


def _judge(name: str):  # type: ignore[no-untyped-def]
    if name == "none":
        return None
    if name.startswith("llm:"):
        from evalgate import llm

        base, model = name[4:].rsplit("#", 1)
        return llm.OpenAICompatibleJudge(llm.Chat(base, model))
    from evalgate.judge import NLIJudge

    return NLIJudge(name)


def _threshold(path: Path, judge_name: str | None) -> tuple[float, dict | None]:  # type: ignore[type-arg]
    if judge_name is None or not path.exists():
        return 0.5, None
    cal = json.loads(path.read_text())
    return (cal["threshold"], cal) if cal.get("judge") == judge_name else (0.5, None)


def cmd_calibrate(args: argparse.Namespace) -> int:
    from evalgate import calibration, kb

    judge = _judge(args.judge)
    assert judge is not None, "calibration needs a judge"
    cal = calibration.calibrate(judge, calibration.build(kb.load(ROOT / "kb")))
    Path(args.out).write_text(json.dumps(asdict(cal), indent=1) + "\n")
    print(f"{cal.judge}: threshold {cal.threshold:.2f}, kappa {cal.kappa:.3f}, accuracy {cal.accuracy:.3f} on {cal.statements} statements")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from evalgate import golden, kb, runner

    agent, config = _agent(Path(args.agent))
    g = golden.load(Path(args.golden), {c.doc for c in kb.load(ROOT / "kb")})
    judge = _judge(args.judge)
    threshold, _ = _threshold(Path(args.calibration), None if judge is None else judge.name)
    result = runner.run(agent, g, judge, threshold, config, repeats=args.repeats)
    runner.save(result, Path(args.out))
    m = result["metrics"]
    print(
        f"{agent.name}: pass rate {m['pass_rate']:.3f} ({sum(c['passed'] for c in result['cases'])}/{len(result['cases'])}), faithfulness {m['faithfulness']:.3f}"
    )
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    from evalgate import gate, runner

    base, cand = runner.load(Path(args.baseline)), runner.load(Path(args.candidate))
    policy = gate.Policy.from_file(Path(args.policy)) if Path(args.policy).exists() else gate.Policy()
    cal = json.loads(Path(args.calibration).read_text()) if Path(args.calibration).exists() else None
    d = gate.decide(base, cand, policy, cal)
    text = gate.markdown(d, base, cand)
    if args.summary:
        Path(args.summary).write_text(text)
    print(text)
    return 1 if d.blocked else 0


def cmd_variants(args: argparse.Namespace) -> int:
    """Run each variant config, gate it against the baseline, and write the report."""
    from evalgate import gate, golden, kb, report, runner

    base = runner.load(Path(args.baseline))
    policy = gate.Policy.from_file(Path(args.policy)) if Path(args.policy).exists() else gate.Policy()
    cal = json.loads(Path(args.calibration).read_text())
    g = golden.load(Path(args.golden), {c.doc for c in kb.load(ROOT / "kb")})
    judge = _judge(args.judge)
    threshold, _ = _threshold(Path(args.calibration), None if judge is None else judge.name)
    rows = []
    for path in sorted(Path(args.variants).glob("*.yaml")):
        agent, config = _agent(path)
        cand = runner.run(agent, g, judge, threshold, config)
        d = gate.decide(base, cand, policy, cal)
        rows.append((path.stem, cand, d))
        print(f"{path.stem:20} {'BLOCKED' if d.blocked else 'passed':8} pass rate {cand['metrics']['pass_rate']:.3f}")
    Path(args.report).write_text(report.markdown(base, cal, rows, policy))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="evg", description=__doc__)
    p.add_argument("--version", action="version", version=f"evg {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("calibrate", help="measure the judge against statements with known labels")
    s.add_argument("--judge", default="cross-encoder/nli-deberta-v3-xsmall")
    s.add_argument("--out", default="judge-calibration.json")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("run", help="run an agent over the golden set")
    s.add_argument("--agent", default="configs/agent.yaml")
    s.add_argument("--golden", default="golden/support.yaml")
    s.add_argument("--judge", default="cross-encoder/nli-deberta-v3-xsmall", help="model name, llm:<base url>#<model>, or none")
    s.add_argument("--calibration", default="judge-calibration.json")
    s.add_argument("--repeats", type=int, default=1)
    s.add_argument("--out", default="runs/candidate.json")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("gate", help="compare a candidate run with the baseline; exit 1 if it is blocked")
    s.add_argument("--baseline", default="baselines/main.json")
    s.add_argument("--candidate", default="runs/candidate.json")
    s.add_argument("--policy", default="gate.yaml")
    s.add_argument("--calibration", default="judge-calibration.json")
    s.add_argument("--summary")
    s.set_defaults(fn=cmd_gate)

    s = sub.add_parser("variants", help="gate every config in a folder against the baseline and write a report")
    s.add_argument("--variants", default="configs/variants")
    s.add_argument("--baseline", default="baselines/main.json")
    s.add_argument("--golden", default="golden/support.yaml")
    s.add_argument("--policy", default="gate.yaml")
    s.add_argument("--judge", default="cross-encoder/nli-deberta-v3-xsmall")
    s.add_argument("--calibration", default="judge-calibration.json")
    s.add_argument("--report", default="docs/eval-report.md")
    s.set_defaults(fn=cmd_variants)

    args = p.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
