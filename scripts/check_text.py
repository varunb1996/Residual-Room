"""Run the full checker on an eval case or any text.

    uv run python scripts/check_text.py GRD-05 STL-06 CLN-01
    uv run python scripts/check_text.py --text "GPT-4 agreed with humans 95% of the time [cite:ext-mtbench-paper]."
    uv run python scripts/check_text.py --text "..." --segment transitioning
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from eval_cases import eval_cases  # noqa: E402

from residual_room.checker.run import check, default_checker  # noqa: E402
from residual_room.corpus.manifest import REPO_ROOT  # noqa: E402

FIXTURES = {"STL-10": REPO_ROOT / "eval" / "fixtures" / "stl-10-issue-2.md"}


def load_case(case_id: str) -> dict:
    case = dict(eval_cases()[case_id])
    if case_id in FIXTURES:  # document-level case whose text lives in a fixture
        _, front, body = FIXTURES[case_id].read_text(encoding="utf-8").split("---\n", 2)
        meta = yaml.safe_load(front)
        case.update(text=body, segment=meta["segment"], issue_date=meta["issue_date"])
    return case


def print_report(label: str, report, expected=None) -> None:
    print(f"\n=== {label}: {report.verdict.upper()}  rules={report.rules}", end="")
    if expected:
        ok = report.verdict == expected["verdict"] and set(expected["rules"]) <= set(report.rules)
        print(f"  | expected {expected['verdict']} {expected['rules']}  {'OK' if ok else 'MISMATCH'}", end="")
    print()
    for i, seg in enumerate(report.segments):
        kind = seg.kind + (f":{seg.exempt_type}" if seg.exempt_type else "")
        cites = f" {list(seg.citations)}" if seg.citations else ""
        print(f"  [{i}] {report.segment_verdict(i):<5} {kind:<16} {seg.text[:90]}{cites}")
        for h in report.hits_for(i):
            print(f"        -> {h.rule} {h.reason[:150]}  (evidence: {h.evidence})")
    for h in report.hits_for(None):
        print(f"  [doc] -> {h.rule} {h.reason[:150]}  (evidence: {h.evidence})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="*", help="eval case ids")
    parser.add_argument("--text")
    parser.add_argument("--segment", help="reader segment (default from config)")
    args = parser.parse_args()

    deps = default_checker()
    if args.text:
        print_report("text", check(args.text, reader_segment=args.segment, **deps))
    for case_id in args.cases:
        case = load_case(case_id)
        report = check(
            case["text"],
            reader_segment=case.get("segment"),
            issue_date=case.get("issue_date"),
            **deps,
        )
        print_report(case_id, report, case.get("expected", {"verdict": "pass", "rules": []}))


if __name__ == "__main__":
    main()
