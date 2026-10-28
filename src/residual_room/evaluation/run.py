"""`make eval`: run eval cases through the full checker and write a report.

    uv run python -m residual_room.evaluation.run --split test
    uv run python -m residual_room.evaluation.run --split all --committable-only   # CI subset
    uv run python -m residual_room.evaluation.run --cases GRD-05 STL-06

Needs `make fetch` + `make ingest` first (several cases cite sources that can't
be committed) and the judge's API key in .env.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

from residual_room.checker.run import check, default_checker
from residual_room.corpus.embed import MODEL_NAME as EMBED_MODEL
from residual_room.evaluation.cases import EVAL_DIR, EvalCase, load_cases
from residual_room.evaluation.metrics import CaseResult, summarize
from residual_room.evaluation.report import write_report
from residual_room.evaluation.split import load_split
from residual_room.llm.client import ResponseCache, load_role

RESULTS_DIR = EVAL_DIR / "results"


def run_case(case: EvalCase, split: str, deps: dict) -> CaseResult:
    report = check(case.text, reader_segment=case.segment, issue_date=case.issue_date, **deps)
    seg_text = lambda i: report.segments[i].text if i is not None else "(document)"  # noqa: E731
    return CaseResult(
        id=case.id,
        split=split,
        hard=case.hard,
        expected_verdict=case.expected_verdict,
        expected_rules=list(case.expected_rules),
        verdict=report.verdict,
        rules=report.rules,
        hits=[
            {"rule": h.rule, "reason": h.reason, "segment_text": seg_text(h.segment), "evidence": h.evidence}
            for h in report.hits
        ],
        segments=[
            {"kind": s.kind, "exempt_type": s.exempt_type, "text": s.text, "citations": list(s.citations)}
            for s in report.segments
        ],
    )


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return sha + (" (uncommitted changes)" if dirty else "")
    except OSError:
        return "unknown"


def main() -> int:
    # Windows consoles default to cp1252, which can't print the ⚠ small-sample flag.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="all")
    parser.add_argument("--cases", nargs="*", help="only these case ids")
    parser.add_argument("--committable-only", action="store_true", help="skip cases citing non-committable sources")
    parser.add_argument("--no-cache", action="store_true", help="force fresh judge calls")
    parser.add_argument("--corpus-version", default="corpus-v1")
    args = parser.parse_args()

    cases, split = load_cases(), load_split()
    split_of = {cid: name for name, ids in split.items() for cid in ids}
    ids = args.cases or [cid for cid in sorted(cases) if args.split == "all" or split_of[cid] == args.split]
    if args.committable_only:
        ids = [cid for cid in ids if cases[cid].committable]

    deps = default_checker(args.corpus_version)
    if args.no_cache:
        deps["judge"] = load_role("judge", cache=ResponseCache(enabled=False))
    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "split": "custom" if args.cases else args.split,
        "cases": len(ids),
        "judge_model": deps["judge"].model,
        "generator_model": load_role("generator").model,
        "embedding_model": EMBED_MODEL,
        "corpus_version": args.corpus_version,
        "git_commit": git_commit(),
        "cache": "off" if args.no_cache else "on",
        "committable_only": args.committable_only,
    }

    results, start = [], time.time()
    for n, cid in enumerate(ids, 1):
        r = run_case(cases[cid], split_of[cid], deps)
        results.append(r)
        mark = "ok " if r.lenient_ok else "XX "
        print(f"[{n:>2}/{len(ids)}] {mark}{cid:<7} expected {r.expected_verdict} {r.expected_rules} "
              f"-> {r.verdict} {r.rules}", flush=True)

    meta["duration_s"] = round(time.time() - start)
    md, js = write_report(results, meta, RESULTS_DIR)
    s = summarize(results)
    print(f"\nlenient {s['lenient']} | strict {s['strict']} | pass FP {s['pass_fp']} | hard {s['hard']}")
    print(f"report: {md.relative_to(EVAL_DIR.parent)}\nraw:    {js.relative_to(EVAL_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
