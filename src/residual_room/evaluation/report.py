"""Write an eval run to eval/results/: a markdown report and the raw JSON."""

import json
from dataclasses import asdict
from pathlib import Path

from residual_room.evaluation.metrics import CaseResult, summarize


def write_report(results: list[CaseResult], meta: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{meta['started_at'][:16].replace(':', '')}-{meta['split']}"
    md_path, json_path = out_dir / f"{stem}.md", out_dir / f"{stem}.json"

    json_path.write_text(
        json.dumps({"meta": meta, "results": [asdict(r) for r in results]}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_path.write_text(render_markdown(results, meta), encoding="utf-8", newline="\n")
    return md_path, json_path


def render_markdown(results: list[CaseResult], meta: dict) -> str:
    lines = [f"# Eval run — {meta['started_at'][:10]} — split: {meta['split']}", ""]
    lines += [f"- **{k}:** {v}" for k, v in meta.items()]
    lines += ["", "Scoring: *lenient* = verdict matches and expected rules ⊆ predicted (the agreed rule); "
              "*strict* = exactly the expected rules. Per-rule precision counts extra rules as false "
              "positives. ⚠ marks fewer than 5 cases: read those as anecdotes, not rates.", ""]

    groups = {"all": results}
    for name in ("dev", "test"):
        part = [r for r in results if r.split == name]
        if part and len(part) != len(results):
            groups[name] = part

    lines += ["## Summary", "", "| | " + " | ".join(groups) + " |", "|---|" + "---|" * len(groups)]
    sums = {name: summarize(rs) for name, rs in groups.items()}
    for label, key in [("Case accuracy (lenient)", "lenient"), ("Case accuracy (strict)", "strict"),
                       ("False positives on pass cases", "pass_fp"), ("Hard cases (lenient)", "hard")]:
        lines.append(f"| {label} | " + " | ".join(str(s[key]) for s in sums.values()) + " |")
    lines.append("| Judge errors (blocking) | " + " | ".join(str(s["judge_errors"]) for s in sums.values()) + " |")

    lines += ["", "## Per rule (all cases in this run)", "",
              "| Rule | TP | FP | FN | Precision | Recall |", "|---|---|---|---|---|---|"]
    for rs in sums["all"]["rules"]:
        lines.append(f"| {rs.rule} | {rs.tp} | {rs.fp} | {rs.fn} | {rs.precision} | {rs.recall} |")

    wrong = [r for r in results if not r.lenient_ok]
    lines += ["", f"## Disagreements with the expected label ({len(wrong)})", ""]
    lines += _case_table(wrong) if wrong else ["None."]

    extras = [r for r in results if r.lenient_ok and r.extra]
    lines += ["", f"## Correct, but with extra rules ({len(extras)})", "",
              "Not errors under the agreed scoring; they lower per-rule precision.", ""]
    lines += _case_table(extras) if extras else ["None."]

    lines += ["", "## Possibly mislabelled (for owner review)", "",
              "_Filled in after reviewing the disagreements. Labels are never changed by the harness._", ""]
    return "\n".join(lines) + "\n"


def _case_table(results: list[CaseResult]) -> list[str]:
    rows = ["| Case | Split | Expected | Got | Missing | Extra | Judge's reasons (evidence) |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        reasons = "<br>".join(
            f"**{h['rule']}** {_cell(h['reason'])}" + (f" ({h['evidence']})" if h.get("evidence") else "")
            for h in r.hits
        ) or "—"
        rows.append(
            f"| {r.id}{' (hard)' if r.hard else ''} | {r.split} | {r.expected_verdict} {r.expected_rules} "
            f"| {r.verdict} {r.rules} | {', '.join(r.missing) or '—'} | {', '.join(r.extra) or '—'} | {reasons} |"
        )
    return rows


def _cell(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")[:220]
