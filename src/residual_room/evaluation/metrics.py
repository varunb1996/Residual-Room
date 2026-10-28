"""Scores for an eval run. Every number carries its raw counts.

  lenient case accuracy  verdict matches AND expected rules ⊆ predicted rules
                         (the agreed scoring rule; extra rules don't fail a case)
  strict case accuracy   verdict matches AND predicted rules == expected rules
  per-rule precision     an extra (unexpected) rule counts as a false positive here,
  and recall             so over-firing shows up even though lenient accuracy forgives it
  pass FP rate           expected-pass cases that got warn or block
  hard-case accuracy     lenient accuracy on the five hard cases
"""

from dataclasses import dataclass, field

RULES = ("P1", "P2", "P3", "P4", "P5", "P6", "S1", "S2", "S3", "S4")
SMALL_N = 5


@dataclass
class CaseResult:
    id: str
    split: str
    hard: bool
    expected_verdict: str
    expected_rules: list[str]
    verdict: str
    rules: list[str]
    hits: list[dict] = field(default_factory=list)  # rule, reason, segment_text, evidence
    segments: list[dict] = field(default_factory=list)

    @property
    def missing(self) -> list[str]:
        return sorted(set(self.expected_rules) - set(self.rules))

    @property
    def extra(self) -> list[str]:
        return sorted(set(self.rules) - set(self.expected_rules))

    @property
    def lenient_ok(self) -> bool:
        return self.verdict == self.expected_verdict and not self.missing

    @property
    def strict_ok(self) -> bool:
        return self.lenient_ok and not self.extra


@dataclass(frozen=True)
class Ratio:
    k: int
    n: int

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a (0 cases)"
        flag = " ⚠ small n" if self.n < SMALL_N else ""
        return f"{self.k}/{self.n} ({100 * self.k / self.n:.0f}%){flag}"


@dataclass(frozen=True)
class RuleScore:
    rule: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> Ratio:
        return Ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> Ratio:
        return Ratio(self.tp, self.tp + self.fn)


def summarize(results: list[CaseResult]) -> dict:
    expected_pass = [r for r in results if r.expected_verdict == "pass"]
    hard = [r for r in results if r.hard]
    return {
        "cases": len(results),
        "lenient": Ratio(sum(r.lenient_ok for r in results), len(results)),
        "strict": Ratio(sum(r.strict_ok for r in results), len(results)),
        "pass_fp": Ratio(sum(r.verdict != "pass" for r in expected_pass), len(expected_pass)),
        "hard": Ratio(sum(r.lenient_ok for r in hard), len(hard)),
        "judge_errors": sum("JUDGE" in r.rules for r in results),
        "rules": [rule_score(rule, results) for rule in RULES],
    }


def rule_score(rule: str, results: list[CaseResult]) -> RuleScore:
    tp = sum(rule in r.rules and rule in r.expected_rules for r in results)
    fp = sum(rule in r.rules and rule not in r.expected_rules for r in results)
    fn = sum(rule not in r.rules and rule in r.expected_rules for r in results)
    return RuleScore(rule, tp, fp, fn)
