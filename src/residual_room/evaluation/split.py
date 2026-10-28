"""The frozen dev/test split (eval/split.yaml).

Stratified by each case's first expected rule ("pass" if none). Hard cases and
rules with a single case always go to test: the headline numbers must come from
cases nobody tuned on. Within a stratum, order is by a hash of the case id, so
the split is deterministic. Prompt changes may only be made against dev.

    uv run python -m residual_room.evaluation.split   # writes once, refuses to overwrite
"""

import hashlib
import sys
from collections import defaultdict

import yaml

from residual_room.evaluation.cases import EVAL_DIR, EvalCase, load_cases

SPLIT_PATH = EVAL_DIR / "split.yaml"


def stratum(case: EvalCase) -> str:
    return case.expected_rules[0] if case.expected_rules else "pass"


def build_split(cases: dict[str, EvalCase]) -> dict[str, list[str]]:
    groups: dict[str, list[EvalCase]] = defaultdict(list)
    for c in cases.values():
        groups[stratum(c)].append(c)
    dev, test = [], []
    for members in groups.values():
        ordered = sorted(members, key=lambda c: hashlib.sha256(c.id.encode()).hexdigest())
        want_dev = len(members) // 2  # 0 for single-case rules: they go to test
        n_dev = 0
        for c in ordered:
            if not c.hard and n_dev < want_dev:
                dev.append(c.id)
                n_dev += 1
            else:
                test.append(c.id)
    return {"dev": sorted(dev), "test": sorted(test)}


def load_split() -> dict[str, list[str]]:
    return yaml.safe_load(SPLIT_PATH.read_text(encoding="utf-8"))


def main() -> int:
    if SPLIT_PATH.exists():
        print(f"{SPLIT_PATH.name} already exists; the split is frozen. Delete it deliberately to rebuild.")
        return 1
    split = build_split(load_cases())
    header = (
        "# Frozen dev/test split. Built once by `python -m residual_room.evaluation.split`.\n"
        "# Stratified by first expected rule; hard cases and single-case rules are in test.\n"
        "# Tune prompts only against dev failures; report test as the headline.\n"
    )
    SPLIT_PATH.write_text(header + yaml.safe_dump(split, sort_keys=True), encoding="utf-8", newline="\n")
    print(f"wrote {SPLIT_PATH.name}: {len(split['dev'])} dev, {len(split['test'])} test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
