"""Load the frozen eval cases by id, for tests that run real case text."""

from functools import lru_cache

import yaml

from residual_room.corpus.manifest import REPO_ROOT


@lru_cache
def eval_cases() -> dict[str, dict]:
    cases = {}
    for path in sorted((REPO_ROOT / "eval").glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for value in data.values():
            if isinstance(value, list):
                cases.update({c["id"]: c for c in value if isinstance(c, dict) and "id" in c})
    return cases


def case(case_id: str) -> dict:
    return eval_cases()[case_id]
