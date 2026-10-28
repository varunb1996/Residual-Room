"""Raw eval cases by id, for tests that run real case text."""

from residual_room.evaluation.cases import raw_cases as eval_cases


def case(case_id: str) -> dict:
    return eval_cases()[case_id]


__all__ = ["case", "eval_cases"]
