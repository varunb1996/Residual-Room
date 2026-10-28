from datetime import date

import yaml

from residual_room.checker.document import check_correction_notice
from residual_room.corpus.manifest import REPO_ROOT

FIXTURE = REPO_ROOT / "eval" / "fixtures" / "stl-10-issue-2.md"


def load_fixture():
    _, front, body = FIXTURE.read_text(encoding="utf-8").split("---\n", 2)
    return yaml.safe_load(front), body


def test_stl10_issue_2_without_notice_is_p5():
    meta, body = load_fixture()
    assert meta["case"] == "STL-10"
    (hit,) = check_correction_notice(body, meta["issue_date"])
    assert hit.rule == "P5" and hit.segment is None and hit.evidence == "exp-003-v2"


def test_same_issue_with_a_notice_passes():
    meta, body = load_fixture()
    notice = "Correction: our explainer on retrieval failures originally said Lost in the Middle tested only GPT-3.5-Turbo.\n"
    assert check_correction_notice(notice + body, meta["issue_date"]) == []


def test_other_issues_and_undated_text_are_not_affected():
    _, body = load_fixture()
    assert check_correction_notice(body, date(2026, 11, 5)) == []  # Issue 1, before the correction
    assert check_correction_notice(body, date(2026, 11, 19)) == []  # Issue 3
    assert check_correction_notice(body, None) == []
