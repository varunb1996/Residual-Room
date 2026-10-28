import itertools

import pytest

from residual_room.workflow.states import (
    TRANSITIONS,
    Action,
    InvalidTransition,
    State,
    approval_blockers,
    next_state,
)


def test_happy_path():
    s = State.DRAFT
    for action in (Action.SUBMIT, Action.CHECKS_DONE, Action.APPROVE, Action.SCHEDULE, Action.SEND):
        s = next_state(s, action)
    assert s == State.SENT


def test_reject_then_edit_returns_to_draft():
    s = next_state(State.IN_REVIEW, Action.REJECT)
    assert s == State.REJECTED
    assert next_state(s, Action.EDIT) == State.DRAFT


def test_failed_checks_return_to_draft():
    assert next_state(State.CHECKS, Action.CHECKS_FAILED) == State.DRAFT


@pytest.mark.parametrize("state,action", list(itertools.product(State, Action)))
def test_every_pair_is_either_allowed_or_refused(state, action):
    if (state, action) in TRANSITIONS:
        assert next_state(state, action) == TRANSITIONS[(state, action)]
    else:
        with pytest.raises(InvalidTransition):
            next_state(state, action)


@pytest.mark.parametrize(
    "state,action",
    [
        (State.DRAFT, Action.APPROVE),  # can't skip the checks
        (State.CHECKS, Action.APPROVE),  # can't approve while checks run
        (State.CHECKS, Action.EDIT),  # can't edit text being checked
        (State.SENT, Action.EDIT),  # sent is final
        (State.SCHEDULED, Action.EDIT),  # unschedule first (Prompt 7)
        (State.APPROVED, Action.APPROVE),
    ],
)
def test_important_refusals(state, action):
    with pytest.raises(InvalidTransition):
        next_state(state, action)


def test_approval_needs_every_blocking_hit_overridden():
    assert approval_blockers([0, 2], overridden={0}) == [2]
    assert approval_blockers([0, 2], overridden={0, 2}) == []
    assert approval_blockers([], overridden=set()) == []
