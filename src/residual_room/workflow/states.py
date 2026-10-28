"""The approval state machine. Pure code: no database, no HTTP.

    draft --submit--> checks --checks_done--> in_review --approve--> approved --schedule--> scheduled --send--> sent
      ^                 |                        |
      |           checks_failed               reject
      +-----------------+                        v
      +-----------edit------------------------ rejected

Every allowed move is in TRANSITIONS; anything else raises InvalidTransition.
Editing creates a new version and returns the piece to draft, so approvals and
overrides never carry over to changed text. A hard-blocked piece still reaches
in_review (with its blocking hits shown); approval is then gated by
`approval_blockers`, not by the state itself.
"""

from enum import StrEnum


class State(StrEnum):
    DRAFT = "draft"
    CHECKS = "checks"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    SCHEDULED = "scheduled"
    SENT = "sent"
    REJECTED = "rejected"


class Action(StrEnum):
    SUBMIT = "submit"
    CHECKS_DONE = "checks_done"
    CHECKS_FAILED = "checks_failed"
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    SCHEDULE = "schedule"  # Prompt 7
    SEND = "send"  # Prompt 7


TRANSITIONS: dict[tuple[State, Action], State] = {
    (State.DRAFT, Action.SUBMIT): State.CHECKS,
    (State.CHECKS, Action.CHECKS_DONE): State.IN_REVIEW,
    (State.CHECKS, Action.CHECKS_FAILED): State.DRAFT,
    (State.IN_REVIEW, Action.APPROVE): State.APPROVED,
    (State.IN_REVIEW, Action.REJECT): State.REJECTED,
    (State.APPROVED, Action.SCHEDULE): State.SCHEDULED,
    (State.SCHEDULED, Action.SEND): State.SENT,
    # Any edit is a new version and goes back to draft.
    (State.DRAFT, Action.EDIT): State.DRAFT,
    (State.IN_REVIEW, Action.EDIT): State.DRAFT,
    (State.REJECTED, Action.EDIT): State.DRAFT,
    (State.APPROVED, Action.EDIT): State.DRAFT,
}


class InvalidTransition(ValueError):
    pass


def next_state(state: State, action: Action) -> State:
    try:
        return TRANSITIONS[(State(state), Action(action))]
    except KeyError:
        raise InvalidTransition(f"can't {action} a piece that is {state}") from None


def approval_blockers(blocking_hits: list[int], overridden: set[int]) -> list[int]:
    """Indexes of blocking hits that still have no override. Approve needs none."""
    return [i for i in blocking_hits if i not in overridden]
