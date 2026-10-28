"""Entry point: text (or pre-segmented generator output) -> CheckReport.

For now only the deterministic rules run; the judge half (P1 support, P2-P4,
P5-semantic, P6, S2, S4, claim extraction) plugs in here next.
"""

from residual_room.checker.claims import split_segments
from residual_room.checker.deterministic import StatusLookup, run_deterministic
from residual_room.checker.types import CheckReport, Segment
from residual_room.config import PublicationConfig, get_config


def check(
    text: str,
    lookup: StatusLookup,
    segments: list[Segment] | None = None,
    config: PublicationConfig | None = None,
) -> CheckReport:
    config = config or get_config()
    segments = segments if segments is not None else split_segments(text, config)
    return CheckReport(segments, run_deterministic(text, segments, lookup, config))
