"""One-off: point already-sent tracked links at the app's public pages.

Each tracked link stores its target when the message is sent, so messages sent
while outbound links used the placeholder site (https://residualroom.example)
keep redirecting there. This rewrites only those targets, keeping the path:
  make relink BASE=https://residual-room.onrender.com
"""

import argparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from residual_room.db.models import Link
from residual_room.db.session import get_engine

PLACEHOLDER = "https://residualroom.example"


def relink(s: Session, base: str, old: str = PLACEHOLDER) -> int:
    base = base.rstrip("/")
    links = s.scalars(select(Link).where(Link.target_url.startswith(old + "/"))).all()
    for link in links:
        link.target_url = base + link.target_url[len(old):]
    return len(links)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="the deployed app's URL, e.g. https://residual-room.onrender.com")
    args = parser.parse_args()
    if not args.base.startswith("https://"):
        raise SystemExit("--base must be the public https:// URL readers can reach")
    with Session(get_engine()) as s, s.begin():
        n = relink(s, args.base)
    print(f"Rewrote {n} link(s) from {PLACEHOLDER} to {args.base.rstrip('/')}.")


if __name__ == "__main__":
    main()
