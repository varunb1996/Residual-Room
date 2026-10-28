"""Query the corpus from the command line.

    uv run python scripts/query.py "position bias in LLM judges"
    uv run python scripts/query.py "publishing day" --include-superseded
"""

import argparse

from residual_room.corpus.embed import LocalEmbedder
from residual_room.corpus.retrieve import latest_version, search
from residual_room.db.session import get_engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--version", help="corpus version (default: latest)")
    parser.add_argument("--include-superseded", action="store_true")
    parser.add_argument("--source", action="append", help="restrict to a source id (repeatable)")
    args = parser.parse_args()

    engine = get_engine()
    version = args.version or latest_version(engine)
    hits = search(
        engine,
        LocalEmbedder(),
        args.query,
        k=args.k,
        corpus_version=version,
        include_superseded=args.include_superseded,
        source_ids=args.source,
    )
    print(f"query: {args.query!r}  corpus: {version}\n")
    for i, h in enumerate(hits, 1):
        preview = " ".join(h.text.split())[:220]
        print(f"{i}. {h.chunk_id}  [{h.status}]  distance={h.distance:.3f}")
        print(f"   {h.heading_path}")
        print(f"   {preview}…\n")


if __name__ == "__main__":
    main()
