"""Grounded generation: retrieve -> generate JSON -> validate (one corrective retry).

    uv run python -m residual_room.generation.generate --explainer exp-004 --segment practitioner
    uv run python -m residual_room.generation.generate --explainer exp-004 --segment transitioning --save

If the output still has structural problems after the retry, it is kept with the
problems recorded: claims without (valid) citations then fail the checker's P1,
so nothing is silently dropped or passed.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field

from residual_room.config import PublicationConfig, get_config
from residual_room.corpus.retrieve import Hit
from residual_room.generation.prompt import build_prompt
from residual_room.generation.schema import GenOutput, problems, render_text
from residual_room.llm.client import LLMError


@dataclass
class Generated:
    output: GenOutput
    text: str
    problems: list[str]
    meta: dict = field(default_factory=dict)


def generate(
    explainer_id: str, segment: str, kind: str, hits: list[Hit], llm, corpus_version: str,
    config: PublicationConfig | None = None,
) -> Generated:
    config = config or get_config()
    explainer = config.explainers[explainer_id]
    system, user = build_prompt(explainer_id, explainer, segment, kind, hits, config)
    allowed = {h.chunk_id for h in hits}

    out = llm.ask_json(system, user, GenOutput)
    found = problems(out, allowed, segment, kind, config)
    attempts = 1
    if found:
        fix = user + "\n\nYour previous draft had these problems; fix them and reply with the full JSON:\n- " + \
            "\n- ".join(found) + f"\n\nPrevious draft:\n{out.model_dump_json()}"
        try:
            out = llm.ask_json(system, fix, GenOutput)
            attempts = 2
            found = problems(out, allowed, segment, kind, config)
        except LLMError as exc:
            found = found + [f"corrective retry failed: {exc}"]
    return Generated(
        output=out,
        text=render_text(out),
        problems=found,
        meta={
            "explainer": explainer_id,
            "segment": segment,
            "kind": kind,
            "generator_model": llm.model,
            "corpus_version": corpus_version,
            "retrieved_chunks": [h.chunk_id for h in hits],
            "attempts": attempts,
            "problems": found,
        },
    )


def generate_for(explainer_id: str, segment: str, kind: str, corpus_version: str = "corpus-v1") -> Generated:
    """The real thing: database retrieval and the configured generator model."""
    from residual_room.corpus.embed import LocalEmbedder
    from residual_room.corpus.manifest import load_manifest
    from residual_room.db.session import get_engine
    from residual_room.generation.retrieve import retrieve_for
    from residual_room.llm.client import load_role

    config = get_config()
    hits = retrieve_for(config.explainers[explainer_id], load_manifest(), get_engine(), LocalEmbedder(), corpus_version)
    return generate(explainer_id, segment, kind, hits, load_role("generator"), corpus_version, config)


def title_for(explainer_id: str, segment: str, kind: str) -> str:
    e = get_config().explainers[explainer_id]
    return f"{e.topic} ({explainer_id}, {segment}, {kind.replace('_', ' ')})"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--explainer", required=True, choices=sorted(get_config().explainers))
    parser.add_argument("--segment", required=True, choices=["transitioning", "practitioner"])
    parser.add_argument("--kind", default="newsletter", choices=["newsletter", "explainer_summary"])
    parser.add_argument("--save", action="store_true", help="save as a draft piece for review")
    args = parser.parse_args()

    g = generate_for(args.explainer, args.segment, args.kind)
    print(json.dumps(g.output.model_dump(), indent=2, ensure_ascii=False))
    non_exempt = [s for s in g.output.segments if s.type == "claim"]
    cited = [s for s in non_exempt if s.chunk_ids]
    print(f"\nclaims citing >=1 chunk: {len(cited)}/{len(non_exempt)} | attempts: {g.meta['attempts']} "
          f"| model: {g.meta['generator_model']}")
    print("problems:", g.problems or "none")
    if args.save:
        from sqlalchemy.orm import Session

        from residual_room.db.session import get_engine
        from residual_room.workflow.service import create_generated_piece

        with Session(get_engine()) as s, s.begin():
            piece = create_generated_piece(s, title_for(args.explainer, args.segment, args.kind),
                                           args.kind, args.segment, None, g)
        print(f"saved as piece {piece.id} (draft)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
