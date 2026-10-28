"""`make ingest`: chunk, embed and store every manifest source as one corpus version.

Internal sources are read from their docs/ path; external ones from the text
`make fetch` saved. A version is written in one transaction and is immutable:
re-ingesting an existing version needs --replace.
"""

import argparse
import hashlib
import sys
from dataclasses import dataclass

from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from residual_room.corpus.chunk import Chunk, chunk_markdown
from residual_room.corpus.embed import Embedder, LocalEmbedder
from residual_room.corpus.manifest import MANIFEST_PATH, REPO_ROOT, Manifest, Source, load_manifest
from residual_room.db.models import ChunkRow, CorpusVersion, SourceRow

DEFAULT_VERSION = "corpus-v1"


@dataclass(frozen=True)
class SourceText:
    source: Source
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


class MissingTextError(RuntimeError):
    pass


def load_texts(manifest: Manifest) -> list[SourceText]:
    texts, missing = [], []
    for s in manifest.sources.values():
        path = REPO_ROOT / s.path if s.origin == "internal" else s.text_dir / "text.md"
        if path.is_file():
            texts.append(SourceText(s, path.read_text(encoding="utf-8")))
        else:
            missing.append(s.id)
    if missing:
        raise MissingTextError(f"no text for {', '.join(missing)}: run `make fetch` first")
    return texts


def ingest(
    engine: Engine,
    version: str,
    texts: list[SourceText],
    embedder: Embedder,
    manifest_sha256: str,
    replace: bool = False,
) -> int:
    chunked: list[tuple[SourceText, list[Chunk]]] = [
        (st, chunk_markdown(st.source.id, st.source.title, st.text)) for st in texts
    ]
    all_chunks = [c for _, chunks in chunked for c in chunks]
    # Embed before opening the transaction: it's the slow part.
    vectors = embedder.embed_passages([c.embed_text for c in all_chunks])

    with Session(engine) as session, session.begin():
        if session.scalar(select(CorpusVersion).where(CorpusVersion.version == version)):
            if not replace:
                raise ValueError(f"{version} already exists; pass --replace to rebuild it")
            session.execute(delete(CorpusVersion).where(CorpusVersion.version == version))

        session.add(
            CorpusVersion(
                version=version,
                embed_model=embedder.name,
                manifest_sha256=manifest_sha256,
                chunk_count=len(all_chunks),
            )
        )
        session.flush()
        for st, _ in chunked:
            s = st.source
            session.add(
                SourceRow(
                    corpus_version=version,
                    source_id=s.id,
                    title=s.title,
                    origin=s.origin,
                    kind=s.kind,
                    status=s.status,
                    license=s.license,
                    commit_allowed=s.committable,
                    url=s.url,
                    text_sha256=st.sha256,
                )
            )
        session.flush()
        by_id = {st.source.id: st.source for st in texts}
        for c, vec in zip(all_chunks, vectors, strict=True):
            s = by_id[c.source_id]
            session.add(
                ChunkRow(
                    corpus_version=version,
                    chunk_id=c.chunk_id,
                    source_id=c.source_id,
                    seq=c.seq,
                    status=s.status,
                    license=s.license,
                    commit_allowed=s.committable,
                    heading_path=c.heading_path,
                    text=c.text,
                    embedding=vec,
                )
            )
    return len(all_chunks)


def main() -> int:
    from residual_room.db.session import get_engine

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--replace", action="store_true", help="rebuild an existing version")
    args = parser.parse_args()

    manifest = load_manifest()
    try:
        texts = load_texts(manifest)
    except MissingTextError as exc:
        print(exc)
        return 1
    manifest_sha = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    n = ingest(get_engine(), args.version, texts, LocalEmbedder(), manifest_sha, args.replace)
    print(f"{args.version}: {n} chunks from {len(texts)} sources")
    return 0


if __name__ == "__main__":
    sys.exit(main())
