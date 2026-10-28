import hashlib
import uuid

import pytest
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from residual_room.corpus.embed import DIM
from residual_room.corpus.ingest import SourceText, ingest
from residual_room.corpus.manifest import Source
from residual_room.corpus.retrieve import search
from residual_room.db.models import ChunkRow, CorpusVersion
from residual_room.db.session import get_engine

pytestmark = pytest.mark.db


class FakeEmbedder:
    """Deterministic stand-in so the test doesn't need the model download."""

    name = "fake"
    dim = DIM

    def embed_passages(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        # Bag of words hashed into DIM buckets: shared words -> similar vectors.
        v = [0.0] * DIM
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        return v


def _source(sid, status):
    return Source(
        id=sid, title=sid, origin="internal", status=status, license="own",
        commit_allowed=True, license_verified=True, path="x.md",
    )


@pytest.fixture
def version():
    engine = get_engine()
    name = f"test-{uuid.uuid4().hex[:8]}"
    texts = [
        SourceText(_source("sched-old", "superseded"), "The newsletter goes out on Tuesdays."),
        SourceText(_source("sched-new", "current"), "The newsletter goes out on Thursdays."),
    ]
    ingest(engine, name, texts, FakeEmbedder(), manifest_sha256="test")
    yield name
    with Session(engine) as s, s.begin():
        s.execute(delete(CorpusVersion).where(CorpusVersion.version == name))


def test_every_chunk_row_has_status_and_corpus_version(version):
    with Session(get_engine()) as s:
        rows = s.scalars(select(ChunkRow).where(ChunkRow.corpus_version == version)).all()
    assert {(r.source_id, r.status, r.corpus_version) for r in rows} == {
        ("sched-old", "superseded", version),
        ("sched-new", "current", version),
    }


def test_superseded_excluded_by_default_and_included_on_request(version):
    engine = get_engine()
    q = "when does the newsletter go out"
    default = search(engine, FakeEmbedder(), q, corpus_version=version)
    assert {h.source_id for h in default} == {"sched-new"}
    everything = search(engine, FakeEmbedder(), q, corpus_version=version, include_superseded=True)
    assert {h.source_id for h in everything} == {"sched-new", "sched-old"}


def test_existing_version_is_not_overwritten_without_replace(version):
    with pytest.raises(ValueError, match="already exists"):
        ingest(get_engine(), version, [], FakeEmbedder(), manifest_sha256="test")
    with Session(get_engine()) as s:
        count = s.scalar(select(func.count()).where(ChunkRow.corpus_version == version))
    assert count == 2
