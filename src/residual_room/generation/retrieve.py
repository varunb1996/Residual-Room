"""Evidence for generation: current chunks only, diversified by source.

Plain top-k is lopsided (for exp-004, 13 of 16 relevant chunks came from the
MT-Bench paper). So: gather candidates from several queries plus the
explainer's own manifest sources, then pick the closest chunks with a cap per
source, making sure each of the explainer's own sources gets at least one slot.
"""

from residual_room.config import Explainer
from residual_room.corpus.embed import Embedder
from residual_room.corpus.manifest import Manifest
from residual_room.corpus.retrieve import Hit, search

PER_SOURCE = 4
TOTAL = 12


def select_chunks(
    candidates: list[Hit], required_sources: list[str], per_source: int = PER_SOURCE, total: int = TOTAL
) -> list[Hit]:
    best = {}
    for h in sorted(candidates, key=lambda h: h.distance):
        best.setdefault(h.chunk_id, h)
    ranked = list(best.values())

    chosen: list[Hit] = []
    counts: dict[str, int] = {}

    def take(h: Hit) -> None:
        chosen.append(h)
        counts[h.source_id] = counts.get(h.source_id, 0) + 1

    for source in required_sources:  # one guaranteed slot for each of the explainer's own sources
        first = next((h for h in ranked if h.source_id == source), None)
        if first and len(chosen) < total:
            take(first)
    for h in ranked:
        if len(chosen) >= total:
            break
        if h not in chosen and counts.get(h.source_id, 0) < per_source:
            take(h)
    return sorted(chosen, key=lambda h: h.distance)


def retrieve_for(
    explainer: Explainer, manifest: Manifest, engine, embedder: Embedder, corpus_version: str
) -> list[Hit]:
    queries = [explainer.topic] + [f"{explainer.topic}: {s}" for s in explainer.subtopics]
    candidates: list[Hit] = []
    for q in queries:
        candidates += search(engine, embedder, q, k=10, corpus_version=corpus_version)
    own = [s.id for s in manifest.external if s.topic in explainer.source_topics]
    for source in own:
        candidates += search(engine, embedder, explainer.topic, k=2, corpus_version=corpus_version,
                             source_ids=[source])
    # search() excludes superseded chunks by default: never ground new content in stale text.
    return select_chunks(candidates, required_sources=own)
