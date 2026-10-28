from residual_room.corpus.retrieve import Hit
from residual_room.generation.retrieve import select_chunks


def hit(source, n, distance):
    return Hit(f"{source}#{n:04d}", source, "current", source, "text", distance)


def test_cap_per_source_and_total():
    candidates = [hit("big", i, 0.10 + i / 100) for i in range(10)] + [hit("small", i, 0.30 + i / 100) for i in range(3)]
    chosen = select_chunks(candidates, required_sources=[], per_source=4, total=6)
    sources = [h.source_id for h in chosen]
    assert sources.count("big") == 4 and sources.count("small") == 2 and len(chosen) == 6


def test_required_sources_get_a_slot_even_when_far():
    candidates = [hit("big", i, 0.10) for i in range(10)] + [hit("own", 0, 0.90)]
    chosen = select_chunks(candidates, required_sources=["own"], per_source=4, total=4)
    assert "own" in {h.source_id for h in chosen}


def test_duplicates_from_several_queries_are_merged_keeping_the_best_distance():
    candidates = [hit("a", 0, 0.5), hit("a", 0, 0.2), hit("b", 0, 0.3)]
    chosen = select_chunks(candidates, required_sources=[])
    assert [(h.chunk_id, h.distance) for h in chosen] == [("a#0000", 0.2), ("b#0000", 0.3)]
