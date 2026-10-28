"""Checks on real fetched text. MT-Bench isn't committable, so these skip on a
fresh clone (or in CI) until `make fetch` has run."""

import pytest

from residual_room.corpus.chunk import chunk_markdown
from residual_room.corpus.manifest import load_manifest

MTBENCH = load_manifest().resolve("ext-mtbench-paper")
MTBENCH_TEXT = MTBENCH.text_dir / "text.md"

pytestmark = pytest.mark.skipif(not MTBENCH_TEXT.is_file(), reason="run `make fetch` first")


@pytest.fixture(scope="module")
def chunks():
    return chunk_markdown(MTBENCH.id, MTBENCH.title, MTBENCH_TEXT.read_text(encoding="utf-8"))


def _table(chunks, caption_start):
    (chunk,) = [c for c in chunks if c.text.startswith(f"[TABLE] {caption_start}")]
    return chunk.text


def test_table_2_position_bias_numbers_intact_with_labels(chunks):
    table = _table(chunks, "Table 2:")
    assert "| Claude-v1 | default | 23.8% |" in table
    assert "| GPT-4 | default | 65.0% |" in table


def test_table_3_verbosity_attack_numbers_intact_with_labels(chunks):
    table = _table(chunks, "Table 3:")
    assert "| Judge | Claude-v1 | GPT-3.5 | GPT-4 |" in table
    assert "| Failure rate | 91.3% | 91.3% | 8.7% |" in table
