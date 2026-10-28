from residual_room.corpus.chunk import MAX_TOKENS, PATH_SEP, chunk_markdown, estimate_tokens

# Padded past MAX_TOKENS so it is split by structure, not kept whole.
PADDING = "\n\n## 9 Appendix\n\n" + " ".join(["Filler sentence for length."] * 120) + "\n"

DOC = """# Paper

## 1 Intro

First paragraph about judges.

Second paragraph about position bias.

## 2 Results

### 2.1 Position

[TABLE] Table 2: Position bias of judges.

| Judge | Consistency |
|---|---|
| Claude-v1 | 23.8% |
| GPT-4 | 65.0% |

Text after the table.
""" + PADDING


def test_small_doc_stays_whole():
    md = "# Publishing Schedule v1\n\n- Newsletter: weekly, Thursdays at 9:00 AM IST.\n"
    (chunk,) = chunk_markdown("int-schedule-v1", "Schedule v1", md)
    assert chunk.chunk_id == "int-schedule-v1#0000"
    assert "Thursdays at 9:00 AM IST" in chunk.text


def test_heading_path_and_ids():
    chunks = chunk_markdown("src", "Paper", DOC)
    assert [c.chunk_id for c in chunks] == [f"src#{i:04d}" for i in range(len(chunks))]
    assert chunks[0].heading_path == PATH_SEP.join(["Paper", "1 Intro"])
    assert "First paragraph" in chunks[0].text and "Second paragraph" in chunks[0].text


def test_table_is_its_own_chunk_with_caption_and_labels():
    chunks = chunk_markdown("src", "Paper", DOC)
    table = next(c for c in chunks if c.text.startswith("[TABLE]"))
    assert table.heading_path.endswith("2.1 Position")
    assert "Table 2: Position bias" in table.text
    assert "| Claude-v1 | 23.8% |" in table.text and "| GPT-4 | 65.0% |" in table.text
    assert "Text after the table" not in table.text


def test_oversized_table_is_split_by_rows_with_header_repeated():
    rows = "\n".join(f"| model-{i} | {i}.0% |" for i in range(400))
    md = f"## T\n\n[TABLE] Table 9: Big.\n\n| Model | Score |\n|---|---|\n{rows}\n"
    chunks = chunk_markdown("src", "Doc", md)
    assert len(chunks) > 1
    for c in chunks:
        assert "Table 9: Big." in c.text
        assert "| Model | Score |" in c.text
        assert estimate_tokens(c.text) <= MAX_TOKENS
    all_rows = "\n".join(c.text for c in chunks)
    assert all(f"| model-{i} |" in all_rows for i in range(400))


def test_long_paragraph_is_split_at_sentences_within_limit():
    para = " ".join(f"Sentence number {i} is here." for i in range(600))
    chunks = chunk_markdown("src", "Doc", f"## S\n\n{para}\n")
    assert len(chunks) > 1
    assert all(estimate_tokens(c.text) <= MAX_TOKENS for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)


def test_code_fence_with_blank_lines_and_hashes_stays_together():
    md = "## Code\n\n```python\n# not a heading\n\nx = embed(['some text to embed'])\n```\n" + PADDING
    chunk = chunk_markdown("src", "Doc", md)[0]
    assert chunk.heading_path == PATH_SEP.join(["Doc", "Code"])
    assert "# not a heading" in chunk.text and "x = embed" in chunk.text
