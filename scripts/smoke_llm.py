"""One live call per LLM role, to check keys and model access.

    uv run python scripts/smoke_llm.py
"""

import time

from residual_room.llm.client import LLMError, ResponseCache, load_role

for role in ("judge", "generator"):
    llm = load_role(role, cache=ResponseCache(enabled=False))
    start = time.time()
    try:
        text = llm.ask_text("Reply with exactly one word.", "Say ready.")
        print(f"{role:<10} {llm.model:<42} OK  {time.time() - start:5.1f}s  -> {text[:40]!r}")
    except LLMError as exc:
        print(f"{role:<10} {llm.model:<42} FAIL {exc}")
