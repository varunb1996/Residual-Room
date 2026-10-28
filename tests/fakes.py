"""Test doubles for the judge LLM and the corpus retriever."""

import re

from residual_room.checker.evidence import Passage


class FakeJudge:
    """Answers ask_json by calling `script(system, user)` -> dict, and records calls."""

    model = "fake-judge"
    max_concurrency = 1

    def __init__(self, script):
        self.script = script
        self.calls: list[tuple[str, str]] = []

    def ask_json(self, system, user, schema):
        self.calls.append((system, user))
        return schema.model_validate(self.script(system, user))


def candidate_indices(user: str) -> list[int]:
    asked = user.split("Label these sentences:\n", 1)[1]
    return [int(m) for m in re.findall(r"^\[(\d+)\]", asked, re.M)]


class FakeRetriever:
    def __init__(self, passages: list[Passage]):
        self.passages = passages
        self.calls: list[tuple] = []

    def for_citation(self, citation, query, k=3):
        self.calls.append(("cite", citation))
        sid = citation.split("#")[0]
        return [p for p in self.passages if p.chunk_id == citation or p.source_id == sid][:k]

    def across_corpus(self, query, k=6):
        self.calls.append(("all", query))
        return self.passages[:k]
