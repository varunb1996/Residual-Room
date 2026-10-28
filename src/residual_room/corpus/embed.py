"""Local, free embeddings: BAAI/bge-small-en-v1.5 via fastembed (ONNX, no PyTorch).

Deterministic: the same text always gives the same vector, so an eval run can be
reproduced exactly against a corpus version. The model (~130 MB) downloads once
into .cache/fastembed/.
"""

from typing import Protocol

from residual_room.corpus.manifest import REPO_ROOT

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384
# bge models expect this instruction on queries (not on passages) for retrieval.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CACHE_DIR = REPO_ROOT / ".cache" / "fastembed"


class Embedder(Protocol):
    name: str
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class LocalEmbedder:
    name = MODEL_NAME
    dim = DIM

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily: loading takes a few seconds

            self._model = TextEmbedding(MODEL_NAME, cache_dir=str(CACHE_DIR))
        return self._model

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._load().embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._load().embed([QUERY_PREFIX + text]))).tolist()
