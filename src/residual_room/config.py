"""Load config/publication.yaml (one publication's brand and policy settings)."""

import re
from functools import cached_property, lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

from residual_room.corpus.manifest import REPO_ROOT

CONFIG_PATH = REPO_ROOT / "config" / "publication.yaml"


class PublicationConfig(BaseModel):
    model_config = {"frozen": True}

    name: str
    segments: list[str]
    default_segment: str
    docs: dict[str, str]
    avoid_list: list[str]
    ctas: dict[str, list[str]]
    exempt_patterns: dict[str, list[str]]

    @cached_property
    def avoid_regex(self) -> re.Pattern:
        words = "|".join(re.escape(w) for w in self.avoid_list)
        return re.compile(rf"(?<![\w-])({words})(?![\w-])", re.IGNORECASE)

    @cached_property
    def cta_regexes(self) -> dict[str, list[re.Pattern]]:
        return {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in self.ctas.items()}

    @cached_property
    def exempt_regexes(self) -> dict[str, list[re.Pattern]]:
        return {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in self.exempt_patterns.items()}


def load_config(path: Path = CONFIG_PATH) -> PublicationConfig:
    return PublicationConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache
def get_config() -> PublicationConfig:
    return load_config()
