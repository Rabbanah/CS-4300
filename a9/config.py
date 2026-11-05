"""Configuration, corpus metadata registry, and environment loading for U0-HW-09."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping


A9_DIR = Path(__file__).resolve().parent
DEFAULT_DOCUMENTS_DIR = A9_DIR / "documents"
DEFAULT_PERSIST_DIR = A9_DIR / "chroma_db"
DEFAULT_RESULTS_DIR = A9_DIR / "results"
DEFAULT_COLLECTION_NAME = "u0_hw_09_corpus"
PLACEHOLDER_KEY = "student-should-use-their-own"


@dataclass(frozen=True)
class CorpusDocumentSpec:
    """Metadata and authority role for an official corpus PDF."""

    filename: str
    title: str
    edition_note: str
    source_role: str


CORPUS_DOCUMENTS: tuple[CorpusDocumentSpec, ...] = (
    CorpusDocumentSpec(
        filename="fda-food-code-2026.pdf",
        title="FDA Food Code 2026",
        edition_note="2026 Edition",
        source_role="FDA model code; jurisdiction-specific legal adoption requires a separate authority.",
    ),
    CorpusDocumentSpec(
        filename="fda-employee-health-personal-hygiene-handbook.pdf",
        title="Employee Health and Personal Hygiene Handbook",
        edition_note="Reflects the 2017 FDA Food Code edition",
        source_role="Operational guidance reflecting an earlier (2017) Food Code edition; subordinate to the 2026 Food Code for current model-code rules.",
    ),
    CorpusDocumentSpec(
        filename="fda-bad-bug-book-second-edition.pdf",
        title="Bad Bug Book, Second Edition",
        edition_note="Second Edition",
        source_role="FDA reference handbook of foodborne pathogenic microorganisms and natural toxins.",
    ),
    CorpusDocumentSpec(
        filename="fda-appendix-3-bacterial-pathogen-growth-inactivation.pdf",
        title="Appendix 3: Bacterial Pathogen Growth and Inactivation",
        edition_note="Draft Guidance",
        source_role="Draft, nonbinding FDA scientific guidance for food processing.",
    ),
    CorpusDocumentSpec(
        filename="utah-risk-factor-study-2016-2017.pdf",
        title="Utah Risk Factor Study, 2016-2017",
        edition_note="2016-2017 Observational Study",
        source_role="Historical Utah retail food inspection observations; does not establish current legal adoption or statutory penalties.",
    ),
)

CORPUS_BY_FILENAME: dict[str, CorpusDocumentSpec] = {
    spec.filename: spec for spec in CORPUS_DOCUMENTS
}

_SOURCE_ALIASES: dict[str, str] = {}
for _spec in CORPUS_DOCUMENTS:
    _SOURCE_ALIASES[_spec.filename.lower()] = _spec.filename
    _SOURCE_ALIASES[_spec.filename.removesuffix(".pdf").lower()] = _spec.filename
    _SOURCE_ALIASES[_spec.title.lower()] = _spec.filename


def resolve_source_filter(source: str) -> str:
    """Validate and resolve a source filename or title to its canonical PDF filename."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source filter must be a non-empty string")
    key = source.strip().lower()
    if key in _SOURCE_ALIASES:
        return _SOURCE_ALIASES[key]
    valid = ", ".join(sorted(CORPUS_BY_FILENAME.keys()))
    raise ValueError(f"Unknown corpus source {source!r}; valid sources: {valid}")


def load_dotenv_file(dotenv_path: Path | None = None, override: bool = False) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file without logging or exposing secret values."""
    path = dotenv_path if dotenv_path is not None else (A9_DIR / ".env")
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


@dataclass(frozen=True)
class ServiceConfig:
    """Connection settings for the OpenAI-compatible embedding and chat services."""

    embedding_base_url: str = "http://golem.cs.utahtech.edu:4000/v1"
    embedding_model: str = "Qwen3-Embedding-8B"
    embedding_api_key: str = ""
    llm_base_url: str = "http://golem:4000/v1"
    llm_model: str = "Qwen3.8-27B"
    llm_api_key: str = ""

    @classmethod
    def from_env(
        cls,
        dotenv_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> ServiceConfig:
        if environ is None:
            load_dotenv_file(dotenv_path)
            env: Mapping[str, str] = os.environ
        else:
            env = environ
        return cls(
            embedding_base_url=env.get(
                "EMBEDDING_BASE_URL", "http://golem.cs.utahtech.edu:4000/v1"
            ).strip(),
            embedding_model=env.get("EMBEDDING_MODEL", "Qwen3-Embedding-8B").strip(),
            embedding_api_key=env.get("EMBEDDING_API_KEY", "").strip(),
            llm_base_url=env.get("LLM_BASE_URL", "http://golem:4000/v1").strip(),
            llm_model=env.get("LLM_MODEL", "Qwen3.8-27B").strip(),
            llm_api_key=env.get("LLM_API_KEY", "").strip(),
        )

    def validate_embedding_credentials(self) -> None:
        if not self.embedding_api_key or self.embedding_api_key == PLACEHOLDER_KEY:
            raise ValueError(
                "EMBEDDING_API_KEY is missing or still set to the example placeholder; "
                "configure a local a9/.env file."
            )

    def validate_llm_credentials(self) -> None:
        if not self.llm_api_key or self.llm_api_key == PLACEHOLDER_KEY:
            raise ValueError(
                "LLM_API_KEY is missing or still set to the example placeholder; "
                "configure a local a9/.env file."
            )


@dataclass(frozen=True)
class RunConfig:
    """Recorded experimental configuration shared across indexing and query runs."""

    embedding_model: str = "Qwen3-Embedding-8B"
    llm_model: str = "Qwen3.8-27B"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_top_k: int = 5
    max_top_k: int = 10
    max_steps: int = 8
    temperature: float = 0.0
    seed: int = 4300
    collection_name: str = DEFAULT_COLLECTION_NAME

    def __post_init__(self) -> None:
        for field_name, val, minimum in (
            ("chunk_size", self.chunk_size, 100),
            ("chunk_overlap", self.chunk_overlap, 0),
            ("retrieval_top_k", self.retrieval_top_k, 1),
            ("max_top_k", self.max_top_k, 1),
            ("max_steps", self.max_steps, 1),
        ):
            if isinstance(val, bool) or not isinstance(val, int) or val < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        if self.retrieval_top_k > self.max_top_k:
            raise ValueError("retrieval_top_k cannot exceed max_top_k")

    def to_metadata_dict(self) -> dict[str, object]:
        """Return safe, non-secret configuration parameters for trace/result recording."""
        return asdict(self)

