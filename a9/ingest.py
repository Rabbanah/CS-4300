"""PDF ingestion, deterministic chunking, persistent ChromaDB indexing, and retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence

import chromadb
import pypdf

from clients import ClientServiceError, EmbeddingClient
from config import (
    CORPUS_BY_FILENAME,
    CORPUS_DOCUMENTS,
    DEFAULT_COLLECTION_NAME,
    CorpusDocumentSpec,
    resolve_source_filter,
)


def make_chunk_id(source: str, page: int, chunk_index: int) -> str:
    """Construct a deterministic chunk identifier from source filename, 1-based page, and index."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty string")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("page must be a one-based integer >= 1")
    if isinstance(chunk_index, bool) or not isinstance(chunk_index, int) or chunk_index < 0:
        raise ValueError("chunk_index must be an integer >= 0")
    return f"{source.strip()}::p{page:04d}::c{chunk_index:02d}"


@dataclass(frozen=True)
class DocumentChunk:
    """A single indexed text chunk tied to an exact physical PDF page and source file."""

    chunk_id: str
    source: str
    source_title: str
    source_role: str
    page: int
    chunk_index: int
    text: str

    def __post_init__(self) -> None:
        if not self.source or not self.chunk_id or not self.text.strip():
            raise ValueError("DocumentChunk requires non-empty source, chunk_id, and text")
        if isinstance(self.page, bool) or not isinstance(self.page, int) or self.page < 1:
            raise ValueError("DocumentChunk.page must be a one-based integer >= 1")

    def to_metadata(self) -> dict[str, str | int]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "source_title": self.source_title,
            "source_role": self.source_role,
            "page": self.page,
            "chunk_index": self.chunk_index,
        }

    def text_for_embedding(self) -> str:
        """Include document title and 1-based page context alongside passage text for embedding."""
        return f"Document: {self.source_title} ({self.source}), Page {self.page}\n{self.text}"


@dataclass(frozen=True)
class RetrievedPassage:
    """A source-labeled passage returned by the retrieval tool."""

    chunk_id: str
    source: str
    source_title: str
    source_role: str
    page: int
    text: str
    distance: float | None = None

    def format_for_observation(self, rank: int) -> str:
        dist_str = f" | distance={self.distance:.4f}" if self.distance is not None else ""
        return (
            f"[{rank}] chunk_id={self.chunk_id} | source={self.source} "
            f"({self.source_title}) | page={self.page}{dist_str}\n"
            f"    role: {self.source_role}\n"
            f"    passage: {self.text}"
        )


class Retriever(Protocol):
    """Protocol for semantic passage retrieval exposed to the ReAct controller."""

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[RetrievedPassage]:
        """Return up to `top_k` source-labeled passages matching `query`."""


_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def normalize_page_text(raw_text: str) -> str:
    """Normalize PDF page text while preserving line breaks for tables and subsections."""
    cleaned = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in cleaned.split("\n")]
    joined = "\n".join(lines).strip()
    return _MULTI_BLANK_RE.sub("\n\n", joined)


def chunk_page_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[str]:
    """Deterministically split a single page's text into overlapping chunks."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 50:
        raise ValueError("chunk_size must be an integer >= 50")
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be an integer >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be strictly less than chunk_size")

    normalized = normalize_page_text(text)
    if not normalized:
        return []
    if len(normalized) <= chunk_size:
        return [normalized]

    chunks: list[str] = []
    start = 0
    length = len(normalized)
    min_break_offset = max(chunk_overlap + 1, chunk_size // 2)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = normalized[start:end]
            # Prefer breaking on paragraph, line, or space boundary in the second half of window
            chosen_rel = -1
            for sep in ("\n\n", "\n", " "):
                pos = window.rfind(sep, min_break_offset)
                if pos != -1:
                    chosen_rel = pos + (len(sep) if sep != " " else 0)
                    break
            if chosen_rel > 0:
                end = start + chosen_rel

        piece = normalized[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break

        next_start = max(start + 1, end - chunk_overlap)
        # Snap overlap start forward to next whitespace/line boundary if reasonable
        if next_start > start + 1 and normalized[next_start - 1].isalnum() and normalized[next_start].isalnum():
            space_pos = normalized.find(" ", next_start, min(end, next_start + 40))
            if space_pos != -1 and space_pos + 1 < end:
                next_start = space_pos + 1
        start = next_start

    return chunks


def extract_pdf_chunks(
    pdf_path: Path,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    doc_spec: CorpusDocumentSpec | None = None,
) -> list[DocumentChunk]:
    """Extract chunks from every non-empty physical page (1-based) of a PDF."""
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    filename = pdf_path.name
    spec = doc_spec or CORPUS_BY_FILENAME.get(
        filename,
        CorpusDocumentSpec(
            filename=filename,
            title=pdf_path.stem,
            edition_note="Test/Custom Document",
            source_role="Supplied source document.",
        ),
    )

    reader = pypdf.PdfReader(str(pdf_path))
    chunks: list[DocumentChunk] = []

    for zero_based_page, pdf_page in enumerate(reader.pages):
        one_based_page = zero_based_page + 1
        raw_text = pdf_page.extract_text() or ""
        page_pieces = chunk_page_text(
            raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        for chunk_idx, piece in enumerate(page_pieces):
            cid = make_chunk_id(spec.filename, one_based_page, chunk_idx)
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    source=spec.filename,
                    source_title=spec.title,
                    source_role=spec.source_role,
                    page=one_based_page,
                    chunk_index=chunk_idx,
                    text=piece,
                )
            )

    return chunks


def extract_corpus_chunks(
    documents_dir: Path,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
    require_official_five: bool = True,
) -> list[DocumentChunk]:
    """Extract chunks across the corpus directory in deterministic document order."""
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    all_chunks: list[DocumentChunk] = []
    if require_official_five:
        missing = [
            spec.filename
            for spec in CORPUS_DOCUMENTS
            if not (documents_dir / spec.filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required official PDF(s) in {documents_dir}: {', '.join(missing)}"
            )
        for spec in CORPUS_DOCUMENTS:
            pdf_path = documents_dir / spec.filename
            all_chunks.extend(
                extract_pdf_chunks(
                    pdf_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    doc_spec=spec,
                )
            )
    else:
        for pdf_path in sorted(documents_dir.glob("*.pdf")):
            all_chunks.extend(
                extract_pdf_chunks(
                    pdf_path,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            )

    return all_chunks


class ChromaCorpusStore:
    """Persistent ChromaDB store holding source- and page-labeled document chunks."""

    def __init__(
        self,
        persist_dir: Path,
        embedder: EmbeddingClient,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        allow_custom_sources: bool = False,
    ) -> None:
        self.persist_dir = Path(persist_dir)
        self.embedder = embedder
        self.collection_name = collection_name
        self.allow_custom_sources = allow_custom_sources
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

    def close(self) -> None:
        """Stop underlying ChromaDB system handles and clear shared system cache."""
        system = getattr(self._client, "_system", None)
        if system is not None and hasattr(system, "stop"):
            try:
                system.stop()
            except Exception:
                pass
        try:
            from chromadb.api.client import SharedSystemClient

            SharedSystemClient.clear_system_cache()
        except Exception:
            pass

    def collection_count(self) -> int:
        collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return int(collection.count())

    def rebuild(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int = 256,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Delete any existing collection and index all supplied chunks."""
        existing = [c.name for c in self._client.list_collections()]
        if self.collection_name in existing:
            self._client.delete_collection(self.collection_name)

        self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self.index_missing(
            chunks, batch_size=batch_size, progress_callback=progress_callback
        )

    def index_missing(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int = 256,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Index any chunks not yet stored in the collection (supports resumable builds)."""
        collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        existing_ids: set[str] = set(collection.get(include=[])["ids"])
        pending = [c for c in chunks if c.chunk_id not in existing_ids]
        total = len(chunks)
        already = len(existing_ids)

        if already and progress_callback is not None:
            progress_callback(already, total)

        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            embeddings = self.embedder.embed_texts(
                [chunk.text_for_embedding() for chunk in batch]
            )
            collection.add(
                ids=[chunk.chunk_id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                embeddings=embeddings,  # type: ignore[arg-type]
                metadatas=[chunk.to_metadata() for chunk in batch],
            )
            if progress_callback is not None:
                progress_callback(min(already + start + len(batch), total), total)

        return int(collection.count())

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[RetrievedPassage]:
        """Embed `query` and retrieve the top-k matching chunks from persistent ChromaDB."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        where_filter: dict[str, str] | None = None
        if source is not None:
            resolved = (
                source.strip()
                if self.allow_custom_sources and source.strip() not in CORPUS_BY_FILENAME
                else resolve_source_filter(source)
            )
            where_filter = {"source": resolved}

        try:
            collection = self._client.get_collection(name=self.collection_name)
        except Exception as exc:
            raise ClientServiceError(
                f"Chroma collection {self.collection_name!r} is not initialized; run build_index.py first."
            ) from exc

        total_items = int(collection.count())
        if total_items == 0:
            return []

        query_vec = self.embedder.embed_texts([query.strip()])[0]
        query_kwargs: dict[str, object] = {
            "query_embeddings": [query_vec],
            "n_results": min(top_k, total_items),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter is not None:
            query_kwargs["where"] = where_filter

        raw = collection.query(**query_kwargs)  # type: ignore[arg-type]
        ids_list = (raw.get("ids") or [[]])[0]
        docs_list = (raw.get("documents") or [[]])[0]
        metas_list = (raw.get("metadatas") or [[]])[0]
        dists_list = (raw.get("distances") or [[]])[0]

        passages: list[RetrievedPassage] = []
        for idx, cid in enumerate(ids_list):
            meta = metas_list[idx] or {}
            doc_text = docs_list[idx] or ""
            dist = float(dists_list[idx]) if idx < len(dists_list) and dists_list[idx] is not None else None
            passages.append(
                RetrievedPassage(
                    chunk_id=str(meta.get("chunk_id", cid)),
                    source=str(meta.get("source", "")),
                    source_title=str(meta.get("source_title", "")),
                    source_role=str(meta.get("source_role", "")),
                    page=int(meta["page"]),
                    text=str(doc_text),
                    distance=dist,
                )
            )
        return passages


class SubstituteRetriever:
    """Offline substitute retriever for deterministic controller and ReAct agent tests."""

    def __init__(
        self,
        passages: Sequence[RetrievedPassage] = (),
        by_query: Mapping[str, Sequence[RetrievedPassage]] | None = None,
        error_to_raise: Exception | None = None,
    ) -> None:
        self._passages = tuple(passages)
        self._by_query = {k: tuple(v) for k, v in (by_query or {}).items()}
        self._error_to_raise = error_to_raise
        self.calls: list[tuple[str, int, str | None]] = []

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[RetrievedPassage]:
        self.calls.append((query, top_k, source))
        if self._error_to_raise is not None:
            raise self._error_to_raise

        resolved_source = resolve_source_filter(source) if source is not None else None
        candidates = self._by_query.get(query, self._passages)
        if resolved_source is not None:
            candidates = tuple(p for p in candidates if p.source == resolved_source)
        return list(candidates[:top_k])
