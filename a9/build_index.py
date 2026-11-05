"""CLI command to build or rebuild the persistent ChromaDB vector store for U0-HW-09."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from clients import OpenAICompatibleEmbeddingClient
from config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_DOCUMENTS_DIR,
    DEFAULT_PERSIST_DIR,
    RunConfig,
    ServiceConfig,
)
from ingest import ChromaCorpusStore, extract_corpus_chunks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest the 5 supplied PDFs into a persistent ChromaDB vector store."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete any existing collection and rebuild the persistent ChromaDB index from scratch.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted index build by embedding only missing chunk_ids.",
    )
    parser.add_argument(
        "--documents-dir",
        type=Path,
        default=DEFAULT_DOCUMENTS_DIR,
        help="Directory containing the 5 supplied PDFs (default: a9/documents).",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Directory for persistent ChromaDB storage (default: a9/chroma_db).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Target maximum characters per chunk (default: 1200).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Character overlap between consecutive chunks on a page (default: 200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of chunks embedded per API request (default: 32).",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=None,
        help="Optional override for EMBEDDING_BASE_URL (e.g. http://golem:4000/v1 on Tailscale).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    services = ServiceConfig.from_env()
    try:
        services.validate_embedding_credentials()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    run_cfg = RunConfig(
        embedding_model=services.embedding_model,
        llm_model=services.llm_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        collection_name=args.collection,
    )

    embedding_base_url = (
        args.embedding_base_url.strip()
        if args.embedding_base_url
        else services.embedding_base_url
    )
    embedder = OpenAICompatibleEmbeddingClient(
        base_url=embedding_base_url,
        model=services.embedding_model,
        api_key=services.embedding_api_key,
        batch_size=args.batch_size,
    )

    store = ChromaCorpusStore(
        persist_dir=args.persist_dir,
        embedder=embedder,
        collection_name=args.collection,
    )

    existing_count = store.collection_count()
    if existing_count > 0 and not (args.rebuild or args.resume):
        print(
            f"Collection {args.collection!r} already contains {existing_count} chunks at "
            f"{args.persist_dir}. Pass --rebuild to recreate from scratch or --resume to finish a partial build."
        )
        return 0

    # Fast pre-flight check before spending ~90s extracting 1,110 PDF pages
    print(f"Verifying embedding service ({services.embedding_model} at {embedding_base_url})...", flush=True)
    probe = embedder.embed_texts(["ChromaDB pre-flight connectivity probe"])
    print(f"Embedding service OK (vector dim={len(probe[0])}).", flush=True)

    print(
        f"Extracting pages from {args.documents_dir} "
        f"(chunk_size={run_cfg.chunk_size}, chunk_overlap={run_cfg.chunk_overlap})..."
    )
    t0 = time.perf_counter()
    chunks = extract_corpus_chunks(
        documents_dir=args.documents_dir,
        chunk_size=run_cfg.chunk_size,
        chunk_overlap=run_cfg.chunk_overlap,
        require_official_five=True,
    )
    extract_secs = time.perf_counter() - t0
    print(f"Extracted {len(chunks)} chunks across 5 PDFs in {extract_secs:.1f}s.", flush=True)

    def _progress(done: int, total: int) -> None:
        pct = 100.0 * done / max(1, total)
        print(f"  Indexed {done}/{total} chunks ({pct:.1f}%)...", flush=True)

    t1 = time.perf_counter()
    super_batch = max(args.batch_size, args.batch_size * 2)
    if args.rebuild:
        indexed_count = store.rebuild(
            chunks, batch_size=super_batch, progress_callback=_progress
        )
    else:
        indexed_count = store.index_missing(
            chunks, batch_size=super_batch, progress_callback=_progress
        )
    embed_secs = time.perf_counter() - t1

    manifest = {
        "collection_name": args.collection,
        "embedding_model": services.embedding_model,
        "chunk_size": run_cfg.chunk_size,
        "chunk_overlap": run_cfg.chunk_overlap,
        "indexed_chunks": indexed_count,
        "sources": sorted({c.source for c in chunks}),
        "build_seconds": round(extract_secs + embed_secs, 2),
    }
    manifest_path = args.persist_dir / "index_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(
        f"Completed ChromaDB build: {indexed_count} chunks stored in {args.persist_dir} "
        f"({embed_secs:.1f}s embedding/indexing)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

