"""Offline unit tests for PDF ingestion, chunk metadata, and persistent ChromaDB retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from clients import DeterministicHashEmbeddingClient
from ingest import (
    ChromaCorpusStore,
    chunk_page_text,
    extract_pdf_chunks,
    make_chunk_id,
)


def _write_minimal_text_pdf(path: Path, page_texts: list[str]) -> None:
    """Create a valid multi-page PDF with selectable text streams using only stdlib."""

    def _escape_pdf_text(s: str) -> str:
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    objects: list[bytes] = []
    # 1: Catalog, 2: Pages, 3: Font
    page_obj_ids = [4 + 2 * i for i in range(len(page_texts))]
    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_texts)} >>".encode("ascii")
    )
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, text in enumerate(page_texts):
        page_id = 4 + 2 * i
        content_id = page_id + 1
        lines = text.splitlines() or [""]
        ops = ["BT", "/F1 11 Tf", "50 750 Td"]
        for line_idx, line in enumerate(lines):
            if line_idx > 0:
                ops.append("0 -16 Td")
            ops.append(f"({_escape_pdf_text(line)}) Tj")
        ops.append("ET")
        stream_bytes = "\n".join(ops).encode("latin-1", errors="replace")

        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        content_obj = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"\nendstream"
        )
        objects.append(page_dict)
        objects.append(content_obj)

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = [0]
    for obj_num, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        out.extend(body)
        out.extend(b"\nendobj\n")

    xref_pos = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    path.write_bytes(bytes(out))


class TestIngestionAndChromaStore(unittest.TestCase):
    """Tests ingestion metadata, 1-based physical PDF pages, stable chunk IDs, and Chroma persistence."""

    def test_make_chunk_id_is_deterministic_and_one_based(self) -> None:
        cid = make_chunk_id("fda-food-code-2026.pdf", page=1, chunk_index=0)
        self.assertEqual(cid, "fda-food-code-2026.pdf::p0001::c00")
        with self.assertRaises(ValueError):
            make_chunk_id("fda-food-code-2026.pdf", page=0, chunk_index=0)
        with self.assertRaises(ValueError):
            make_chunk_id("fda-food-code-2026.pdf", page=True, chunk_index=0)  # type: ignore[arg-type]

    def test_pdf_extraction_preserves_source_one_based_page_and_stable_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "fda-food-code-2026.pdf"
            long_page_2 = "\n".join(
                f"Section 2-201.12 Clause {i}: Exclude food employee with vomiting or diarrhea."
                for i in range(1, 25)
            )
            _write_minimal_text_pdf(
                pdf_path,
                [
                    "FDA Food Code 2026 Preface Page One.",
                    long_page_2,
                    "Section 2-201.13 Reinstatement after 24 hours asymptomatic.",
                ],
            )

            chunks_first = extract_pdf_chunks(pdf_path, chunk_size=400, chunk_overlap=80)
            chunks_second = extract_pdf_chunks(pdf_path, chunk_size=400, chunk_overlap=80)

            # Stable chunk IDs across repeated extractions
            self.assertEqual(
                [c.chunk_id for c in chunks_first],
                [c.chunk_id for c in chunks_second],
            )
            self.assertGreaterEqual(len(chunks_first), 4)

            # Check 1-based physical pages (1, 2, 3 — never 0)
            pages = {c.page for c in chunks_first}
            self.assertEqual(pages, {1, 2, 3})
            self.assertNotIn(0, pages)

            # Check metadata dictionary contract
            meta0 = chunks_first[0].to_metadata()
            self.assertEqual(meta0["source"], "fda-food-code-2026.pdf")
            self.assertEqual(meta0["source_title"], "FDA Food Code 2026")
            self.assertEqual(meta0["page"], 1)
            self.assertEqual(meta0["chunk_id"], "fda-food-code-2026.pdf::p0001::c00")

    def test_persistent_chromadb_rebuild_reload_and_source_filtering(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp_dir = Path(tmp)
            docs_dir = tmp_dir / "docs"
            chroma_dir = tmp_dir / "chroma_store"
            docs_dir.mkdir()

            pdf_fc = docs_dir / "fda-food-code-2026.pdf"
            pdf_bbb = docs_dir / "fda-bad-bug-book-second-edition.pdf"
            _write_minimal_text_pdf(
                pdf_fc,
                [
                    "2-201.12 Exclude a food employee experiencing vomiting or diarrhea.",
                    "2-201.13 Reinstate after asymptomatic for at least 24 hours.",
                ],
            )
            _write_minimal_text_pdf(
                pdf_bbb,
                [
                    "Norovirus infective dose is very low, estimated at 18 viral particles.",
                ],
            )

            chunks = extract_pdf_chunks(pdf_fc, chunk_size=500, chunk_overlap=50) + extract_pdf_chunks(
                pdf_bbb, chunk_size=500, chunk_overlap=50
            )
            embedder = DeterministicHashEmbeddingClient(dimensions=64)

            store = ChromaCorpusStore(
                persist_dir=chroma_dir,
                embedder=embedder,
                collection_name="test_hw09",
            )
            count = store.rebuild(chunks, batch_size=2)
            self.assertEqual(count, 3)
            store.close()

            # Reopen persistent store from disk to verify persistence
            reopened = ChromaCorpusStore(
                persist_dir=chroma_dir,
                embedder=embedder,
                collection_name="test_hw09",
            )
            self.assertEqual(reopened.collection_count(), 3)

            results = reopened.retrieve("norovirus infective dose viral particles", top_k=2)
            self.assertGreaterEqual(len(results), 1)
            self.assertEqual(results[0].source, "fda-bad-bug-book-second-edition.pdf")
            self.assertEqual(results[0].page, 1)
            self.assertEqual(
                results[0].chunk_id, "fda-bad-bug-book-second-edition.pdf::p0001::c00"
            )

            # Source filter restricts results strictly to FDA Food Code 2026
            filtered = reopened.retrieve(
                "vomiting asymptomatic 24 hours",
                top_k=5,
                source="FDA Food Code 2026",
            )
            self.assertEqual(len(filtered), 2)
            self.assertTrue(
                all(p.source == "fda-food-code-2026.pdf" for p in filtered)
            )

            # Rebuild replaces collection cleanly without duplicating chunks
            rebuilt_count = reopened.rebuild(chunks, batch_size=4)
            self.assertEqual(rebuilt_count, 3)
            reopened.close()

    def test_chunk_page_text_validation(self) -> None:
        with self.assertRaises(ValueError):
            chunk_page_text("sample", chunk_size=100, chunk_overlap=100)
        self.assertEqual(chunk_page_text("   \n\n  ", chunk_size=200, chunk_overlap=20), [])


if __name__ == "__main__":
    unittest.main()
