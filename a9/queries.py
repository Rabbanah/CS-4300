"""Parser for `U0-HW-09-required-queries.md` ensuring verbatim query execution."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from config import A9_DIR


DEFAULT_QUERIES_PATH = A9_DIR / "U0-HW-09-required-queries.md"
_HEADING_RE = re.compile(r"^##\s+(Q\d+):\s*(.+)$")


@dataclass(frozen=True)
class RequiredQuery:
    query_id: str
    title: str
    question: str


def load_required_queries(path: Path = DEFAULT_QUERIES_PATH) -> tuple[RequiredQuery, ...]:
    """Parse `U0-HW-09-required-queries.md` and return each query's exact prompt text."""
    if not path.is_file():
        raise FileNotFoundError(f"Required queries file not found: {path}")

    current_id: str | None = None
    current_title: str = ""
    buffer: list[str] = []
    results: list[RequiredQuery] = []

    def _flush() -> None:
        if current_id is not None:
            raw_block = "\n".join(buffer).strip()
            # Collapse single markdown line wraps into clean spaces while preserving exact wording
            paragraphs = [
                " ".join(line.strip() for line in para.splitlines() if line.strip())
                for para in raw_block.split("\n\n")
                if para.strip()
            ]
            question_text = "\n\n".join(paragraphs)
            results.append(
                RequiredQuery(
                    query_id=current_id,
                    title=current_title,
                    question=question_text,
                )
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            _flush()
            current_id = m.group(1).strip()
            current_title = m.group(2).strip()
            buffer = []
        elif current_id is not None:
            buffer.append(line)

    _flush()
    return tuple(results)

