"""Typed tool schemas, strict JSON parser, citation extractor, and observations for U0-HW-09."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Sequence, Union

from config import CORPUS_DOCUMENTS, resolve_source_filter
from ingest import RetrievedPassage


class ToolName(str, Enum):
    RETRIEVE = "retrieve"
    SUBMIT_ANSWER = "submit_answer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class ClaimItem:
    """A single factual claim paired with the chunk identifiers that support it."""

    claim: str
    citations: tuple[str, ...]


@dataclass(frozen=True)
class RetrieveArgs:
    """Validated arguments for semantic retrieval over the ChromaDB corpus."""

    query: str
    source: str | None = None
    top_k: int | None = None


@dataclass(frozen=True)
class SubmitAnswerArgs:
    """Validated arguments for a source-supported final answer."""

    answer: str
    citations: tuple[str, ...]
    claims: tuple[ClaimItem, ...] = ()


@dataclass(frozen=True)
class InsufficientEvidenceArgs:
    """Validated arguments when the corpus does not support answering the question."""

    explanation: str
    citations: tuple[str, ...] = ()


ToolArgs = Union[RetrieveArgs, SubmitAnswerArgs, InsufficientEvidenceArgs]


@dataclass(frozen=True)
class ToolCall:
    tool: ToolName
    args: ToolArgs
    thought: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool.value,
            "thought": self.thought,
            "args": asdict(self.args),
        }


class ToolValidationError(ValueError):
    """Raised when a model response violates the JSON envelope or typed tool schema."""


@dataclass(frozen=True)
class ToolObservation:
    """Structured observation returned by the controller after validating/executing a step."""

    tool: str
    accepted: bool
    step: int
    remaining_steps: int
    message: str
    passages: tuple[RetrievedPassage, ...] = ()
    rejection_reason: str | None = None
    terminal: bool = False
    terminal_reason: str | None = None

    def format_for_prompt(self) -> str:
        header = (
            f"OBSERVATION [step={self.step}, remaining_steps={self.remaining_steps}, "
            f"tool={self.tool}, accepted={self.accepted}]"
        )
        lines = [header]
        if self.rejection_reason:
            lines.append(f"REJECTION_REASON: {self.rejection_reason}")
        if self.message:
            lines.append(f"SUMMARY: {self.message}")
        if self.passages:
            lines.append("RETRIEVED_PASSAGES:")
            for rank, p in enumerate(self.passages, 1):
                lines.append(p.format_for_observation(rank))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "accepted": self.accepted,
            "step": self.step,
            "remaining_steps": self.remaining_steps,
            "message": self.message,
            "rejection_reason": self.rejection_reason,
            "terminal": self.terminal,
            "terminal_reason": self.terminal_reason,
            "passages": [asdict(p) for p in self.passages],
        }


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
CHUNK_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+::p\d{4}::c\d{2}")


def extract_inline_chunk_ids(text: str) -> tuple[str, ...]:
    """Extract any chunk_id-formatted tokens appearing inline in answer prose."""
    seen: list[str] = []
    for match in CHUNK_ID_PATTERN.findall(text or ""):
        if match not in seen:
            seen.append(match)
    return tuple(seen)


def _no_duplicate_keys_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    obj: dict[str, Any] = {}
    for k, v in pairs:
        if k in obj:
            raise ToolValidationError(f"Duplicate JSON key: {k!r}")
        obj[k] = v
    return obj


def _extract_json_and_think(raw_text: str) -> tuple[str, str]:
    """Separate optional `<think>...</think>` reasoning blocks and markdown code fences."""
    think_parts: list[str] = []

    def _collect(match: re.Match[str]) -> str:
        think_parts.append(match.group(1).strip())
        return ""

    stripped = _THINK_BLOCK_RE.sub(_collect, raw_text).strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    return stripped, "\n".join(p for p in think_parts if p)


def _require_nonempty_str(val: Any, field: str, max_len: int = 16000) -> str:
    if not isinstance(val, str) or not val.strip():
        raise ToolValidationError(f"Field {field!r} must be a non-empty string")
    cleaned = val.strip()
    if len(cleaned) > max_len:
        raise ToolValidationError(f"Field {field!r} exceeds maximum length {max_len}")
    return cleaned


def _require_citation_list(val: Any, field: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(val, list) or isinstance(val, (str, bytes)):
        raise ToolValidationError(f"Field {field!r} must be a JSON array of chunk_id strings")
    cleaned: list[str] = []
    for item in val:
        if not isinstance(item, str) or not item.strip():
            raise ToolValidationError(f"Every entry in {field!r} must be a non-empty string")
        cid = item.strip()
        if cid not in cleaned:
            cleaned.append(cid)
    if not allow_empty and not cleaned:
        raise ToolValidationError(f"Field {field!r} must contain at least one chunk_id")
    return tuple(cleaned)


def validate_retrieve_args(
    arg_dict: dict[str, Any],
    max_top_k: int = 10,
    validate_official_sources: bool = True,
) -> RetrieveArgs:
    """Validate arguments for the `retrieve` tool."""
    unknown = set(arg_dict.keys()) - {"query", "source", "top_k"}
    if unknown:
        raise ToolValidationError(f"Unknown argument(s) for retrieve: {sorted(unknown)}")
    if "query" not in arg_dict:
        raise ToolValidationError("retrieve requires non-empty string field 'query'")

    query = _require_nonempty_str(arg_dict["query"], "query", max_len=600)

    source: str | None = None
    if "source" in arg_dict and arg_dict["source"] is not None:
        raw_source = _require_nonempty_str(arg_dict["source"], "source", max_len=200)
        if validate_official_sources:
            try:
                source = resolve_source_filter(raw_source)
            except ValueError as exc:
                raise ToolValidationError(str(exc)) from exc
        else:
            source = raw_source

    top_k: int | None = None
    if "top_k" in arg_dict and arg_dict["top_k"] is not None:
        raw_k = arg_dict["top_k"]
        if isinstance(raw_k, bool) or not isinstance(raw_k, int):
            raise ToolValidationError("Field 'top_k' must be an integer")
        if raw_k < 1 or raw_k > max_top_k:
            raise ToolValidationError(f"Field 'top_k' must be between 1 and {max_top_k}")
        top_k = raw_k

    return RetrieveArgs(query=query, source=source, top_k=top_k)


def parse_tool_call(
    raw_text: str,
    max_top_k: int = 10,
    validate_official_sources: bool = True,
) -> ToolCall:
    """Strictly parse and validate one typed JSON tool call from raw model output."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ToolValidationError("Model response must be a non-empty JSON object")

    json_candidate, think_prefix = _extract_json_and_think(raw_text)
    if not json_candidate:
        raise ToolValidationError("Model response contained no JSON object after <think> block")

    try:
        payload = json.loads(json_candidate, object_pairs_hook=_no_duplicate_keys_hook)
    except ToolValidationError:
        raise
    except Exception as exc:
        raise ToolValidationError(f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ToolValidationError("Top-level JSON value must be an object")

    if "tool" not in payload or not isinstance(payload["tool"], str):
        raise ToolValidationError("JSON object must include string field 'tool'")

    try:
        tool_name = ToolName(payload["tool"].strip())
    except ValueError as exc:
        valid = ", ".join(t.value for t in ToolName)
        raise ToolValidationError(
            f"Unknown tool {payload['tool']!r}; valid tools: {valid}"
        ) from exc

    thought = think_prefix
    if "thought" in payload and payload["thought"] is not None:
        if not isinstance(payload["thought"], str):
            raise ToolValidationError("Field 'thought' must be a string")
        explicit_thought = payload["thought"].strip()
        thought = explicit_thought if explicit_thought else think_prefix

    if "args" in payload:
        extra_top = set(payload.keys()) - {"tool", "thought", "args"}
        if extra_top:
            raise ToolValidationError(
                f"Cannot combine nested 'args' with top-level fields {sorted(extra_top)}"
            )
        if not isinstance(payload["args"], dict):
            raise ToolValidationError("Field 'args' must be a JSON object")
        arg_dict: dict[str, Any] = payload["args"]
    else:
        arg_dict = {k: v for k, v in payload.items() if k not in ("tool", "thought")}

    if tool_name == ToolName.RETRIEVE:
        args = validate_retrieve_args(
            arg_dict,
            max_top_k=max_top_k,
            validate_official_sources=validate_official_sources,
        )
        return ToolCall(tool=tool_name, args=args, thought=thought)

    if tool_name == ToolName.SUBMIT_ANSWER:
        unknown = set(arg_dict.keys()) - {"answer", "citations", "claims"}
        if unknown:
            raise ToolValidationError(
                f"Unknown argument(s) for submit_answer: {sorted(unknown)}"
            )
        if "answer" not in arg_dict or "citations" not in arg_dict:
            raise ToolValidationError(
                "submit_answer requires 'answer' and 'citations'"
            )
        answer = _require_nonempty_str(arg_dict["answer"], "answer")
        citations = _require_citation_list(
            arg_dict["citations"], "citations", allow_empty=False
        )
        claims: list[ClaimItem] = []
        if "claims" in arg_dict and arg_dict["claims"] is not None:
            raw_claims = arg_dict["claims"]
            if not isinstance(raw_claims, list):
                raise ToolValidationError("Field 'claims' must be a JSON array")
            for idx, item in enumerate(raw_claims):
                if not isinstance(item, dict) or set(item.keys()) != {"claim", "citations"}:
                    raise ToolValidationError(
                        f"claims[{idx}] must be an object with exact keys 'claim' and 'citations'"
                    )
                c_text = _require_nonempty_str(item["claim"], f"claims[{idx}].claim")
                c_cites = _require_citation_list(
                    item["citations"], f"claims[{idx}].citations", allow_empty=False
                )
                claims.append(ClaimItem(claim=c_text, citations=c_cites))
        return ToolCall(
            tool=tool_name,
            args=SubmitAnswerArgs(
                answer=answer, citations=citations, claims=tuple(claims)
            ),
            thought=thought,
        )

    if tool_name == ToolName.INSUFFICIENT_EVIDENCE:
        unknown = set(arg_dict.keys()) - {"explanation", "citations"}
        if unknown:
            raise ToolValidationError(
                f"Unknown argument(s) for insufficient_evidence: {sorted(unknown)}"
            )
        if "explanation" not in arg_dict:
            raise ToolValidationError(
                "insufficient_evidence requires non-empty string field 'explanation'"
            )
        explanation = _require_nonempty_str(arg_dict["explanation"], "explanation")
        citations = ()
        if "citations" in arg_dict and arg_dict["citations"] is not None:
            citations = _require_citation_list(
                arg_dict["citations"], "citations", allow_empty=True
            )
        return ToolCall(
            tool=tool_name,
            args=InsufficientEvidenceArgs(explanation=explanation, citations=citations),
            thought=thought,
        )

    raise AssertionError(f"Unhandled tool: {tool_name}")


def build_system_prompt(default_top_k: int, max_top_k: int, max_steps: int) -> str:
    """Construct the developer system prompt describing source roles, policies, and JSON schemas."""
    sources_lines = "\n".join(
        f"- `{spec.filename}` ({spec.title}, {spec.edition_note}): {spec.source_role}"
        for spec in CORPUS_DOCUMENTS
    )
    return (
        "You are a bounded ReAct regulatory & food-safety research agent.\n"
        "You answer questions EXCLUSIVELY from passages retrieved during the current run via the `retrieve` tool.\n\n"
        "CORPUS DOCUMENTS & AUTHORITY ROLES:\n"
        f"{sources_lines}\n\n"
        "STRICT EVIDENCE & POLICY RULES:\n"
        f"1. Step budget: You have at most {max_steps} total tool steps. Plan focused searches and use the optional `source` filter when a question targets a specific document.\n"
        "2. Grounding: Every factual claim in `submit_answer` MUST follow directly from passages returned by `retrieve` in THIS run. Do NOT use outside/background knowledge (even if true in the real world or pop culture).\n"
        "3. Citation provenance: Every `chunk_id` listed in `citations` (or inline in `answer`) MUST exactly match a `chunk_id` returned by `retrieve` during this run. Never invent or guess a `chunk_id`.\n"
        "4. Source roles & versions: Distinguish model codes (FDA Food Code 2026) from earlier-edition operational guidance (Employee Health Handbook, based on 2017 Food Code), pathogen reference material (Bad Bug Book), draft nonbinding processing guidance (Appendix 3), and historical observational studies (Utah Risk Factor Study 2016-2017). Note that the corpus does NOT contain current Utah statutory/regulatory adoption rules or legal penalties.\n"
        "5. Insufficient evidence: If, after searching relevant corpus sources with `retrieve`, the retrieved documents do not support answering the question (e.g., missing jurisdictional law/penalties or off-topic questions), call `insufficient_evidence` explaining clearly what the corpus lacks.\n\n"
        "OUTPUT FORMAT:\n"
        "Output ONLY a single valid JSON object (no markdown prose outside JSON, no duplicate keys) matching one of these three tool schemas:\n"
        "1. Retrieve passages:\n"
        f'   {{"thought": "<concise search plan>", "tool": "retrieve", "query": "<focused search terms>", "source": "<optional exact filename from list above or null>", "top_k": {default_top_k}}}\n'
        f"   (`top_k` is optional, integer 1..{max_top_k}).\n"
        "2. Submit a source-supported answer:\n"
        '   {"thought": "<verification that every claim is backed by retrieved chunk_ids>", "tool": "submit_answer", "answer": "<clear, thorough answer citing [chunk_id] inline>", "claims": [{"claim": "<specific claim>", "citations": ["<chunk_id>"]}], "citations": ["<chunk_id_1>", "<chunk_id_2>"]}\n'
        "3. Report insufficient evidence:\n"
        '   {"thought": "<why retrieved passages do not answer the question>", "tool": "insufficient_evidence", "explanation": "<clear statement of what cannot be answered from the corpus and why>", "citations": ["<optional retrieved chunk_id showing scope limits>"]}'
    )

