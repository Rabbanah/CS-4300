"""Bounded ReAct retrieval agent and evidence/citation/termination controller."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from clients import ChatClient
from config import RunConfig
from ingest import RetrievedPassage, Retriever
from tools import (
    ClaimItem,
    InsufficientEvidenceArgs,
    RetrieveArgs,
    SubmitAnswerArgs,
    ToolCall,
    ToolName,
    ToolObservation,
    ToolValidationError,
    build_system_prompt,
    extract_inline_chunk_ids,
    parse_tool_call,
)


class TerminalReason(str, Enum):
    """Canonical terminal outcomes enforced by the ReAct controller."""

    ANSWER = "answer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ERROR = "error"
    STEP_LIMIT = "step_limit"


@dataclass(frozen=True)
class TraceStep:
    """Single inspectable action-observation step in a ReAct episode."""

    step: int
    raw_model_output: str
    parsed_call: ToolCall | None
    observation: ToolObservation

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "raw_model_output": self.raw_model_output,
            "parsed_call": self.parsed_call.to_dict() if self.parsed_call else None,
            "observation": self.observation.to_dict(),
        }


@dataclass(frozen=True)
class AgentRunResult:
    """Complete outcome, provenance citations, configuration, and trace for one query run."""

    question: str
    terminal_reason: TerminalReason
    final_output: str
    citations: tuple[str, ...]
    cited_passages: tuple[RetrievedPassage, ...]
    claims: tuple[ClaimItem, ...]
    steps_used: int
    max_steps: int
    runtime_seconds: float
    config: dict[str, Any]
    trace: tuple[TraceStep, ...]
    query_id: str | None = None
    error_detail: str | None = None

    @property
    def answered(self) -> bool:
        return self.terminal_reason == TerminalReason.ANSWER

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "question": self.question,
            "terminal_reason": self.terminal_reason.value,
            "final_output": self.final_output,
            "citations": list(self.citations),
            "claims": [asdict(c) for c in self.claims],
            "cited_passages": [asdict(p) for p in self.cited_passages],
            "steps_used": self.steps_used,
            "max_steps": self.max_steps,
            "runtime_seconds": round(self.runtime_seconds, 4),
            "error_detail": self.error_detail,
            "config": self.config,
            "trace": [step.to_dict() for step in self.trace],
        }


def _sanitize_exception(exc: Exception) -> str:
    """Produce a safe diagnostic string without leaking credentials or secret headers."""
    exc_type = type(exc).__name__
    msg = str(exc)
    # Strip any potential Bearer token or key-like substring
    if "bearer " in msg.lower() or "api_key" in msg.lower() or "sk-" in msg.lower():
        return f"{exc_type}: service/authentication failure (redacted)"
    return f"{exc_type}: {msg[:200]}"


class ReActRAGAgent:
    """Bounded ReAct agent enforcing retrieval provenance, citation validation, and step limits."""

    def __init__(
        self,
        chat_client: ChatClient,
        retriever: Retriever,
        config: RunConfig | None = None,
        validate_official_sources: bool = True,
    ) -> None:
        self.chat_client = chat_client
        self.retriever = retriever
        self.config = config if config is not None else RunConfig()
        self.validate_official_sources = validate_official_sources

    def _build_messages(
        self,
        question: str,
        trace: list[TraceStep],
    ) -> list[dict[str, str]]:
        system_prompt = build_system_prompt(
            default_top_k=self.config.retrieval_top_k,
            max_top_k=self.config.max_top_k,
            max_steps=self.config.max_steps,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Step 1 of {self.config.max_steps}: Select your first JSON tool call."
                ),
            },
        ]
        for entry in trace:
            clean_assistant = re.sub(
                r"<think>.*?</think>", "", entry.raw_model_output, flags=re.DOTALL | re.IGNORECASE
            ).strip()
            messages.append(
                {
                    "role": "assistant",
                    "content": clean_assistant or entry.raw_model_output,
                }
            )
            next_step = entry.step + 1
            prompt_suffix = (
                f"\n\nStep {next_step} of {self.config.max_steps}: "
                "Select your next single JSON tool call."
                if entry.observation.remaining_steps > 0
                else ""
            )
            messages.append(
                {
                    "role": "user",
                    "content": entry.observation.format_for_prompt() + prompt_suffix,
                }
            )
        return messages

    def _validate_citation_provenance(
        self,
        all_citations: tuple[str, ...],
        retrieved_by_id: dict[str, RetrievedPassage],
    ) -> str | None:
        """Return an error reason if any cited chunk_id was not retrieved in this run."""
        unretrieved = [cid for cid in all_citations if cid not in retrieved_by_id]
        if unretrieved:
            available = sorted(retrieved_by_id.keys())
            avail_preview = ", ".join(available[:10]) if available else "none"
            return (
                f"Invalid citation(s) not returned by retrieval in this run: {unretrieved}. "
                f"Retrieved chunk_ids so far: [{avail_preview}]."
            )
        return None

    def run(self, question: str, query_id: str | None = None) -> AgentRunResult:
        """Execute a bounded ReAct run for `question` and return an inspectable AgentRunResult."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        started = time.perf_counter()
        trace: list[TraceStep] = []
        retrieved_by_id: dict[str, RetrievedPassage] = {}
        accepted_retrievals = 0

        for step_num in range(1, self.config.max_steps + 1):
            remaining = self.config.max_steps - step_num
            messages = self._build_messages(question, trace)

            # 1. Invoke language model
            try:
                raw_output = self.chat_client.chat(messages)
            except Exception as exc:
                err_str = _sanitize_exception(exc)
                obs = ToolObservation(
                    tool="chat_client",
                    accepted=False,
                    step=step_num,
                    remaining_steps=remaining,
                    message="Model service call failed.",
                    rejection_reason=err_str,
                    terminal=True,
                    terminal_reason=TerminalReason.ERROR.value,
                )
                trace.append(
                    TraceStep(
                        step=step_num,
                        raw_model_output=f"<chat_error:{type(exc).__name__}>",
                        parsed_call=None,
                        observation=obs,
                    )
                )
                return AgentRunResult(
                    query_id=query_id,
                    question=question,
                    terminal_reason=TerminalReason.ERROR,
                    final_output="",
                    citations=(),
                    cited_passages=(),
                    claims=(),
                    steps_used=step_num,
                    max_steps=self.config.max_steps,
                    runtime_seconds=time.perf_counter() - started,
                    config=self.config.to_metadata_dict(),
                    trace=tuple(trace),
                    error_detail=err_str,
                )

            # 2. Parse & validate typed JSON tool schema
            try:
                call = parse_tool_call(
                    raw_output,
                    max_top_k=self.config.max_top_k,
                    validate_official_sources=self.validate_official_sources,
                )
            except ToolValidationError as exc:
                obs = ToolObservation(
                    tool="invalid_tool_call",
                    accepted=False,
                    step=step_num,
                    remaining_steps=remaining,
                    message="Tool call rejected by schema validator.",
                    rejection_reason=str(exc),
                    terminal=(remaining == 0),
                    terminal_reason=(
                        TerminalReason.STEP_LIMIT.value if remaining == 0 else None
                    ),
                )
                trace.append(
                    TraceStep(
                        step=step_num,
                        raw_model_output=raw_output,
                        parsed_call=None,
                        observation=obs,
                    )
                )
                continue

            # 3. Dispatch & enforce controller policies
            if call.tool == ToolName.RETRIEVE:
                assert isinstance(call.args, RetrieveArgs)
                k = (
                    call.args.top_k
                    if call.args.top_k is not None
                    else self.config.retrieval_top_k
                )
                try:
                    passages = self.retriever.retrieve(
                        query=call.args.query,
                        top_k=k,
                        source=call.args.source,
                    )
                except Exception as exc:
                    err_str = _sanitize_exception(exc)
                    obs = ToolObservation(
                        tool=ToolName.RETRIEVE.value,
                        accepted=False,
                        step=step_num,
                        remaining_steps=remaining,
                        message="Retrieval service error.",
                        rejection_reason=err_str,
                        terminal=True,
                        terminal_reason=TerminalReason.ERROR.value,
                    )
                    trace.append(
                        TraceStep(
                            step=step_num,
                            raw_model_output=raw_output,
                            parsed_call=call,
                            observation=obs,
                        )
                    )
                    return AgentRunResult(
                        query_id=query_id,
                        question=question,
                        terminal_reason=TerminalReason.ERROR,
                        final_output="",
                        citations=(),
                        cited_passages=(),
                        claims=(),
                        steps_used=step_num,
                        max_steps=self.config.max_steps,
                        runtime_seconds=time.perf_counter() - started,
                        config=self.config.to_metadata_dict(),
                        trace=tuple(trace),
                        error_detail=err_str,
                    )

                accepted_retrievals += 1
                for p in passages:
                    retrieved_by_id[p.chunk_id] = p

                src_msg = f" (source={call.args.source})" if call.args.source else ""
                obs = ToolObservation(
                    tool=ToolName.RETRIEVE.value,
                    accepted=True,
                    step=step_num,
                    remaining_steps=remaining,
                    message=(
                        f"Retrieved {len(passages)} passage(s) for query={call.args.query!r}{src_msg}."
                    ),
                    passages=tuple(passages),
                    terminal=(remaining == 0),
                    terminal_reason=(
                        TerminalReason.STEP_LIMIT.value if remaining == 0 else None
                    ),
                )
                trace.append(
                    TraceStep(
                        step=step_num,
                        raw_model_output=raw_output,
                        parsed_call=call,
                        observation=obs,
                    )
                )
                continue

            if call.tool == ToolName.SUBMIT_ANSWER:
                assert isinstance(call.args, SubmitAnswerArgs)
                if accepted_retrievals == 0:
                    reason = (
                        "Cannot submit_answer before retrieving evidence with `retrieve`."
                    )
                    obs = ToolObservation(
                        tool=ToolName.SUBMIT_ANSWER.value,
                        accepted=False,
                        step=step_num,
                        remaining_steps=remaining,
                        message="Submission rejected by evidence policy.",
                        rejection_reason=reason,
                        terminal=(remaining == 0),
                        terminal_reason=(
                            TerminalReason.STEP_LIMIT.value if remaining == 0 else None
                        ),
                    )
                    trace.append(
                        TraceStep(
                            step=step_num,
                            raw_model_output=raw_output,
                            parsed_call=call,
                            observation=obs,
                        )
                    )
                    continue

                # Gather explicit citations, per-claim citations, and inline answer chunk_ids
                combined_cites: list[str] = list(call.args.citations)
                for claim_item in call.args.claims:
                    for cid in claim_item.citations:
                        if cid not in combined_cites:
                            combined_cites.append(cid)
                for inline_cid in extract_inline_chunk_ids(call.args.answer):
                    if inline_cid not in combined_cites:
                        combined_cites.append(inline_cid)

                prov_err = self._validate_citation_provenance(
                    tuple(combined_cites), retrieved_by_id
                )
                if prov_err is not None:
                    obs = ToolObservation(
                        tool=ToolName.SUBMIT_ANSWER.value,
                        accepted=False,
                        step=step_num,
                        remaining_steps=remaining,
                        message="Submission rejected due to unretrieved citation.",
                        rejection_reason=prov_err,
                        terminal=(remaining == 0),
                        terminal_reason=(
                            TerminalReason.STEP_LIMIT.value if remaining == 0 else None
                        ),
                    )
                    trace.append(
                        TraceStep(
                            step=step_num,
                            raw_model_output=raw_output,
                            parsed_call=call,
                            observation=obs,
                        )
                    )
                    continue

                final_cites = tuple(combined_cites)
                cited_passages = tuple(retrieved_by_id[cid] for cid in final_cites)
                obs = ToolObservation(
                    tool=ToolName.SUBMIT_ANSWER.value,
                    accepted=True,
                    step=step_num,
                    remaining_steps=remaining,
                    message=f"Answer accepted with {len(final_cites)} verified citation(s).",
                    terminal=True,
                    terminal_reason=TerminalReason.ANSWER.value,
                )
                trace.append(
                    TraceStep(
                        step=step_num,
                        raw_model_output=raw_output,
                        parsed_call=call,
                        observation=obs,
                    )
                )
                return AgentRunResult(
                    query_id=query_id,
                    question=question,
                    terminal_reason=TerminalReason.ANSWER,
                    final_output=call.args.answer,
                    citations=final_cites,
                    cited_passages=cited_passages,
                    claims=call.args.claims,
                    steps_used=step_num,
                    max_steps=self.config.max_steps,
                    runtime_seconds=time.perf_counter() - started,
                    config=self.config.to_metadata_dict(),
                    trace=tuple(trace),
                )

            if call.tool == ToolName.INSUFFICIENT_EVIDENCE:
                assert isinstance(call.args, InsufficientEvidenceArgs)
                if accepted_retrievals == 0:
                    reason = (
                        "Cannot declare insufficient_evidence before searching the corpus with `retrieve`."
                    )
                    obs = ToolObservation(
                        tool=ToolName.INSUFFICIENT_EVIDENCE.value,
                        accepted=False,
                        step=step_num,
                        remaining_steps=remaining,
                        message="Insufficient-evidence declaration rejected.",
                        rejection_reason=reason,
                        terminal=(remaining == 0),
                        terminal_reason=(
                            TerminalReason.STEP_LIMIT.value if remaining == 0 else None
                        ),
                    )
                    trace.append(
                        TraceStep(
                            step=step_num,
                            raw_model_output=raw_output,
                            parsed_call=call,
                            observation=obs,
                        )
                    )
                    continue

                inline_cites = extract_inline_chunk_ids(call.args.explanation)
                combined_cites = list(call.args.citations)
                for cid in inline_cites:
                    if cid not in combined_cites:
                        combined_cites.append(cid)

                prov_err = self._validate_citation_provenance(
                    tuple(combined_cites), retrieved_by_id
                )
                if prov_err is not None:
                    obs = ToolObservation(
                        tool=ToolName.INSUFFICIENT_EVIDENCE.value,
                        accepted=False,
                        step=step_num,
                        remaining_steps=remaining,
                        message="Insufficient-evidence declaration cited unretrieved chunk.",
                        rejection_reason=prov_err,
                        terminal=(remaining == 0),
                        terminal_reason=(
                            TerminalReason.STEP_LIMIT.value if remaining == 0 else None
                        ),
                    )
                    trace.append(
                        TraceStep(
                            step=step_num,
                            raw_model_output=raw_output,
                            parsed_call=call,
                            observation=obs,
                        )
                    )
                    continue

                final_cites = tuple(combined_cites)
                cited_passages = tuple(retrieved_by_id[cid] for cid in final_cites)
                obs = ToolObservation(
                    tool=ToolName.INSUFFICIENT_EVIDENCE.value,
                    accepted=True,
                    step=step_num,
                    remaining_steps=remaining,
                    message="Insufficient-evidence outcome accepted after corpus search.",
                    terminal=True,
                    terminal_reason=TerminalReason.INSUFFICIENT_EVIDENCE.value,
                )
                trace.append(
                    TraceStep(
                        step=step_num,
                        raw_model_output=raw_output,
                        parsed_call=call,
                        observation=obs,
                    )
                )
                return AgentRunResult(
                    query_id=query_id,
                    question=question,
                    terminal_reason=TerminalReason.INSUFFICIENT_EVIDENCE,
                    final_output=call.args.explanation,
                    citations=final_cites,
                    cited_passages=cited_passages,
                    claims=(),
                    steps_used=step_num,
                    max_steps=self.config.max_steps,
                    runtime_seconds=time.perf_counter() - started,
                    config=self.config.to_metadata_dict(),
                    trace=tuple(trace),
                )

        return AgentRunResult(
            query_id=query_id,
            question=question,
            terminal_reason=TerminalReason.STEP_LIMIT,
            final_output="",
            citations=(),
            cited_passages=(),
            claims=(),
            steps_used=len(trace),
            max_steps=self.config.max_steps,
            runtime_seconds=time.perf_counter() - started,
            config=self.config.to_metadata_dict(),
            trace=tuple(trace),
        )

