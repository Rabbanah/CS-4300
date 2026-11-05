"""CLI runner for single questions and the U0-HW-09 required query suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from agent import AgentRunResult, ReActRAGAgent
from clients import OpenAICompatibleChatClient, OpenAICompatibleEmbeddingClient
from config import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    DEFAULT_RESULTS_DIR,
    RunConfig,
    ServiceConfig,
)
from ingest import ChromaCorpusStore
from queries import DEFAULT_QUERIES_PATH, RequiredQuery, load_required_queries


def format_markdown_report(
    config: RunConfig,
    results: Sequence[tuple[RequiredQuery | None, AgentRunResult]],
) -> str:
    """Render an inspectable Markdown report containing configuration, traces, and final outputs."""
    lines = [
        "# U0-HW-09 Required Queries — Complete Traces & Results",
        "",
        "## Recorded Configuration",
        "",
        f"- **Embedding Model**: `{config.embedding_model}`",
        f"- **Chat Model**: `{config.llm_model}`",
        f"- **Chunk Size**: `{config.chunk_size}` characters",
        f"- **Chunk Overlap**: `{config.chunk_overlap}` characters",
        f"- **Retrieval Result Count (`top_k`)**: `{config.retrieval_top_k}` (max `{config.max_top_k}`)",
        f"- **Model/Tool Step Limit (`max_steps`)**: `{config.max_steps}`",
        f"- **Temperature**: `{config.temperature}`",
        f"- **Seed**: `{config.seed}`",
        "",
        "## Summary Table",
        "",
        "| Query | Title | Terminal Reason | Steps Used | Citations | Runtime (s) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for rq, res in results:
        qid = rq.query_id if rq else (res.query_id or "Ad-hoc")
        title = rq.title if rq else "Custom Question"
        lines.append(
            f"| `{qid}` | {title} | `{res.terminal_reason.value}` | "
            f"{res.steps_used}/{res.max_steps} | {len(res.citations)} | {res.runtime_seconds:.2f} |"
        )

    for rq, res in results:
        qid = rq.query_id if rq else (res.query_id or "Ad-hoc")
        title = f"{qid}: {rq.title}" if rq else qid
        lines.extend(
            [
                "",
                "---",
                "",
                f"## {title}",
                "",
                "### Exact Prompt",
                "",
                f"> {res.question}",
                "",
                f"- **Terminal Reason**: `{res.terminal_reason.value}`",
                f"- **Steps Used**: `{res.steps_used} / {res.max_steps}`",
                f"- **Verified Citations**: {', '.join(f'`{c}`' for c in res.citations) if res.citations else 'None'}",
                "",
                "### Final Output",
                "",
                res.final_output if res.final_output else f"*(No final text; ended with `{res.terminal_reason.value}`)*",
                "",
            ]
        )
        if res.claims:
            lines.append("### Structured Claims & Provenance")
            lines.append("")
            for idx, cl in enumerate(res.claims, 1):
                cites_str = ", ".join(f"`{c}`" for c in cl.citations)
                lines.append(f"{idx}. {cl.claim} — [{cites_str}]")
            lines.append("")

        if res.cited_passages:
            lines.append("### Cited Source Passages")
            lines.append("")
            for p in res.cited_passages:
                lines.append(
                    f"- **`{p.chunk_id}`** (`{p.source}`, *{p.source_title}*, physical page **{p.page}**):"
                )
                excerpt = p.text.replace("\n", " ")
                lines.append(f"  > {excerpt}")
            lines.append("")

        lines.append("### Complete Action-Observation Trace")
        lines.append("")
        for step in res.trace:
            tool_label = step.parsed_call.tool.value if step.parsed_call else "unparsed"
            lines.append(
                f"#### Step {step.step} (`{tool_label}`, accepted=`{step.observation.accepted}`, "
                f"remaining_steps=`{step.observation.remaining_steps}`)"
            )
            if step.parsed_call and step.parsed_call.thought:
                lines.append(f"- **Thought**: {step.parsed_call.thought}")
            if step.parsed_call:
                lines.append(
                    f"- **Tool Arguments**: `{json.dumps(step.parsed_call.to_dict()['args'], ensure_ascii=False)}`"
                )
            else:
                raw_clean = step.raw_model_output.replace("`", "'")
                lines.append(f"- **Raw Model Output**: `{raw_clean}`")
            lines.append(f"- **Observation Message**: {step.observation.message}")
            if step.observation.terminal:
                lines.append(
                    f"- **Terminal Transition**: `terminal=True`, `terminal_reason={step.observation.terminal_reason}`"
                )
            if step.observation.rejection_reason:
                lines.append(f"- **Rejection Reason**: `{step.observation.rejection_reason}`")
            if step.observation.passages:
                lines.append("- **Retrieved Passages Observation**:")
                for p in step.observation.passages:
                    full_text = " ".join(p.text.split())
                    dist_str = f"{p.distance:.4f}" if p.distance is not None else "n/a"
                    lines.append(
                        f"  - **`{p.chunk_id}`** | source=`{p.source}` (*{p.source_title}*, role: *{p.source_role}*) "
                        f"| page=`{p.page}` | distance=`{dist_str}`: {full_text}"
                    )
            lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the U0-HW-09 ReAct retrieval agent against the persistent ChromaDB store."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--required",
        action="store_true",
        help="Run the required queries from U0-HW-09-required-queries.md.",
    )
    group.add_argument(
        "--question",
        type=str,
        help="Run a single custom question.",
    )
    parser.add_argument(
        "--query-id",
        action="append",
        default=[],
        help="Optional filter when using --required (e.g. --query-id Q1 --query-id Q8).",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to U0-HW-09-required-queries.md.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Directory of persistent ChromaDB store (default: a9/chroma_db).",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
        help=f"Chroma collection name (default: {DEFAULT_COLLECTION_NAME}).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Default retrieval result count per tool call (default: 5).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum model/tool steps per run (default: 8).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Recorded chunk size matching the built index (default: 1200).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Recorded chunk overlap matching the built index (default: 200).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory to write JSON and Markdown trace outputs (default: a9/results).",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=None,
        help="Optional override for EMBEDDING_BASE_URL (e.g. http://golem:4000/v1 on Tailscale).",
    )
    parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional override for LLM_MODEL (e.g. Qwen3.8-Flash-Next or Gemma-4-26B-A4B).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    services = ServiceConfig.from_env()
    try:
        services.validate_embedding_credentials()
        services.validate_llm_credentials()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    llm_model = args.llm_model.strip() if args.llm_model else services.llm_model

    config = RunConfig(
        embedding_model=services.embedding_model,
        llm_model=llm_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        retrieval_top_k=args.top_k,
        max_steps=args.max_steps,
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
    )
    chat_client = OpenAICompatibleChatClient(
        base_url=services.llm_base_url,
        model=llm_model,
        api_key=services.llm_api_key,
        temperature=config.temperature,
        seed=config.seed,
    )
    store = ChromaCorpusStore(
        persist_dir=args.persist_dir,
        embedder=embedder,
        collection_name=args.collection,
    )

    if store.collection_count() == 0:
        print(
            f"Error: Chroma collection {args.collection!r} at {args.persist_dir} is empty. "
            "Run `python build_index.py --rebuild` first.",
            file=sys.stderr,
        )
        return 2

    agent = ReActRAGAgent(
        chat_client=chat_client,
        retriever=store,
        config=config,
    )

    if args.question:
        result = agent.run(args.question, query_id="custom")
        print(
            f"[custom] terminal_reason={result.terminal_reason.value} "
            f"steps={result.steps_used}/{result.max_steps} citations={list(result.citations)}"
        )
        print(result.final_output)
        return 0

    all_queries = load_required_queries(args.queries_file)
    if args.query_id:
        wanted = {q.upper() for q in args.query_id}
        selected = [q for q in all_queries if q.query_id.upper() in wanted]
    else:
        selected = list(all_queries)

    paired_results: list[tuple[RequiredQuery | None, AgentRunResult]] = []
    for rq in selected:
        print(f"Running {rq.query_id}: {rq.title}...", flush=True)
        res = agent.run(rq.question, query_id=rq.query_id)
        print(
            f"  -> terminal_reason={res.terminal_reason.value} | "
            f"steps={res.steps_used}/{res.max_steps} | "
            f"citations={len(res.citations)} | time={res.runtime_seconds:.2f}s",
            flush=True,
        )
        paired_results.append((rq, res))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "configuration": config.to_metadata_dict(),
        "runs": [res.to_dict() for _, res in paired_results],
    }
    json_path = args.output_dir / "required_queries_results.json"
    md_path = args.output_dir / "required_queries_report.md"

    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(
        format_markdown_report(config, paired_results), encoding="utf-8"
    )
    print(f"Saved complete JSON traces to {json_path}")
    print(f"Saved Markdown report to {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

