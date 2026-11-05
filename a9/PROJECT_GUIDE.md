# U0-HW-09 System Design, Policies, and Usage Guide

## 1. Architecture Overview

| Module | Responsibility |
|---|---|
| [`config.py`](config.py) | Official 5-PDF corpus registry (titles, editions, authority roles), `.env` loader, `ServiceConfig`, and `RunConfig`. |
| [`clients.py`](clients.py) | `EmbeddingClient` and `ChatClient` protocols, live `OpenAICompatibleEmbeddingClient` (`Qwen3-Embedding-8B`) and `OpenAICompatibleChatClient` (`Qwen3.8-Flash-Next` / `Qwen3.8-27B`), plus offline test doubles (`DeterministicHashEmbeddingClient`, `ScriptedChatClient`). |
| [`ingest.py`](ingest.py) | `pypdf` 1-based physical page extraction, newline-aware overlapping chunker, deterministic `chunk_id` builder (`<filename>::p<page:04d>::c<idx:02d>`), persistent `ChromaCorpusStore`, and offline `SubstituteRetriever`. |
| [`tools.py`](tools.py) | Validated typed tool schemas (`retrieve`, `submit_answer`, `insufficient_evidence`), duplicate-key-rejecting JSON parser (`<think>...</think>` aware), inline `chunk_id` extractor, and source-labeled `ToolObservation` formatter. |
| [`agent.py`](agent.py) | `ReActRAGAgent` bounded controller enforcing step limits, mandatory corpus search prior to terminal submission, strict per-run citation provenance validation, credential redaction, and the four `TerminalReason` outcomes. |
| [`queries.py`](queries.py) | Verbatim parser for `U0-HW-09-required-queries.md` (`Q1`–`Q9`). |
| [`build_index.py`](build_index.py) | CLI command (`--rebuild`) to ingest all 5 PDFs (1,110 pages) into persistent ChromaDB (`a9/chroma_db/`). |
| [`run_queries.py`](run_queries.py) | CLI runner executing ad-hoc questions or `Q1`–`Q9`, saving complete JSON and Markdown action-observation traces to `a9/results/`. |

---

## 2. Recorded Configuration

All nine required runs (`Q1`–`Q9`) in [`results/required_queries_report.md`](results/required_queries_report.md) and [`results/required_queries_results.json`](results/required_queries_results.json) share one recorded configuration:
- **Embedding Model**: `Qwen3-Embedding-8B` (3,054 chunks across 1,110 physical PDF pages)
- **Chat Model**: `Qwen3.8-Flash-Next`
- **Chunk Size**: `1200` characters
- **Chunk Overlap**: `200` characters
- **Retrieval Result Count (`top_k`)**: `5` (validated range `1..10`)
- **Model/Tool Step Limit (`max_steps`)**: `8`
- **Temperature**: `0.0` (`seed = 4300`)

---

## 3. Controller Policies & Terminal Reasons

1. **Retrieval Validation**: `retrieve` requires a non-empty `query`, validates optional `top_k` (`int` in `1..10`, rejecting `bool`), and validates optional `source` against the five official corpus documents. Each observation returns passage text together with `chunk_id`, `source` filename and title, authority role, and **1-based physical PDF page**.
2. **Citation Provenance**: Every `chunk_id` appearing in `citations`, `claims[i].citations`, or inline `[<chunk_id>]` inside `answer`/`explanation` must belong to the set of chunks returned by `retrieve` *during that specific run*. Unretrieved or fabricated chunk IDs are rejected by the controller.
3. **Mandatory Search Policy**: Neither `submit_answer` nor `insufficient_evidence` is accepted before at least one `retrieve` call succeeds in the current run.
4. **Four Terminal Reasons (`TerminalReason`)**:
   - `answer`: Source-supported answer accepted with verified retrieved citations (`Q1`–`Q7`).
   - `insufficient_evidence`: Corpus does not support an answer after searching (`Q8`–`Q9`).
   - `error`: Unrecoverable model, embedding, or vector-store error (with sensitive headers/keys redacted).
   - `step_limit`: `max_steps` reached before an accepted terminal tool call.

---

## 4. Running Offline Automated Tests

No `.env`, network access, or live services are needed:

```powershell
.\a9\.venv\Scripts\python.exe -m unittest discover -v -s a9 -t a9
```

---

## 5. Building the Index & Running Required Queries (`Q1`–`Q9`)

1. Copy [`model-service.env.example`](model-service.env.example) to `a9/.env` (ignored by Git) and fill in your assigned course API key while connected to the Utah Tech course network / Tailscale:
   ```powershell
   Copy-Item a9\model-service.env.example a9\.env
   # Edit a9\.env to set EMBEDDING_API_KEY and LLM_API_KEY
   ```
2. Build (or rebuild) the persistent ChromaDB vector index from all 5 supplied PDFs (1,110 pages):
   ```powershell
   .\a9\.venv\Scripts\python.exe a9\build_index.py --rebuild --embedding-base-url http://golem:4000/v1
   ```
   *(Over Tailscale MagicDNS, `http://golem:4000/v1` resolves directly; on campus, the default `EMBEDDING_BASE_URL` in `.env` also works.)*
3. Run all nine required queries (`Q1`–`Q9`) exactly as written and record complete traces + final outputs:
   ```powershell
   .\a9\.venv\Scripts\python.exe a9\run_queries.py --required --llm-model Qwen3.8-Flash-Next --embedding-base-url http://golem:4000/v1
   ```
   Outputs are written to:
   - `a9/results/required_queries_results.json`
   - `a9/results/required_queries_report.md`

