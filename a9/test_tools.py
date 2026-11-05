"""Offline unit tests for typed retrieval-tool validation and source-labeled observations."""

from __future__ import annotations

import unittest

from ingest import RetrievedPassage
from queries import DEFAULT_QUERIES_PATH, load_required_queries
from tools import (
    InsufficientEvidenceArgs,
    RetrieveArgs,
    SubmitAnswerArgs,
    ToolName,
    ToolObservation,
    ToolValidationError,
    parse_tool_call,
)


class TestToolValidationAndFormatting(unittest.TestCase):
    """Tests typed tool argument validation and source/page/chunk observation formatting."""

    def test_valid_retrieve_call_and_source_alias_normalization(self) -> None:
        call = parse_tool_call(
            '{"thought": "Look up 2-201.12", "tool": "retrieve", '
            '"query": "2-201.12 vomiting exclude", "source": "FDA Food Code 2026", "top_k": 4}',
            max_top_k=10,
        )
        self.assertEqual(call.tool, ToolName.RETRIEVE)
        self.assertEqual(
            call.args,
            RetrieveArgs(
                query="2-201.12 vomiting exclude",
                source="fda-food-code-2026.pdf",
                top_k=4,
            ),
        )

    def test_think_tag_and_fenced_json_parsed_cleanly(self) -> None:
        raw = (
            "<think>Need to query Appendix 3 Table 3-A for Salmonella and Listeria.</think>\n"
            "```json\n"
            '{"tool": "retrieve", "query": "Table 3-A Salmonella Listeria", '
            '"source": "fda-appendix-3-bacterial-pathogen-growth-inactivation.pdf"}\n'
            "```"
        )
        call = parse_tool_call(raw)
        self.assertEqual(call.tool, ToolName.RETRIEVE)
        self.assertIn("Appendix 3", call.thought)
        assert isinstance(call.args, RetrieveArgs)
        self.assertEqual(
            call.args.source,
            "fda-appendix-3-bacterial-pathogen-growth-inactivation.pdf",
        )

    def test_invalid_retrieve_arguments_rejected(self) -> None:
        bad_payloads = [
            "",  # empty
            "not json",  # prose
            '{"tool": "retrieve"}',  # missing query
            '{"tool": "retrieve", "query": "   "}',  # blank query
            '{"tool": "retrieve", "query": "norovirus", "top_k": 0}',  # top_k < 1
            '{"tool": "retrieve", "query": "norovirus", "top_k": 99}',  # top_k > max
            '{"tool": "retrieve", "query": "norovirus", "top_k": true}',  # bool as int
            '{"tool": "retrieve", "query": "norovirus", "source": "wikipedia.pdf"}',  # unknown source
            '{"tool": "retrieve", "query": "a", "query": "b"}',  # duplicate key
            '{"tool": "retrieve", "query": "norovirus", "extra": 1}',  # unknown field
        ]
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ToolValidationError):
                    parse_tool_call(payload, max_top_k=10)

    def test_submit_answer_and_insufficient_evidence_schema_validation(self) -> None:
        ans_call = parse_tool_call(
            '{"tool": "submit_answer", "answer": "Exclude the employee.", '
            '"citations": ["fda-food-code-2026.pdf::p0072::c00"], '
            '"claims": [{"claim": "Exclude employee with vomiting.", '
            '"citations": ["fda-food-code-2026.pdf::p0072::c00"]}]}'
        )
        self.assertEqual(ans_call.tool, ToolName.SUBMIT_ANSWER)
        self.assertIsInstance(ans_call.args, SubmitAnswerArgs)

        insuff_call = parse_tool_call(
            '{"tool": "insufficient_evidence", '
            '"explanation": "Corpus contains no Utah statute adopting the 2026 Food Code."}'
        )
        self.assertEqual(insuff_call.tool, ToolName.INSUFFICIENT_EVIDENCE)
        self.assertIsInstance(insuff_call.args, InsufficientEvidenceArgs)

        # Empty citations in submit_answer must fail validation
        with self.assertRaises(ToolValidationError):
            parse_tool_call('{"tool": "submit_answer", "answer": "Yes.", "citations": []}')

    def test_retrieval_observation_contains_passage_source_page_and_chunk_labels(self) -> None:
        passage = RetrievedPassage(
            chunk_id="utah-risk-factor-study-2016-2017.pdf::p0011::c01",
            source="utah-risk-factor-study-2016-2017.pdf",
            source_title="Utah Risk Factor Study, 2016-2017",
            source_role="Historical Utah retail food inspection observations.",
            page=11,
            text="Four inspection items had elevated summer rate ratios.",
            distance=0.1234,
        )
        obs = ToolObservation(
            tool="retrieve",
            accepted=True,
            step=1,
            remaining_steps=7,
            message="Retrieved 1 passage(s).",
            passages=(passage,),
        )
        rendered = obs.format_for_prompt()
        self.assertIn("chunk_id=utah-risk-factor-study-2016-2017.pdf::p0011::c01", rendered)
        self.assertIn("source=utah-risk-factor-study-2016-2017.pdf", rendered)
        self.assertIn("Utah Risk Factor Study, 2016-2017", rendered)
        self.assertIn("page=11", rendered)
        self.assertIn("Four inspection items had elevated summer rate ratios.", rendered)

    def test_required_queries_markdown_loads_all_nine_queries(self) -> None:
        if DEFAULT_QUERIES_PATH.is_file():
            queries = load_required_queries(DEFAULT_QUERIES_PATH)
            self.assertEqual([q.query_id for q in queries], [f"Q{i}" for i in range(1, 10)])
            self.assertIn("2-201.12(A)(1)", queries[0].question)
            self.assertIn("strongest Avenger", queries[8].question)


if __name__ == "__main__":
    unittest.main()

