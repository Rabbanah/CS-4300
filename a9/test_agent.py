"""Offline unit tests for ReAct controller policies, citation provenance, step limits, and terminal reasons."""

from __future__ import annotations

import json
import unittest

from agent import ReActRAGAgent, TerminalReason
from clients import ClientServiceError, ScriptedChatClient
from config import RunConfig
from ingest import RetrievedPassage, SubstituteRetriever


FOOD_CODE_PASSAGE_12 = RetrievedPassage(
    chunk_id="fda-food-code-2026.pdf::p0072::c00",
    source="fda-food-code-2026.pdf",
    source_title="FDA Food Code 2026",
    source_role="FDA model code; jurisdiction-specific legal adoption requires a separate authority.",
    page=72,
    text=(
        "2-201.12 Exclusions and Restrictions. (A) Exclude a food employee when "
        "symptomatic with vomiting or diarrhea and no noninfectious condition is known."
    ),
)

FOOD_CODE_PASSAGE_13 = RetrievedPassage(
    chunk_id="fda-food-code-2026.pdf::p0075::c01",
    source="fda-food-code-2026.pdf",
    source_title="FDA Food Code 2026",
    source_role="FDA model code; jurisdiction-specific legal adoption requires a separate authority.",
    page=75,
    text=(
        "2-201.13 Removal, Adjustment, or Retention of Exclusions and Restrictions. "
        "(A)(1) Reinstate an excluded food employee after being asymptomatic for at least 24 hours "
        "or providing medical documentation."
    ),
)

UTAH_STUDY_PASSAGE = RetrievedPassage(
    chunk_id="utah-risk-factor-study-2016-2017.pdf::p0004::c00",
    source="utah-risk-factor-study-2016-2017.pdf",
    source_title="Utah Risk Factor Study, 2016-2017",
    source_role="Historical Utah retail food inspection observations.",
    page=4,
    text=(
        "The 2016-2017 Utah Risk Factor Study summarizes baseline observational inspection "
        "frequencies in retail food establishments and does not enact regulations or penalties."
    ),
)


class TestReActRAGControllerAndTerminals(unittest.TestCase):
    """Tests citation provenance, step limits, all 4 terminal reasons, and scripted runs."""

    def test_supported_answer_run_and_claim_follows_from_cited_passage(self) -> None:
        """Terminal reason 1 (`answer`): supported run with claim-to-passage verification."""
        retriever = SubstituteRetriever(
            passages=(FOOD_CODE_PASSAGE_12, FOOD_CODE_PASSAGE_13)
        )
        chat = ScriptedChatClient(
            [
                json.dumps(
                    {
                        "thought": "Retrieve FDA Food Code 2026 sections 2-201.12 and 2-201.13.",
                        "tool": "retrieve",
                        "query": "2-201.12 2-201.13 vomiting diarrhea exclude reinstate",
                        "source": "fda-food-code-2026.pdf",
                        "top_k": 5,
                    }
                ),
                json.dumps(
                    {
                        "thought": "Both exclusion and 24-hour asymptomatic reinstatement are supported.",
                        "tool": "submit_answer",
                        "answer": (
                            "Under FDA Food Code 2026 (a model code), the person in charge must "
                            "exclude a symptomatic food employee with vomiting or diarrhea "
                            "[fda-food-code-2026.pdf::p0072::c00] and may reinstate the employee "
                            "after at least 24 hours asymptomatic or with medical documentation "
                            "[fda-food-code-2026.pdf::p0075::c01]."
                        ),
                        "claims": [
                            {
                                "claim": "Exclude a food employee symptomatic with vomiting or diarrhea.",
                                "citations": ["fda-food-code-2026.pdf::p0072::c00"],
                            },
                            {
                                "claim": "Reinstate after asymptomatic for at least 24 hours or medical documentation.",
                                "citations": ["fda-food-code-2026.pdf::p0075::c01"],
                            },
                        ],
                        "citations": [
                            "fda-food-code-2026.pdf::p0072::c00",
                            "fda-food-code-2026.pdf::p0075::c01",
                        ],
                    }
                ),
            ]
        )
        agent = ReActRAGAgent(chat, retriever, config=RunConfig(max_steps=6))
        result = agent.run(
            "What must the person in charge do when an employee has vomiting, and when reinstate?",
            query_id="Q1",
        )

        self.assertEqual(result.terminal_reason, TerminalReason.ANSWER)
        self.assertTrue(result.answered)
        self.assertEqual(result.steps_used, 2)
        self.assertEqual(
            result.citations,
            (
                "fda-food-code-2026.pdf::p0072::c00",
                "fda-food-code-2026.pdf::p0075::c01",
            ),
        )
        # Verify each claim's cited chunk exists in cited_passages and supports the claim's content
        passage_by_id = {p.chunk_id: p.text.lower() for p in result.cited_passages}
        self.assertIn("exclude", passage_by_id["fda-food-code-2026.pdf::p0072::c00"])
        self.assertIn("vomiting", passage_by_id["fda-food-code-2026.pdf::p0072::c00"])
        self.assertIn("24 hours", passage_by_id["fda-food-code-2026.pdf::p0075::c01"])
        self.assertIn("asymptomatic", passage_by_id["fda-food-code-2026.pdf::p0075::c01"])

    def test_insufficient_evidence_run_with_scripted_actions(self) -> None:
        """Terminal reason 2 (`insufficient_evidence`): corpus lacks Utah 2026 adoption & penalties."""
        retriever = SubstituteRetriever(passages=(UTAH_STUDY_PASSAGE, FOOD_CODE_PASSAGE_12))
        chat = ScriptedChatClient(
            [
                json.dumps(
                    {
                        "thought": "Search Utah study and Food Code for Utah adoption of 2026 Food Code and penalties.",
                        "tool": "retrieve",
                        "query": "Utah regulation adopts FDA Food Code 2026 penalties",
                    }
                ),
                json.dumps(
                    {
                        "thought": "Retrieved passages show FDA Food Code 2026 is a model code and Utah study is 2016-2017 observational data without statutory adoption or penalties.",
                        "tool": "insufficient_evidence",
                        "explanation": (
                            "The supplied corpus does not identify any current Utah regulation adopting "
                            "the 2026 FDA Food Code or any restaurant violation penalties. The FDA Food "
                            "Code 2026 is a model code requiring separate jurisdictional adoption, and "
                            "the Utah Risk Factor Study (2016-2017) reports historical inspection "
                            "observations rather than current legal rules or penalties."
                        ),
                        "citations": ["utah-risk-factor-study-2016-2017.pdf::p0004::c00"],
                    }
                ),
            ]
        )
        agent = ReActRAGAgent(chat, retriever, config=RunConfig(max_steps=6))
        result = agent.run(
            "What Utah regulation currently adopts the FDA Food Code 2026, and what penalties does it impose?",
            query_id="Q8",
        )
        self.assertEqual(result.terminal_reason, TerminalReason.INSUFFICIENT_EVIDENCE)
        self.assertFalse(result.answered)
        self.assertEqual(result.steps_used, 2)
        self.assertEqual(
            result.citations, ("utah-risk-factor-study-2016-2017.pdf::p0004::c00",)
        )

    def test_rejects_unretrieved_citations_and_enforces_cross_run_isolation(self) -> None:
        """Controller must reject citations not returned by retrieval during the current run."""
        retriever = SubstituteRetriever(passages=(FOOD_CODE_PASSAGE_12,))
        chat = ScriptedChatClient(
            [
                # Attempt 1: submit before any retrieval -> rejected
                json.dumps(
                    {
                        "tool": "submit_answer",
                        "answer": "Exclude immediately.",
                        "citations": ["fda-food-code-2026.pdf::p0072::c00"],
                    }
                ),
                # Step 2: retrieve FOOD_CODE_PASSAGE_12
                json.dumps(
                    {
                        "tool": "retrieve",
                        "query": "2-201.12 vomiting",
                    }
                ),
                # Step 3: cite a fabricated/unretrieved chunk_id -> rejected!
                json.dumps(
                    {
                        "tool": "submit_answer",
                        "answer": "Exclude and reinstate after 24h.",
                        "citations": [
                            "fda-food-code-2026.pdf::p0072::c00",
                            "fda-food-code-2026.pdf::p0999::c00",
                        ],
                    }
                ),
                # Step 4: sneak unretrieved chunk_id inline in answer text -> rejected!
                json.dumps(
                    {
                        "tool": "submit_answer",
                        "answer": "Exclude [fda-food-code-2026.pdf::p0888::c00].",
                        "citations": ["fda-food-code-2026.pdf::p0072::c00"],
                    }
                ),
                # Step 5: cite only the actually retrieved chunk_id -> accepted!
                json.dumps(
                    {
                        "tool": "submit_answer",
                        "answer": "Exclude the symptomatic employee [fda-food-code-2026.pdf::p0072::c00].",
                        "citations": ["fda-food-code-2026.pdf::p0072::c00"],
                    }
                ),
            ]
        )
        agent = ReActRAGAgent(chat, retriever, config=RunConfig(max_steps=6))
        res1 = agent.run("What does 2-201.12 require?")
        self.assertEqual(res1.terminal_reason, TerminalReason.ANSWER)
        self.assertEqual(res1.steps_used, 5)
        self.assertFalse(res1.trace[0].observation.accepted)
        self.assertFalse(res1.trace[2].observation.accepted)
        self.assertIn("fda-food-code-2026.pdf::p0999::c00", res1.trace[2].observation.rejection_reason or "")
        self.assertFalse(res1.trace[3].observation.accepted)
        self.assertIn("fda-food-code-2026.pdf::p0888::c00", res1.trace[3].observation.rejection_reason or "")
        self.assertTrue(res1.trace[4].observation.accepted)

        # Second run on the SAME agent instance: citing p0072 without retrieving in Run 2 must fail!
        chat2 = ScriptedChatClient(
            [
                json.dumps({"tool": "retrieve", "query": "utah study"}),
                json.dumps(
                    {
                        "tool": "submit_answer",
                        "answer": "Reusing chunk from previous run.",
                        "citations": ["fda-food-code-2026.pdf::p0072::c00"],
                    }
                ),
            ]
        )
        agent2 = ReActRAGAgent(
            chat2,
            SubstituteRetriever(passages=(UTAH_STUDY_PASSAGE,)),
            config=RunConfig(max_steps=2),
        )
        res2 = agent2.run("Second run isolation check")
        self.assertEqual(res2.terminal_reason, TerminalReason.STEP_LIMIT)
        self.assertFalse(res2.trace[1].observation.accepted)

    def test_step_limit_terminal_reason(self) -> None:
        """Terminal reason 3 (`step_limit`): halts cleanly when max_steps is reached."""
        retriever = SubstituteRetriever(passages=(FOOD_CODE_PASSAGE_12,))
        chat = ScriptedChatClient(
            [
                json.dumps({"tool": "retrieve", "query": "search one"}),
                json.dumps({"tool": "retrieve", "query": "search two"}),
                json.dumps({"tool": "retrieve", "query": "search three"}),
            ]
        )
        agent = ReActRAGAgent(chat, retriever, config=RunConfig(max_steps=3))
        result = agent.run("Endless retrieval test")
        self.assertEqual(result.terminal_reason, TerminalReason.STEP_LIMIT)
        self.assertEqual(result.steps_used, 3)
        self.assertEqual(len(result.trace), 3)
        self.assertTrue(result.trace[-1].observation.terminal)
        self.assertEqual(result.trace[-1].observation.terminal_reason, "step_limit")

    def test_error_terminal_reason_and_credential_redaction(self) -> None:
        """Terminal reason 4 (`error`): records error outcome and redacts sensitive text."""
        secret_err = ClientServiceError("HTTP 401 with Authorization: Bearer sk-secret-token-123")
        chat = ScriptedChatClient([secret_err])
        retriever = SubstituteRetriever(passages=(FOOD_CODE_PASSAGE_12,))
        agent = ReActRAGAgent(chat, retriever, config=RunConfig(max_steps=5))

        res_chat_err = agent.run("Test chat failure")
        self.assertEqual(res_chat_err.terminal_reason, TerminalReason.ERROR)
        self.assertEqual(res_chat_err.steps_used, 1)
        self.assertNotIn("sk-secret-token-123", res_chat_err.error_detail or "")
        self.assertNotIn(
            "sk-secret-token-123",
            res_chat_err.trace[0].observation.rejection_reason or "",
        )

        # Also test retriever error path
        chat_ok = ScriptedChatClient([json.dumps({"tool": "retrieve", "query": "norovirus"})])
        failing_retriever = SubstituteRetriever(
            error_to_raise=ClientServiceError("Embedding service unreachable")
        )
        agent_ret_err = ReActRAGAgent(chat_ok, failing_retriever, config=RunConfig(max_steps=5))
        res_ret_err = agent_ret_err.run("Test retriever failure")
        self.assertEqual(res_ret_err.terminal_reason, TerminalReason.ERROR)
        self.assertEqual(res_ret_err.steps_used, 1)


if __name__ == "__main__":
    unittest.main()

