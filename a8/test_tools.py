"""Unit tests for `tools.py` derived from SPECIFICATIONS.md (SPEC-TOOLS-1..3)."""

from __future__ import annotations

import unittest

from tools import (
    InspectArgs,
    KBAddArgs,
    KBQueryArgs,
    PlaceArgs,
    SubmitArgs,
    ToolName,
    ToolParseError,
    UndoArgs,
    parse_tool_call,
)


class TestTypedToolsSpec(unittest.TestCase):
    """Black-box tests for typed tool schema validation and JSON parsing."""

    # --- SPEC-TOOLS-1 & SPEC-TOOLS-2: Valid Tool Calls ---

    def test_parse_inspect_variants(self) -> None:
        c_board = parse_tool_call('{"tool": "inspect", "target": "board"}')
        self.assertEqual(c_board.tool, ToolName.INSPECT)
        self.assertEqual(c_board.args, InspectArgs(target="board"))

        c_cell = parse_tool_call(
            '{"thought": "check (7,8)", "tool": "inspect", "args": {"target": "cell", "row": 7, "col": 8}}'
        )
        self.assertEqual(c_cell.tool, ToolName.INSPECT)
        self.assertEqual(c_cell.thought, "check (7,8)")
        self.assertEqual(c_cell.args, InspectArgs(target="cell", row=7, col=8))

        c_row = parse_tool_call('{"tool": "inspect", "target": "row", "row": 4}')
        self.assertEqual(c_row.args, InspectArgs(target="row", row=4))

        c_col = parse_tool_call('{"tool": "inspect", "target": "col", "col": 2}')
        self.assertEqual(c_col.args, InspectArgs(target="col", col=2))

        c_box = parse_tool_call('{"tool": "inspect", "target": "box", "box": 8}')
        self.assertEqual(c_box.args, InspectArgs(target="box", box=8))

    def test_parse_place_undo_kb_and_submit(self) -> None:
        c_place = parse_tool_call('{"tool": "place", "row": 7, "col": 8, "digit": 5}')
        self.assertEqual(c_place.tool, ToolName.PLACE)
        self.assertEqual(c_place.args, PlaceArgs(row=7, col=8, digit=5))

        c_undo_default = parse_tool_call('{"tool": "undo"}')
        self.assertEqual(c_undo_default.tool, ToolName.UNDO)
        self.assertEqual(c_undo_default.args, UndoArgs(steps=1))

        c_undo_two = parse_tool_call('{"tool": "undo", "steps": 2}')
        self.assertEqual(c_undo_two.args, UndoArgs(steps=2))

        c_kb_add = parse_tool_call(
            '{"tool": "kb_add", "kind": "rule", "formula": "c_r7_c8_d5 | c_r7_c8_d6"}'
        )
        self.assertEqual(c_kb_add.tool, ToolName.KB_ADD)
        self.assertEqual(
            c_kb_add.args, KBAddArgs(kind="rule", formula="c_r7_c8_d5 | c_r7_c8_d6")
        )

        c_kb_query = parse_tool_call('{"tool": "kb_query", "formula": "c_r7_c8_d5"}')
        self.assertEqual(c_kb_query.tool, ToolName.KB_QUERY)
        self.assertEqual(c_kb_query.args, KBQueryArgs(formula="c_r7_c8_d5"))

        c_submit = parse_tool_call('```json\n{"tool": "submit"}\n```')
        self.assertEqual(c_submit.tool, ToolName.SUBMIT)
        self.assertEqual(c_submit.args, SubmitArgs())

    # --- SPEC-TOOLS-1: Rejection of Malformed, Duplicate-Key, or Out-of-Schema Inputs ---

    def test_reject_duplicate_json_keys(self) -> None:
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "submit", "tool": "submit"}')
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "place", "row": 1, "row": 2, "col": 3, "digit": 4}')

    def test_reject_unknown_tools_extra_fields_and_wrong_types(self) -> None:
        with self.assertRaises(ToolParseError):
            parse_tool_call("I solved the puzzle!")
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "solve_magic"}')
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "submit", "extra": 123}')
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "place", "row": true, "col": 0, "digit": 5}')
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "place", "row": 0, "col": 0, "digit": 10}')
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "inspect", "target": "cell", "row": 0}')
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "kb_add", "kind": "guess", "formula": "a"}')
        with self.assertRaises(ToolParseError):
            parse_tool_call('{"tool": "undo", "steps": 0}')


if __name__ == "__main__":
    unittest.main()

