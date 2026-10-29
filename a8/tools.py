"""Typed tool schemas, strict JSON parser, and structured observations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union


class ToolName(str, Enum):
    INSPECT = "inspect"
    PLACE = "place"
    UNDO = "undo"
    KB_ADD = "kb_add"
    KB_QUERY = "kb_query"
    SUBMIT = "submit"


@dataclass(frozen=True)
class InspectArgs:
    target: str
    row: int | None = None
    col: int | None = None
    box: int | None = None


@dataclass(frozen=True)
class PlaceArgs:
    row: int
    col: int
    digit: int


@dataclass(frozen=True)
class UndoArgs:
    steps: int = 1


@dataclass(frozen=True)
class KBAddArgs:
    kind: str
    formula: str


@dataclass(frozen=True)
class KBQueryArgs:
    formula: str


@dataclass(frozen=True)
class SubmitArgs:
    pass


ToolArgs = Union[
    InspectArgs,
    PlaceArgs,
    UndoArgs,
    KBAddArgs,
    KBQueryArgs,
    SubmitArgs,
]


@dataclass(frozen=True)
class ToolCall:
    tool: ToolName
    args: ToolArgs
    thought: str = ""

    def signature(self) -> str:
        """Return a canonical deterministic string signature of the tool call (ignoring thought)."""
        return f"{self.tool.value}:{self.args!r}"


class ToolParseError(ValueError):
    """Raised when a model response violates the JSON envelope or tool schema."""


@dataclass(frozen=True)
class ToolObservation:
    tool: str
    accepted: bool
    remaining_budget: int
    message: str = ""
    board_view: str | None = None
    kb_consistent: bool | None = None
    query_status: str | None = None
    rejection_reason: str | None = None
    terminal: bool = False
    terminal_reason: str | None = None

    def format_for_prompt(self) -> str:
        """Format this observation into a structured string for the language model."""
        parts = [
            f"tool={self.tool}",
            f"accepted={self.accepted}",
            f"remaining_budget={self.remaining_budget}",
        ]
        if self.kb_consistent is not None:
            parts.append(f"kb_consistent={self.kb_consistent}")
        if self.query_status is not None:
            parts.append(f"query_status={self.query_status}")
        if self.rejection_reason:
            parts.append(f"rejection_reason={self.rejection_reason!r}")
        if self.terminal:
            parts.append(f"terminal=True ({self.terminal_reason})")
        if self.message:
            parts.append(f"message={self.message!r}")
        header = "OBSERVATION [" + ", ".join(parts) + "]"
        if self.board_view:
            return f"{header}\n{self.board_view}"
        return header


def _no_duplicate_keys_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolParseError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _strip_code_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _require_int(value: Any, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ToolParseError(f"Field {field!r} must be an integer, got {type(value).__name__}")
    if value < low or value > high:
        raise ToolParseError(f"Field {field!r} must be in {low}..{high}, got {value}")
    return value


def _require_nonempty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolParseError(f"Field {field!r} must be a non-empty string")
    return value.strip()


def parse_tool_call(raw_text: str) -> ToolCall:
    """Strictly parse and validate a single typed JSON tool call from model text."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ToolParseError("Model response must be a non-empty JSON object string")

    cleaned = _strip_code_fence(raw_text)
    try:
        payload = json.loads(cleaned, object_pairs_hook=_no_duplicate_keys_hook)
    except ToolParseError:
        raise
    except Exception as exc:
        raise ToolParseError(f"Invalid JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ToolParseError("Top-level JSON value must be an object")

    if "tool" not in payload or not isinstance(payload["tool"], str):
        raise ToolParseError("JSON object must contain string field 'tool'")

    raw_tool = payload["tool"]
    try:
        tool_name = ToolName(raw_tool)
    except ValueError as exc:
        valid = ", ".join(t.value for t in ToolName)
        raise ToolParseError(f"Unknown tool {raw_tool!r}; valid tools: {valid}") from exc

    thought = ""
    if "thought" in payload:
        if not isinstance(payload["thought"], str):
            raise ToolParseError("Field 'thought' must be a string")
        thought = payload["thought"]

    # Extract tool arguments either from nested "args" dict or flat fields
    if "args" in payload:
        extra_top = set(payload.keys()) - {"tool", "thought", "args"}
        if extra_top:
            raise ToolParseError(
                f"Cannot mix top-level fields {sorted(extra_top)} with nested 'args'"
            )
        if not isinstance(payload["args"], dict):
            raise ToolParseError("Field 'args' must be a JSON object")
        arg_dict: dict[str, Any] = payload["args"]
    else:
        arg_dict = {k: v for k, v in payload.items() if k not in ("tool", "thought")}

    if tool_name == ToolName.INSPECT:
        allowed = {"target", "row", "col", "box"}
        unknown = set(arg_dict.keys()) - allowed
        if unknown:
            raise ToolParseError(f"Unknown argument(s) for inspect: {sorted(unknown)}")
        if "target" not in arg_dict or not isinstance(arg_dict["target"], str):
            raise ToolParseError("inspect requires string field 'target'")
        target = arg_dict["target"]
        if target == "board":
            if any(k in arg_dict and arg_dict[k] is not None for k in ("row", "col", "box")):
                raise ToolParseError("inspect target='board' does not accept row/col/box")
            return ToolCall(tool_name, InspectArgs(target="board"), thought)
        if target == "cell":
            if "row" not in arg_dict or "col" not in arg_dict or arg_dict.get("box") is not None:
                raise ToolParseError("inspect target='cell' requires exactly 'row' and 'col'")
            r = _require_int(arg_dict["row"], "row", 0, 8)
            c = _require_int(arg_dict["col"], "col", 0, 8)
            return ToolCall(tool_name, InspectArgs(target="cell", row=r, col=c), thought)
        if target == "row":
            if "row" not in arg_dict or arg_dict.get("col") is not None or arg_dict.get("box") is not None:
                raise ToolParseError("inspect target='row' requires only 'row'")
            r = _require_int(arg_dict["row"], "row", 0, 8)
            return ToolCall(tool_name, InspectArgs(target="row", row=r), thought)
        if target == "col":
            if "col" not in arg_dict or arg_dict.get("row") is not None or arg_dict.get("box") is not None:
                raise ToolParseError("inspect target='col' requires only 'col'")
            c = _require_int(arg_dict["col"], "col", 0, 8)
            return ToolCall(tool_name, InspectArgs(target="col", col=c), thought)
        if target == "box":
            if "box" not in arg_dict or arg_dict.get("row") is not None or arg_dict.get("col") is not None:
                raise ToolParseError("inspect target='box' requires only 'box'")
            b = _require_int(arg_dict["box"], "box", 0, 8)
            return ToolCall(tool_name, InspectArgs(target="box", box=b), thought)
        raise ToolParseError(
            f"Invalid inspect target {target!r}; expected board, cell, row, col, or box"
        )

    if tool_name == ToolName.PLACE:
        if set(arg_dict.keys()) != {"row", "col", "digit"}:
            raise ToolParseError("place requires exactly 'row', 'col', and 'digit'")
        r = _require_int(arg_dict["row"], "row", 0, 8)
        c = _require_int(arg_dict["col"], "col", 0, 8)
        d = _require_int(arg_dict["digit"], "digit", 1, 9)
        return ToolCall(tool_name, PlaceArgs(row=r, col=c, digit=d), thought)

    if tool_name == ToolName.UNDO:
        unknown = set(arg_dict.keys()) - {"steps"}
        if unknown:
            raise ToolParseError(f"Unknown argument(s) for undo: {sorted(unknown)}")
        steps = 1
        if "steps" in arg_dict:
            steps = _require_int(arg_dict["steps"], "steps", 1, 81)
        return ToolCall(tool_name, UndoArgs(steps=steps), thought)

    if tool_name == ToolName.KB_ADD:
        if set(arg_dict.keys()) != {"kind", "formula"}:
            raise ToolParseError("kb_add requires exactly 'kind' and 'formula'")
        kind = _require_nonempty_str(arg_dict["kind"], "kind")
        if kind not in ("fact", "rule"):
            raise ToolParseError(f"kb_add kind must be 'fact' or 'rule', got {kind!r}")
        formula = _require_nonempty_str(arg_dict["formula"], "formula")
        return ToolCall(tool_name, KBAddArgs(kind=kind, formula=formula), thought)

    if tool_name == ToolName.KB_QUERY:
        if set(arg_dict.keys()) != {"formula"}:
            raise ToolParseError("kb_query requires exactly 'formula'")
        formula = _require_nonempty_str(arg_dict["formula"], "formula")
        return ToolCall(tool_name, KBQueryArgs(formula=formula), thought)

    if tool_name == ToolName.SUBMIT:
        if arg_dict:
            raise ToolParseError(f"submit takes no arguments, got {sorted(arg_dict.keys())}")
        return ToolCall(tool_name, SubmitArgs(), thought)

    raise AssertionError(f"Unhandled tool: {tool_name}")


def tool_schema_prompt() -> str:
    """Return developer prompt documentation for all permitted JSON tool schemas."""
    return (
        "Output ONLY a single valid JSON object (no extra prose, no duplicate keys).\n"
        "Permitted tool schemas:\n"
        '1. Inspect board/region: {"thought": "<reason>", "tool": "inspect", "target": "board"|"cell"|"row"|"col"|"box", "row": 0..8, "col": 0..8, "box": 0..8}\n'
        '2. Place digit:          {"thought": "<reason>", "tool": "place", "row": 0..8, "col": 0..8, "digit": 1..9}\n'
        '3. Undo placement(s):    {"thought": "<reason>", "tool": "undo", "steps": 1..81}\n'
        '4. Add KB fact or rule:  {"thought": "<reason>", "tool": "kb_add", "kind": "fact"|"rule", "formula": "<propositional_formula>"}\n'
        '5. Query KB proposition: {"thought": "<reason>", "tool": "kb_query", "formula": "<propositional_formula>"}\n'
        '6. Submit finished grid: {"thought": "<reason>", "tool": "submit"}\n'
        "Propositional symbol convention: c_r<row>_c<col>_d<digit> (e.g., c_r7_c8_d5) with operators ~, &, |, ->, <->."
    )
