"""OpenAI-compatible client, deterministic test client, and rule-based ReAct client."""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence


class ChatClient(Protocol):
    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        """Return assistant text for the supplied bounded conversation."""


@dataclass
class OpenAICompatibleClient:
    model: str
    api_key: str
    base_url: str = "http://golem:8000/v1"
    temperature: float = 0.0
    seed: int = 0
    timeout_seconds: float = 120.0
    max_tokens: int = 256

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        if not self.api_key:
            raise ValueError("api_key must be provided")
        payload = json.dumps(
            {
                "model": self.model,
                "messages": list(messages),
                "stream": False,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])


class ScriptedClient:
    """Deterministic fake for tests; responses are returned in order."""

    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self._responses = iter(responses)
        self.messages_seen: list[tuple[dict[str, str], ...]] = []

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        self.messages_seen.append(tuple(messages))
        try:
            item = next(self._responses)
        except StopIteration as error:
            raise RuntimeError("scripted responses exhausted") from error
        if isinstance(item, Exception):
            raise item
        return item


_ROW_LINE_RE = re.compile(r"^r([0-8])\s+(.+)$")


def _extract_latest_grid(user_content: str) -> list[list[int]] | None:
    """Parse the most recent 9x9 ASCII board rendered in the user prompt."""
    blocks: list[dict[int, list[int]]] = []
    current: dict[int, list[int]] = {}
    for raw_line in user_content.splitlines():
        line = raw_line.strip()
        match = _ROW_LINE_RE.match(line)
        if not match:
            continue
        r = int(match.group(1))
        rest = match.group(2)
        tokens = [tok for tok in rest.split() if tok != "|"]
        if len(tokens) != 9:
            continue
        row_vals = [0 if tok == "." else int(tok) for tok in tokens]
        if r == 0 and current:
            if len(current) == 9:
                blocks.append(current)
            current = {}
        current[r] = row_vals
        if len(current) == 9:
            blocks.append(current)
            current = {}
    if not blocks:
        return None
    latest = blocks[-1]
    return [latest[r] for r in range(9)]


def _candidates_for(grid: list[list[int]], r: int, c: int) -> list[int]:
    used: set[int] = set()
    for col in range(9):
        if col != c and grid[r][col] != 0:
            used.add(grid[r][col])
    for row in range(9):
        if row != r and grid[row][c] != 0:
            used.add(grid[row][c])
    r0, c0 = 3 * (r // 3), 3 * (c // 3)
    for br in range(r0, r0 + 3):
        for bc in range(c0, c0 + 3):
            if (br != r or bc != c) and grid[br][bc] != 0:
                used.add(grid[br][bc])
    return [d for d in range(1, 10) if d not in used]


class RuleBasedReActClient:
    """Autonomous deterministic client that solves Sudoku via ReAct JSON tool calls.

    Demonstrates the full ReAct tool lifecycle using only prompt observations:
    1. Inspects the first target cell (`inspect`),
    2. Encodes a propositional rule and peer elimination facts (`kb_add`),
    3. Queries entailment (`kb_query`),
    4. Places forced naked/hidden singles (`place`),
    5. Uses trial placement + `undo` recovery when encountering a two-candidate branch,
    6. Submits the completed board (`submit`).
    """

    def __init__(self) -> None:
        self._did_initial_inspect = False
        self._did_kb_rule = False
        self._did_kb_query = False
        self._kb_target: tuple[int, int, int, int] | None = None
        # Stack of (row, col, tried_digits, placements_since_branch)
        self._branch_stack: list[list[object]] = []

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        user_text = messages[-1]["content"] if messages else ""
        grid = _extract_latest_grid(user_text)
        if grid is None:
            return json.dumps(
                {
                    "thought": "Inspect full board to observe current puzzle state.",
                    "tool": "inspect",
                    "target": "board",
                }
            )

        empty_cells: list[tuple[int, int, list[int]]] = []
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    empty_cells.append((r, c, _candidates_for(grid, r, c)))

        # Step 1: If board is complete, submit!
        if not empty_cells:
            return json.dumps(
                {
                    "thought": "All 81 cells are filled without conflicts; submitting board.",
                    "tool": "submit",
                }
            )

        # Sort empty cells by MRV (minimum remaining values)
        empty_cells.sort(key=lambda item: (len(item[2]), item[0], item[1]))
        r, c, cands = empty_cells[0]

        # Step 2: If any empty cell has 0 candidates, a prior branch choice failed -> undo!
        if len(cands) == 0:
            if self._branch_stack:
                top = self._branch_stack[-1]
                steps_to_undo = int(top[3])  # type: ignore[arg-type]
                top[3] = 0
                return json.dumps(
                    {
                        "thought": f"Dead end at ({r},{c}) with 0 candidates; undoing {steps_to_undo} branch placement(s).",
                        "tool": "undo",
                        "steps": max(1, steps_to_undo),
                    }
                )
            return json.dumps(
                {
                    "thought": f"Contradiction at ({r},{c}); undoing last placement.",
                    "tool": "undo",
                    "steps": 1,
                }
            )

        # Step 3: Exercise inspect + propositional KB reasoning on first deduction
        if not self._did_initial_inspect:
            self._did_initial_inspect = True
            return json.dumps(
                {
                    "thought": f"Inspect cell ({r},{c}) to observe its candidates.",
                    "tool": "inspect",
                    "target": "cell",
                    "row": r,
                    "col": c,
                }
            )

        if not self._did_kb_rule:
            self._did_kb_rule = True
            d_true = cands[0]
            d_alt = cands[1] if len(cands) > 1 else (1 if d_true != 1 else 2)
            self._kb_target = (r, c, d_true, d_alt)
            sym_true = f"c_r{r}_c{c}_d{d_true}"
            sym_alt = f"c_r{r}_c{c}_d{d_alt}"
            if len(cands) == 1:
                formula = f"({sym_true} | {sym_alt}) & ~{sym_alt}"
            else:
                formula = " | ".join(f"c_r{r}_c{c}_d{d}" for d in cands)
            return json.dumps(
                {
                    "thought": f"Record candidate rule for ({r},{c}) in propositional KB.",
                    "tool": "kb_add",
                    "kind": "rule",
                    "formula": formula,
                }
            )

        if not self._did_kb_query and self._kb_target is not None:
            self._did_kb_query = True
            kr, kc, kd, _ = self._kb_target
            return json.dumps(
                {
                    "thought": f"Query KB whether c_r{kr}_c{kc}_d{kd} is entailed or unresolved.",
                    "tool": "kb_query",
                    "formula": f"c_r{kr}_c{kc}_d{kd}",
                }
            )

        # Step 4: Check if we just undid back to a branch point
        if self._branch_stack and self._branch_stack[-1][3] == 0:
            top = self._branch_stack[-1]
            br, bc = int(top[0]), int(top[1])  # type: ignore[arg-type]
            tried: list[int] = top[2]  # type: ignore[assignment]
            rem = [d for d in _candidates_for(grid, br, bc) if d not in tried]
            if rem:
                chosen = rem[0]
                tried.append(chosen)
                top[3] = 1
                return json.dumps(
                    {
                        "thought": f"After undo, try next candidate {chosen} at branch cell ({br},{bc}).",
                        "tool": "place",
                        "row": br,
                        "col": bc,
                        "digit": chosen,
                    }
                )
            else:
                self._branch_stack.pop()
                if self._branch_stack:
                    prev = self._branch_stack[-1]
                    steps_to_undo = int(prev[3])  # type: ignore[arg-type]
                    prev[3] = 0
                    return json.dumps(
                        {
                            "thought": f"All candidates exhausted at ({br},{bc}); backtracking {steps_to_undo} step(s).",
                            "tool": "undo",
                            "steps": max(1, steps_to_undo),
                        }
                    )

        # Step 5: Naked single -> place immediately
        if len(cands) == 1:
            digit = cands[0]
            if self._branch_stack:
                self._branch_stack[-1][3] = int(self._branch_stack[-1][3]) + 1  # type: ignore[arg-type]
            return json.dumps(
                {
                    "thought": f"Cell ({r},{c}) has single legal candidate {digit}; placing {digit}.",
                    "tool": "place",
                    "row": r,
                    "col": c,
                    "digit": digit,
                }
            )

        # Step 6: Branching choice -> record branch frame and try first candidate
        chosen = cands[0]
        self._branch_stack.append([r, c, [chosen], 1])
        return json.dumps(
            {
                "thought": f"Cell ({r},{c}) has candidates {cands}; hypothesizing {chosen}.",
                "tool": "place",
                "row": r,
                "col": c,
                "digit": chosen,
            }
        )
