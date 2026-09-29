"""Interactive terminal agent."""

from collections.abc import Callable

from jetan import JetanBoard, Move


class TerminalUserAgent:
    """Ask a person at the terminal to select a legal move."""

    def __init__(
        self,
        name: str,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], object] = print,
    ) -> None:
        self.name = name
        self._input = input_fn
        self._output = output_fn

    def choose_action(self, board: JetanBoard) -> Move:
        actions = board.actions()
        if not actions:
            raise RuntimeError("TerminalUserAgent received a terminal board")
        action_by_text = {str(action): action for action in actions}
        self._output(str(board))
        self._output(
            f"{self.name} to move. Enter a move such as a1-b2, an action "
            "number, or 'list'."
        )
        while True:
            try:
                response = self._input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt) as error:
                raise RuntimeError(
                    "terminal input ended before a move was selected"
                ) from error
            if response == "list":
                self._output(
                    "\n".join(
                        f"{number}: {action}" for number, action in enumerate(actions, 1)
                    )
                )
                continue
            if response.isdecimal():
                number = int(response)
                if 1 <= number <= len(actions):
                    return actions[number - 1]
            elif response in action_by_text:
                return action_by_text[response]
            self._output(
                f"Invalid selection. Enter a legal move, 1-{len(actions)}, or 'list'."
            )
