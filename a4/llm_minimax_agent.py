"""Minimax agent using a student-designed LLM cutoff evaluator."""

from jetan import JetanBoard, Move
from llm_evaluation import LLMEvaluationFunction
from minimax_agent import DepthLimitedMinimaxAgent


class LLMMinimaxAgent(DepthLimitedMinimaxAgent):
    def __init__(
        self,
        name: str,
        depth: int,
        evaluation_function: LLMEvaluationFunction,
    ) -> None:
        self.llm_evaluation = evaluation_function
        super().__init__(name, depth, evaluation_function)

    def choose_action(self, board: JetanBoard) -> Move:
        self.llm_evaluation.begin_search()
        return super().choose_action(board)
