"""Register the five required agents from student strategy functions."""

from agent_builders import (
    register_direct_llm_agent,
    register_llm_minimax_agent,
    register_minimax_agent,
)
from student_strategies import (
    build_evaluation_prompt,
    build_move_prompt,
    choose_fallback,
    evaluate_position_1,
    evaluate_position_2,
    evaluate_position_3,
    parse_evaluation_response,
    parse_move_response,
)


register_minimax_agent("minimax_1", evaluate_position_1, "Minimax 1")
register_minimax_agent("minimax_2", evaluate_position_2, "Minimax 2")
register_minimax_agent("minimax_3", evaluate_position_3, "Minimax 3")
# Replace evaluate_position_3 here if Stage A supports a different fallback.
register_llm_minimax_agent(
    "llm_evaluator",
    build_evaluation_prompt,
    parse_evaluation_response,
    evaluate_position_3,
    "LLM Evaluator",
)
register_direct_llm_agent(
    "llm_direct",
    build_move_prompt,
    parse_move_response,
    "Direct LLM",
    choose_fallback,
)
