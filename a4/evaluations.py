"""Interface for student-designed Jetan evaluation functions."""

from collections.abc import Callable

from jetan import JetanBoard, Player

EvaluationFunction = Callable[[JetanBoard, Player], float]
