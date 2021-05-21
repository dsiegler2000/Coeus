import os
import unittest

from evaluation import SimplifiedEvaluator, ViceEvaluator
from movegeneration import SearchBoard


class TestEvaluation(unittest.TestCase):
    def test_simplified_evaluator_mirror(self):
        evaluator = SimplifiedEvaluator(os.path.join("config", "evaluators", "simple_evaluator_config.json"))
        for board in test_boards():
            e1 = evaluator.evaluate_board(board)
            e2 = evaluator.evaluate_board(board.mirror())
            self.assertEqual(e1.board_evaluation, e2.board_evaluation, board.fen())

    def test_vice_evaluator_mirror(self):
        evaluator = ViceEvaluator(os.path.join("config", "evaluators", "vice_evaluator_config.json"))
        for board in test_boards():
            e1 = evaluator.evaluate_board(board)
            e2 = evaluator.evaluate_board(board.mirror())
            self.assertEqual(e1.board_evaluation, e2.board_evaluation, board.fen())


def test_boards():
    with open(os.path.join("testing/mirror.epd"), "r", encoding="iso8859-1") as mirror:
        while True:
            line = mirror.readline()
            if line is None or line.strip() == "":
                break
            position_fen = line.split(" ")[0]
            fen = f"{position_fen} w KQkq - 0 1"
            board = SearchBoard(fen=fen)
            yield board
