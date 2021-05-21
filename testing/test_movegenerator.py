import time
import unittest

import chess


class TestMoveGenerator(unittest.TestCase):
    """
    Tests the move generator, which is currently completely handled by the chess package
    """
    def test_perft_scores(self, max_ply=4):
        # Maps ply to number of nodes
        target_perft_counts = {
            1: 20,
            2: 400,
            3: 8902,
            4: 197281,
            5: 4865609,
            6: 119060324,
            7: 3195901860,
            8: 84998978956,
            9: 2439530234167,
            10: 69352859712417
        }

        global leaf_nodes
        leaf_nodes = 0

        def perft(depth: int, board: chess.Board):
            global leaf_nodes
            if depth == 0:
                leaf_nodes += 1
                return

            for move in board.legal_moves:
                board.push(move)
                perft(depth - 1, board)
                board.pop()

        start_time = time.perf_counter()
        perft(max_ply, chess.Board())
        end_time = time.perf_counter()
        self.assertEqual(leaf_nodes, target_perft_counts[max_ply])
        dt = end_time - start_time
        print(f"Generated {leaf_nodes} in {dt:3.4f}s, {leaf_nodes / dt:5.2f} nodes/s")
