import functools
import random
from collections import Iterator
from typing import List, Generator

import chess

from movegeneration import SearchBoard


@functools.total_ordering
class _ScoredMove:
    """
    A move with a score for ordering.
    """
    def __init__(self, move: chess.Move, score: int):
        self.move = move
        self.score = score

    def __eq__(self, other):
        return other.score == self.score

    def __lt__(self, other):
        return self.score < other.score

    def __str__(self):
        return f"{str(self.move)}, score={self.score}"

    def __repr__(self):
        return f"_ScoredMove({self.move.__repr__()}, {self.score})"


def ordered_move_generator(moves: List[_ScoredMove]) -> Generator[chess.Move, None, None]:
    """
    Generates the highest scored moves dynamically using an iteration of selection sort each time.
    Note that the generator does reorder the list of moves.
    :param moves: Moves to generate from
    :return: The highest move so far
    """
    n = 0
    while n < len(moves):
        best_score = -1_000_000
        best_move = None
        best_move_idx = -1
        i = n
        for m in moves[n:]:
            # print(f"considering {m}")
            if m.score > best_score:
                best_score = m.score
                best_move = m
                best_move_idx = i
            i += 1
        if best_move:
            moves[n], moves[best_move_idx] = moves[best_move_idx], moves[n]
            n += 1
            yield best_move.move
        else:
            return


def _generate_mvv_lva_scores() -> List[List[int]]:
    """
    Generates the most valuable victim, least valuable attacker score table, indexed by victim, attacker.
    Ex. `scores[chess.ROOK][chess.BISHOP]` means the score for a bishop attacking a rook.
    :return:
    """
    scores = [[0 for i in range(chess.KING + 1)] for j in range(chess.KING + 1)]
    victim_scores = {
        chess.PAWN: 100,
        chess.KNIGHT: 200,
        chess.BISHOP: 300,
        chess.ROOK: 400,
        chess.QUEEN: 500,
        chess.KING: 600
    }
    for attacker in chess.PIECE_TYPES:
        for victim in chess.PIECE_TYPES:
            scores[victim][attacker] = victim_scores[victim] + 6 - (victim_scores[attacker] // 100)
    return scores


MVV_LVA_SCORES = _generate_mvv_lva_scores()


def generate_ordered_moves_v1(board: SearchBoard, pv_move: chess.Move, killer_moves: List[List[chess.Move]],
                              search_history: List[List[int]], captures_only: bool = False) \
        -> Generator[chess.Move, None, None]:
    """
    Generates ordered moves using the following heuristics in this order:
    - principal variation move
    - most valuable victim, least valuable attacker captures (MVV LVA)
    - killer moves (beta cutoff but aren't captures)
    - search history (alpha cutoff but not beta cutoff and not a capture)
    :param board: The board to generate moves for
    :param pv_move: The principal variation move
    :param killer_moves: The killer moves table, a 2 x max_depth size array with the 0th index containing current
    killers and the 1st index containing next killers
    :param search_history: The search history table, a 64 x 64 size array that provides a heatmap of "relevant"
    (alpha cutoff meeting but not beta or capture) moves
    :param captures_only: True if only captures should be generated, False otherwise
    :return:
    """
    scored_moves: List[_ScoredMove] = []
    moves = board.generate_legal_captures() if captures_only else board.legal_moves
    for move in moves:
        if board.is_capture(move):
            attacker = board.piece_at(move.from_square).piece_type
            victim_piece = board.piece_at(move.to_square)
            if not victim_piece:  # en passant
                victim = chess.PAWN
            else:
                victim = victim_piece.piece_type
            score = MVV_LVA_SCORES[victim][attacker] + 10_000_000
        # "Quiet" moves
        elif killer_moves[0][board.searching_ply] == move:  # Current killer moves
            score = 9_000_000
        elif killer_moves[1][board.searching_ply] == move:  # Next killer moves
            score = 8_000_000
        else:  # Simply use search history heatmap
            score = search_history[move.from_square][move.to_square]

        # Principal variation should always be considered first
        if move == pv_move:
            score = 20_000_000
        scored_moves.append(_ScoredMove(move, score))
    return ordered_move_generator(scored_moves)

