"""
All searching code
"""
import json
import time
from typing import List, Optional, Callable, Tuple

import chess
import chess.polyglot
import chess.gaviota

import moveordering
from evaluation import BaseEvaluator
from movegeneration import SearchBoard, TranspositionTableFlag
from utils import call_repeatedly


class _SearchInfo:
    """
    Private data class to store all relevant information about a specific search.
    """
    def __init__(self, time_to_think: Optional[float] = None, nodes: int = None):
        self.start_time: float = time.time()

        self.nodes: int = 0
        self.curr_depth: int = 0
        self.fail_high: int = 0
        self.fail_high_first: int = 0

        # Fixed time
        self.end_time: Optional[float]
        if time_to_think:
            self.end_time = self.start_time + time_to_think
        else:
            self.end_time = None

        # Target nodes
        self.target_nodes: Optional[int] = nodes

        # Excluded moves
        self.excluded_moves = set()

        # Flag to stop
        self.stop: bool = False


class AlphaBetaMinimaxSearcher:
    """
    Implements iterative deepening minimax with alpha-beta pruning with the following features:
    - transposition table
    - iterative deepening
    - killer moves and search history (see move order)
    - repetition detection
    - quiescence search
    This searcher can only handle one board at a time.
    """
    MOVE_ORDERERS = ["v1"]

    def __init__(self, config_filepath: str, evaluator: BaseEvaluator):
        """
        Creates a searching instance that uses the given evaluator.
        :param evaluator: Evaluator to use to get board position evaluations
        :param config_filepath: Filepath to the config file
        """
        self.config_filepath = config_filepath
        self.evaluator = evaluator

        with open(self.config_filepath, "r+") as config_f:
            self.config = json.load(config_f)

        self._parse_config()
        self._init_search_info()

    def _parse_config(self):
        move_orderer = self.config["move_orderer"].lower()
        if move_orderer == "v1":
            def move_orderer_shortened(board, pv_move, captures_only=False):
                return moveordering.generate_ordered_moves_v1(board, pv_move=pv_move, killer_moves=self.search_killers,
                                                              search_history=self.search_history,
                                                              captures_only=captures_only)

            self.move_orderer = move_orderer_shortened
        else:
            raise ValueError(f"move_orderer must be one of {AlphaBetaMinimaxSearcher.MOVE_ORDERERS}")

        self.reduction_factor = self.config["reduction_factor"]

        self.inf = self.config["inf"]
        self.max_depth = self.config["max_depth"]
        self.mate_value = self.inf - self.max_depth

        self.opening_book_filepath = self.config["opening_book_filepath"]
        self.opening_book_min_weight = self.config["opening_book_min_weight"]

        self.endgame_tablebase_filepath = self.config["endgame_tablebase_filepath"]
        self.endgame_tablebase_num_men = self.config["endgame_tablebase_num_men"]

    def _checkup(self):
        if self.search_info.end_time and time.time() > self.search_info.end_time:
            self.search_info.stop = True
        if self.search_info.target_nodes and self.search_info.nodes >= self.search_info.target_nodes:
            self.search_info.stop = True

    def _init_search_info(self, board: SearchBoard = None, time_to_think: float = None,
                          nodes: int = None) -> SearchBoard:
        """
        Clears all relevant searching arrays and sets up the search information.
        :param board: The board to clear out too, if needed
        :param time_to_think: Time to think in seconds
        :param nodes: Number of nodes to search
        :return: The updated board, if passed in
        """
        self.search_info = _SearchInfo(time_to_think=time_to_think, nodes=nodes)
        self.search_history: List[List[int]] = [[0 for s in range(len(chess.SQUARES))]
                                                for p in range(len(chess.SQUARES))]
        self.search_killers: List[List[chess.Move]] = [[chess.Move.null() for d in range(self.max_depth)]
                                                       for _ in range(2)]
        if board:  # Reset the board's information
            board.searching_ply = 0
        return board

    @staticmethod
    def _zugzwang(board: SearchBoard) -> bool:
        """
        Detects so called zugzwang positions, wherein not moving is actually more benificial than moving (often happens
        in the end game when trying to mate the king). Note that this method is only a heuristic and technically
        doesn't cover every case.
        :param board: Board to check
        :return: True if zugzwang, False otherwise
        """
        # TODO improve zugzwang detection (see brucemo.com)
        return board.pieces_mask(chess.KNIGHT, board.turn) == 0 and \
               board.pieces_mask(chess.BISHOP, board.turn) == 0 and \
               board.pieces_mask(chess.ROOK, board.turn) == 0 and \
               board.pieces_mask(chess.QUEEN, board.turn) == 0

    @staticmethod
    def _terminal_condition(board: SearchBoard) -> bool:
        return board.is_fifty_moves() or board.is_insufficient_material() or \
               board.is_repetition(count=2) or board.is_repetition(count=4)

    def _quiescence(self, alpha: int, beta: int, board: SearchBoard) -> int:
        self.search_info.nodes += 1

        evaluation = self.evaluator.evaluate_board(board)

        if self._terminal_condition(board):
            return 0

        score = evaluation.board_evaluation

        if board.searching_ply > self.max_depth - 1:
            return score

        if score >= beta:  # Beta cutoff
            return beta

        if score > alpha:  # Standing pattern
            alpha = score

        # From here, it is basically minimax alpha beta using only capture moves
        num_legal_moves = 0

        for move in self.move_orderer(board, None, captures_only=True):
            if move in self.search_info.excluded_moves:
                continue
            num_legal_moves += 1
            board.push(move)
            score = -self._quiescence(-beta, -alpha, board)
            board.pop()

            if self.search_info.stop:
                return 0

            if score > alpha:
                if score >= beta:
                    self.search_info.fail_high_first += 1 if num_legal_moves == 1 else 0
                    self.search_info.fail_high += 1
                    return beta
                alpha = score

        return alpha

    def _minimax_alpha_beta(self, alpha: int, beta: int, board: SearchBoard, depth: int, null_pruning: bool) -> int:
        # Check depth terminal condition
        if depth <= 0:
            return self._quiescence(alpha, beta, board)
        self.search_info.nodes += 1

        if self._terminal_condition(board) and board.searching_ply > 0:  # Stalemate/draw condition
            return 0

        # Evaluate
        evaluation = self.evaluator.evaluate_board(board)

        if board.searching_ply > self.max_depth - 1:  # Max depth
            return evaluation.board_evaluation

        # In-check test to "get out of check"
        in_check = board.is_check()
        depth += 1 if in_check else 0

        # Prove transposition table and return early if we have a hit
        pv_move, score = board.probe_transposition_table(alpha, beta, depth, self.mate_value)

        if pv_move:
            return score

        # Null move pruning (must also verify that the side isn't in check and not in zugzwang scenario)
        if null_pruning and not in_check and board.searching_ply > 0 \
                and not self._zugzwang(board) and depth >= self.reduction_factor + 1:
            board.push(chess.Move.null())
            score = -self._minimax_alpha_beta(-beta, -beta + 1, board, depth - self.reduction_factor - 1, False)
            board.pop()
            if self.search_info.stop:
                return 0
            # If the null move option improves on beta and isn't a mate then return it
            if score >= beta and abs(score) < self.mate_value:
                return beta

        old_alpha = alpha
        best_move = None
        best_score = -self.inf
        num_legal_moves = 0
        found_pv = False

        for move in self.move_orderer(board, pv_move):
            if move in self.search_info.excluded_moves:
                continue
            num_legal_moves += 1
            board.push(move)

            if found_pv:  # PVS (principal variation search)
                score = -self._minimax_alpha_beta(-alpha - 1, -alpha, board, depth - 1, True)
                if alpha < score < beta:
                    score = -self._minimax_alpha_beta(-beta, -alpha, board, depth - 1, True)
            else:
                score = -self._minimax_alpha_beta(-beta, -alpha, board, depth - 1, True)

            board.pop()

            if self.search_info.stop:
                return 0

            capture = board.is_capture(move)

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:  # Alpha cutoff
                    if score >= beta:  # Beta cutoff
                        self.search_info.fail_high_first += 1 if num_legal_moves == 1 else 0
                        self.search_info.fail_high += 1
                        if not capture:  # Killer move (beta cutoff but not capture)
                            self.search_killers[1][board.searching_ply] = self.search_killers[0][board.searching_ply]
                            self.search_killers[0][board.searching_ply] = move
                        board.store_transposition_table_entry(best_move, beta, depth,
                                                              TranspositionTableFlag.BETA, self.mate_value)
                        return beta
                    found_pv = True
                    alpha = score
                    if not capture:  # Alpha cutoff that isn't a capture
                        self.search_history[best_move.from_square][best_move.to_square] += depth

        if num_legal_moves == 0:  # Checkmate cases
            if in_check:
                return -self.inf + board.searching_ply
            else:  # Draw
                return 0

        if alpha != old_alpha:  # Principal variation
            board.store_transposition_table_entry(best_move, best_score, depth,
                                                  TranspositionTableFlag.EXACT, self.mate_value)
        else:
            board.store_transposition_table_entry(best_move, alpha, depth,
                                                  TranspositionTableFlag.ALPHA, self.mate_value)

        return alpha

    def _probe_opening_book(self, board: SearchBoard, weighted_choice: bool = True) -> Optional[chess.Move]:
        """
        Probes the opening book, returning a move if the board is in the book.
        :param board: Board to probe
        :param weighted_choice: Whether to return a weighted choice of options or just the best one
        :return: Move to make, if there is one
        """
        with chess.polyglot.open_reader(self.opening_book_filepath) as reader:
            exclude_moves = []
            num_moves = 0
            best_move = None
            best_weight = None
            for e in reader.find_all(board):
                if e.weight < self.opening_book_min_weight:
                    exclude_moves.append(e.move)
                    if best_weight is None or e.weight > best_weight:
                        best_move = e.move
                        best_weight = e.weight
                else:
                    num_moves += 1
            if num_moves > 0:
                if weighted_choice:
                    return reader.weighted_choice(board, exclude_moves=exclude_moves).move
                else:
                    return best_move
        return None

    def _probe_endgame_tablebase(self, board: SearchBoard) -> Tuple[Optional[chess.Move], Optional[int]]:
        with chess.gaviota.PythonTablebase() as tablebase:
            tablebase.add_directory(self.endgame_tablebase_filepath)
            curr_dtm = tablebase.get_dtm(board)
            # If playing for a stalemate then no point in probing
            if curr_dtm is not None and curr_dtm != 0:
                max_dtm = -1_000_000
                max_dtm_move = None
                for move in board.legal_moves:
                    board.push(move)
                    outcome = board.outcome()
                    considering_dtm = tablebase.get_dtm(board)
                    no_draw = outcome is None or outcome.termination == chess.Termination.CHECKMATE
                    mate_move = outcome is not None and outcome.termination == chess.Termination.CHECKMATE

                    # The case where a DTM of 0 means that we are mating
                    if 0 <= curr_dtm <= 3 and mate_move and considering_dtm == 0:
                        board.pop()
                        return move, considering_dtm
                    else:  # Otherwise, DTM of 0 means draw
                        if curr_dtm > 0:
                            optimizes = max_dtm < considering_dtm < 0
                        else:
                            optimizes = considering_dtm > max_dtm and considering_dtm > 0
                        if considering_dtm is not None and no_draw and optimizes:
                            max_dtm = considering_dtm
                            max_dtm_move = move
                        board.pop()
                return max_dtm_move, max_dtm
            return None, None

    def mate_in(self, score: int) -> Optional[int]:
        """
        Determines if the given score is a mate and if so mate in how many moves.
        :param score: Score
        :return: A positive number for mate in x moves, negative for being mated in x moves, and None if the score
        doesn't represent a mate
        """
        if score is None:
            return None
        if score < -self.mate_value:
            return score + self.inf
        elif score >= self.mate_value:
            return self.inf - score
        else:
            return None

    def search(self, board: SearchBoard, fixed_time: Optional[float], target_depth: int, target_nodes: Optional[int],
               on_depth_completed: Callable[[int, int, List[chess.Move], str], None] = None,
               log_func: Callable[[str], None] = None,
               use_opening_book: bool = True, use_endgame_tablebase: bool = True) -> List[chess.Move]:
        """
        Performs iterative deepening minimax (negamax) search with alpha-beta pruning. Also handles checking opening
        and closing books.
        :param board: Board to search
        :param fixed_time: Time to use for the search, in seconds
        :param target_depth: Depth to search to
        :param target_nodes: Number of nodes to search
        :param on_depth_completed: A callback that is run after each successive depth is completed
        :param log_func: Logging function to use
        :param use_opening_book: Whether to use the opening book
        :param use_endgame_tablebase: Whether to use the endgame tablebase
        :return: The PV line found
        """
        # Parse parameters
        if target_depth is None:
            target_depth = self.max_depth
        if log_func is None:
            log_func = lambda s: None

        board = self._init_search_info(board, time_to_think=fixed_time, nodes=target_nodes)

        # Check for opening book entry
        if use_opening_book:
            move = self._probe_opening_book(board)
            if move is not None:
                log_func("opening book hit")
                return [move]

        # Check endgame tables
        if use_endgame_tablebase:
            move, dtm = self._probe_endgame_tablebase(board)
            if move is not None:
                log_func(f"endgame table hit, DTM: {dtm}")
                return [move]

        # Clear the transposition table when there is a transition to endgame
        if len(board.move_stack) > 0:
            move = board.pop()
            prev_evaluation = self.evaluator.evaluate_board(board)
            board.push(move)
            evaluation = self.evaluator.evaluate_board(board)
            if evaluation.white_endgame != prev_evaluation.white_endgame or \
                    evaluation.black_endgame != prev_evaluation.black_endgame:
                board.clear_transposition_table()
                log_func("transposition table cleared due to transition to endgame")

        # Set up the checkup daemon thread
        timer = call_repeatedly(0.5, self._checkup)

        # Iterative deepening
        prev_elapsed_time = 0
        best_score = None
        max_depth = 0
        err_log = dict()
        for curr_depth in range(1, target_depth + 1):
            st = time.time()
            self.search_info.curr_depth = curr_depth
            best_score = self._minimax_alpha_beta(-self.inf, self.inf, board, curr_depth, True)
            self._checkup()

            if self.search_info.stop:
                break

            max_depth = curr_depth
            ordering = self.search_info.fail_high_first / self.search_info.fail_high \
                if self.search_info.fail_high > 0 else 0.0

            # elapsed_time = time.time() - st
            # log_func(f"Depth {curr_depth} took time {elapsed_time}")
            # if curr_depth > 2:
            #     err = abs(predicted_time_next_iteration - elapsed_time)
            #     err_log[curr_depth] = err
            # if curr_depth > 1:
            #     # Effective branching factor
            #     time_ebf = elapsed_time / prev_elapsed_time
            #     predicted_time_next_iteration = time_ebf * elapsed_time
            #     log_func(f"Predicting that depth {curr_depth + 1} will take {round(predicted_time_next_iteration, 2)}s")
            #     # # TODO finish up the timecontrol classes and delegate to that via callbacks
            #     # # TODO this doesn't work when coming off of the transposition table, or maybe the error calculation doesn't work??
            #     # if self.search_info.end_time is not None:
            #     #     time_remaining = self.search_info.end_time - time.time()
            #     #     if time_remaining < 0.75 * predicted_time_next_iteration:
            #     #         break
            #
            # prev_elapsed_time = elapsed_time

            on_depth_completed(curr_depth, best_score, board.generate_pv_line(curr_depth))

        # Stop the timer
        timer()

        log_func("Errors:")
        for d in range(1, target_depth + 1):
            if d in err_log:
                log_func(f"Depth {d}: {round(err_log[d], 2)}")

        return board.generate_pv_line(max_depth)
