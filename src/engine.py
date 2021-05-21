"""
Brings together the search, evaluation, and move generation into one interface (mainly a wrapper)
"""
import json
import logging
import os
import time
from enum import Enum
from typing import List, Optional, Callable, Any

import chess

import evaluation
from movegeneration import SearchBoard
from search import AlphaBetaMinimaxSearcher
from timecontrol import EBFFixedTimeController
from utils import call_repeatedly

logger = logging.getLogger(os.path.basename(__file__))


class _TimeMode(Enum):
    TIME_CONTROL = "time_control"
    FIXED_TIME = "fixed_time"
    DEPTH = "depth"
    NODES = "nodes"
    INFINITE = "infinite"


class EngineGoParams:
    def __init__(self):
        self.mode: _TimeMode = _TimeMode.TIME_CONTROL
        self._clear_mode()

    def _clear_mode(self):
        # Time control values
        self.wtime: Optional[int] = None  # in ms
        self.btime: Optional[int] = None
        self.winc: Optional[int] = None  # time increment per move (ms)
        self.binc: Optional[int] = None
        self.moves_to_go: Optional[int] = None  # moves until next time control

        # Other modes
        self.target_depth: Optional[int] = None
        self.target_nodes: Optional[int] = None
        self.fixed_time: Optional[int] = None  # in ms

        # Mode to use
        self.mode: Optional[_TimeMode] = None

        # Opening book flag
        self.use_opening_book = True
        self.use_endgame_tablebase = True

    def clear_mode(self):
        """
        Clears the mode and all related variables. Should be called each iteration before setting any of the variables
        relevant to controlling the go command.
        :return: None
        """
        self._clear_mode()

    def set_mode(self, use_time_control: bool, use_fixed_time: bool, use_depth: bool, use_nodes: bool, infinite: bool):
        """
        Sets the mode by resolving potentially conflicting boolean flags.
        :param use_time_control: Whether to use time control mode
        :param use_fixed_time: Whether to use fixed time mode
        :param use_depth: Whether to use depth control mode
        :param use_nodes: Whether to use nodes control mode
        :param infinite: Whether to use infinite time control
        :return: True if the mode was resolved properly, False otherwise
        """
        self.mode = None
        if use_time_control and self.wtime and self.btime:
            self.mode = _TimeMode.TIME_CONTROL
        elif use_fixed_time and self.fixed_time:
            self.mode = _TimeMode.FIXED_TIME
        elif use_depth and self.target_depth:
            self.mode = _TimeMode.DEPTH
        elif use_nodes and self.target_nodes:
            self.mode = _TimeMode.NODES

        # Infinite
        if infinite:
            self.mode = _TimeMode.INFINITE

        return True


class CoeusEngine:
    """
    The engine largely just wraps the searcher to provide proper UCI protocol control and time control.
    This includes managing pondering.
    In order to use the engine, follow these steps:
    `engine.clear_mode()`
    `# set relevant engine parameters such as fixed_time, winc, etc.`
    `engine.set_mode(relevant mode booleans)`
    `engine.go(on_completed)`
    """
    def __init__(self, config_filepath: str, log_func: Callable[[str], None]):
        """
        Creates an engine using the given searcher.
        :param config_filepath: File path to the config file
        :param log_func: The function to use to log data
        """
        self.config_filepath = config_filepath
        self.board: SearchBoard = SearchBoard()

        with open(self.config_filepath, "r+") as config_f:
            self.config = json.load(config_f)

        self._parse_config()

        # Stop flag (for pondering)
        self._stop = False

        # Logging information
        self.log_func: Callable[[str], None] = log_func
        self._time_last_log: Optional[float] = None
        self._nodes_last_log: Optional[int] = None

    def _parse_config(self):
        self.name = self.config["name"]
        self.version = self.config["version"]

        # Evaluator and searcher
        self.evaluator = evaluation.evaluator_from_config(self.config["evaluator_config"])
        self.searcher = AlphaBetaMinimaxSearcher(self.config["searcher_config"], self.evaluator)

        # Ponder settings
        self._base_ponder_depth = self.config["base_ponder_depth"]
        self._ponder_pv_depth_offset = self.config["ponder_pv_depth_offset"]

    def _reset_logging(self):
        self._time_last_log = None
        self._nodes_last_log = None

    def _log_info(self):
        # Depth, nodes, and time
        elapsed_search_time_ms = int(1000 * (time.time() - self.searcher.search_info.start_time))
        self.log_func(f"info depth {self.searcher.search_info.curr_depth} "
                      f"nodes {self.searcher.search_info.nodes} "
                      f"time {elapsed_search_time_ms}")

        # Nodes per second
        if self._time_last_log and self._nodes_last_log:
            elapsed_time = time.time() - self._time_last_log
            nodes_since_last_log = self.searcher.search_info.nodes - self._nodes_last_log
            nps = nodes_since_last_log / elapsed_time
            self.log_func(f"info nps {nps}")

        self._time_last_log = time.time()
        self._nodes_last_log = self.searcher.search_info.nodes

    def _log_completed_depth(self, depth_completed: Optional[int], best_score: Optional[int],
                             pv_line: Optional[List[chess.Move]], info_str: str = None):
        if depth_completed is not None and best_score is not None and pv_line is not None and len(pv_line) > 0:
            # Log the best score, depth, nodes, time, and PV line
            elapsed_search_time_ms = int(1000 * (time.time() - self.searcher.search_info.start_time))
            pv_line_str = " ".join(str(m) for m in pv_line)
            mate_in = self.searcher.mate_in(best_score)
            score_str = f"mate {mate_in}" if mate_in else f"cp {best_score}"
            self.log_func(f"info score {score_str} "
                          f"depth {depth_completed} "
                          f"nodes {self.searcher.search_info.nodes} "
                          f"time {elapsed_search_time_ms} "
                          f"pv {pv_line_str}")
        if info_str:
            self.log_func(f"info string {info_str}")

    def set_position(self, lag: List[str], starting_fen=chess.STARTING_FEN):
        """
        Sets the internal board state to that specified by the series of long algebraic notation moves given
        (starting from the specified start position).
        :param lag: List of long algebraic notation moves
        :param starting_fen: The FEN to start the board at
        :return: None
        """
        self.board = SearchBoard(fen=starting_fen)
        for move in lag:
            self.board.push(chess.Move.from_uci(move))

    def new_game(self):
        """
        Resets everything to be ready for a new game.
        :return: None
        """
        self.board = SearchBoard()

    def stop(self):
        """
        Stops the engine from the current search.
        :return: None
        """
        self._stop = True
        self.searcher.search_info.stop = True

    def go(self, params: EngineGoParams, on_completed: Callable[[List[chess.Move]], Any] = None,
           log_time_quantum: float = 0.25, ponder: bool = False) -> List[chess.Move]:
        """
        Finds the best move considering the time constraints and returns it.
        :param params: All parameters associated with a call to go, including the proper mode information
        :param on_completed: The completion callback (argument is the output)
        :param log_time_quantum: The time quantum that searching info will be logged every, in seconds
        :param ponder: Whether to ponder
        :return: The principal variation line found
        """
        logger.debug(f"Searching from board:\n")
        logger.debug(f"\n{str(self.board)}")
        logger.debug(f"FEN: {self.board.fen()}")
        logger.debug(f"mode={params.mode}")
        self.log_func(f"info string {self.board.transposition_table_size()} transposition table entries")
        if params.mode == _TimeMode.INFINITE or ponder:
            fixed_time = None
            depth = None
            nodes = None
        elif params.mode == _TimeMode.TIME_CONTROL:
            moves_to_go = 30 if params.moves_to_go is None else params.moves_to_go
            time_left = params.wtime if self.board.turn == chess.WHITE else params.btime
            inc = params.winc if self.board.turn == chess.WHITE else params.binc
            if inc is None:
                inc = 0

            # Time control calculation
            fixed_time = ((time_left // moves_to_go) + inc) / 1_000
            depth = self.searcher.max_depth
            nodes = None
        elif params.mode == _TimeMode.FIXED_TIME:
            fixed_time = params.fixed_time / 1_000
            depth = self.searcher.max_depth
            nodes = None
        elif params.mode == _TimeMode.DEPTH:
            fixed_time = None
            depth = params.target_depth
            nodes = None
        elif params.mode == _TimeMode.NODES:
            fixed_time = None
            depth = None
            nodes = params.target_nodes
        else:
            raise ValueError("The mode is not set properly!")

        # Set a separate timer to log information
        self._reset_logging()
        if ponder and abs(log_time_quantum - 0.25) < 1e-3:
            log_time_quantum = 10.0
        timer = call_repeatedly(log_time_quantum, self._log_info)

        if ponder:
            # In ponder mode, use a kind of iterative deepening, namely:
            # - Search to a base depth plus an offset for the opponent's PV move
            # - Search to a base depth for all other of the opponent's moves
            # - Increase the base depth and repeat
            pondering_board: SearchBoard = self.board.copy_transposition_table_referenced()

            # Recall that this PV line is dynamically updated so we simply take the 1st element
            pv_line = pondering_board.generate_pv_line(depth=1)
            # Note that in theory the PV line could be messed up so verify that the entry is actual a legal move
            suspected_opponent_pv_move = pv_line[0] if len(pv_line) > 0 else None
            opponent_pv_move = None
            other_opponent_moves = []
            for move in pondering_board.legal_moves:
                if suspected_opponent_pv_move == move:
                    opponent_pv_move = suspected_opponent_pv_move
                else:
                    other_opponent_moves.append(move)
            ponder_depth = self._base_ponder_depth
            kwargs = {
                "on_depth_completed": self._log_completed_depth,
                "log_func": lambda s: self.log_func(f"info string {s}"),
                "use_opening_book": False,
                "use_endgame_tablebase": False
            }
            while not self._stop and ponder_depth + self._ponder_pv_depth_offset < self.searcher.max_depth:
                if opponent_pv_move:
                    if self._stop:
                        self.stop()
                        break
                    self._reset_logging()
                    pondering_board.push(opponent_pv_move)
                    self.searcher.search(pondering_board, None, ponder_depth + self._ponder_pv_depth_offset,
                                         None, **kwargs)
                    pondering_board.pop()
                for move in other_opponent_moves:
                    if self._stop:
                        self.stop()
                        break
                    self._reset_logging()
                    pondering_board.push(move)
                    self.searcher.search(pondering_board, None, ponder_depth, None, **kwargs)
                    pondering_board.pop()
                ponder_depth += 1
            # Simply set for returning
            pv_line = None
        else:
            pv_line = self.searcher.search(self.board, fixed_time, depth, nodes,
                                           on_depth_completed=self._log_completed_depth,
                                           log_func=lambda s: self.log_func(f"info string {s}"),
                                           use_opening_book=params.use_opening_book,
                                           use_endgame_tablebase=params.use_endgame_tablebase)

        # Stop the logging timer
        timer()

        # Reset stop flag
        self._stop = False

        if on_completed:
            on_completed(pv_line)

        return pv_line
