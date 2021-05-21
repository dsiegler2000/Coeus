"""
All code to communicate with an interface. Also includes main driver and logging setup currently.
"""
import datetime
import logging
import os
import sys
import threading
import traceback
from typing import List, Optional, Tuple

import chess
import chess.polyglot
import chess.pgn

from engine import CoeusEngine, EngineGoParams

logger = logging.getLogger(os.path.basename(__file__))


class UCIHandler:
    """
    Implements a handler for the UCI protocol with threading to independently handle input and computation.
    Follows the protocol outlined at http://wbec-ridderkerk.nl/html/UCIProtocol.html.
    """
    def __init__(self, engine: CoeusEngine):
        self.engine = engine
        self.debug = False

    def start(self):
        logger.debug(f"Starting {self.__class__.__name__}")
        self._listen()

    def cleanup(self):
        """
        Cleanup method when the engine needs to quit.
        :return: None
        """
        pass  # For now, do nothing

    def _identify(self):
        """
        Identifies the engine in response to a "uci" command
        :return: None
        """
        UCIHandler.output(f"id name {self.engine.name}")
        UCIHandler.output(f"id author Dylan Siegler")
        UCIHandler.output(f"uciok")
        # No options can be set so no option command is sent

    def _go(self, split):
        """
        Handles to "go" command.
        :param split: The full split argument string (including the go)
        :return: None
        """
        # Go from the back picking off arguments, keep flags for the go mode/type
        use_time_control = False
        use_fixed_time = False
        use_depth = False
        use_nodes = False
        infinite = False
        params = EngineGoParams()
        for i, s in reversed(list(enumerate(split))):
            if s == "searchmoves":  # Search only the following moves (e.g. consider only e2e4 and d2d4)
                UCIHandler.output("info string searchmoves currently not supported!")
            elif s == "ponder":  # "Ponder" (search during opponent's move)
                UCIHandler.output(f"info string ponder currently not supported!")
            elif s == "wtime":
                params.wtime = int(split[i + 1])
                use_time_control = True
            elif s == "btime":
                params.btime = int(split[i + 1])
                use_time_control = True
            elif s == "winc":
                params.winc = int(split[i + 1])
                use_time_control = True
            elif s == "binc":
                params.binc = int(split[i + 1])
                use_time_control = True
            elif s == "movestogo":
                params.moves_to_go = int(split[i + 1])
                use_time_control = True
            elif s == "depth":
                params.target_depth = int(split[i + 1])
                use_depth = True
            elif s == "nodes":
                params.target_nodes = int(split[i + 1])
                use_nodes = True
            elif s == "mate":
                UCIHandler.output(f"info string mate currently not supported!")
            elif s == "movetime":
                self.engine.fixed_time = int(split[i + 1])
                use_fixed_time = True
            elif s == "infinite":
                infinite = True
        valid_mode = params.set_mode(use_time_control, use_fixed_time, use_depth, use_nodes, infinite)
        if not valid_mode:
            UCIHandler.output(f"info string unsupported go command!")

        def on_engine_completed(pv_line: List[chess.Move]):
            if len(pv_line) == 1:
                UCIHandler.output(f"bestmove {pv_line[0].uci()}")
            elif len(pv_line) > 1:
                UCIHandler.output(f"bestmove {pv_line[0].uci()} ponder {pv_line[1].uci()}")
        t = threading.Thread(target=self.engine.go, args=[params], kwargs={"on_completed": on_engine_completed},
                             daemon=False)
        t.start()

    def _listen(self):
        """
        Listens for all commands and properly handles them.
        :return: None
        """
        counter = 0
        while True:
            try:
                line = input()
                logger.debug(f"UCI(received):{line}")
                counter += 1
                split = line.split(" ")
                if line == "uci":  # Engine must identify itself
                    self._identify()
                elif line.startswith("debug"):  # Set debug mode
                    self.debug = split[1] == "on"
                elif line == "isready":  # Synchronization command, for now simply echo the readyok response
                    UCIHandler.output(f"readyok")
                elif line.startswith("setoption"):  # No options are currently supported
                    UCIHandler.output(f"info string {split[1]} is not a currently supported option!")
                elif line.startswith("register"):  # No registration needed
                    UCIHandler.output(f"info string this engine doesn't need registration!")
                elif line == "ucinewgame":  # No specific handling for new game
                    self.engine.new_game()
                    UCIHandler.output(f"info string ready for new game")
                elif line.startswith("position"):  # Handle the position parsing
                    if split[1] == "fen":
                        self.engine.set_position(split[9:], starting_fen=" ".join(split[2:8]))
                    elif split[1] == "startpos":
                        moves = split[3:]  # In long algebraic notation
                        self.engine.set_position(moves)
                elif line.startswith("go"):
                    self._go(split)
                elif line == "stop":
                    self.engine.stop()
                elif line.startswith("ponderhit"):
                    UCIHandler.output(f"info string ponderhit is currently not supported!")
                elif line == "quit":
                    self.cleanup()
                    break
            except IndexError as e:
                logger.warning(f"UCIHandler encountered an error")
                tb = traceback.format_exc()
                logger.warning(tb)
                UCIHandler.output(f"info string encountered a UCI exception {e}!")

    @staticmethod
    def output(line):
        print(line)
        logger.debug(f"UCI(sent):{line}")


class ConsoleHandler:
    """
    Implements a handler for playing in the console that uses the following simple commands:
    - undo/t: takeback move
    - pv: print PV line
    - pfen: print position FEN
    - fen: print FEN
    - s/search: search to depth 6 and print the result
    - q/quit: quit
    - any move string: move and if it is black's turn the engine will recommend a move to make (note that to accept the
    recommendation the user then has to type in that move and hit enter again)
    """
    def __init__(self, engine: CoeusEngine, profile: bool = False):
        self.engine = engine
        self.profile = profile
        self.profile_commands = ["s", "q"]

    def start(self):
        print_next_loop = True
        post = True
        force = False
        ponder = False
        use_opening_book = True
        use_endgame_tablebase = True
        ponder_thread = None
        fixed_time = 30 * 1_000
        depth = None
        time_control: Tuple[Optional[int], Optional[int]] = (None, None)  # t in x moves
        i = 0
        ConsoleHandler.output("Welcome to Coeus Chess Engine - Console Mode")
        ConsoleHandler.output(f"Loaded engine from {self.engine.config_filepath}")
        ConsoleHandler.output(f"Loaded searcher from {self.engine.searcher.config_filepath}")
        ConsoleHandler.output(f"Loaded evaluator from {self.engine.evaluator.config_filepath}")
        with chess.polyglot.open_reader(self.engine.searcher.opening_book_filepath) as reader:
            ConsoleHandler.output(f"Loaded opening book from {self.engine.searcher.opening_book_filepath} "
                                  f"with {len(reader)} entries")
        ConsoleHandler.output("type `help` or `h` for help at any time")
        while not self.engine.board.outcome():
            if print_next_loop:
                ConsoleHandler.output("Current board state:")
                ConsoleHandler.output(self.engine.board)
                ConsoleHandler.output(f"FEN: {self.engine.board.fen()}")
            else:
                print_next_loop = True
            if self.profile:
                if i > len(self.profile_commands):
                    break
                else:
                    input_str = self.profile_commands[i]
            else:
                input_str = input("> ")
                logger.debug(f"Console(received):{input_str}")
            i += 1
            split = input_str.split(" ")
            input_str = input_str.lower()
            if input_str == "help" or input_str == "h":  # Help
                ConsoleHandler.output("Commands:")
                ConsoleHandler.output("quit/q - quit")
                ConsoleHandler.output("undo/t - takeback the previous move")
                ConsoleHandler.output("pv [depth] - print the current principal variation line up to depth (default 6)")
                ConsoleHandler.output("pfen - print the current position FEN")
                ConsoleHandler.output("fen - print the current FEN")
                ConsoleHandler.output("openbook - print the entries in the current opening book for the board position")
                ConsoleHandler.output("useopenbook/noopenbook - whether to use the opening book (default true)")
                ConsoleHandler.output("useendgametablebase/noendgametablebase - whether to use the endgame tablebase\n"
                                      "                                         (default true)")
                ConsoleHandler.output("pgn filename - save the current game to a PGN file")
                ConsoleHandler.output("search/s - set the engine to think using the currently set parameters\n"
                                      "           (will NOT execute the move it found)")
                ConsoleHandler.output("probe/p move - probe the line for the given move in the transposition table ")
                ConsoleHandler.output("ponder/noponder - whether the engine will ponder, or think during the \n"
                                      "                  opponent's time, recommended nopost if pondering is enabled,\n"
                                      "                  and pondering will only happen if force is disabled\n"
                                      "                  (default noponder)")
                ConsoleHandler.output("stopponder - stop the engine from pondering if it currently is")
                ConsoleHandler.output("post/nopost/noponderpost - whether to print or not print the debug information\n"
                                      "                           of the engine, noponderpost means that pondering\n"
                                      "                           won't print the debug information but regular\n"
                                      "                           searching will")
                ConsoleHandler.output("force/noforce - whether the engine will automatically think after player move,\n"
                                      "noforce means that it will think, force means that it will (default noforce)")
                ConsoleHandler.output("newgame - tell the engine that a new game has begun")
                ConsoleHandler.output("setboard fen - sets the board to the specified FEN")
                ConsoleHandler.output("depth d - sets the engine to search to the specified depth (time ignored)")
                ConsoleHandler.output("time t - sets the engine to search to the specified time in seconds \n"
                                      "         (depth ignored), default is 30s")
                ConsoleHandler.output("settimecontrol t in x [moves] - sets the time control for both the player and \n"
                                      "the engine to t minutes in x moves")
                ConsoleHandler.output("view - view the current engine parameters and console settings")
                ConsoleHandler.output("")
                ConsoleHandler.output("Enter moves using e7e8q format")
                ConsoleHandler.output("Press return with an empty line to view the board")
                ConsoleHandler.output("Note that the engine thinking blocks input")
                print_next_loop = False
            elif input_str == "undo" or input_str == "t":  # Takeback
                try:
                    self.engine.board.pop()
                except IndexError:
                    ConsoleHandler.output("Cannot take back!", err=True)
            elif split[0].lower() == "pv":  # Print principal variation line
                try:
                    depth = int(split[1]) if len(split) > 1 else 6
                    pv_line = [m.uci() for m in self.engine.board.generate_pv_line(depth)]
                    if len(pv_line) == 0:
                        ConsoleHandler.output("Engine has no PV line, may have accessed opening book, "
                                              "not searched yet, or pondering is enabled!", err=True)
                    else:
                        ConsoleHandler.output("PV line: " + " ".join(pv_line))
                except ValueError:
                    ConsoleHandler.output(f"Invalid depth {split[1]}", err=True)
                print_next_loop = False
            elif input_str == "pfen":
                ConsoleHandler.output(self.engine.board.position_fen())
                print_next_loop = False
            elif input_str == "fen":
                ConsoleHandler.output(self.engine.board.fen())
                print_next_loop = False
            elif input_str == "openbook":
                with chess.polyglot.open_reader(self.engine.searcher.opening_book_filepath) as reader:
                    weight_sum = sum(e.weight for e in reader.find_all(self.engine.board))
                    for e in reader.find_all(self.engine.board):
                        pct = round(100 * e.weight / weight_sum, 2)
                        if pct > 1:
                            print(f"({pct: >5}%): {e.move.uci()} (weight {e.weight})")
                print_next_loop = False
            elif input_str == "useopenbook":
                use_opening_book = True
                ConsoleHandler.output("opening book enabled")
                print_next_loop = False
            elif input_str == "noopenbook":
                use_opening_book = False
                ConsoleHandler.output("opening book disabled")
                print_next_loop = False
            elif input_str == "useendgametablebase":
                use_endgame_tablebase = True
                ConsoleHandler.output("endgame tablebase enabled")
                print_next_loop = False
            elif input_str == "noendgametablebase":
                use_endgame_tablebase = False
                ConsoleHandler.output("endgame tablebase disabled")
                print_next_loop = False
            elif split[0].lower() == "pgn":
                if len(split) == 2:
                    filepath = split[1] + ("" if split[1].endswith(".pgn") else ".pgn")
                    self._save_to_pgn(filepath)
                else:
                    ConsoleHandler.output("Usage: pgn filepath!", err=True)
                print_next_loop = False
            elif input_str == "s" or input_str == "search":  # Search
                params = EngineGoParams()
                params.fixed_time = fixed_time
                params.target_depth = depth
                params.set_mode(False, bool(fixed_time), bool(depth), False, False)
                pv_line = self.engine.go(params)
                ConsoleHandler.output("PV line: " + " ".join([m.uci() for m in pv_line]))
            elif split[0] == "probe" or split[0] == "p":
                try:
                    move = chess.Move.from_uci(split[1].lower())
                    if move in self.engine.board.legal_moves:
                        self.engine.board.push(move)
                        pv_line = [m.uci() for m in self.engine.board.generate_pv_line(depth=6)]
                        self.engine.board.pop()
                        if len(pv_line) == 0:
                            ConsoleHandler.output(f"Engine has no line for move {move.uci()}, may have accessed "
                                                  f"opening book, not searched that line yet, or pondering is enabled!",
                                                  err=True)
                        else:
                            ConsoleHandler.output(f"Line following move {move.uci()}: " + " ".join(pv_line))
                    else:
                        raise ValueError("Illegal move!")
                    print_next_loop = False
                except (ValueError, IndexError, AttributeError):
                    ConsoleHandler.output("Invalid move, usage: probe move", err=True)
            elif input_str == "q" or input_str == "quit":
                break
            elif input_str == "post":  # Post output
                self.engine.log_func = ConsoleHandler.output
                post = True
                ConsoleHandler.output("post enabled")
                print_next_loop = False
            elif input_str == "nopost":  # Don't post output
                self.engine.log_func = lambda s: None
                post = False
                ConsoleHandler.output("post disabled")
                print_next_loop = False
            elif input_str == "noponderpost":
                post = "noponderpost"
                self.engine.log_func = lambda s: None
                ConsoleHandler.output("post set to noponderpost")
                print_next_loop = False
            elif input_str == "force":  # Force inputs (engine WON'T move)
                force = True
                ConsoleHandler.output("force enabled")
                print_next_loop = False
            elif input_str == "noforce":  # No forcing inputs (engine WILL move)
                force = False
                ConsoleHandler.output("force disabled")
                print_next_loop = False
            elif input_str == "ponder":  # Ponder
                ponder = True
                post = "noponderpost"
                ConsoleHandler.output("ponder enabled (post set to noponderpost)")
                print_next_loop = False
            elif input_str == "noponder":  # Don't ponder
                ponder = False
                post = True
                self.engine.log_func = ConsoleHandler.output
                self.engine.stop()
                ponder_thread = self._kill_ponder_thread(ponder_thread, post)
                ConsoleHandler.output("ponder disabled (and post enabled)")
                print_next_loop = False
            elif input_str == "stopponder":
                self.engine.stop()
                if ponder_thread:
                    ponder_thread = self._kill_ponder_thread(ponder_thread, post)
                    ConsoleHandler.output("Stopped pondering")
                else:
                    ConsoleHandler.output("The engine was not pondering!", err=True)
                print_next_loop = False
            elif input_str == "newgame":  # Start new game
                if ponder:
                    ConsoleHandler.output("ponder must be disabled to start newgame!", err=True)
                else:
                    ConsoleHandler.output("New game created")
                    self.engine.new_game()
            elif split[0].lower() == "setboard":  # Set board
                try:
                    if ponder:
                        ConsoleHandler.output("ponder must be disabled to set board!", err=True)
                    else:
                        if len(split) > 1:
                            fen = " ".join(split[1:])
                            self.engine.board.set_fen(fen)
                        else:
                            ConsoleHandler.output("Usage: setboard fen", err=True)
                except ValueError:
                    ConsoleHandler.output(f"{split[1]} is not a valid FEN!", err=True)
            elif split[0].lower() == "time":  # Set time
                try:
                    if len(split) > 1:
                        t = int(split[1])
                        fixed_time = t * 1_000
                        depth = None
                        time_control = (None, None)
                        ConsoleHandler.output(f"time set to {t}s")
                    else:
                        ConsoleHandler.output("Usage: time t", err=True)
                except ValueError:
                    ConsoleHandler.output(f"{split[1]} is not a valid time!", err=True)
                print_next_loop = False
            elif split[0].lower() == "depth":  # Set depth
                try:
                    if len(split) > 1:
                        fixed_time = None
                        d = int(split[1])
                        depth = d
                        time_control = (None, None)
                        ConsoleHandler.output(f"depth set to {d}")
                    else:
                        ConsoleHandler.output("Usage: depth d", err=True)
                except ValueError:
                    ConsoleHandler.output(f"{split[1]} is not a valid depth!", err=True)
                print_next_loop = False
            elif split[0].lower() == "settimecontrol":  # settimecontrol t in x [moves]
                try:
                    t = int(split[1])
                    x = int(split[3])
                    fixed_time = None
                    depth = None
                    time_control = (t, x)
                except (ValueError, IndexError):
                    pass
            elif input_str == "view":  # View searching parameters
                if fixed_time:
                    ConsoleHandler.output(f"Fixed time mode: {fixed_time // 1_000}s time limit")
                if depth:
                    ConsoleHandler.output(f"Fixed depth mode: {depth} plies limit")
                if post == "noponderpost":
                    post_str = "noponderpost"
                else:
                    post_str = ("enabled" if post else "disabled")
                ConsoleHandler.output("             post: " + post_str)
                ConsoleHandler.output("            force: " + ("enabled" if force else "disabled"))
                ConsoleHandler.output("           ponder: " + ("enabled" if ponder else "disabled"))
                ConsoleHandler.output("     opening book: " + ("enabled" if use_opening_book else "disabled"))
                ConsoleHandler.output("endgame tablebase: " + ("enabled" if use_endgame_tablebase else "disabled"))
                print_next_loop = False
            elif input_str != "":
                try:
                    move = chess.Move.from_uci(input_str.replace(" ", "").replace("-", ""))
                    if self.engine.board.is_legal(move):
                        self.engine.stop()
                        if ponder_thread:
                            ponder_thread = self._kill_ponder_thread(ponder_thread, post)
                            if post == "noponderpost":
                                self.engine.log_func = ConsoleHandler.output
                        self.engine.board.push(move)
                        ConsoleHandler.output(f"Received move {move.uci()}")

                        if not force:
                            ConsoleHandler.output(f"Engine thinking using specified parameters...")
                            # Play the engine's recommended move
                            params = EngineGoParams()
                            params.fixed_time = fixed_time
                            params.target_depth = depth
                            params.use_opening_book = use_opening_book
                            params.use_endgame_tablebase = use_endgame_tablebase
                            params.set_mode(False, bool(fixed_time), bool(depth), False, False)
                            pv_line = self.engine.go(params)
                            if len(pv_line) > 0:
                                pv_move = pv_line[0]
                                self.engine.board.push(pv_move)
                                # Print the result as a colored string
                                ConsoleHandler.output(f"\033[92mCoeus moved {pv_move.uci()}\033[0m")
                            if ponder:  # Pondering
                                if post == "noponderpost":
                                    self.engine.log_func = lambda s: None
                                ponder_params = EngineGoParams()
                                params.set_mode(False, False, False, False, False)
                                ponder_thread = threading.Thread(target=self.engine.go, args=[ponder_params],
                                                                 kwargs={"ponder": True}, daemon=False)
                                ponder_thread.start()
                    else:  # Invalid move
                        raise ValueError("Non-legal move!")
                except ValueError:
                    ConsoleHandler.output(f"{input_str}: Invalid move or unknown command, "
                                          f"type `help` or `h` for help!", err=True)
        outcome = self.engine.board.outcome()
        if outcome.winner is not None:
            winner = "White" if outcome.winner == chess.WHITE else "Black"
            ConsoleHandler.output(f"{winner} won!")
        else:
            ConsoleHandler.output(f"Draw/stalemate due to {str(outcome.termination)}")
        self._kill_ponder_thread(ponder_thread, post)
        pgn = input("Save to PGN? [Y/n]: ").lower()
        if pgn == "y" or pgn == "yes":
            timestamp = datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S")
            filepath = os.path.join(f"logs/{timestamp}.pgn")
            self._save_to_pgn(filepath)
        ConsoleHandler.output("Thank you for playing!")

    def _save_to_pgn(self, filepath):
        game = chess.pgn.Game.from_board(self.engine.board)
        with open(filepath, "w") as f:
            f.write(str(game))
        ConsoleHandler.output(f"Saved PGN to {filepath}")

    def _kill_ponder_thread(self, ponder_thread: threading.Thread, post) -> Optional[bool]:
        if ponder_thread:
            ponder_thread.join(timeout=0.5)
            while ponder_thread.is_alive():
                import time
                time.sleep(0.5)
                ConsoleHandler.output("Waiting to kill ponder thread...")
            if post == "noponderpost":
                self.engine.log_func = ConsoleHandler.output
        return None

    @staticmethod
    def output(line, err=False):
        logger.debug(line)
        print(line, file=sys.stderr if err else sys.stdout)
