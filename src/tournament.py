import traceback

import chess
import chess.engine
import chess.pgn
import chess.gaviota
import logging
import os
import datetime

from engine import CoeusEngine, EngineGoParams

# tournament vs VICE

logger = logging.getLogger(os.path.basename(__file__))


def configure_logging(logging_level=logging.DEBUG):
    filepath = os.path.join("logs", datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S") + ".log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    handler = logging.FileHandler(filepath, "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
    root_logger.addHandler(handler)


def play_vice(vice, depth=6, coeus_plays=chess.WHITE, starting_fen=None):
    engine = CoeusEngine(os.path.join("config", "engine.json"), output)
    if starting_fen:
        engine.set_position([], starting_fen)
    output(f"Starting a game, depth={depth}, coeus_plays={chess.BLACK}")
    while not engine.board.is_game_over():
        if engine.board.turn == coeus_plays:
            params = EngineGoParams()
            params.target_depth = depth
            params.set_mode(False, False, True, False, False)
            pv_line = engine.go(params)
            if len(pv_line) > 0:
                result = pv_line[0]
            else:
                output(">>> PV line not generated!!")
                result = list(engine.board.legal_moves)[0]
            player = "coeus"
        else:
            result = vice.play(engine.board, chess.engine.Limit(depth=depth))
            if result:
                result = result.move
            player = "vice"
        engine.board.push(result)
        output(f"{player} plays move {result.uci()}, new board state: {engine.board.fen()}")
    output(f">>> Outcome: {engine.board.outcome()}")
    output(f">>> Moves: {[m.uci() for m in engine.board.move_stack]}")
    return engine.board.outcome()


def output(s):
    logger.debug(s)
    print(s)


def tournament(n=50):
    vice = chess.engine.SimpleEngine.popen_uci("/Users/dsiegler/PycharmProjects/Coeus/vice")
    won = 0
    loss = 0
    draw = 0
    try:
        for i in range(n):
            side = bool(i % 2)
            output(f">>> Coeus playing {side}")
            outcome = play_vice(vice, coeus_plays=side)
            if outcome.winner == side:
                output(f">>> COEUS WON")
                won += 1
            elif outcome.winner == (not side):
                output(f">>> COEUS LOST")
                loss += 1
            else:
                output(f">>> DRAW")
                draw += 1
            output(f">>> {won}-{loss}-{draw}")
    except:
        tb = traceback.format_exc()
        output(tb)
    finally:
        vice.quit()


def play_from_fen(fen, coeus_plays, depth):
    vice = chess.engine.SimpleEngine.popen_uci("/Users/dsiegler/PycharmProjects/Coeus/vice")
    try:
        print(play_vice(vice, coeus_plays=coeus_plays, starting_fen=fen, depth=depth))
    except:
        tb = traceback.format_exc()
        output(tb)
    finally:
        vice.close()


def convert_log(log_filepath):
    with open(log_filepath) as f:
        n = 1
        while True:
            lines = f.readlines(-1)
            if lines is None or len(lines) == 0:
                break
            for line in lines:
                if "Moves:" in line:
                    moves = eval(" ".join(line.split(" ")[3:]))
                    game = chess.pgn.Game()
                    node = game.add_main_variation(chess.Move.from_uci(moves[0]))
                    for m in moves[1:]:
                        node = node.add_main_variation(chess.Move.from_uci(m))
                    with open(f"logs/game_{n}.pgn", "w+") as output:
                        output.write(str(game))
                    n += 1
                    print(n)


def fen_plus_moves_to_pgn(fen, moves):
    game = chess.pgn.Game.from_board(chess.Board(fen=fen))
    node = game.add_main_variation(moves[0])
    for m in moves[1:]:
        node = node.add_main_variation(m)
    print(game)


def probe_eg_tb_debug(fen):
    board = chess.Board(fen=fen)
    with chess.gaviota.PythonTablebase() as tablebase:
        tablebase.add_directory("data/endgame_tablebases/gaviota")
        curr_dtm = tablebase.get_dtm(board)
        print(f"curr dtm: {curr_dtm}")
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
                print(f"considering: {move.uci()}, dtm: {considering_dtm}")

                # The case where a DTM of 0 means that we are mating
                if 0 <= curr_dtm <= 3 and mate_move and considering_dtm == 0:
                    print(f"why here?!?!?!")
                    board.pop()
                    return move, considering_dtm
                else:  # Otherwise, DTM of 0 means draw
                    if curr_dtm > 0:
                        optimizes = max_dtm < considering_dtm < 0
                    else:
                        optimizes = considering_dtm > max_dtm and considering_dtm > 0
                    if considering_dtm is not None and no_draw and optimizes:
                        print(f"new max!")
                        max_dtm = considering_dtm
                        max_dtm_move = move
                    board.pop()
            return max_dtm_move, max_dtm
        return None, None


if __name__ == "__main__":
    configure_logging()
    convert_log("logs/21-05-20 11:29:30.log")
    # tournament(50)
    # play_from_fen("1Q6/8/7p/6p1/6k1/4p3/K7/8 w - - 0 52", coeus_plays=chess.WHITE, depth=6)
    # probe_eg_tb_debug("8/8/8/4B3/K2P4/1pk5/8/8 w - - 0 83")
