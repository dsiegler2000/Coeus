"""
Simply a main file to house the driver code in a sensible place
"""
import argparse
import datetime
import logging
import os

from communication import UCIHandler, ConsoleHandler
from engine import CoeusEngine

logger = logging.getLogger(os.path.basename(__file__))


def configure_logging(logging_level=logging.DEBUG):
    filepath = os.path.join("logs", datetime.datetime.now().strftime("%y-%m-%d %H:%M:%S") + ".log")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging_level)
    handler = logging.FileHandler(filepath, "w", "utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s:%(name)s:%(levelname)s:%(message)s"))
    root_logger.addHandler(handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--console", help="flag to put the engine into console only mode (no UCI)", action="store_true")
    parser.add_argument("--profile", help="argument to put the engine into profiling mode, "
                                          "must pass a starting FEN with it", type=str)
    args = parser.parse_args()

    configure_logging()
    engine = CoeusEngine(os.path.join("config", "engine.json"), UCIHandler.output)

    if args.profile:
        console_handler = ConsoleHandler(engine, profile=True)
        console_handler.engine.set_position([], starting_fen=args.profile)
        console_handler.start()
    elif args.console:
        console_handler = ConsoleHandler(engine)
        console_handler.start()
    else:
        uci_handler = UCIHandler(engine)
        uci_handler.start()


if __name__ == "__main__":
    # TODO
    #  why is NPS is randomly dropping to very low numbers (might just be b/c other processes are running or GC)
    #  add pondering and opening books and endgame tablebase to the UCI handler
    #  see the last episode of the youtube series, chess programming wiki for ideas
    #  all kinds of evaluation function ideas, though careful w/ testing these
    #  come up with testing regiment
    #  > improve time control (see the wiki for more info, use EBF and like increase time right after no longer hitting on opening table)
    #  >> run a tournament tonight against whatever engine was mentioned in the YT series
    #  save to pgn feature
    #  find an efficient way to add endgame tablebase to the actual alpha beta/quiescence

    main()
