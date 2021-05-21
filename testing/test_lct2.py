import os
import traceback
import unittest

from engine import CoeusEngine

# Tests just about everything by running through positions in lct2.edp and looking for errors


class TestLCT2(unittest.TestCase):
    def test_all(self):
        log_func = lambda s: print(s)
        engine = CoeusEngine(os.path.join("config", "engine.json"), log_func)
        with open(os.path.join("testing/lct2.epd"), "r", encoding="iso8859-1") as lct2:
            line_num = 1
            while True:
                try:
                    line = lct2.readline()
                    if line is None or line.strip() == "":
                        break
                    position_fen = line.split(" ")[0]
                    fen = f"{position_fen} w KQkq - 0 1"
                    print(f"============ STARTING LINE {line_num} ============")
                    print(f"FEN: {fen}")
                    line_num += 1
                    engine.new_game()
                    engine.set_position([], fen)
                    engine.clear_mode()
                    engine.fixed_time = 10 * 60 * 1_000  # 10 minutes per board
                    engine.set_mode(False, True, False, False, False)
                    engine.go(log_time_quantum=30.0)
                except Exception:
                    traceback.print_exc()
