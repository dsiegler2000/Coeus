from threading import Event, Thread
from typing import List, Any

import chess


def call_repeatedly(interval, func, *args):
    """
    Calls the given function every interval. To stop, simply call the returned function.
    :param interval: Interval to call
    :param func: Function to call
    :param args: Arguments to the function
    :return: Stopper, a function to call to stop this timer.
    """
    stopped = Event()

    def loop():
        while not stopped.wait(interval):
            func(*args)
    Thread(target=loop, daemon=False).start()
    return stopped.set


def count_bin_ones(num: int) -> int:
    """
    Counts the number of 1s in the binary representation of the number.
    :param num: Number to consider
    :return: Number of 1s in the binary representation
    """
    cnt = 0
    while num > 0:
        cnt += 1 if num & 1 else 0
        num >>= 1
    return cnt


def print_bitboard(bb: chess.Bitboard):
    bb_bin = bin(bb)[2:].zfill(64)
    for i in range(0, 64, 8):
        file = bb_bin[i:i + 8]
        file = file[::-1].replace("1", " x ").replace("0", " . ")
        print(file)


def array_to_bitboard(arr: List[List[Any]]):
    bb = 0
    i = 0
    for r in reversed(arr):
        for s in r:
            bb |= (1 if s else 0) << i
            i += 1
    return bb
