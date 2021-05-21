from enum import Enum
from typing import List, Union, Tuple, Optional

import chess
import chess.polyglot

from lru import LRU

# TODO add more efficient support for piece list (every time push or pop the move list is update and cached)


class PieceList:
    """
    Piece list mapping color, piece type to a list of squares it occupies
    """
    def __init__(self, white_piece_list: List[List[chess.Square]], black_piece_list: List[List[chess.Square]],
                 both_piece_list: List[List[chess.Square]]):
        self.white_piece_list: List[List[chess.Square]] = white_piece_list
        self.black_piece_list: List[List[chess.Square]] = black_piece_list
        self.both_piece_list: List[List[chess.Square]] = both_piece_list

    def square_list(self, piece_type: chess.PieceType, color: Union[int, chess.Color]) -> List[chess.Square]:
        """
        Gets the square list for a given piece type and color.
        :param piece_type: Piece type
        :param color: Color, -1 for both
        :return: The appropriate square list
        """
        if color == chess.WHITE:
            return self.white_piece_list[piece_type]
        elif color == chess.BLACK:
            return self.black_piece_list[piece_type]
        elif color == -1:
            return self.both_piece_list[piece_type]

    def color_piece_list(self, color: Union[int, chess.Color]) -> List[List[chess.Square]]:
        """
        Gets the piece list for the specified color.
        :param color:
        :return: The appropriate raw piece list
        """
        if color == chess.WHITE:
            return self.white_piece_list
        elif color == chess.BLACK:
            return self.black_piece_list
        elif color == -1:
            return self.both_piece_list

    def __str__(self):
        s = ""
        for piece_type in chess.PIECE_TYPES:
            s += f"w{chess.PIECE_SYMBOLS[piece_type]}: {self.white_piece_list[piece_type]}\n"
            s += f"b{chess.PIECE_SYMBOLS[piece_type]}: {self.black_piece_list[piece_type]}\n"
        return s[:-1]


class TranspositionTableFlag(Enum):
    ALPHA = 0
    BETA = 1
    EXACT = 2


class _TranspositionTableEntry:
    def __init__(self, move: chess.Move, score: int, depth: int, flag: TranspositionTableFlag):
        self.move: chess.Move = move
        self.score: int = score
        self.depth: int = depth
        self.flag: TranspositionTableFlag = flag


class SearchBoard(chess.Board):
    """
    Extension to chess.Board that adds all of the support needed to make searching efficient.
    """
    def __init__(self, num_transposition_table_entries: int = 1_000_000, *args, **kwargs):
        """
        Creates a search board with the specified number of transposition table entries. Each entry is at somewhere
        around 980 bytes, so a 1,000,000 entry table takes up about 1gb.
        :param num_transposition_table_entries: Number of transposition table entries
        :param args: Args to the parent chess.Board initializer
        :param kwargs: Keyword args to the parent chess.Board initializer
        """
        super().__init__(*args, **kwargs)

        self.num_transposition_table_entries = num_transposition_table_entries
        # Maps position FENs to an entry containing score/searching information (including PV information)
        self._transposition_table: Optional[LRU] = LRU(self.num_transposition_table_entries) \
            if self.num_transposition_table_entries else None

        # Stores a ply used for searching (reset before each search)
        self.searching_ply: int = self.ply()

    def push(self, move: chess.Move) -> None:
        super().push(move)
        self.searching_ply += 1

    def pop(self) -> chess.Move:
        self.searching_ply -= 1
        return super().pop()

    def probe_transposition_table(self, alpha: int, beta: int, depth: int,
                                  mate_score: int) -> Tuple[Optional[chess.Move], Optional[int]]:
        """
        Probes the transposition table for an entry with the appropriate values. See brucemo.com for a more extended
        explanation of the logic.
        :param alpha: Alpha
        :param beta: Beta
        :param depth: Depth
        :param mate_score: Mate score that is used (inf - max_depth)
        :return: The move, and the score if there was a hit, otherwise None, None
        """
        if self._transposition_table is None:
            raise ValueError("Cannot probe transposition table on a copy!")
        if self.zobrist_hash() in self._transposition_table:
            entry: _TranspositionTableEntry = self._transposition_table[self.zobrist_hash()]
            move = entry.move
            if entry.depth >= depth:
                score = entry.score
                if score > mate_score:
                    score -= self.searching_ply
                elif score < -mate_score:
                    score += self.searching_ply

                # Check the flag
                if entry.flag == TranspositionTableFlag.ALPHA and score <= alpha:
                    return move, alpha
                elif entry.flag == TranspositionTableFlag.BETA and score >= beta:
                    return move, beta
                elif entry.flag == TranspositionTableFlag.EXACT:
                    return move, score
        return None, None

    def store_transposition_table_entry(self, move: chess.Move, score: int, depth: int,
                                        flag: TranspositionTableFlag, mate_score: int):
        """
        Stores a move in the transposition table.
        :param move: Move to store
        :param score: Score
        :param depth: Depth
        :param flag: Appropriate flag for this entry
        :param mate_score: Score of a mate
        :return: None
        """
        if self._transposition_table is None:
            raise ValueError("Cannot store to transposition table on a copy!")
        # Set the score back to inf if it is a mate
        if score > mate_score:
            score += self.searching_ply
        elif score < -mate_score:
            score -= self.searching_ply

        self._transposition_table[self.zobrist_hash()] = _TranspositionTableEntry(move, score, depth, flag)

    def probe_pv(self) -> chess.Move:
        """
        Probes the transposition table for the current PV
        :return: The PV move, if it is stored, otherwise None
        """
        if self._transposition_table is None:
            raise ValueError("Cannot probe PV on a copy!")
        pfen = self.zobrist_hash()
        if pfen in self._transposition_table:
            return self._transposition_table[pfen].move

    def clear_transposition_table(self):
        """
        Clears the transposition table. Should be done at the beginning of each game.
        :return: None
        """
        if self._transposition_table is None:
            raise ValueError("Cannot clear transposition table on a copy!")
        self._transposition_table: LRU = LRU(self.num_transposition_table_entries)

    def transposition_table_size(self) -> int:
        """
        Simply returns the current length of the transposition table
        :return: Length of transposition table
        """
        return len(self._transposition_table)

    def generate_pv_line(self, depth) -> List[chess.Move]:
        """
        Generates the principal variation line and returns it as a list of Moves.
        :return: List of moves, the principal variation
        """
        move = self.probe_pv()
        pv_line = []
        while move and len(pv_line) < depth:
            if self.is_legal(move):
                self.push(move)
                pv_line.append(move)
            else:
                break
            move = self.probe_pv()
        for _ in range(len(pv_line)):
            self.pop()
        return pv_line

    def generate_piece_list(self) -> PieceList:
        """
        Generates a piece list for this board.
        :return: Piece list
        """
        white_piece_list = [[] for _ in range(chess.KING + 1)]
        black_piece_list = [[] for _ in range(chess.KING + 1)]
        both_piece_list = [[] for _ in range(chess.KING + 1)]
        for sq, piece in self.piece_map().items():
            if piece.color == chess.WHITE:
                white_piece_list[piece.piece_type].append(sq)
            else:
                black_piece_list[piece.piece_type].append(sq)
            both_piece_list[piece.piece_type].append(sq)
        return PieceList(white_piece_list, black_piece_list, both_piece_list)

    def position_fen(self) -> str:
        """
        Gets the position FEN (board plus side plus castling plus en passant)
        :return: The position FEN string
        """
        castling = "{}{}{}{}".format(("K" if self.has_kingside_castling_rights(chess.WHITE) else ""),
                                     ("Q" if self.has_queenside_castling_rights(chess.WHITE) else ""),
                                     ("k" if self.has_kingside_castling_rights(chess.BLACK) else ""),
                                     ("q" if self.has_queenside_castling_rights(chess.BLACK) else ""))
        if len(castling) == 0:
            castling = "-"
        side = ("w" if self.turn else "b")
        ep = (str(self.ep_square) if self.ep_square is not None else "-")
        return "{} {} {} {}".format(self.board_fen(), side, castling, ep)

    def zobrist_hash(self) -> int:
        """
        Gets the Zobrist hash of the current position.
        :return: Zobrist hash value
        """
        return chess.polyglot.zobrist_hash(self)

    def copy_transposition_table_referenced(self) -> "SearchBoard":
        """
        Creates and returns a copy just of the board with the transposition table referencing the transposition table of
        the original board (`self`).
        :return: Copy of self, except for transposition table (reference copy)
        """
        b = SearchBoard(num_transposition_table_entries=1)
        b.searching_ply = self.searching_ply
        b.set_fen(self.fen())
        b._transposition_table = self._transposition_table
        b.num_transposition_table_entries = self.num_transposition_table_entries
        return b

    def __str__(self) -> str:
        files = "".join([f"{chr(i): <3}" for i in range(ord('a'), ord('h') + 1)])
        footer = "     " + files
        board_lines = super().__str__().split("\n")
        for i in range(len(board_lines)):
            board_lines[i] = board_lines[i].replace(" ", "  ")
            board_lines[i] = str(8 - i) + "    " + board_lines[i]
        board_lines.append("")
        board_lines.append(footer)
        return "\n".join(board_lines)
