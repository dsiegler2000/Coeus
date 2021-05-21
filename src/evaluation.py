"""
All code related to evaluating a board state
"""
import json
import logging
import os
from typing import Union, List, Optional, Tuple

import chess

import utils
from movegeneration import SearchBoard, PieceList
from utils import count_bin_ones

logger = logging.getLogger(os.path.basename(__file__))


class BoardEvaluation:
    """
    Basic data class to store board evaluations. A positive evaluation means `self.side_to_move` is ahead.
    """
    def __init__(self, white_evaluation: float, black_evaluation: float, side_to_move: chess.Color,
                 white_endgame: bool, black_endgame: bool, num_men: int):
        self.white_evaluation = white_evaluation
        self.black_evaluation = black_evaluation
        self.side_to_move = side_to_move
        self.white_endgame = white_endgame
        self.black_endgame = black_endgame
        self.num_men = num_men
        self.board_evaluation = self.black_evaluation - self.white_evaluation
        self.board_evaluation *= -1 if self.side_to_move == chess.WHITE else 1

        # Mate information
        self.mate = False
        self.mate_in = -1

    def set_mate(self, mate_in: int, mate_val: int = 1_000_000):
        """
        Sets it so that this evaluation is for a checkmate position in `mate_in` moves.
        :param mate_in: Number of moves until mate.
        :param mate_val: Dummy value to use as the mate value.
        :return: None
        """
        self.mate = True
        self.mate_in = mate_in
        self.board_evaluation = -mate_val + mate_in
        self.white_evaluation = None
        self.black_evaluation = None

    def __str__(self):
        to_move = "w" if self.side_to_move == chess.WHITE else "b"
        if self.mate:
            return f"mate in {self.mate_in: <3}, evaluation: {self.board_evaluation: <3}, {to_move} to move"
        return f"evaluation: {self.board_evaluation: <3} (white: {self.white_evaluation: <3}, " \
               f"black: {self.black_evaluation: <3}, {to_move} to move)"


class BoardHeatmap:
    """
    Data class to store heatmap information on a chess board, including the white and black versions of the heatmap.
    Heatmap refers to an array of ints that gives a weighting to each square on the board for a given piece.
    """
    def __init__(self, heatmap: Union[str, List[List[int]], List[int]], piece_type: chess.PieceType, color=chess.WHITE):
        """
        Parses a board heatmap (such as a piece square table) in either string or array form.
        :param heatmap: The heatmap to parse. The heatmap must be either a 2D array of values or a string where the
        ranks are separated by newlines and the values separated by commas
        :param piece_type: The piece type that this map relates to
        :param color: The color that the input is assumed to be for ("w" for white, "b" for black)
        """
        self.piece_type = piece_type
        if isinstance(heatmap, str):
            heatmap_list = []
            for rank in heatmap.split("\n"):
                heatmap_list.append([int(v) for v in rank.split(",") if len(v) > 0])
            heatmap = heatmap_list
        if isinstance(heatmap, list):
            if isinstance(heatmap[0], int):  # "Flattened"
                heatmap_list = []
                for i in range(0, 64, 8):
                    heatmap_list.append(heatmap[i:i + 8])
                heatmap = heatmap_list
            else:
                if len(heatmap) != 8 or sum(len(rank) for rank in heatmap) != 64:
                    raise ValueError("the provided heatmap has an invalid format or size!")
        # Generate the horizontally flipped heatmap to have both white and black versions
        heatmap_flipped = [heatmap[len(heatmap) - i - 1] for i in range(len(heatmap))]

        # Note that this is actually flipped from what is expected because of how chess reports pieces, and also we
        # only need the "white" oriented version because pieces are always reported relative to white in the bitmask
        white_heatmap_2d = heatmap_flipped if color is chess.WHITE else heatmap
        black_heatmap_2d = heatmap if color is chess.WHITE else heatmap_flipped

        # Finally flatten the arrays
        self._white_heatmap = [v for r in white_heatmap_2d for v in r]
        self._black_heatmap = [v for r in black_heatmap_2d for v in r]

        # Save a copy for printing/display
        self._printing_heatmap = white_heatmap_2d if color is chess.WHITE else heatmap_flipped

    def apply_heatmap(self, color: chess.Color, piece_list: PieceList):
        """
        Applies a heatmap to the given piece list
        :param color:
        :param piece_list:
        :return:
        """
        heatmap = self._white_heatmap if color == chess.WHITE else self._black_heatmap
        return sum(heatmap[sq] for sq in piece_list.square_list(self.piece_type, color))

    def apply_heatmap_dynamic(self, board: SearchBoard, color: chess.Color) -> int:
        """
        Applies a heatmap to a given board. This implementation dynamically locates pieces and thus it is slower than
        `apply_heatmap`, which should be used if possible.
        :param board: The board to apply the heatmap to
        :param color: The color to apply the heatmap to
        :return: The sum of the heatmap times 1 if a piece of the given color and `self.piece_type`
        """
        piece_mask = board.pieces_mask(self.piece_type, color)
        heatmap = self._white_heatmap if color == chess.WHITE else self._black_heatmap
        return sum(v * (1 if piece_mask & (1 << i) else 0) for i, v in enumerate(heatmap))

    def __str__(self):
        files = "".join([f"{chr(i): <5}" for i in range(ord('a'), ord('h') + 1)])
        header = "   " * 5 + files
        footer = "(w)    " + files
        board_lines = [str(8 - i) + "  " + "".join([f"{str(s)[:4]: >5}" for s in r]) for i, r in
                       enumerate(self._printing_heatmap)]
        board_lines.insert(0, header)
        board_lines.append("")
        board_lines.append(footer)
        return "\n".join(board_lines) + "\n"


class BaseEvaluator:
    """
    Base interface for all board evaluators, including some helper methods that are useful to all evaluators.
    """
    def __init__(self, config_filepath: str):
        """
        The initializer simply reads in the config file given the filepath. This config file should contain all relevant
        information on the parameters of the evaluator.
        All evaluators must contain a name, version, and description.
        :param config_filepath: Filepath to the config json
        """
        self.config_filepath = config_filepath
        with open(self.config_filepath, "r+") as config_f:
            self.config = json.load(config_f)
        self._parse_base_config()
        self._parse_config()

    def _parse_base_config(self):
        """
        Internal method to parse the fields that are required for all evaluators.
        :return: None
        """
        try:
            self.name = self.config["name"]
            self.description = self.config["description"]
            self.version = self.config["version"]
            self.type = self.config["type"]
        except KeyError as e:
            logger.warning(e)

    def _parse_config(self):
        """
        Abstract method that all child classes must implement to parse the relevant fields in their configurations.
        :return: None (sets fields)
        """
        raise NotImplementedError("_parse_config must be implemented by all evaluators!")

    def _parse_board_heatmap(self, heatmap_key: str, piece_type: chess.PieceType,
                             color_key: str = "piece_square_table_color") -> BoardHeatmap:
        """
        Parses a board heatmap (such as a piece square table) from the config and returns the heatmap object.
        :param heatmap_key: The key into the config file for this heatmap. The value must be either a 2D array of values
        or a string where the ranks are separated by newlines. '/'s can be included in the key to index into
        sub-dictionaries. This is all relative to the JSON root
        :param piece_type: The piece type for this heatmap
        :param color_key: The key into the config file for the color that the input is assumed to be for ("w" for white,
        "b" for black).
        :return: The BoardHeatmap for the given heatmap
        """
        key_split = heatmap_key.split("/")
        heatmap = self.config[key_split[0]]
        for s in key_split[1:]:
            heatmap = heatmap[s]
        color = chess.WHITE if self.config[color_key][0].lower() == "w" else chess.BLACK
        return BoardHeatmap(heatmap, piece_type, color=color)

    def _parse_piece_value_table(self, key) -> List[Optional[int]]:
        """
        Helper method to parse piece value tables from the config into dictionary form.
        :param key: The key in the config for the piece value dictionary
        :return: The piece value dictionary
        """
        piece_values = [None for _ in range(max(chess.PIECE_TYPES) + 1)]
        for piece_symbol, value in self.config[key].items():
            piece_values[chess.Piece.from_symbol(piece_symbol).piece_type] = value
        return piece_values

    def _log_base_config(self):
        """
        Logs the base configuration of name, description, and version to debug
        :return: None
        """
        logger.debug(f"\tname={self.name}")
        logger.debug(f"\tdescription={self.description}")
        logger.debug(f"\tversion={self.version}")

    def evaluate_board(self, board: SearchBoard) -> BoardEvaluation:
        """
        Evaluates a board position, generating a score for white and black.
        :param board: The board to evaluate
        :return: A BoardEvaluation object with the evaluations for both white and black and `board_evaluation` set to
        positive if the move to side to move is white and white is winning, also positive if side to move is black and
        black is winning, etc.
        """
        raise NotImplementedError("evaluate_board must be implemented by all evaluators!")

    @staticmethod
    def _count_piece_type(board: SearchBoard, piece_type: chess.PieceType) -> Tuple[int, int]:
        """
        Counts how many occurrences of the given piece type are on the board.
        :param board: Board to consider
        :param piece_type: Piece type to count
        :return: The piece count for white, black
        """
        return (count_bin_ones(board.pieces_mask(piece_type, chess.WHITE)),
                count_bin_ones(board.pieces_mask(piece_type, chess.BLACK)))

    @staticmethod
    def _compute_materials_dynamic(value_map: List[int], board: SearchBoard) -> Tuple[int, int, int]:
        """
        Counts up piece values on each side using the provided mapping of pieces to values, returning the piece values
        on each side. Dynamically counts pieces, which is much slower than the preferred _sum_materials method.
        :param value_map: A list of values for each piece
        :param board: Board to consider
        :return: Tuple of white piece values, left piece values, number of total pieces
        """
        white_materials, black_materials, num_men = 0, 0, 0
        for piece_type in chess.PIECE_TYPES:
            white_count, black_count = BaseEvaluator._count_piece_type(board, piece_type)
            white_materials += value_map[piece_type] * white_count
            black_materials += value_map[piece_type] * black_count
            num_men += white_count + black_count
        return white_materials, black_materials, num_men

    @staticmethod
    def _compute_materials(value_map: List[int], piece_list: PieceList) -> Tuple[int, int, int]:
        """
        Counts up piece values on each side using the provided mapping of pieces to values, returning the piece values
        on each side. Uses a piece list and therefore is much faster.
        :param value_map: A list of values for each piece
        :param piece_list: Piece list
        :return: Tuple of white piece values, left piece values, number of total pieces
        """
        white_materials, black_materials, num_men = 0, 0, 0
        for piece_type in chess.PIECE_TYPES:
            wn = len(piece_list.square_list(piece_type, chess.WHITE))
            bn = len(piece_list.square_list(piece_type, chess.BLACK))
            white_materials += value_map[piece_type] * wn
            black_materials += value_map[piece_type] * bn
            num_men += wn + bn
        return white_materials, black_materials, num_men


class SimplifiedEvaluator(BaseEvaluator):
    """
    Evaluator based on https://www.chessprogramming.org/Simplified_Evaluation_Function.
    The config file must contain the piece value dictionary, appropriate piece square tables (pawn, knight, bishop,
    rook, queen, king middle game, king end game, with the proper names/nesting and color), the endgame criteria, and
    the position weighting (material values are assumed to be a weight of 1).
    The endgame criteria is either "queens" or "queens+pieces" representing the following respective criteria:
    Both sides have no queens ("queens")
    Every side which has a queen has additionally no other pieces or one minorpiece maximum ("queens+pieces")
    """
    def __init__(self, config_filepath):
        """
        All SimplifiedEvaluators must contain a piece_values dict.
        :param config_filepath: Filepath to config file
        """
        super().__init__(config_filepath)
        logger.debug(f"Created a SimplifiedEvaluator: config_filepath={config_filepath}")
        self._log_base_config()

    def _parse_config(self):
        """
        Currently parses and sets piece values and heatmaps.
        :return: None
        """
        # Piece values
        self.piece_values = self._parse_piece_value_table("piece_values")

        # Piece square tables
        self.pawn_pst = self._parse_board_heatmap("piece_square_tables/pawn", chess.PAWN)
        self.knight_pst = self._parse_board_heatmap("piece_square_tables/knight", chess.KNIGHT)
        self.bishop_pst = self._parse_board_heatmap("piece_square_tables/bishop", chess.BISHOP)
        self.rook_pst = self._parse_board_heatmap("piece_square_tables/rook", chess.ROOK)
        self.queen_pst = self._parse_board_heatmap("piece_square_tables/queen", chess.QUEEN)
        self.king_middle_game_pst = self._parse_board_heatmap("piece_square_tables/king/middle_game", chess.KING)
        self.king_end_game_pst = self._parse_board_heatmap("piece_square_tables/king/end_game", chess.KING)

        # Endgame criteria
        self.endgame_criteria = self.config["endgame_criteria"].lower()
        if "queen" in self.endgame_criteria and "pieces" in self.endgame_criteria:
            self.endgame_criteria = "queen+pieces"
        elif "queen" in self.endgame_criteria:
            self.endgame_criteria = "queen"
        else:
            raise ValueError("endgame_criteria in config must be either 'queen' or 'queen+pieces'!")

    def _compute_positions(self, board: SearchBoard, color: chess.Color, endgame: bool) -> int:
        """
        Computes the position score for a given board, color, and whether it is endgame using the PSTs.
        :param board: The board to consider
        :param color: The color to consider
        :param endgame: Whether it is currently endgame or not (precomputed)
        :return: The position score for the given color, board, and endgame
        """
        return sum([
            self.pawn_pst.apply_heatmap_dynamic(board, color),
            self.knight_pst.apply_heatmap_dynamic(board, color),
            self.bishop_pst.apply_heatmap_dynamic(board, color),
            self.rook_pst.apply_heatmap_dynamic(board, color),
            self.queen_pst.apply_heatmap_dynamic(board, color),
            self.king_end_game_pst.apply_heatmap_dynamic(board, color) if endgame else
            self.king_middle_game_pst.apply_heatmap_dynamic(board, color),
        ])

    def evaluate_board(self, board: SearchBoard) -> BoardEvaluation:
        """
        Evaluates the board position using https://www.chessprogramming.org/Simplified_Evaluation_Function.
        :param board: Board to evaluate
        :return: A `BoardEvaluation` object
        """
        # Materials
        white_materials, black_materials, num_men = self._compute_materials_dynamic(self.piece_values, board)
        # Positions
        endgame = self._is_endgame(board, white_materials, black_materials)
        white_positions, black_positions = (self._compute_positions(board, chess.WHITE, endgame),
                                            self._compute_positions(board, chess.BLACK, endgame))
        return BoardEvaluation(white_materials + white_positions,
                               black_materials + black_positions, board.turn, endgame, endgame, num_men)

    def _is_endgame(self, board: SearchBoard, white_materials: int, black_materials: int) -> bool:
        """
        Detects the endgame using the appropriate criteria
        :return: True if in the endgame, False otherwise
        """
        white_queens, black_queens = self._count_piece_type(board, chess.QUEEN)
        if self.endgame_criteria == "queen":
            return white_queens == black_queens == 0
        elif self.endgame_criteria == "queen+pieces":
            # If the side has a queen then they have 1 other minorpiece maximum
            return not ((white_queens > 0 and white_materials > self.piece_values[chess.BISHOP]) or
                        (black_queens > 0 and black_materials > self.piece_values[chess.BISHOP]))
        else:
            return False


class ViceEvaluator(BaseEvaluator):
    """
    Evaluator based off the vice engine (https://www.youtube.com/playlist?list=PLZ1QII7yudbc-Ky058TEaOstZHVbT-2hg).
    Implements the following features:
    - position square tables
    - passed pawns
    - isolated pawns
    - open and semi-open squares for rooks and queens
    - bishop pairs
    - uses piece lists for efficiency
    """
    def __init__(self, config_filepath: str):
        super().__init__(config_filepath)
        logger.debug(f"Created a ViceEvaluatorV2: config_filepath={config_filepath}")
        self._log_base_config()

        self._generate_bitmasks()

    def _generate_bitmasks(self):
        """
        Generates the needed bitmasks for this evaluator (rank, file, passed, and isolated)
        :return:
        """
        # Rank and file bitmasks (note that rank 1 is 0th index and file A is 0th index)
        self._rank_bbs = chess.BB_RANKS
        self._file_bbs = chess.BB_FILES

        # Passed pawns, & opposite side pieces BB with the square in question
        # True means the pawn is NOT passed, False means that it is passed
        self._white_passed_pawns_bbs = [0 for _ in chess.SQUARES]
        self._black_passed_pawns_bbs = [0 for _ in chess.SQUARES]
        for rank in range(8):
            for file in range(8):
                # White
                board = [[0 for i in range(8)] for j in range(8)]
                for rank_offset in range(1, 8):
                    for file_offset in range(-1, 2):
                        target_rank = rank + rank_offset
                        target_file = file + file_offset
                        if 0 <= target_rank <= 7 and 0 <= target_file <= 7:
                            board[7 - target_rank][target_file] = 1
                self._white_passed_pawns_bbs[8 * rank + file] = utils.array_to_bitboard(board)

                # Black
                board = [[0 for i in range(8)] for j in range(8)]
                for rank_offset in range(-1, -8, -1):  # Simply go in reverse here
                    for file_offset in range(-1, 2):
                        target_rank = rank + rank_offset
                        target_file = file + file_offset
                        if 0 <= target_rank <= 7 and 0 <= target_file <= 7:
                            board[7 - target_rank][target_file] = 1
                self._black_passed_pawns_bbs[8 * rank + file] = utils.array_to_bitboard(board)

        # Isolated pawns, again & with same side pawns and True means the pawn is NOT isolated, False means it is
        self._isolated_pawns_bbs = [0 for _ in chess.SQUARES]
        for rank in range(8):
            for file in range(8):
                board = [[0 for i in range(8)] for j in range(8)]
                for rank_offset in range(-7, 8):
                    for file_offset in [-1, 1]:
                        target_rank = rank + rank_offset
                        target_file = file + file_offset
                        if 0 <= target_rank <= 7 and 0 <= target_file <= 7:
                            board[7 - target_rank][target_file] = 1
                self._isolated_pawns_bbs[8 * rank + file] = utils.array_to_bitboard(board)

    def _parse_config(self):
        self.piece_values = self._parse_piece_value_table("piece_values")

        self.pawn_pst = self._parse_board_heatmap("piece_square_tables/pawn", chess.PAWN)
        self.knight_pst = self._parse_board_heatmap("piece_square_tables/knight", chess.KNIGHT)
        self.bishop_pst = self._parse_board_heatmap("piece_square_tables/bishop", chess.BISHOP)
        self.rook_pst = self._parse_board_heatmap("piece_square_tables/rook", chess.ROOK)
        self.king_opening_game_pst = self._parse_board_heatmap("piece_square_tables/king/opening_game", chess.KING)
        self.king_end_game_pst = self._parse_board_heatmap("piece_square_tables/king/end_game", chess.KING)

        # Bishop pairs
        self.bishop_pair = self.config["bishop_pair"]

        # Endgame material value cutoff
        self.endgame_materials_cutoff = sum(v * self.piece_values[chess.Piece.from_symbol(p).piece_type]
                                            for p, v in self.config["endgame_cutoff_material"].items())

        # Pawn positioning
        self.pawn_isolated: int = self.config["pawn_isolated"]
        self.pawn_passed: List[int] = self.config["pawn_passed"]

        # Rook positioning
        self.rook_open_file = self.config["rook_open_file"]
        self.rook_semiopen_file = self.config["rook_semiopen_file"]

        # Queen positioning
        self.queen_open_file = self.config["queen_open_file"]
        self.queen_semiopen_file = self.config["queen_semiopen_file"]

    def _compute_positions(self, color: chess.Color, piece_list: PieceList,
                           white_materials: int, black_materials: int) -> int:
        """
        Computes position score for the given color.
        :param color: Color
        :param piece_list: Piece list
        :param white_materials: White material values
        :param black_materials: Black material values
        :return: Position score
        """
        king_pst = self.king_end_game_pst if self._is_endgame(color, white_materials, black_materials) \
            else self.king_opening_game_pst
        return sum([
            self.pawn_pst.apply_heatmap(color, piece_list),
            self.knight_pst.apply_heatmap(color, piece_list),
            self.bishop_pst.apply_heatmap(color, piece_list),
            self.rook_pst.apply_heatmap(color, piece_list),
            king_pst.apply_heatmap(color, piece_list)
        ])

    def _compute_passed_isolated_pawns(self, board: SearchBoard, piece_list: PieceList) -> Tuple[int, int, int, int]:
        """
        Computes the score for white and black passed and isolated pawns.
        :param board: Board to compute on
        :param piece_list: Piece list
        :return: white passed pawn score, black, white isolated pawn score, black
        """
        white_pawns = board.pieces(chess.PAWN, chess.WHITE).mask
        black_pawns = board.pieces(chess.PAWN, chess.BLACK).mask
        white_passed, black_passed = 0, 0
        white_isolated, black_isolated = 0, 0
        for sq in piece_list.square_list(chess.PAWN, chess.WHITE):
            # Pawn present and it is passed (note that pawn is guaranteed to be here)
            if not self._white_passed_pawns_bbs[sq] & black_pawns:
                white_passed += self.pawn_passed[sq // 8]
            # Pawn present and is isolated
            if not self._isolated_pawns_bbs[sq] & white_pawns:
                white_isolated += self.pawn_isolated
        for sq in piece_list.square_list(chess.PAWN, chess.BLACK):
            if not self._black_passed_pawns_bbs[sq] & white_pawns:
                black_passed += self.pawn_passed[7 - (sq // 8)]
            if not self._isolated_pawns_bbs[sq] & black_pawns:
                black_isolated += self.pawn_isolated
        return white_passed, black_passed, white_isolated, black_isolated

    def _compute_open_files(self, board: SearchBoard, piece_list: PieceList) -> Tuple[int, int]:
        """
        Computes the score for occupying open files (for rooks and queens)
        :param board: Board to consider
        :param piece_list: Piece list of the board
        :return: White open file score, black open file score
        """
        white_pawns = board.pieces(chess.PAWN, chess.WHITE).mask
        black_pawns = board.pieces(chess.PAWN, chess.BLACK).mask
        pawns = white_pawns | black_pawns
        open_file = lambda s: not (pawns & self._file_bbs[s % 8])
        white_semiopen_file = lambda s: not (white_pawns & self._file_bbs[s % 8])
        black_semiopen_file = lambda s: not (black_pawns & self._file_bbs[s % 8])
        white, black = 0, 0
        for sq in piece_list.square_list(chess.ROOK, chess.WHITE):
            if open_file(sq):
                white += self.rook_open_file
            elif white_semiopen_file(sq):
                white += self.rook_semiopen_file
        for sq in piece_list.square_list(chess.ROOK, chess.BLACK):
            if open_file(sq):
                black += self.rook_open_file
            elif black_semiopen_file(sq):
                black += self.rook_semiopen_file

        # Queens
        for sq in piece_list.square_list(chess.QUEEN, chess.WHITE):
            if open_file(sq):
                white += self.queen_open_file
            elif white_semiopen_file(sq):
                white += self.queen_semiopen_file
        for sq in piece_list.square_list(chess.QUEEN, chess.BLACK):
            if open_file(sq):
                black += self.queen_open_file
            elif black_semiopen_file(sq):
                black += self.queen_semiopen_file
        return white, black

    def _is_endgame(self, color: chess.Color, white_materials: int, black_materials: int) -> bool:
        """
        Returns True if the given color is in the end game.
        :param color: Color to check
        :param white_materials: White materials
        :param black_materials: White materials
        :return: True if in end game, False otherwise
        """
        materials = black_materials if color == chess.WHITE else white_materials
        materials_minus_king = materials - self.piece_values[chess.KING]
        return materials_minus_king <= self.endgame_materials_cutoff

    def _is_material_draw(self, piece_list: PieceList) -> bool:
        """
        Determines if the board is a material draw (minus pawns) using a heuristic from sjeng 11.2 and
        https://www.youtube.com/watch?v=4ozHuSRDyfE&list=PLZ1QII7yudbc-Ky058TEaOstZHVbT-2hg&index=82.
        Note that pawns must also be checked to truly determine if the board is a material draw.
        :param piece_list: Piece list
        :return: True if material draw, otherwise False
        """
        if len(piece_list.square_list(chess.ROOK, chess.WHITE)) == 0 and \
                len(piece_list.square_list(chess.ROOK, chess.BLACK)) == 0 and \
                len(piece_list.square_list(chess.QUEEN, chess.WHITE)) == 0 and \
                len(piece_list.square_list(chess.QUEEN, chess.BLACK)) == 0:
            if len(piece_list.square_list(chess.BISHOP, chess.WHITE)) == 0 and \
                    len(piece_list.square_list(chess.BISHOP, chess.BLACK)) == 0:
                return len(piece_list.square_list(chess.KNIGHT, chess.WHITE)) < 3 and \
                       len(piece_list.square_list(chess.KNIGHT, chess.BLACK)) < 3
            elif len(piece_list.square_list(chess.KNIGHT, chess.WHITE)) == 0 and \
                    len(piece_list.square_list(chess.KNIGHT, chess.BLACK)) == 0:
                return abs(len(piece_list.square_list(chess.BISHOP, chess.WHITE)) -
                           len(piece_list.square_list(chess.BISHOP, chess.BLACK))) < 2
            elif (len(piece_list.square_list(chess.KNIGHT, chess.WHITE)) < 3
                  and len(piece_list.square_list(chess.BISHOP, chess.WHITE)) == 0) or \
                    (len(piece_list.square_list(chess.BISHOP, chess.WHITE)) == 1
                     and len(piece_list.square_list(chess.KNIGHT, chess.WHITE)) == 0):
                return (len(piece_list.square_list(chess.KNIGHT, chess.BLACK)) < 3
                        and len(piece_list.square_list(chess.BISHOP, chess.BLACK)) == 0) or \
                       (len(piece_list.square_list(chess.BISHOP, chess.BLACK)) == 1
                        and len(piece_list.square_list(chess.KNIGHT, chess.BLACK)) == 0)
        elif len(piece_list.square_list(chess.QUEEN, chess.WHITE)) == 0 and \
                len(piece_list.square_list(chess.QUEEN, chess.BLACK)) == 0:
            if len(piece_list.square_list(chess.ROOK, chess.WHITE)) == 1 and \
                    len(piece_list.square_list(chess.ROOK, chess.BLACK)) == 1:
                return len(piece_list.square_list(chess.KNIGHT, chess.WHITE)) + \
                       len(piece_list.square_list(chess.KNIGHT, chess.WHITE)) < 2 and \
                       len(piece_list.square_list(chess.KNIGHT, chess.BLACK)) + \
                       len(piece_list.square_list(chess.KNIGHT, chess.BLACK)) < 2
            elif len(piece_list.square_list(chess.ROOK, chess.WHITE)) == 1 and \
                    len(piece_list.square_list(chess.ROOK, chess.BLACK)) == 0:
                return len(piece_list.square_list(chess.KNIGHT, chess.WHITE)) + \
                       len(piece_list.square_list(chess.BISHOP, chess.WHITE)) == 0 and \
                       1 <= len(piece_list.square_list(chess.KNIGHT, chess.BLACK)) + \
                       len(piece_list.square_list(chess.BISHOP, chess.BLACK)) <= 2
            elif len(piece_list.square_list(chess.ROOK, chess.BLACK)) == 1 and \
                    len(piece_list.square_list(chess.ROOK, chess.WHITE)) == 0:
                return len(piece_list.square_list(chess.KNIGHT, chess.BLACK)) + \
                       len(piece_list.square_list(chess.BISHOP, chess.BLACK)) == 0 and \
                       1 <= len(piece_list.square_list(chess.KNIGHT, chess.WHITE)) + \
                       len(piece_list.square_list(chess.BISHOP, chess.WHITE)) <= 2
        return False

    def evaluate_board(self, board: SearchBoard) -> BoardEvaluation:
        piece_list = board.generate_piece_list()

        # Pawns must also be checked
        if len(piece_list.square_list(chess.PAWN, chess.WHITE)) == 0 and \
                len(piece_list.square_list(chess.PAWN, chess.BLACK)) == 0 and \
                self._is_material_draw(piece_list):
            return BoardEvaluation(0, 0, board.turn, True, True, 0)

        # Compute all values (all in centipawns)
        white_materials, black_materials, num_men = self._compute_materials(self.piece_values, piece_list)
        white_positions = self._compute_positions(chess.WHITE, piece_list, white_materials, black_materials)
        black_positions = self._compute_positions(chess.BLACK, piece_list, white_materials, black_materials)
        passed_isolated = self._compute_passed_isolated_pawns(board, piece_list)
        white_passed, black_passed, white_isolated, black_isolated = passed_isolated
        white_open_files, black_open_files = self._compute_open_files(board, piece_list)
        white_bishop_pair = self.bishop_pair if len(piece_list.square_list(chess.BISHOP, chess.WHITE)) >= 2 else 0
        black_bishop_pair = self.bishop_pair if len(piece_list.square_list(chess.BISHOP, chess.BLACK)) >= 2 else 0

        # Sum it all up
        white = white_materials + white_positions + white_passed + white_isolated + white_open_files + white_bishop_pair
        black = black_materials + black_positions + black_passed + black_isolated + black_open_files + black_bishop_pair

        return BoardEvaluation(white, black, board.turn,
                               self._is_endgame(chess.WHITE, white_materials, black_materials),
                               self._is_endgame(chess.BLACK, white_materials, black_materials), num_men)


def evaluator_from_config(config_filepath: str) -> BaseEvaluator:
    with open(config_filepath, "r+") as config_f:
        config = json.load(config_f)
        eval_type = config["type"]
    if eval_type == "SimpleEvaluator":
        return SimplifiedEvaluator(config_filepath)
    elif eval_type == "ViceEvaluator":
        return ViceEvaluator(config_filepath)
    raise ValueError("the given config does not contain a valid type key!")
