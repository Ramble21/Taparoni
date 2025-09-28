import chess
from hyperparams import *

def material_val(piece):
    material_values = {
        chess.PAWN: PAWN_VALUE,
        chess.KNIGHT: KNIGHT_VALUE,
        chess.BISHOP: BISHOP_VALUE,
        chess.ROOK: ROOK_VALUE,
        chess.QUEEN: QUEEN_VALUE,
        chess.KING: 0
    }
    return material_values[piece.piece_type]
def fen_material_balance(fen=None, board=None):
    """
    Returns material balance in centipawns (e.g. -100 means Black up a pawn)
    """
    if board is None:
        board = chess.Board(fen)
    score = 0.0
    for sq, piece in board.piece_map().items():
        v = material_val(piece) if piece is not None else 0
        score += v if piece.color == chess.WHITE else -v
    return score
def total_material(board):
    """
    Returns total material on the board
    (e.g. an endgame with 1 black pawn and 1 white pawn will return 2)
    Kings excluded
    """
    total = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            total += (material_val(piece) / 100)
    return total
def is_passed_pawn(board, square):
    piece = board.piece_at(square)
    if not piece or piece.piece_type != chess.PAWN:
        return False

    color = piece.color
    file = chess.square_file(square)
    rank = chess.square_rank(square)

    step = 1 if color == chess.WHITE else -1
    files = [f for f in [file - 1, file, file + 1] if 0 <= f < 8]

    r = rank + step
    while 0 <= r < 8:
        for f in files:
            sq = chess.square(f, r)
            p = board.piece_at(sq)
            if p and p.piece_type == chess.PAWN and p.color != color:
                return False
        r += step

    return True

def lambda_value(board):
    """
    Gets a scalar value λ where 0 <= λ <= 1 that determines how "endgame-y" a position is
    1.0 = complete endgame, 0.0 = opening / complex middlegame
    """
    material = total_material(board)
    if material >= MAX_MATERIAL:
        return 0
    elif material <= MIN_MATERIAL:
        return 1
    return (MAX_MATERIAL - material) / (MAX_MATERIAL - MIN_MATERIAL)

def heuristic_eval(board, move):
    """
    Hardcoded endgame heuristics to improve endgame play
    """
    score = 0

    # simplify while ahead, don't simplify while behind
    if board.is_capture(move):
        engine_color = board.turn
        engine_material = sum(material_val(p) for p in board.piece_map().values() if p.color == engine_color)
        opp_material = sum(material_val(p) for p in board.piece_map().values() if p.color != engine_color)
        if engine_material > opp_material:
            score += PASSED_PAWN_BONUS
        else:
            score -= PASSED_PAWN_BONUS

    # push passed pawns
    if board.piece_type_at(move.from_square) == chess.PAWN:
        if chess.square_rank(move.to_square) != chess.square_rank(move.from_square):
            if is_passed_pawn(board, move.to_square):
                rank = chess.square_rank(move.to_square) + 1
                score += SIMPLIFICATION_BONUS * rank

    # rooks behind passed pawns
    if board.piece_type_at(move.from_square) == chess.ROOK:
        to_sq = move.to_square
        engine_color = board.turn

        for sq, piece in board.piece_map().items():
            if piece.piece_type == chess.PAWN and piece.color == engine_color:
                if chess.square_file(sq) == chess.square_file(to_sq) and is_passed_pawn(board, sq):
                    if (engine_color == chess.WHITE and chess.square_rank(to_sq) < chess.square_rank(sq)) or \
                            (engine_color == chess.BLACK and chess.square_rank(to_sq) > chess.square_rank(sq)):
                        score += ROOK_BEHIND_PAWN_BONUS

    return score

