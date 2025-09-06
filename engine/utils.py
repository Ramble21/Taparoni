import chess
def move_to_plane(move: chess.Move, white_to_move):
    """
    0-6 = moving forward 1-7 spaces from perspective (e2e4 is forward for white, e7e5 is forward for black)
    7-13 = moving backward ~, 14-20 = moving east from side's perspective, 21-27 = moving west ~
    28-34 = NW, 35-41 = NE, 42-48 = SW, 49-55 = SE
    56-63 = knight moves, ordered clockwise starting at the move 2 up 1 left
    Underpromotions ignored (lumped together with regular promotions) for simplicity
    """
    from_sq = move.from_square
    to_sq = move.to_square
    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    tx, ty = chess.square_file(to_sq), chess.square_rank(to_sq)
    dx, dy = tx - fx, ty - fy
    # Flip perspective for black: "forward" always means toward opponent
    if not white_to_move:
        dx, dy = -dx, -dy
    if move.promotion is not None:
        if dx == 1:
            return 64
        elif dx == 0:
            return 65
        elif dx == -1:
            return 66
    # Knight moves
    if (abs(dx), abs(dy)) in [(1, 2), (2, 1)]:
        # Order: clockwise starting at (dx=-1, dy=2)
        knight_order = [(-1, 2), (1, 2), (2, 1), (2, -1),
                        (1, -2), (-1, -2), (-2, -1), (-2, 1)]
        idx = knight_order.index((dx, dy))
        return 56 + idx
    # Queen moves
    if dx == 0 and dy > 0:  # forward
        return dy - 1
    elif dx == 0 and dy < 0:  # backward
        return 7 + (-dy - 1)
    elif dy == 0 and dx > 0:  # east
        return 14 + (dx - 1)
    elif dy == 0 and dx < 0:  # west
        return 21 + (-dx - 1)
    elif dx == dy and dx > 0:  # NE
        return 35 + (dx - 1)
    elif dx == dy and dx < 0:  # SW
        return 42 + (-dx - 1)
    elif dx == -dy and dx < 0:  # NW
        return 28 + (-dx - 1)
    elif dx == -dy and dx > 0:  # SE
        return 49 + (dx - 1)
    raise ValueError(f"Move {move.uci()} not encodable")

def plane_to_uci(from_sq, plane, white_to_move) -> str:
    """
    Convert (from square index 0..63, plane index 0..63, turn) into UCI string.
    """
    fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
    dx, dy = plane_to_delta(plane)
    # Flip perspective for black
    if not white_to_move:
        dx, dy = -dx, -dy
    tx, ty = fx + dx, fy + dy
    if not (0 <= tx < 8 and 0 <= ty < 8):
        raise ValueError(f"Target off board: from {from_sq}, plane {plane}, white to move {white_to_move}")
    from_uci = chess.square_name(from_sq)
    to_uci = chess.square_name(chess.square(tx, ty))
    if 64 <= plane <= 66:
        return from_uci + to_uci + 'q' # promotion
    return from_uci + to_uci
def plane_to_delta(plane):
    """
    Helper function to return (dx,dy) for a given plane index
    """
    if 0 <= plane <= 6:  # forward
        return 0, plane + 1
    elif 7 <= plane <= 13:  # backward
        return 0, -(plane - 7 + 1)
    elif 14 <= plane <= 20:  # east
        return plane - 14 + 1, 0
    elif 21 <= plane <= 27:  # west
        return -(plane - 21 + 1), 0
    elif 28 <= plane <= 34:  # NW
        d = plane - 28 + 1
        return -d, d
    elif 35 <= plane <= 41:  # NE
        d = plane - 35 + 1
        return d, d
    elif 42 <= plane <= 48:  # SW
        d = plane - 42 + 1
        return -d, -d
    elif 49 <= plane <= 55:  # SE
        d = plane - 49 + 1
        return d, -d
    elif 56 <= plane <= 63:  # knights
        knight_order = [(-1,2), (1,2), (2,1), (2,-1),
                        (1,-2), (-1,-2), (-2,-1), (-2,1)]
        return knight_order[plane - 56]
    elif 64 <= plane <= 66:
        dx = 0
        if plane == 64:
            dx = 1
        elif plane == 66:
            dx = -1
        return dx, 1
    else:
        raise ValueError(f"Invalid plane index {plane}")

def decode_all_predictions(probs, fens):
    """
    Returns a list of tuples with every move and the probability that it is the "best" move
    If B > 1, returns a list of dictionaries for each
    """
    B = probs.shape[0]
    results = []
    # sort from highest to lowest prob
    sorted_idx = probs.argsort(dim=1, descending=True) # (B, 67*64)

    for b in range(B):
        nonzero = [i for i in sorted_idx[b].tolist() if probs[b, i] > 0]
        fen = fens[b]
        board = chess.Board(fen)
        white_to_move = board.turn == chess.WHITE
        moves = []
        for idx in nonzero:
            from_sq = idx % 64
            plane = idx // 64
            prob = probs[b, idx].item()
            uci = plane_to_uci(from_sq, plane, white_to_move)
            moves.append((uci, prob))
        results.append(moves)
    if B == 1:
        return results[0]
    return results