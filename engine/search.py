import math
from main import eval_fen, load_old_model
from heuristics import *
import chess.polyglot

def evaluate_position(fen, threefold_lookup, model):
    board = chess.Board(fen)
    key = chess.polyglot.zobrist_hash(board)
    if board.is_checkmate():
        halfmove_number = board.fullmove_number * 2 - (0 if board.turn == chess.WHITE else 1)
        mate_distance_adjustment = halfmove_number / 10
        if board.turn == chess.WHITE:
            return -MATE_VALUE + mate_distance_adjustment, True
        return MATE_VALUE - mate_distance_adjustment, True
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or threefold_lookup[key] >= 3:
        return 0, True
    else:
        evaluation, _ = eval_fen(fen, model)
        return evaluation, False

def get_next_move(board, model, depth, max_lines, starting_position, TT, q_depth, root_depth):

    def init_lookup(final_board):
        lookup = {}
        temp_board = chess.Board(starting_position) if starting_position != "" else chess.Board()
        for mv in final_board.move_stack:
            temp_board.push(mv)
            key = chess.polyglot.zobrist_hash(temp_board)
            lookup[key] = lookup.get(key, 0) + 1
        return lookup

    fen = board.fen()
    threefold_lookup = init_lookup(board)
    score, move = minimax(fen, threefold_lookup, model, depth, max_lines, float('-inf'), float('inf'), TT, q_depth, root_depth, root=True)
    candidate_moves, _ = get_candidate_moves(fen, model, max_lines)
    return move, candidate_moves

def quiescence_search(fen, threefold_lookup, model, alpha, beta, q_depth):
    eval_raw, game_over = evaluate_position(fen, threefold_lookup, model)

    if q_depth <= 0 or game_over:
        return eval_raw

    board = chess.Board(fen)
    forcing = [mv for mv in board.legal_moves if board.is_capture(mv)]
    if not forcing:
        return eval_raw

    def piece_val(pt):
        return {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9}.get(pt, 0)

    def score_move(move):
        check_flag = 1 if board.gives_check(move) else 0
        cap_val = 0
        if board.is_capture(move):
            if board.is_en_passant(move):
                cap_val = 1
            else:
                cap_piece = board.piece_at(move.to_square)
                cap_val = piece_val(cap_piece.piece_type) if cap_piece else 0
        promo_bonus = 8 if move.promotion else 0
        return check_flag, cap_val + promo_bonus

    forcing.sort(key=score_move, reverse=True)

    if board.turn == chess.WHITE:
        if eval_raw > alpha:
            alpha = eval_raw
        if alpha >= beta:
            return alpha
        best = alpha
        for mv in forcing:
            board.push(mv)
            key = chess.polyglot.zobrist_hash(board)
            threefold_lookup[key] = threefold_lookup.get(key, 0) + 1
            score = quiescence_search(board.fen(), threefold_lookup, model, best, beta, q_depth - 1)
            threefold_lookup[key] -= 1
            board.pop()
            if score > best:
                best = score
                if best >= beta:
                    return best
        return best
    else:
        if eval_raw < beta:
            beta = eval_raw
        if beta <= alpha:
            return beta
        best = beta
        for mv in forcing:
            board.push(mv)
            key = chess.polyglot.zobrist_hash(board)
            threefold_lookup[key] = threefold_lookup.get(key, 0) + 1
            score = quiescence_search(board.fen(), threefold_lookup, model, alpha, best, q_depth - 1)
            threefold_lookup[key] -= 1
            board.pop()
            if score < best:
                best = score
                if best <= alpha:
                    return best
        return best

def minimax(fen, threefold_lookup, model, depth, max_lines, alpha, beta, TT, q_depth, root_depth, root=False):
    board = chess.Board(fen)
    key = chess.polyglot.zobrist_hash(board)
    orig_alpha, orig_beta = alpha, beta
    threefold_lookup[key] = threefold_lookup.get(key, 0) + 1

    if key in TT:
        entry = TT[key]
        if entry['depth'] >= depth:
            if entry['flag'] == 'EXACT':
                return entry['value'], entry.get('best_move', None)
            if entry['flag'] == 'LOWERBOUND':
                alpha = max(alpha, entry['value'])
            elif entry['flag'] == 'UPPERBOUND':
                beta = min(beta, entry['value'])
            if alpha >= beta:
                return entry['value'], entry.get('best_move', None)

    if depth == 0 or board.is_game_over(claim_draw=True) or threefold_lookup[key] >= 3:
        value = quiescence_search(fen, threefold_lookup, model, alpha, beta, q_depth)
        return value, None

    candidate_moves, candidate_preds = get_candidate_moves(fen, model, max_lines)
    white_to_move = board.turn == chess.WHITE

    best_value_unweighted = float('-inf') if white_to_move else float('inf')
    best_value_weighted = float('-inf') if white_to_move else float('inf')
    best_move_weighted = None
    best_move_unweighted = None

    for i in range(len(candidate_moves)):
        mv_uci = candidate_moves[i]
        move = chess.Move.from_uci(mv_uci)
        endgame_lambda = lambda_value(board)
        heuri_eval = heuristic_eval(board, move)
        board.push(move)
        child_key = chess.polyglot.zobrist_hash(board)
        value_raw, _ = minimax(board.fen(), threefold_lookup, model, depth-1, max_lines, alpha, beta, TT, q_depth, root_depth, root=False)
        value_unweighted = value_raw + endgame_lambda * heuri_eval
        threefold_lookup[child_key] -= 1
        board.pop()

        prob = candidate_preds[i]
        weight = PRED_WEIGHT * math.log(max(prob, 1e-8))
        value_weighted = value_unweighted
        if root:
            value_weighted = value_unweighted + (weight if white_to_move else -weight)

        if depth == root_depth:
            print(f"depth={depth} -- {mv_uci}, raw={value_raw:.4f}, unw={value_unweighted:.4f}, prob={prob:.4f}, w={value_weighted:.4f}")

        if white_to_move:
            if value_weighted > best_value_weighted:
                best_value_weighted, best_move_weighted = value_weighted, mv_uci
            if value_unweighted > best_value_unweighted:
                best_value_unweighted, best_move_unweighted = value_unweighted, mv_uci
            alpha = max(alpha, value_unweighted)
        else:
            if value_weighted < best_value_weighted:
                best_value_weighted, best_move_weighted = value_weighted, mv_uci
            if value_unweighted < best_value_unweighted:
                best_value_unweighted, best_move_unweighted = value_unweighted, mv_uci
            beta = min(beta, value_unweighted)
        if beta <= alpha:
            break
    if best_value_unweighted <= orig_alpha:
        flag = 'UPPERBOUND'
    elif best_value_unweighted >= orig_beta:
        flag = 'LOWERBOUND'
    else:
        flag = 'EXACT'
    TT[key] = {
        'depth': depth,
        'value': best_value_unweighted,
        'flag': flag,
        'best_move': best_move_unweighted
    }
    if root:
        return best_value_weighted, best_move_weighted
    return best_value_unweighted, best_move_unweighted

def get_candidate_moves(fen, model, max_lines):
    # top moves by raw preds
    _, move_preds = eval_fen(fen, model)
    moves, preds = zip(*move_preds)
    moves, preds = list(moves), list(preds)
    moves_cut, preds_cut = moves, preds
    if max_lines is not None:
        moves_cut = moves[:max_lines]
        preds_cut = preds[:max_lines]

    # tactical moves (checks and captures)
    board = chess.Board(fen)
    for move in board.legal_moves:
        if (board.is_capture(move) or board.gives_check(move)) and move.uci() not in moves_cut:
            moves_cut.append(move.uci())
            found_pred = None
            for m_uci, p in zip(moves, preds):
                if m_uci == move.uci():
                    found_pred = p
                    break
            if found_pred is None:
                found_pred = 0.001
            preds_cut.append(found_pred)

    # hanging pieces safety net
    engine_color = board.turn
    attacked_squares = []
    for sq, piece in board.piece_map().items():
        if piece.color == engine_color and piece.piece_type != chess.PAWN:
            if board.is_attacked_by(not engine_color, sq):
                attacked_squares.append(sq)
    for sq in attacked_squares:
        piece_moves = [mv.uci() for mv in board.legal_moves if mv.from_square == sq]
        moves_found = 0
        for mv, pred in zip(moves, preds):
            if moves_found >= 2:
                break
            if mv in piece_moves and mv not in moves_cut:
                moves_found += 1
                moves_cut.append(mv)
                preds_cut.append(pred)

    # renormalize
    preds_cut = [pred / sum(preds_cut) for pred in preds_cut]
    return moves_cut, preds_cut

class Game:

    def __init__(self, model, bot_color=None, starting_position=""):
        def random(s):
            import random
            return random.choice(s)
        self.bot_color = 'w' if bot_color == 'w' else 'b' if bot_color == 'b' else random('wb')
        self.player_color = 'b' if self.bot_color == 'w' else 'w'
        self.board = chess.Board() if starting_position == "" else chess.Board(starting_position)
        self.model_depth = MODEL_DEPTH
        self.model = model
        self.max_lines = MAX_LINES
        self.starting_position = starting_position
        self.bot_candidate_moves = []
        self.q_depth = QUIESCENCE_DEPTH
        self.TT = {}
        if self.bot_color == 'w':
            print("Bot is playing the white pieces, and will make the first move.")
            self.bot_move()
        else:
            print("Player is playing the white pieces, and will make the first move.")
            self.prompt_player_move()

    def bot_move(self):
        if self.board.is_game_over():
            print("Game over! Result:", self.board.result())
            return
        best_move, candidate_moves = get_next_move(self.board, model=self.model, depth=self.model_depth, max_lines=self.max_lines, starting_position=self.starting_position, TT=self.TT, q_depth=self.q_depth, root_depth=self.model_depth)
        self.bot_candidate_moves = candidate_moves
        best_move = chess.Move.from_uci(best_move)
        san = self.board.san(best_move)
        self.board.push(best_move)
        response = f"Bot plays {san}. {'White' if self.player_color == 'w' else 'Black'} to move."
        print(response)
        self.prompt_player_move()

    def prompt_player_move(self):
        if self.board.is_game_over():
            print("Game over! Result: ", self.board.result())
            return
        while True:
            user_input = input("Your move in UCI notation: ")
            try:
                if user_input == 'show_moves':
                    print(self.bot_candidate_moves)
                    continue
                move = chess.Move.from_uci(user_input)
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.bot_move()
                    break
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid UCI format. Try again.")

if __name__ == '__main__':
    m = load_old_model()
    Game(m, bot_color='w', starting_position="rn1k3r/pp3B2/3p2pp/8/3PNn2/2N2b2/PPP4P/2KR2R1 w - - 2 18")
