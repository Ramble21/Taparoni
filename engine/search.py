import math
from main import eval_boards, load_old_model
from heuristics import *
import chess.polyglot

def eval_boards_memoized(boards, model, eval_cache):
    uncached_boards = []
    uncached_keys = []
    results = {}
    for board in boards:
        key = chess.polyglot.zobrist_hash(board)
        if key in eval_cache:
            results[key] = eval_cache.get(key)
        else:
            uncached_boards.append(board)
            uncached_keys.append(key)

    if uncached_boards:
        evs, preds = eval_boards(uncached_boards, model)
        for key, ev, pred in zip(uncached_keys, evs, preds):
            eval_cache[key] = (ev, pred)
            results[key] = (ev, pred)

    evs_out, preds_out = [], []
    for board in boards:
        key = chess.polyglot.zobrist_hash(board)
        ev, pred = results[key]
        evs_out.append(ev)
        preds_out.append(pred)
    return evs_out, preds_out

def evaluate_position(board, threefold_lookup, model, eval_cache):
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
        evaluation, _ = eval_boards_memoized([board], model, eval_cache)
        return evaluation[0], False

def get_child_eval_batch(board, model, eval_cache):
    child_boards = []
    child_fens = []
    for move in list(board.legal_moves):
        board.push(move)
        child_boards.append(board.copy())
        child_fens.append(board.fen())
        board.pop()
    child_batch_evals, child_batch_preds = eval_boards_memoized(child_boards, model, eval_cache)
    child_batch = [(fen, ev, pred) for fen, ev, pred in zip(child_fens, child_batch_evals, child_batch_preds)]
    return child_batch

def get_next_move(board, model, depth, max_lines, starting_position, TT, eval_cache, q_depth, root_depth):

    def init_lookup(final_board):
        lookup = {}
        temp_board = chess.Board(starting_position) if starting_position != "" else chess.Board()
        for mv in final_board.move_stack:
            temp_board.push(mv)
            key = chess.polyglot.zobrist_hash(temp_board)
            lookup[key] = lookup.get(key, 0) + 1
        return lookup

    threefold_lookup = init_lookup(board)

    score, move = minimax(board=board, threefold_lookup=threefold_lookup, model=model, depth=depth,
                          max_lines=max_lines, alpha=float('-inf'), beta=float('inf'), TT=TT,
                          eval_cache=eval_cache, q_depth=q_depth, root_depth=root_depth, eval_batch=None)
    return move

def minimax(board, threefold_lookup, model, depth, max_lines, alpha, beta, TT, eval_cache, q_depth, root_depth, eval_batch):
    key = chess.polyglot.zobrist_hash(board)
    orig_alpha, orig_beta = alpha, beta
    threefold_lookup[key] = threefold_lookup.get(key, 0) + 1

    try:
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
                    if entry['flag'] == 'LOWERBOUND':
                        return alpha, entry.get('best_move', None)
                    else:
                        return beta, entry.get('best_move', None)

        if depth == 0 or board.is_game_over(claim_draw=True) or threefold_lookup[key] >= 3:
            value = quiescence_search(board=board, threefold_lookup=threefold_lookup, model=model,
                                      alpha=alpha, beta=beta, q_depth=q_depth, eval_cache=eval_cache)
            return value, None

        candidate_moves, candidate_preds, eval_set = get_candidate_moves(board, eval_batch, max_lines, model, eval_cache)
        white_to_move = board.turn == chess.WHITE
        if key in TT and TT[key].get('best_move') in candidate_moves:
            candidate_moves.remove(TT[key]['best_move'])
            candidate_moves.insert(0, TT[key]['best_move'])
        child_eval_batch = get_child_eval_batch(board, model, eval_cache)

        best_value_weighted = float('-inf') if white_to_move else float('inf')
        best_move_weighted = None

        for i in range(len(candidate_moves)):
            mv_uci = candidate_moves[i]
            move = chess.Move.from_uci(mv_uci)
            endgame_lambda = lambda_value(board)
            heuri_eval = heuristic_eval(board, move)

            board.push(move)
            value_raw, _ = minimax(board=board, threefold_lookup=threefold_lookup, model=model,
                                   depth=depth-1, max_lines=max_lines, alpha=alpha, beta=beta, TT=TT,
                                   eval_cache=eval_cache, q_depth=q_depth, root_depth=root_depth, eval_batch=child_eval_batch)
            value_unweighted = value_raw + endgame_lambda * heuri_eval
            board.pop()

            prob = candidate_preds[i]
            weight = PRED_WEIGHT * math.log(prob)
            value_weighted = value_unweighted + (weight if white_to_move else -weight)

            if depth == root_depth:
                print(f"depth={depth} -- {mv_uci}, unw={value_unweighted:.3f}, weight={weight:.3f}, prob={prob:.3f}, w={value_weighted:.3f}")

            if white_to_move:
                if value_weighted > best_value_weighted:
                    best_value_weighted, best_move_weighted = value_weighted, mv_uci
                    alpha = max(alpha, value_weighted)
            else:
                if value_weighted < best_value_weighted:
                    best_value_weighted, best_move_weighted = value_weighted, mv_uci
                    beta = min(beta, value_weighted)
            if beta <= alpha:
                break

        if best_value_weighted <= orig_alpha:
            flag = 'UPPERBOUND'
        elif best_value_weighted >= orig_beta:
            flag = 'LOWERBOUND'
        else:
            flag = 'EXACT'
        TT[key] = {
            'depth': depth,
            'value': best_value_weighted,
            'flag': flag,
            'best_move': best_move_weighted,
            'eval_set': eval_set
        }
        return best_value_weighted, best_move_weighted
    finally:
        threefold_lookup[key] -= 1
        if threefold_lookup[key] <= 0:
            del threefold_lookup[key]

def quiescence_search(board, threefold_lookup, model, alpha, beta, q_depth, eval_cache):
    eval_raw, game_over = evaluate_position(board, threefold_lookup, model, eval_cache)

    if q_depth <= 0 or game_over:
        return eval_raw

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
            score = quiescence_search(board, threefold_lookup, model, best, beta, q_depth - 1, eval_cache)
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
            score = quiescence_search(board, threefold_lookup, model, alpha, best, q_depth - 1, eval_cache)
            threefold_lookup[key] -= 1
            board.pop()
            if score < best:
                best = score
                if best <= alpha:
                    return best
        return best

def get_candidate_moves(board, eval_batch, max_lines, model, eval_cache):
    move_preds, ev_set = None, None

    if eval_batch is not None:
        for eval_set in eval_batch:
            if eval_set[0] == board.fen():
                _, _, move_preds = eval_set
                ev_set = eval_set
                break
        if move_preds is None:
            raise RuntimeError("big bug")
    else:
        _, move_preds = eval_boards_memoized([board], model, eval_cache)

    moves, preds = zip(*move_preds)
    moves, preds = list(moves), list(preds)
    if max_lines is not None:
        moves_cut = moves[:max_lines]
        preds_cut = preds[:max_lines]
    else:
        moves_cut, preds_cut = moves, preds

    # tactical moves (checks and captures)
    for move in board.legal_moves:
        if (board.is_capture(move) or board.gives_check(move)) and move.uci() not in moves_cut:
            target_square = move.to_square
            is_defended = board.is_attacked_by(not board.turn, target_square)
            piece = None if not board.is_capture(move) else chess.Piece(chess.PAWN, not board.turn) if board.is_en_passant(move) else board.piece_at(target_square)
            if board.is_capture(move) and not board.gives_check(move) and piece == chess.PAWN and not is_defended:
                # don't include capturing defended pawns as a tactical move
                continue
            moves_cut.append(move.uci())
            if board.is_capture(move) and not is_defended:
                found_pred = HANGING_PIECE_PROB * material_val(piece)
            else:
                found_pred = 0.01
                for m_uci, p in zip(moves, preds):
                    if m_uci == move.uci():
                        found_pred = p
                        break
            preds_cut.append(found_pred)

    # hanging pieces safety net
    engine_color = board.turn
    attacked_squares = []
    for sq, piece in board.piece_map().items():
        if piece.color == engine_color:
            if piece.piece_type == chess.PAWN and board.is_attacked_by(engine_color, sq):
                # don't include defended pawns as hanging pieces
                continue
            if board.is_attacked_by(not engine_color, sq):
                attacked_squares.append(sq)
    for sq in attacked_squares:
        piece_moves = [mv.uci() for mv in board.legal_moves if mv.from_square == sq]
        moves_found = 0
        for mv, pred in zip(moves, preds):
            if moves_found >= HP_SN_MAX:
                break
            if mv in piece_moves and mv not in moves_cut:
                moves_found += 1
                moves_cut.append(mv)
                preds_cut.append(pred)

    # renormalize
    preds_cut = [pred / sum(preds_cut) for pred in preds_cut]
    return moves_cut, preds_cut, ev_set

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
        self.q_depth = QUIESCENCE_DEPTH
        self.TT = {}
        self.eval_cache = {}
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
        best_move = get_next_move(self.board, model=self.model, depth=self.model_depth, max_lines=self.max_lines,
                                  starting_position=self.starting_position, TT=self.TT, eval_cache=self.eval_cache,
                                  q_depth=self.q_depth, root_depth=self.model_depth)
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
    Game(m, bot_color='w', starting_position="")
