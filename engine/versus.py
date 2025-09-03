import chess

from main import eval_fen, load_old_model
from hyperparams import MATE_VALUE
import chess.polyglot
import random

def evaluate_position(fen, threefold_lookup, model):
    board = chess.Board(fen)
    key = chess.polyglot.zobrist_hash(board)
    if board.is_checkmate():
        return (MATE_VALUE if board.turn == chess.BLACK else -MATE_VALUE), True
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or threefold_lookup[key] >= 3:
        return 0, True
    else:
        evaluation, _ = eval_fen(fen, model)
        return evaluation, False

def get_next_move(board, model, depth, max_lines, starting_position, q_depth):

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
    score, move = minimax(fen, threefold_lookup, model, depth, max_lines, float('-inf'), float('inf'), q_depth)
    candidate_moves, _ = get_legal_moves(fen, model, max_lines)
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

    if q_depth <= 0 or game_over:
        return eval_raw

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

def minimax(fen, threefold_lookup, model, depth, max_lines, alpha, beta, q_depth):

    board = chess.Board(fen)
    key = chess.polyglot.zobrist_hash(board)
    threefold_lookup[key] = threefold_lookup.get(key, 0) + 1

    if depth == 0 or board.is_game_over(claim_draw=True) or threefold_lookup[key] >= 3:
        value = quiescence_search(fen, threefold_lookup, model, alpha, beta, q_depth)
        return value, None

    legal_moves, legal_fens = get_legal_moves(fen, model, max_lines)
    white_to_move = board.turn == chess.WHITE

    if white_to_move:
        best_value = float('-inf')
        best_move = None
        for i in range(len(legal_fens)):
            mv, f = legal_moves[i], legal_fens[i]
            value, _ = minimax(f, threefold_lookup, model, depth-1, max_lines, alpha, beta, q_depth)
            if value > best_value:
                best_value, best_move = value, mv
            alpha = max(alpha, value)
            if beta <= alpha:
                break
        return best_value, best_move
    else:
        best_value = float('inf')
        best_move = None
        for i in range(len(legal_fens)):
            mv, f = legal_moves[i], legal_fens[i]
            value, _ = minimax(f, threefold_lookup, model, depth-1, max_lines, alpha, beta, q_depth)
            if value < best_value:
                best_value, best_move = value, mv
            beta = min(beta, value)
            if beta <= alpha:
                break
        return best_value, best_move

def get_legal_moves(fen, model, max_lines):
    _, moves = eval_fen(fen, model)
    if max_lines is not None:
        moves = moves[:max_lines]

    board = chess.Board(fen)
    for move in board.legal_moves:
        if (board.is_capture(move) or board.gives_check(move)) and move.uci() not in moves:
            moves.append(move.uci())

    fens = []
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
        fens.append(board.fen())
        board.pop()
    return moves, fens

class Game:

    def __init__(self, model, bot_color=None, starting_position="", model_depth=1, q_depth=1, max_lines=None):
        self.bot_color = 'w' if bot_color == 'w' else 'b' if bot_color == 'b' else random.choice('wb')
        self.player_color = 'b' if self.bot_color == 'w' else 'w'
        self.board = chess.Board() if starting_position == "" else chess.Board(starting_position)
        self.model_depth = model_depth
        self.model = model
        self.max_lines = max_lines
        self.starting_position = starting_position
        self.bot_candidate_moves = []
        self.q_depth = q_depth
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
        best_move, candidate_moves = get_next_move(self.board, model=self.model, depth=self.model_depth, max_lines=self.max_lines, starting_position=self.starting_position, q_depth=self.q_depth)
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
    Game(m, bot_color='w', model_depth=3, max_lines=5, starting_position="")
