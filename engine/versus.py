import chess

from main import eval_fen, load_old_model
import chess.polyglot
import random

def evaluate_position(fen, threefold_lookup, model):
    board = chess.Board(fen)
    key = chess.polyglot.zobrist_hash(board)
    if board.is_checkmate():
        return 100 if board.turn == chess.BLACK else -100
    elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_fifty_moves() or threefold_lookup[key] >= 3:
        return 0
    else:
        evaluation, _ = eval_fen(fen, model)
        return evaluation

def get_next_move(board, model, depth, max_lines, starting_position):

    def init_lookup(final_board):
        lookup = {}
        temp_board = chess.Board(starting_position) if starting_position is not None else chess.Board()
        for mv in final_board.move_stack:
            temp_board.push(mv)
            key = chess.polyglot.zobrist_hash(temp_board)
            lookup[key] = lookup.get(key, 0) + 1
        return lookup

    fen = board.fen()
    threefold_lookup = init_lookup(board)
    score, move = minimax(fen, threefold_lookup, model, depth, max_lines, float('-inf'), float('inf'))
    return move

def minimax(fen, threefold_lookup, model, depth, max_lines, alpha, beta):

    board = chess.Board(fen)
    key = chess.polyglot.zobrist_hash(board)
    threefold_lookup[key] = threefold_lookup.get(key, 0) + 1

    if depth == 0 or board.is_game_over(claim_draw=True) or threefold_lookup[key] >= 3:
        value = evaluate_position(fen, threefold_lookup, model)
        threefold_lookup[key] -= 1
        return value, None

    legal_moves, legal_fens = get_legal_moves(fen, model, max_lines)
    white_to_move = fen.split()[1] == 'w'

    if white_to_move:
        best_value = float('-inf')
        best_move = None
        for i in range(len(legal_fens)):
            mv, f = legal_moves[i], legal_fens[i]
            value, _ = minimax(f, threefold_lookup, model, depth-1, max_lines, alpha, beta)
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
            value, _ = minimax(f, threefold_lookup, model, depth-1, max_lines, alpha, beta)
            if value < best_value:
                best_value, best_move = value, mv
            beta = min(beta, value)
            if beta <= alpha:
                break
        return best_value, best_move

def get_legal_moves(fen, model, max_lines):
    _, moves = eval_fen(fen, model)
    white_to_move = fen.split()[1] == 'w'
    board = chess.Board(fen)
    if not white_to_move:
        moves.reverse()
    fens = []
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
        fens.append(board.fen())
        board.pop()
    if max_lines is None:
        return moves, fens
    return moves[:max_lines], fens[:max_lines]

class Game:

    def __init__(self, model, bot_color=None, starting_position=None, model_depth=1, max_lines=None):
        self.bot_color = 'w' if bot_color == 'w' else 'b' if bot_color == 'b' else random.choice('wb')
        self.player_color = 'b' if self.bot_color == 'w' else 'w'
        self.board = chess.Board() if starting_position is None else chess.Board(starting_position)
        self.white_to_move = True
        self.model_depth = model_depth
        self.model = model
        self.max_lines = max_lines
        self.starting_position = starting_position
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
        best_move = get_next_move(self.board, model=self.model, depth=self.model_depth, max_lines=self.max_lines, starting_position=self.starting_position)
        best_move = chess.Move.from_uci(best_move)
        san = self.board.san(best_move)
        self.board.push(best_move)
        response = f"Bot plays {san}. {"White" if self.player_color == 'w' else "Black"} to move."
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
                    break
                else:
                    print("Illegal move. Try again.")
            except ValueError:
                print("Invalid UCI format. Try again.")
        self.board.push(move)
        self.bot_move()

if __name__ == '__main__':
    m = load_old_model()
    Game(m, bot_color='w', model_depth=4, max_lines=5, starting_position="Q7/7k/8/2K5/8/8/8/8 w - - 1 66")
