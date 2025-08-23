from main import *
import random

def get_next_move(fen, model, depth):
  legal_moves, legal_fens = get_legal_moves(fen)
  white_to_move = fen.split()[1] == 'w'
  if not legal_moves:
    raise RuntimeError("No legal moves!")

  if white_to_move:
    best_score = float('-inf')
    best_move, best_fen = None, None
    for move, fen in zip(legal_moves, legal_fens):
      score = minimax(fen, model, depth-1, float('-inf'), float('inf'))
      if score > best_score:
        best_score = score
        best_move, best_fen = move, fen
    return best_move, best_fen
  else:
    best_score = float('inf')
    best_move, best_fen = None, None
    for move, fen in zip(legal_moves, legal_fens):
      score = minimax(fen, model, depth-1, float('-inf'), float('inf'))
      if score < best_score:
        best_score = score
        best_move, best_fen = move, fen
    return best_move, best_fen

def minimax(fen, model, depth, alpha, beta):
  legal_moves, legal_fens = get_legal_moves(fen)
  white_to_move = fen.split()[1] == 'w'
  if depth == 0 or not legal_moves:
    return eval_fen(fen, model)

  if white_to_move:
    best_value = float('-inf')
    for f in legal_fens:
      value = minimax(f, model, depth-1, alpha, beta)
      best_value = max(value, best_value)
      alpha = max(alpha, value)
      if beta <= alpha:
        break
    return best_value
  else:
    best_value = float('inf')
    for f in legal_fens:
      value = minimax(f, model, depth-1, alpha, beta)
      best_value = min(best_value, value)
      beta = min(beta, value)
      if beta <= alpha:
        break
    return best_value

def get_legal_moves(fen):
  board = chess.Board(fen)
  moves = list(board.legal_moves)
  fens = []
  for move in moves:
    board.push(move)
    fens.append(board.fen())
    board.pop()
  return moves, fens

class Game:

    def __init__(self, model, bot_color=None, model_depth=1):
        self.bot_color = 'w' if bot_color == 'w' else 'b' if bot_color == 'b' else random.choice('wb')
        self.player_color = 'b' if self.bot_color == 'w' else 'w'
        self.board = chess.Board()
        self.white_to_move = True
        self.model_depth = model_depth
        self.model = model
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
        fen = self.board.fen()
        best_move, best_fen = get_next_move(fen=fen, model=self.model, depth=self.model_depth)
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
    Game(m, bot_color='b', model_depth=2)
