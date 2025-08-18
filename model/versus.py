from main import *
import random

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
        fen = self.board.fen()
        best_move, best_fen = get_next_move(fen=fen, model=self.model, depth=self.model_depth)
        san = self.board.san(best_move)
        self.board.push(best_move)
        response = f"Bot plays {san}. {"White" if self.player_color == 'w' else "Black"} to move."
        print(response)
        self.prompt_player_move()

    def prompt_player_move(self):
        disclaimer = "Bot only accepts UCI notation (start square -> end square ex. d3d5, g1f3)"
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

m = load_old_model()
Game(m, bot_color='b', model_depth=2)
# known bug: lots of errors are thrown when game ends as of now