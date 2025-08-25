import chess.pgn
import os
import re
import json
import numpy as np
from hyperparams import MAX_CENTIPAWNS

def get_dataset():
    """
    Retrieves information from "evals.json" in order to create the raw features and labels
    Returns both features and labels in raw Python lists to later be turned into tensors
    Saves created lists as a Numpy array file (.npz) to avoid recomputing in consecutive grabs
    If data file contains these arrays saved as a Numpy file, retrieves directly from there instead
    """
    path = '../data/dataset.npz'
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        features = data['features'].tolist()
        labels = data['labels'].tolist()
        return features, labels
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "data", "evals.json")

        with open(file_path, 'r') as f:
            data = json.load(f)

        feature_list = [entry["wnn"] for entry in data]
        label_list = [entry["eval"] for entry in data]
        np.savez(path, features=feature_list, labels=label_list)
        return feature_list, label_list

def get_filtered_games(log_freq, max_num_games=None):
    """
    Converts a .pgn file named "games.pgn" taken from Lichess's database into a .pgn file named "games_f.pgn"
    "games_f.pgn" is a parsed version of the file that contains the first max_num_games in "games.pgn" that have
    pre-installed Stockfish annotation. If there aren't max_num_games qualifying games, it will return all that do
    qualify.
    """
    games = []
    with open("../data/games.pgn") as pgn:
        i = 0
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            has_eval = False
            for node in game.mainline():
                comment = node.comment
                if "[%eval" in comment:
                    has_eval = True
                    break
            if has_eval:
                games.append(game)
                i += 1
                if i % log_freq == 0:
                    print(f"{i} games processed")
            if max_num_games is not None and i == max_num_games:
                break
    print(f"{len(games)} games compiled!")

    with open("../data/games_f.pgn", "w", encoding="utf-8") as out_pgn:
        for g in games:
            exporter = chess.pgn.FileExporter(out_pgn)
            g.accept(exporter)

def fen_to_wnn(fen):
    """
    Converts a FEN string into the more neural network-friendly WNN.
    WNN contains information about every square and all blank squares are . (instead of numbers like 2 and 3 to indicate
    2 or 3 blank squares), castling rights are condensed into one character (B instead of KQ for both), and en passant /
    the 50 move rule are ignored for simplicity
    """
    fen_fields = fen.split()
    wnn = []
    for c in fen_fields[0]:
        if c.isdigit():
            for _ in range(int(c)):
                wnn.append(".")
        elif c != '/':
            wnn.append(c)
    wnn.append(fen_fields[1]) # turn indicator
    def conv_castle(s_fen):
        """
        Helper function for the castling rights condenser
        """
        w_kingside = 'K' in s_fen
        w_queenside = 'Q' in s_fen
        b_kingside = 'k' in s_fen
        b_queenside = 'q' in s_fen

        if w_kingside and w_queenside:
            white_code = 'B'
        elif w_kingside:
            white_code = 'K'
        elif w_queenside:
            white_code = 'Q'
        else:
            white_code = 'N'

        if b_kingside and b_queenside:
            black_code = 'b'
        elif b_kingside:
            black_code = 'k'
        elif b_queenside:
            black_code = 'q'
        else:
            black_code = 'n'

        return white_code + black_code

    wnn.append(conv_castle(fen_fields[2]))
    return "".join(wnn)

def wnn_to_fen(wnn):
    """
    Converts a WNN string back into a FEN string. (for testing purposes)
    Ignores en passant and fifty-move rule (assumes 0 moves, no en passant)
    """
    board_part = wnn[:64]
    turn = wnn[64]
    castle_code = wnn[65:67]

    fen_rows = []
    for rank in range(8):
        row = board_part[rank*8:(rank+1)*8]
        fen_row = ""
        empty_count = 0
        for c in row:
            if c == '.':
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += c
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    fen_board = "/".join(fen_rows)

    def decode_castle(code):
        w, b = code[0], code[1]
        castle = ""
        if w == 'B':
            castle += 'KQ'
        elif w == 'K':
            castle += 'K'
        elif w == 'Q':
            castle += 'Q'
        if b == 'b':
            castle += 'kq'
        elif b == 'k':
            castle += 'k'
        elif b == 'q':
            castle += 'q'
        return castle if castle else "-"
    fen_castle = decode_castle(castle_code)
    return f"{fen_board} {turn} {fen_castle} - 0 1"

def get_evals_json():
    """
        Converts a .pgn file named "games_f.pgn" into a JSON named "evals.json"
        "fen_evals.json" contains a list of dictionaries containing every move from every game in
        "games_f.pgn" (as WNNs) corresponding to the Stockfish evaluation of that position (in centipawns)
    """
    fens = []
    mate_value = MAX_CENTIPAWNS
    eval_regex = re.compile(r"\[%eval ([^]]+)]")
    with open("../data/games_f.pgn") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            for node in game.mainline():
                match = eval_regex.search(node.comment)
                if match:
                    eval_str = match.group(1)
                    if eval_str.startswith("#"):
                        mate_in = int(eval_str[1:])
                        eval_value = mate_value if mate_in > 0 else -mate_value
                    else:
                        eval_value = float(eval_str) * 100
                        if eval_value > mate_value:
                            eval_value = mate_value
                        elif eval_value < -mate_value:
                            eval_value = -mate_value
                    value = int(eval_value)
                    fens.append((fen_to_wnn(board.fen()), value))
                board.push(node.move)

    fens_dict = [{"wnn": fen, "eval": eval_value} for fen, eval_value in fens]
    max_eval = max(fens_dict, key=lambda x: x["eval"])
    print(max_eval)

    with open("../data/evals.json", "w", encoding="utf-8") as f:
        json.dump(fens_dict, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    get_evals_json()