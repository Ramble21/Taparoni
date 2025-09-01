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
        evals = data['evals'].tolist()
        preds = data['preds'].tolist()
        return features, evals, preds
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "..", "data", "evals.json")

        with open(file_path, 'r') as f:
            data = json.load(f)

        features = [entry["fnn"] for entry in data]
        evals = [entry["eval"] for entry in data]
        preds = [entry['next_move'] for entry in data]
        np.savez(path, features=features, evals=evals, preds=preds)
        return features, evals, preds

def get_filtered_games(log_freq=1000, max_games=None, max_positions=None,
                    filter_elite=False, elite_filter_min=2100):
    re_eval = re.compile(r"\[%eval")
    re_white = re.compile(r'\[WhiteElo "(\d+)"]')
    re_black = re.compile(r'\[BlackElo "(\d+)"]')
    re_event = re.compile(r'\[Event "([^"]+)"]')
    re_result = re.compile(r"(1-0|0-1|1/2-1/2|\*)")

    total_positions = 0
    games_written = 0
    buffer = []

    with open("../data/games.pgn", encoding="utf-8") as f_in, open("../data/games_f.pgn", "w", encoding="utf-8") as f_out:
        for line in f_in:
            if line.strip() == "":
                # blank line, but don’t decide yet, just store it
                buffer.append(line)
                continue

            buffer.append(line)

            # If this line has the game result, we know the game ended
            if re_result.search(line):
                game_txt = "".join(buffer)
                buffer.clear()

                if filter_elite:
                    m_white = re_white.search(game_txt)
                    m_black = re_black.search(game_txt)
                    white = int(m_white.group(1)) if m_white else 0
                    black = int(m_black.group(1)) if m_black else 0
                    event = re_event.search(game_txt)
                    event = event.group(1).lower() if event else ""
                    if white < elite_filter_min or black < elite_filter_min or "bullet" in event:
                        continue

                if re_eval.search(game_txt):
                    pos_count = game_txt.count("[%eval")
                    total_positions += pos_count
                    games_written += 1
                    f_out.write(game_txt + "\n\n")

                    if games_written % log_freq == 0:
                        print(f"{games_written} games, {total_positions} positions processed")

                    if max_games and games_written >= max_games:
                        break
                    if max_positions and total_positions >= max_positions:
                        break

    print(f"{games_written} games ({total_positions} positions) compiled!")

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

def get_evals_json(log_freq):
    """
        Converts a .pgn file named "games_f.pgn" into a JSON named "evals.json"
        "fen_evals.json" contains a list of dictionaries containing every move from every game in
        "games_f.pgn" (as FENs) corresponding to the Stockfish evaluation of that position (in centipawns)
    """
    dataset = []
    mate_value = MAX_CENTIPAWNS
    eval_regex = re.compile(r"\[%eval ([^]]+)]")
    num_positions = 0
    num_games = 0
    with open("../data/games_f.pgn") as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            num_games += 1
            if game is None:
                break
            board = game.board()
            mainline_nodes = list(game.mainline())
            for i, node in enumerate(mainline_nodes):

                match = eval_regex.search(node.comment)
                if not match:
                    board.push(node.move)
                    continue

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
                if i + 1 < len(mainline_nodes):
                    next_move = mainline_nodes[i].move.uci()
                    dataset.append({
                        "fnn": board.fen(),
                        "eval": value,
                        "next_move": next_move
                    })
                    num_positions += 1
                board.push(node.move)
            if num_games % log_freq == 0:
                print(f"{num_games} games processed ({num_positions} positions)")

    print(f"{num_games} games and {num_positions} positions processed in total!")
    with open("../data/evals.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    get_evals_json(250)