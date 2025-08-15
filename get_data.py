import chess.pgn
import re
import json

def get_fens():
    fens = []
    mate_value = 10000
    eval_regex = re.compile(r"\[%eval ([^]]+)]")
    with open("games_f.pgn") as pgn:
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
                    fens.append((board.fen(), eval_value))
                board.push(node.move)

    fens_dict = [{"fen": fen, "eval": eval_value} for fen, eval_value in fens]
    max_eval = max(fens_dict, key=lambda x: x["eval"])
    print(max_eval)

    with open("fen_evals.json", "w", encoding="utf-8") as f:
        json.dump(fens_dict, f, ensure_ascii=False, indent=2)


def get_data(max_num_games, log_freq):
    games = []
    with open("games.pgn") as pgn:
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
            if i == max_num_games:
                break
    print(f"{len(games)} games compiled!")

    with open("games_f.pgn", "w", encoding="utf-8") as out_pgn:
        for g in games:
            exporter = chess.pgn.FileExporter(out_pgn)
            g.accept(exporter)

get_fens()