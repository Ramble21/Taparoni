import chess.pgn

def get_data(num_games, log_freq):
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
            if i == num_games:
                break
    print(f"{len(games)} games compiled!")

    with open("games_f.pgn", "w", encoding="utf-8") as out_pgn:
        for g in games:
            exporter = chess.pgn.FileExporter(out_pgn)
            g.accept(exporter)

get_data(num_games=1000, log_freq=10)