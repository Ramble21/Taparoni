import chess

from engine.heuristics import fen_material_balance
from engine.plane_utils import move_to_plane, decode_all_predictions
from parse_data import *
from model import TwoHeadTransformer
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math
from hyperparams import *

# Seed for consistent random numbers
SEED = 8675309
torch.manual_seed(SEED)

# Open dataset
features_raw, evals_raw, preds_raw = get_dataset()
print(f"Dataset size: {len(features_raw):,} FENs")

# --------------------- Pretraining Tokenization --------------------------- #
# Convert centipawns into deca-pawns? (centipawns / 1000)
evals_raw_deca = [x / 1000 for x in evals_raw]
with open("../ucis.txt", "r") as f:
    ucis_raw = [line.strip() for line in f if line.strip()]

vocab_size = len(ucis_raw)
print(f"{vocab_size} pre-train tokens")

# --------------------- Finetuning Tokenization --------------------------- #

# Piece tokenization
piece_to_pi = {
    '.': 0,  # empty square
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,  # white pieces
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6  # black pieces
}
encode_pieces = lambda s: [piece_to_pi.get(p) for p in s]
# Color tokenization
piece_to_ci = {
    '.': 0,  # empty square
    'P': 1, 'N': 1, 'B': 1, 'R': 1, 'Q': 1, 'K': 1,  # white pieces
    'p': 2, 'n': 2, 'b': 2, 'r': 2, 'q': 2, 'k': 2  # black pieces
}
encode_colors = lambda s: [piece_to_ci.get(p) for p in s]
# Turn to move tokenization
encode_ttm = lambda c: [0] if c == 'w' else [1]

dataset_path = '../data/dataset_encoded.pt'
if os.path.exists(dataset_path):
    saved = torch.load(dataset_path)
    piece_features = saved["piece_features"]
    color_features = saved["color_features"]
    ttm_features = saved["ttm_features"]
    eval_labels = saved["eval_labels"]
    pred_labels = saved["pred_labels"]
    print(f"Loaded encoded tensors from {dataset_path}")
else:
    def create_pred_labels(fens, ucis):
        labels = []
        for fen, uci in zip(fens, ucis):
            board = chess.Board(fen)
            white_to_move = board.turn == chess.WHITE
            move = chess.Move.from_uci(uci)
            plane = move_to_plane(move, white_to_move)
            labels.append(plane * 64 + move.from_square)
        return torch.tensor(labels, dtype=torch.long, device=DEVICE)
    # Only encode first 64 characters of WNN that represent the board into piece and color features
    piece_features = torch.stack([torch.tensor(encode_pieces(fen_to_wnn(f)[:64]), dtype=torch.long) for f in features_raw])
    color_features = torch.stack([torch.tensor(encode_colors(fen_to_wnn(f)[:64]), dtype=torch.long) for f in features_raw])
    # Turn to move encoding (character 65 of a WNN string gives turn to move)
    ttm_features = torch.stack([torch.tensor(encode_ttm(fen_to_wnn(f)[64]), dtype=torch.long) for f in features_raw])
    eval_labels = torch.tensor(evals_raw_deca, dtype=torch.float).unsqueeze(1)
    pred_labels = create_pred_labels(features_raw, preds_raw)
    torch.save({
        "piece_features": piece_features,
        "color_features": color_features,
        "ttm_features": ttm_features,
        "eval_labels": eval_labels,
        "pred_labels": pred_labels,
    }, dataset_path)
    print(f"Saved encoded tensors to {dataset_path}")

# Split into training and dev
n = int(TRAINING_SIZE * len(piece_features))
features_raw_tr, features_raw_dev = features_raw[:n], features_raw[n:]
piece_features_tr, piece_features_dev = piece_features[:n], piece_features[n:]
color_features_tr, color_features_dev = color_features[:n], color_features[n:]
ttm_features_tr, ttm_features_dev = ttm_features[:n], ttm_features[n:]
eval_labels_tr, eval_labels_dev = eval_labels[:n], eval_labels[n:]
pred_labels_tr, pred_labels_dev = pred_labels[:n], pred_labels[n:]

def test_loss(model, eval_weight):
    with torch.no_grad():
        pieces_batch, colors_batch, ttm_batch, eval_labels_batch, pred_labels_batch, index = get_batch(
            'train', BATCH_SIZE)
        loss_t = model(pieces_batch, colors_batch, ttm_batch, index=index, eval_weight=MAX_FINETUNE_EVAL_WEIGHT,
                       pred_targets=pred_labels_batch, eval_targets=eval_labels_batch, split='train')

        pieces_batch, colors_batch, ttm_batch, eval_labels_batch, pred_labels_batch, index = get_batch(
            'dev', BATCH_SIZE)
        loss_d = model(pieces_batch, colors_batch, ttm_batch, index=index, eval_weight=MAX_FINETUNE_EVAL_WEIGHT,
                       pred_targets=pred_labels_batch, eval_targets=eval_labels_batch, split='dev')

        print(f"Training loss (eval_weight={eval_weight}): {loss_t.item()}")
        print(f"Dev loss (eval_weight={eval_weight}): {loss_d.item()}")

def graph(log, bucket_size, zero_y_axis=False, graph_type="loss", model_title="unknown", save_dir="../graphs"):
    log_tensor = torch.tensor(log)
    num_steps = len(log_tensor)
    trunc_len = (num_steps // bucket_size) * bucket_size
    log_tensor = log_tensor[:trunc_len]
    loss_mean = log_tensor.view(-1, bucket_size).mean(1)
    steps = torch.arange(len(loss_mean)) * bucket_size

    plt.plot(steps, loss_mean)
    plt.xlabel("Training step")
    plt.ylabel(graph_type.capitalize())
    plt.title(f"{graph_type.capitalize()} benchmark graph for model \"{model_title}\"")
    if zero_y_axis:
        plt.ylim(bottom=0)

    # format x-axis in k notation
    def k_formatter(x, pos):
        if pos % 2 == 1:
            return ""
        if x >= 1000:
            return f"{int(x / 1000)}k"
        return str(int(x))

    plt.gca().xaxis.set_major_formatter(FuncFormatter(k_formatter))

    save_path = os.path.join(save_dir, f"{graph_type}_{model_title}.png")
    plt.savefig(save_path)
    plt.close()

def encode_board(board):
    piece_to_pi_pychess = {
        None: 0,  # empty
        chess.PAWN: 1, chess.KNIGHT: 2, chess.BISHOP: 3,
        chess.ROOK: 4, chess.QUEEN: 5, chess.KING: 6,
    }
    piece_to_ci_pychess = {
        None: 0,  # empty
        chess.WHITE: 1,
        chess.BLACK: 2,
    }

    pieces_enc = []
    colors_enc = []

    for rank in range(7, -1, -1):  # 7 down to 0
        for file in range(8):  # 0 up to 7
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is None:
                pieces_enc.append(piece_to_pi_pychess[None])
                colors_enc.append(piece_to_ci_pychess[None])
            else:
                pieces_enc.append(piece_to_pi_pychess[piece.piece_type])
                colors_enc.append(piece_to_ci_pychess[piece.color])

    ttm_enc = [0] if board.turn == chess.WHITE else [1]

    pieces = torch.tensor(pieces_enc, dtype=torch.long, device=DEVICE)
    colors = torch.tensor(colors_enc, dtype=torch.long, device=DEVICE)
    ttm = torch.tensor(ttm_enc, dtype=torch.long, device=DEVICE)
    return pieces, colors, ttm

def eval_board(board, model):
    pieces, colors, ttm = encode_board(board)

    with torch.no_grad():
        evaluation, pred_probs = model(pieces, colors, ttm, board=board, return_preds=True)
        preds = decode_all_predictions(pred_probs, [board])
        return 10 * evaluation.item(), preds

def get_batch(split, size, return_fens=False):
    if split == 'train':
        pf, cf, tf, el, pl, fr = piece_features_tr, color_features_tr, ttm_features_tr, eval_labels_tr, pred_labels_tr, features_raw_tr
    else:
        pf, cf, tf, el, pl, fr = piece_features_dev, color_features_dev, ttm_features_dev, eval_labels_dev, pred_labels_dev, features_raw_dev

    index = torch.randint(len(pf), (size,))
    pieces_batch = pf[index].to(DEVICE, non_blocking=True)
    colors_batch = cf[index].to(DEVICE, non_blocking=True)
    ttm_batch    = tf[index].to(DEVICE, non_blocking=True)
    eval_labels_batch = el[index].to(DEVICE, non_blocking=True)
    pred_labels_batch = pl[index].to(DEVICE, non_blocking=True)

    if return_fens:
        batch_fens = [fr[i.item()] for i in index]
        return pieces_batch, colors_batch, ttm_batch, eval_labels_batch, pred_labels_batch, index, batch_fens
    return pieces_batch, colors_batch, ttm_batch, eval_labels_batch, pred_labels_batch, index

def get_boards_for_lmm(index, split):
    features_raw_x = features_raw_tr if split == 'train' else features_raw_dev
    return [chess.Board(features_raw_x[i]) for i in index.tolist()]

def get_legal_move_mask(boards):
    mask = torch.zeros((len(boards), NUM_PLANES, 8, 8), dtype=torch.bool, device=DEVICE)
    for i, board in enumerate(boards):
        white_to_move = board.turn == chess.WHITE
        for move in board.legal_moves:
            plane = move_to_plane(move, white_to_move)
            from_sq = move.from_square
            fx, fy = chess.square_file(from_sq), chess.square_rank(from_sq)
            mask[i, plane, fy, fx] = True
    return mask

def compute_material_tensor(index, split, board, batch_size):
    if board is None:
        batch_boards = get_boards_for_lmm(index, split)
        mats = [fen_material_balance(board=batch_board) for batch_board in batch_boards]
    else:
        mats = [fen_material_balance(board=board)]
    mat_tensor = torch.tensor(mats, dtype=torch.float32, device=DEVICE).view(batch_size, 1)
    return mat_tensor / 10.0

def train(model, num_steps, model_name='taparoni', cosine_weighting=True):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_log = []
    weight_log = []
    for i in range(1, num_steps + 1):
        # Forward pass
        pieces_batch, colors_batch, ttm_batch, eval_labels_batch, pred_labels_batch, index = get_batch('train', BATCH_SIZE)
        progress = i / num_steps
        current_eval_weight = 0 if model_name == 'pretrain' else MAX_FINETUNE_EVAL_WEIGHT * (0.5 - 0.5 * math.cos(math.pi * progress))
        if not cosine_weighting:
            current_eval_weight = MAX_FINETUNE_EVAL_WEIGHT
        loss = model(pieces_batch, colors_batch, ttm_batch, index=index, eval_weight=current_eval_weight,
                     pred_targets=pred_labels_batch, eval_targets=eval_labels_batch, split='train')
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # Log
        loss_log.append(loss.item())
        weight_log.append(current_eval_weight)
        if i % LOSS_BENCH == 0:
            recent_losses = loss_log[-LOSS_BENCH:]
            approx_loss = sum(recent_losses) / len(recent_losses)
            print(f"{i} / {num_steps}: bucket loss={approx_loss:.4f}")
        # Training save checkpoints
        if i % TRN_SC == 0:
            save_model_weights(model)

    model.eval()
    print(f"model.{model_name} training finished!")
    return weight_log, loss_log

def load_saved_weights(path="../data/saved_weights.pt"):
    if not os.path.exists(path):
        raise RuntimeError(f"Path {path} doesn't exist!")
    m_eval = TwoHeadTransformer()
    m_eval.load_state_dict(torch.load(path))
    m_eval.to(DEVICE)
    m_eval.eval()
    return m_eval

def save_model_weights(model, path="../data/saved_weights.pt"):
    torch.save(model.state_dict(), path)
    print("Weights successfully saved!")

def train_new_model():
    model = TwoHeadTransformer().to(DEVICE)
    weight_log, finetune_log = train(model, NUM_STEPS_FINETUNE, model_name='finetune')
    graph(finetune_log, LOSS_BENCH, zero_y_axis=True, graph_type='loss', model_title="finetune")
    graph(weight_log, LOSS_BENCH, graph_type='weight', model_title='finetune')
    print("All training finished!")

    save_model_weights(model)
    test_loss(model, MAX_FINETUNE_EVAL_WEIGHT)
    return model

def continue_training_old_model():
    model = load_old_model()
    weight_log, finetune_log = train(model, NUM_STEPS_FINETUNE, model_name='finetune', cosine_weighting=False)
    graph(finetune_log, LOSS_BENCH, zero_y_axis=True, graph_type='loss', model_title="finetune")
    graph(weight_log, LOSS_BENCH, graph_type='weight', model_title='finetune')
    print("All training finished!")

    save_model_weights(model)
    test_loss(model, MAX_FINETUNE_EVAL_WEIGHT)
    return model

def load_old_model():
    m_eval = load_saved_weights()
    test_loss(m_eval, MAX_FINETUNE_EVAL_WEIGHT)
    return m_eval

# --------------- Transformer ---------------
if __name__ == '__main__':
    m = load_old_model()