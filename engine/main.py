import chess

from parse_data import *
from model import TwoHeadTransformer
import matplotlib.pyplot as plt
import random
import math
from hyperparams import *

# Seed for consistent random numbers
SEED = 8675309
random.seed(SEED)
torch.manual_seed(SEED)

# Open dataset
features_raw, evals_raw, preds_raw = get_dataset()
print(f"Dataset size: {len(features_raw):,} FENs")

# Convert centipawns into deca-pawns? (centipawns / 1000)
evals_raw_deca = [x / 1000 for x in evals_raw]

# --------------------- Pretraining Tokenization --------------------------- #
with open("../ucis.txt", "r") as f:
    ucis_raw = [line.strip() for line in f if line.strip()]

vocab_size = len(ucis_raw)
move_to_i = {move: i for i, move in enumerate(ucis_raw)}
i_to_move = {i: move for i, move in enumerate(ucis_raw)}
decode_moves = lambda s: [i_to_move.get(i) for i in s.tolist()]
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
    to_pred_labels = saved["to_pred_labels"]
    from_pred_labels = saved["from_pred_labels"]
    print(f"Loaded encoded tensors from {dataset_path}")
else:
    def create_pred_labels(raw_preds):
        from_labels, to_labels = [], []
        for raw_pred in raw_preds:
            from_label = torch.zeros(64)
            to_label = torch.zeros(64)
            move = chess.Move.from_uci(raw_pred)
            from_sq, to_sq = move.from_square, move.to_square
            from_label[from_sq] = 1
            to_label[to_sq] = 1
            from_labels.append(from_label)
            to_labels.append(to_label)
        from_labels_t = torch.stack(from_labels, dim=0)
        to_labels_t = torch.stack(to_labels, dim=0)
        return to_labels_t, from_labels_t
    # Only encode first 64 characters of WNN that represent the board into piece and color features

    piece_features = torch.stack([torch.tensor(encode_pieces(fen_to_wnn(f)[:64]), dtype=torch.long) for f in features_raw])
    color_features = torch.stack([torch.tensor(encode_colors(fen_to_wnn(f)[:64]), dtype=torch.long) for f in features_raw])
    # Turn to move encoding (character 65 of a WNN string gives turn to move)
    ttm_features = torch.stack([torch.tensor(encode_ttm(fen_to_wnn(f)[64]), dtype=torch.long) for f in features_raw])
    eval_labels = torch.tensor(evals_raw_deca, dtype=torch.float).unsqueeze(1)
    to_pred_labels, from_pred_labels = create_pred_labels(preds_raw)
    torch.save({
        "piece_features": piece_features,
        "color_features": color_features,
        "ttm_features": ttm_features,
        "eval_labels": eval_labels,
        "to_pred_labels": to_pred_labels,
        "from_pred_labels": from_pred_labels
    }, dataset_path)
    print(f"Saved encoded tensors to {dataset_path}")

# Split into training and dev
n = int(TRAINING_SIZE * len(piece_features))
features_raw_tr, features_raw_dev = features_raw[:n], features_raw[n:]
piece_features_tr, piece_features_dev = piece_features[:n], piece_features[n:]
color_features_tr, color_features_dev = color_features[:n], color_features[n:]
ttm_features_tr, ttm_features_dev = ttm_features[:n], ttm_features[n:]
eval_labels_tr, eval_labels_dev = eval_labels[:n], eval_labels[n:]
from_pred_labels_tr, from_pred_labels_dev = from_pred_labels[:n], from_pred_labels[n:]
to_pred_labels_tr, to_pred_labels_dev = to_pred_labels[:n], to_pred_labels[n:]


def test_loss(model, eval_weight):
    with torch.no_grad():
        pieces_batch, colors_batch, ttm_batch, eval_labels_batch, to_pred_labels_batch, from_pred_labels_batch, index = get_batch(
            'train', BATCH_SIZE)
        loss_t = model(pieces_batch, colors_batch, ttm_batch, index=index, eval_weight=MAX_FINETUNE_EVAL_WEIGHT,
                       pred_targets_to=to_pred_labels_batch, pred_targets_from=from_pred_labels_batch,
                       eval_targets=eval_labels_batch, split='train')

        pieces_batch, colors_batch, ttm_batch, eval_labels_batch, to_pred_labels_batch, from_pred_labels_batch, index = get_batch(
            'dev', BATCH_SIZE)
        loss_d = model(pieces_batch, colors_batch, ttm_batch, index=index, eval_weight=MAX_FINETUNE_EVAL_WEIGHT,
                       pred_targets_to=to_pred_labels_batch, pred_targets_from=from_pred_labels_batch,
                       eval_targets=eval_labels_batch, split='dev')

        print(f"Training loss (eval_weight={eval_weight}): {loss_t.item()}")
        print(f"Dev loss (eval_weight={eval_weight}): {loss_d.item()}")

def graph(log, bucket_size, graph_type="loss", model_title="unknown", save_dir="../data/loss"):
    log_tensor = torch.tensor(log)
    num_steps = len(log_tensor)
    trunc_len = (num_steps // bucket_size) * bucket_size
    log_tensor = log_tensor[:trunc_len]
    loss_mean = log_tensor.view(-1, bucket_size).mean(1)

    plt.plot(loss_mean)
    plt.title(f"{graph_type.capitalize()} benchmark graph for model \"{model_title}\"")

    save_path = os.path.join(save_dir, f"{graph_type}_{model_title}.png")
    plt.savefig(save_path)
    plt.close()

def eval_fen(fen, model):
    model.eval()
    wnn = fen_to_wnn(fen)
    pieces = torch.tensor(encode_pieces(wnn[:64]), dtype=torch.long, device=DEVICE)
    colors = torch.tensor(encode_colors(wnn[:64]), dtype=torch.long, device=DEVICE)
    ttm = torch.tensor(encode_ttm(wnn[64]), dtype=torch.long, device=DEVICE)
    evaluation, pred_probs = model(pieces, colors, ttm, fen=fen, return_preds=True)
    return 10 * evaluation.item(), pred_probs.squeeze(0).tolist()

def get_batch(split, size, return_fens=False):
    piece_features_x =   piece_features_tr   if split == 'train' else piece_features_dev
    color_features_x =   color_features_tr   if split == 'train' else color_features_dev
    ttm_features_x =     ttm_features_tr     if split == 'train' else ttm_features_dev
    eval_labels_x  =     eval_labels_tr      if split == 'train' else eval_labels_tr
    to_pred_labels_x =   to_pred_labels_tr   if split == 'train' else to_pred_labels_dev
    from_pred_labels_x = from_pred_labels_tr if split == 'train' else from_pred_labels_dev
    features_raw_x =     features_raw_tr     if split == 'train' else features_raw_dev
    index = torch.randint(len(piece_features_x), (size,))

    pieces_batch = torch.stack([piece_features_x[i.item()] for i in index])
    colors_batch = torch.stack([color_features_x[i.item()] for i in index])
    ttm_batch = torch.stack([ttm_features_x[i.item()] for i in index])
    eval_labels_batch = torch.stack([eval_labels_x[i.item()] for i in index])
    to_pred_labels_batch = to_pred_labels_x[index]
    from_pred_labels_batch = from_pred_labels_x[index]

    pieces_batch, colors_batch, ttm_batch, eval_labels_batch, to_pred_labels_batch, from_pred_labels_batch = (
        pieces_batch.to(DEVICE), colors_batch.to(DEVICE), ttm_batch.to(DEVICE), eval_labels_batch.to(DEVICE),
        to_pred_labels_batch.to(DEVICE), from_pred_labels_batch.to(DEVICE)
    )

    if return_fens:
        batch_fens = [features_raw_x[i.item()] for i in index]
        return pieces_batch, colors_batch, ttm_batch, eval_labels_batch, to_pred_labels_batch, from_pred_labels_batch, index, batch_fens
    return pieces_batch, colors_batch, ttm_batch, eval_labels_batch, to_pred_labels_batch, from_pred_labels_batch, index

def get_fens_for_lmm(index, split):
    features_raw_x = features_raw_tr if split == 'train' else features_raw_dev
    return [features_raw_x[i] for i in index.tolist()]

def get_legal_move_masks(fens):
    to_masks, from_masks = [], []
    for fen in fens:
        board = chess.Board(fen)
        to_mask = torch.zeros(64, dtype=torch.bool, device=DEVICE)
        from_mask = torch.zeros(64, dtype=torch.bool, device=DEVICE)
        for move in board.legal_moves:
            to_mask[move.to_square] = True
            from_mask[move.from_square] = True
        to_masks.append(to_mask)
        from_masks.append(from_mask)
    to_masks, from_masks = torch.stack(to_masks, dim=0), torch.stack(from_masks, dim=0)
    return to_masks, from_masks

def get_pred_target_raw(i):
    def move_from_fens(fen_a, fen_b):
        board = chess.Board(fen_a)
        target = chess.Board(fen_b)

        for legal_move in board.legal_moves:
            board_copy = board.copy()
            board_copy.push(legal_move)
            if board_copy.board_fen() == target.board_fen() and board_copy.turn == target.turn:
                return legal_move
        return None

    fen1 = wnn_to_fen(features_raw_tr[i])
    if i + 1 == len(features_raw_tr):
        return None
    fen2 = wnn_to_fen(features_raw_tr[i + 1])
    move = move_from_fens(fen1, fen2)
    if move is None:
        return None
    return move.uci()

def train(model, num_steps, model_name='taparoni'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_log = []
    weight_log = []
    for i in range(1, num_steps + 1):
        # Forward pass
        pieces_batch, colors_batch, ttm_batch, eval_labels_batch, to_pred_labels_batch, from_pred_labels_batch, index = get_batch('train', BATCH_SIZE)
        progress = i / num_steps
        current_eval_weight = MAX_FINETUNE_EVAL_WEIGHT * (0.5 - 0.5 * math.cos(math.pi * progress))
        loss = model(pieces_batch, colors_batch, ttm_batch, index=index, eval_weight=current_eval_weight,
                     pred_targets_to=to_pred_labels_batch, pred_targets_from=from_pred_labels_batch,
                     eval_targets=eval_labels_batch, split='train')
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

def train_new_model():
    model = TwoHeadTransformer().to(DEVICE)
    _, pretrain_log = train(model, NUM_STEPS_PRETRAIN, model_name='pretrain')
    graph(pretrain_log, LOSS_BENCH, graph_type='loss', model_title="pretrain")
    weight_log, finetune_log = train(model, NUM_STEPS_FINETUNE, model_name='finetune')
    graph(finetune_log, LOSS_BENCH, graph_type='loss', model_title="finetune")
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
    m = train_new_model()