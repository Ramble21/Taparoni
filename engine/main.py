from parse_data import *
from model import TransformerBody, TransformerEval, TransformerPretrain
import matplotlib.pyplot as plt
import random
from hyperparams import *

# Seed for consistent random numbers
SEED = 8675309
random.seed(SEED)
torch.manual_seed(SEED)

# Open dataset
features_raw, labels_raw = get_dataset()
print(f"Dataset size: {len(features_raw):,} FENs")

# Convert centipawns into deca-pawns? (centipawns / 1000)
labels_raw_deca = [x / 1000 for x in labels_raw]

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
    labels = saved["labels"]
    print(f"Loaded encoded tensors from {dataset_path}")
else:
    # Only encode first 64 characters of WNN that represent the board into piece and color features
    piece_features = torch.stack([torch.tensor(encode_pieces(f[:64]), dtype=torch.long) for f in features_raw])
    color_features = torch.stack([torch.tensor(encode_colors(f[:64]), dtype=torch.long) for f in features_raw])
    # Turn to move encoding (character 65 of a WNN string gives turn to move)
    ttm_features = torch.stack([torch.tensor(encode_ttm(f[65]), dtype=torch.long) for f in features_raw])
    labels = torch.tensor(labels_raw_deca, dtype=torch.float).unsqueeze(1)
    torch.save({
        "piece_features": piece_features,
        "color_features": color_features,
        "ttm_features": ttm_features,
        "labels": labels,
    }, dataset_path)
    print(f"Saved encoded tensors to {dataset_path}")

# Split into training and dev
n = int(TRAINING_SIZE * len(piece_features))
features_raw_tr, features_raw_dev = features_raw[:n], features_raw[n:]
piece_features_tr, piece_features_dev = piece_features[:n], piece_features[n:]
color_features_tr, color_features_dev = color_features[:n], color_features[n:]
ttm_features_tr, ttm_features_dev = ttm_features[:n], ttm_features[n:]
labels_tr, labels_dev = labels[:n], labels[n:]

def get_batch(split, size, return_wnns=False, return_index=False):
    piece_features_x = piece_features_tr if split == 'train' else piece_features_dev
    color_features_x = color_features_tr if split == 'train' else color_features_dev
    ttm_features_x =   ttm_features_tr   if split == 'train' else ttm_features_dev
    labels_x =         labels_tr         if split == 'train' else labels_dev
    features_raw_x =   features_raw_tr   if split == 'train' else features_raw_dev

    index = torch.randint(len(piece_features_x), (size,))
    pieces_batch = torch.stack([piece_features_x[i.item()] for i in index])
    colors_batch = torch.stack([color_features_x[i.item()] for i in index])
    ttm_batch = torch.stack([ttm_features_x[i.item()] for i in index])
    labels_batch = torch.stack([labels_x[i.item()] for i in index])

    pieces_batch, colors_batch, ttm_batch, labels_batch = pieces_batch.to(DEVICE), colors_batch.to(
        DEVICE), ttm_batch.to(DEVICE), labels_batch.to(DEVICE)

    if return_wnns:
        batch_wnns = [features_raw_x[i.item()] for i in index]
        return pieces_batch, colors_batch, ttm_batch, labels_batch, batch_wnns
    elif return_index:
        return pieces_batch, colors_batch, ttm_batch, labels_batch, index
    return pieces_batch, colors_batch, ttm_batch, labels_batch

def get_legal_move_mask(index):
    wnns = [features_raw_tr[i] for i in index.tolist()]
    fens = [wnn_to_fen(wnn) for wnn in wnns]
    masks = []

    for i in range(len(fens)):
        fen = fens[i]
        board = chess.Board(fen)
        ucis = [move.uci() for move in board.legal_moves]
        encoded = [move_to_i.get(uci) for uci in ucis]
        mask = torch.full((vocab_size,), False, device=DEVICE)
        mask[encoded] = True
        masks.append(mask)
    result = torch.stack(masks, dim=0)  # (B, vocab_size)
    return result

def test_loss(model):
    p_t, c_t, ttm_t, labels_t = get_batch('train', BATCH_SIZE)
    preds_t, loss_t = model(p_t, c_t, ttm_t, targets=labels_t)
    p_d, c_d, ttm_d, labels_d = get_batch('dev', BATCH_SIZE)
    preds_d, loss_d = model(p_d, c_d, ttm_d, targets=labels_d)
    print("Training loss: ", loss_t.item())
    print("Dev loss: ", loss_d.item())


def random_sample(model, num_samples):
    from versus import get_next_move
    for i in range(num_samples):
        p, c, ttm, l, wnns = get_batch('dev', 1, return_wnns=True)
        preds, loss = model(p, c, ttm, l)
        wnn = wnns[0]
        print()
        print("Sample", i)
        print(f"WNN: {wnn}")
        print(f"FEN: {wnn_to_fen(wnn)}")
        print(f"Prediction (pawns): {preds.view(-1).item() * 10}")
        print(f"Actual: {l.view(-1).item() * 10}")
        print(f"Next FEN d=1: {get_next_move(wnn_to_fen(wnn), model, 1)}")
        print(f"Next FEN d=2: {get_next_move(wnn_to_fen(wnn), model, 2)}")


def graph_loss(l_log, bucket_size, model_title="unknown", save_dir="../data/loss"):
    l_log_tensor = torch.tensor(l_log)
    num_steps = len(l_log_tensor)
    trunc_len = (num_steps // bucket_size) * bucket_size
    l_log_tensor = l_log_tensor[:trunc_len]
    loss_mean = l_log_tensor.view(-1, bucket_size).mean(1)

    plt.plot(loss_mean)
    plt.title(f"Loss benchmark graph for model \"{model_title}\"")

    save_path = os.path.join(save_dir, f"loss_{model_title}.png")
    plt.savefig(save_path)
    plt.close()

def eval_fen(fen, model):
    model.eval()
    wnn = fen_to_wnn(fen)
    pieces = torch.tensor(encode_pieces(wnn[:64]), dtype=torch.long, device=DEVICE)
    colors = torch.tensor(encode_colors(wnn[:64]), dtype=torch.long, device=DEVICE)
    ttm = torch.tensor(encode_ttm(wnn[64]), dtype=torch.long, device=DEVICE)
    preds, loss = model(pieces, colors, ttm)
    return 10 * preds.item()

def get_pretrain_target(i):
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
    return move_to_i.get(move.uci())

def train(model, num_steps, model_type):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_log = []
    for i in range(1, num_steps + 1):
        pieces_batch, colors_batch, ttm_batch, labels_batch, index = get_batch('train', BATCH_SIZE, return_index=True)

        # Forward pass
        if model_type == 'pretrain':
            index_l = index.tolist()
            pretrain_targets = [get_pretrain_target(i) for i in index_l]

            # Remove None labels and their corresponding targets (positions with no next move)
            keep_indices = [i for i, t in enumerate(pretrain_targets) if t is not None]
            index_tensor = torch.tensor(keep_indices, device=DEVICE, dtype=torch.long)
            index_f = torch.tensor([index_l[i] for i in range(len(index_l)) if i in keep_indices])
            index_f.to(DEVICE)

            pretrain_targets_f = torch.tensor([pretrain_targets[i] for i in keep_indices]).to(DEVICE)
            pieces_batch_f = pieces_batch.index_select(0, index_tensor).to(DEVICE)
            colors_batch_f = colors_batch.index_select(0, index_tensor).to(DEVICE)
            ttm_batch_f = ttm_batch.index_select(0, index_tensor).to(DEVICE)

            if len(pretrain_targets_f) == 0:
                preds, loss = None, None
            else:
                preds, loss = model(pieces_batch_f, colors_batch_f, ttm_batch_f, index=index_f,
                                    targets=pretrain_targets_f)
        else:
            preds, loss = model(pieces_batch, colors_batch, ttm_batch, targets=labels_batch)

        if loss is not None:
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # Log
            loss_log.append(loss.item())
            if i % LOSS_BENCH == 0:
                recent_losses = loss_log[-LOSS_BENCH:]
                approx_loss = sum(recent_losses) / len(recent_losses)
                print(f"{i} / {num_steps}: bucket loss={approx_loss:.4f}")

    model.eval()
    print(f"model.{model_type} training finished!")
    return loss_log

def load_saved_weights(path="../data/saved_weights.pt"):
    if not os.path.exists(path):
        raise RuntimeError(f"Path {path} doesn't exist!")
    m_eval = TransformerEval()
    m_eval.load_state_dict(torch.load(path))
    m_eval.to(DEVICE)
    m_eval.eval()
    return m_eval

def save_model_weights(model, path="../data/saved_weights.pt"):
    torch.save(model.state_dict(), path)

def train_new_model():
    m_body = TransformerBody().to(DEVICE)
    m_pretrain = TransformerPretrain(m_body, vocab_size).to(DEVICE)
    m_eval = TransformerEval(m_body).to(DEVICE)

    pretrain_log = train(m_pretrain, NUM_STEPS_PRETRAIN, 'pretrain')
    graph_loss(pretrain_log, LOSS_BENCH, model_title="pretrain")
    finetune_log = train(m_eval, NUM_STEPS_FINETUNE, 'finetune')
    graph_loss(finetune_log, LOSS_BENCH, model_title="finetune")
    print("All training finished!")

    save_model_weights(m_eval)
    test_loss(m_eval)
    return m_eval

def load_old_model():
    m_eval = load_saved_weights()
    test_loss(m_eval)
    print()
    return m_eval

# --------------- Transformer ---------------
if __name__ == '__main__':
    m = load_old_model()
    random_sample(m, 5)