from parse_data import *
from model import Transformer
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

# Character-level tokenization
chars = sorted(list({c for f in features_raw for c in f}))
vocab_size = len(chars)
s_to_i = {ch: i for i, ch in enumerate(chars)}
i_to_s = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join([i_to_s[i] for i in l])

# Piece tokenization
piece_to_pi = {
    '.': 0,                                         # empty square
    'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6, # white pieces
    'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6  # black pieces
}
encode_pieces = lambda s: [piece_to_pi.get(p) for p in s]
# Color tokenization
piece_to_ci = {
    '.': 0,                                         # empty square
    'P': 1, 'N': 1, 'B': 1, 'R': 1, 'Q': 1, 'K': 1, # white pieces
    'p': 2, 'n': 2, 'b': 2, 'r': 2, 'q': 2, 'k': 2  # black pieces
}
encode_colors = lambda s: [piece_to_ci.get(p) for p in s]
# Turn to move tokenization
encode_ttm = lambda c: [0] if c == 'w' else [1]

# Only encode first 64 characters of WNN that represent the board into piece and color features
piece_features = torch.stack([torch.tensor(encode_pieces(f[:64]), dtype=torch.long) for f in features_raw])
color_features = torch.stack([torch.tensor(encode_colors(f[:64]), dtype=torch.long) for f in features_raw])
# Turn to move encoding (character 65 of a WNN string gives turn to move)
ttm_features = torch.stack([torch.tensor(encode_ttm(f[65]), dtype=torch.long) for f in features_raw])


labels = torch.tensor(labels_raw_deca, dtype=torch.float).unsqueeze(1)

# Split into training and dev
n = int(TRAINING_SIZE * len(piece_features))
features_raw_tr, features_raw_dev = features_raw[:n], features_raw[n:]
piece_features_tr, piece_features_dev = piece_features[:n], piece_features[n:]
color_features_tr, color_features_dev = color_features[:n], color_features[n:]
ttm_features_tr, ttm_features_dev = ttm_features[:n], ttm_features[n:]
labels_tr, labels_dev = labels[:n], labels[n:]

def get_batch(split, size, return_wnns=False):
  piece_features_x = piece_features_tr if split == 'train' else piece_features_dev
  color_features_x = color_features_tr if split == 'train' else color_features_dev
  ttm_features_x = ttm_features_tr if split == 'train' else ttm_features_dev
  labels_x = labels_tr if split == 'train' else labels_dev

  index = torch.randint(len(piece_features_x), (size, ))
  pieces_batch = torch.stack([piece_features_x[i] for i in index])
  colors_batch = torch.stack([color_features_x[i] for i in index])
  ttm_batch = torch.stack([ttm_features_x[i] for i in index])
  labels_batch = torch.stack([labels_x[i] for i in index])
  pieces_batch, colors_batch, ttm_batch, labels_batch = pieces_batch.to(DEVICE), colors_batch.to(DEVICE), ttm_batch.to(DEVICE), labels_batch.to(DEVICE)

  if return_wnns:
    features_raw_x = features_raw_tr if split == 'train' else features_raw_dev
    batch_wnns = torch.stack([features_raw_x[i] for i in index])
    batch_wnns.to(DEVICE)
    return pieces_batch, colors_batch, ttm_batch, labels_batch, batch_wnns

  return pieces_batch, colors_batch, ttm_batch, labels_batch

def test_loss(model):
  p_t, c_t, ttm_t, labels_t = get_batch('train', BATCH_SIZE)
  preds_t, loss_t = model(p_t, c_t, ttm_t, targets=labels_t)
  p_d, c_d, ttm_d, labels_d = get_batch('dev', BATCH_SIZE)
  preds_d, loss_d = model(p_d, c_d, ttm_d, targets=labels_d)
  print("Training loss: ", loss_t.item())
  print("Dev loss: ", loss_d.item())

def random_sample(m, num_samples):
  for i in range(num_samples):
    p, c, ttm, l, wnns = get_batch('dev', 1, return_wnns=True)
    preds, loss = m(p, c, ttm, l)
    wnn = wnns.item()
    print()
    print("Sample", i)
    print(f"WNN: {wnn}")
    print(f"FEN: {wnn_to_fen(wnn)}")
    print(f"Prediction (pawns): {preds.view(-1).item() * 10}")
    print(f"Actual: {l.view(-1).item() * 10}")
    print(f"Next FEN d=1: {get_next_move(wnn_to_fen(wnn), m, 1)}")
    print(f"Next FEN d=2: {get_next_move(wnn_to_fen(wnn), m, 2)}")

def graph_loss(l_log, bucket_size):
  mean_calc = NUM_STEPS // bucket_size
  loss_mean = torch.tensor(l_log).view(-1, mean_calc).mean(1)
  plt.plot(loss_mean)
  plt.show()

def load_new_model():
  model = Transformer(vocab_size)
  model.to(DEVICE)
  return model

def load_saved_weights():
  model = Transformer(vocab_size)
  model.load_state_dict(torch.load("../data/saved_weights.pt"))
  model.to(DEVICE)
  model.eval()
  return model

def save_model_weights(model):
  torch.save(model.state_dict(), "../data/saved_weights.pt")

def eval_fen(fen, model):
  model.eval()
  wnn = fen_to_wnn(fen)
  pieces = torch.tensor(encode_pieces(wnn[:64]), dtype=torch.long)
  colors = torch.tensor(encode_colors(wnn[:64]), dtype=torch.long)
  ttm = torch.tensor(encode_ttm(wnn[64]), dtype=torch.long)
  preds, loss = model(pieces, colors, ttm)
  return 10 * preds.item()

def train(model):
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
  loss_log = []
  for i in range(1, NUM_STEPS + 1):
    pieces_batch, colors_batch, ttm_batch, labels_batch = get_batch('train', BATCH_SIZE)
    # Forward pass
    preds, loss = model(pieces_batch, colors_batch, ttm_batch, targets=labels_batch)
    # Backward pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # Log
    loss_log.append(loss.item())
    if i % LOSS_BENCH == 0:
        recent_losses = loss_log[-LOSS_BENCH:]
        approx_loss = sum(recent_losses) / len(recent_losses)
        print(f"{i} / {NUM_STEPS}: approx. loss={approx_loss:.4f}")
  model.eval()
  print("Training finished!")
  return loss_log

# --------------------Next Move Prediction---------------------- #

def get_next_move(fen, model, depth):
  legal_moves, legal_fens = get_legal_moves(fen)
  white_to_move = fen.split()[1] == 'w'
  if not legal_moves:
    raise RuntimeError("No legal moves!")

  if white_to_move:
    best_score = float('-inf')
    best_move, best_fen = None, None
    for m,f in zip(legal_moves, legal_fens):
      score = minimax(f, model, depth-1, float('-inf'), float('inf'))
      if score > best_score:
        best_score = score
        best_move, best_fen = m, f
    return best_move, best_fen
  else:
    best_score = float('inf')
    best_move, best_fen = None, None
    for m,f in zip(legal_moves, legal_fens):
      score = minimax(f, model, depth-1, float('-inf'), float('inf'))
      if score < best_score:
        best_score = score
        best_move, best_fen = m, f
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

# --------------------- Transformer --------------------------- #

def new_model():
  m = load_new_model()
  log = train(m)
  test_loss(m)
  save_model_weights(m)
  graph_loss(log, LOSS_BENCH)
  return m

def load_old_model():
  m = load_saved_weights()
  test_loss(m)
  print()
  return m

load_old_model()