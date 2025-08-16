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

# Character-level tokenization
chars = sorted(list({c for f in features_raw for c in f}))
vocab_size = len(chars)
s_to_i = {ch: i for i, ch in enumerate(chars)}
i_to_s = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [s_to_i[c] for c in s]
decode = lambda l: ''.join([i_to_s[i] for i in l])

# Convert centipawns into deca-pawns? (centipawns / 1000)
labels_raw_deca = [x / 1000 for x in labels_raw]

features = torch.stack([torch.tensor(encode(f), dtype=torch.long) for f in features_raw])
labels = torch.tensor(labels_raw_deca, dtype=torch.float).unsqueeze(1)

# 90% training, 10% dev
n = int(0.9*len(features))
features_tr =  features[:n]
features_dev = features[n:]
labels_tr = labels[:n]
labels_dev = labels[n:]

def get_batch(split, size):
  features_x = features_tr if split == 'train' else features_dev
  labels_x = labels_tr if split == 'train' else labels_dev
  index = torch.randint(len(features_x), (size, ))
  x = torch.stack([features_x[i] for i in index])
  y = torch.stack([labels_x[i] for i in index])
  x, y = x.to(DEVICE), y.to(DEVICE)
  return x, y

def test_loss(model):
  X_t, Y_t = get_batch('train', BATCH_SIZE)
  preds_t, loss_t = model(X_t, Y_t)
  X_d, Y_d = get_batch('dev', BATCH_SIZE)
  preds_d, loss_d = model(X_d, Y_d)
  print("Training loss: ", loss_t.item())
  print("Dev loss: ", loss_d.item())

def random_sample(m, num_samples):
  for i in range(num_samples):

    X_b, Y_b = get_batch('dev', 1)
    preds, loss = m(X_b, Y_b)
    wnn = decode(X_b.view(-1).tolist())
    print()
    print("Sample", i)
    print(f"WNN: {wnn}")
    print(f"FEN: {wnn_to_fen(wnn)}")
    print(f"Prediction (pawns): {preds.view(-1).item() * 10}")
    print(f"Actual: {Y_b.view(-1).item() * 10}")
    print(f"Next FEN d=1: {get_next_move(m, wnn_to_fen(wnn), 1)}")
    print(f"Next FEN d=2: {get_next_move(m, wnn_to_fen(wnn), 2)}")

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
  batch = torch.tensor(encode(wnn), dtype=torch.long)
  batch = batch.view(1, -1)
  preds, loss = model(batch)
  return 10 * preds.item()

def train(model):
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
  loss_log = []
  for i in range(1, NUM_STEPS + 1):
    X_b, Y_b = get_batch('train', BATCH_SIZE)
    # Forward pass
    preds, loss = model(X_b, Y_b)
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

def get_next_move(model, fen, depth):
  legal_moves = get_legal_moves_as_fens(fen)
  if not legal_moves:
    return "NO LEGAL MOVES", 0.0
  white_to_move = fen.split()[1] == 'w'
  evals = [eval_fen(f, model) for f in legal_moves]
  zipped = zip(legal_moves, evals)
  if depth == 1:
    best_fen, best_eval = max(zipped, key=lambda x: x[1]) if white_to_move else min(zipped, key=lambda x: x[1])
  else:
    scored_moves = []
    for move in legal_moves:
      _, score, _ = get_next_move(model, move, depth - 1)
      scored_moves.append((move, score))
    best_fen, best_eval = max(scored_moves, key=lambda x: x[1]) if white_to_move else min(scored_moves, key=lambda x: x[1])

  # Convert FENs to a readable move for debugging
  board = chess.Board(fen)
  move_readable = "UNKNOWN"
  for m in board.legal_moves:
    board.push(m)
    if board.fen() == best_fen:
      move_readable = board.uci(m)
      board.pop()
      break
    board.pop()

  return best_fen, best_eval, move_readable

def get_legal_moves_as_fens(fen):
  board = chess.Board(fen)
  moves = list(board.legal_moves)
  fens = []
  for move in moves:
    board.push(move)
    fens.append(board.fen())
    board.pop()
  return fens

# --------------------- Transformer ----------------------------

def new_model():
  m = load_new_model()
  log = train(m)
  test_loss(m)
  save_model_weights(m)
  graph_loss(log, LOSS_BENCH)
  random_sample(m, NUM_SAMPLES)
  return m

def load_old_model():
  m = load_saved_weights()
  test_loss(m)
  random_sample(m, NUM_SAMPLES)
  return m

load_old_model()