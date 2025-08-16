from parse_data import get_dataset, wnn_to_fen
from model import Transformer
import matplotlib.pyplot as plt
import random

# Hyperparameters
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

def sample(m, num_samples):
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

def train(model):
  # Pytorch optimizer object, used for training models in production
  # torch.optim.SGD represents the classic stochastic gradient descent optimization, but AdamW is more advanced and popular
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
  # Train our model
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

# ---------------- Transformer ---------------------------------
def new_model():
  m = load_new_model()
  log = train(m)
  test_loss(m)
  save_model_weights(m)
  graph_loss(log, LOSS_BENCH)
  sample(m, NUM_SAMPLES)

def load_old_model():
  m = load_saved_weights()
  test_loss(m)
  sample(m, NUM_SAMPLES)
